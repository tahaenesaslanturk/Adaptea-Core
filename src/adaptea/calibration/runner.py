from __future__ import annotations

import asyncio
import csv
import html
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptea.calibration.aggregate import aggregate_samples, recommend
from adaptea.calibration.profile import build_profile
from adaptea.calibration.workloads import Workload, workloads
from adaptea.config import Config
from adaptea.inference import InferenceBackend, configured_model
from adaptea.lmstudio.client import select_loaded_model
from adaptea.lmstudio.models import LMModel
from adaptea.lmstudio.telemetry import poll_ps
from adaptea.models import TelemetrySample, utc_now


def interleaved_order(candidates: list[int], repetitions: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    base = sorted(set(candidates))
    order: list[int] = []
    previous: list[int] | None = None
    for _ in range(repetitions):
        row = base.copy()
        rng.shuffle(row)
        if row == previous and len(row) > 1:
            row = row[1:] + row[:1]
        order.extend(row)
        previous = row
    return order


def cap_candidates(candidates: list[int], parallel: int, ceiling: int | None = None) -> list[int]:
    """Keep every concurrency a run could actually ask for.

    ``parallel`` is what the backend serves without queueing. ``ceiling`` is the highest
    concurrency the scheduler is allowed to request — the user's ``worker.max_agents``.
    Measuring above ``parallel`` is the point of the sweep rather than a mistake: the only
    way to know whether eight agents beat four on this machine is to run eight and watch
    what queueing does to useful throughput. Dropping those rows is why a max_agents of 8
    was never measured.
    """
    limit = max(parallel, ceiling or 0)
    values = {value for value in candidates if 0 < value <= limit}
    if ceiling and 0 < ceiling <= limit:
        values.add(ceiling)
    return sorted(values)


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One model to measure, and every loaded instance requests may be sent to."""

    key: str
    destinations: tuple[str, ...]
    parallel_limit: int
    model: LMModel

    @property
    def capacity(self) -> int:
        """Requests this model serves concurrently across all of its instances."""
        return max(1, self.parallel_limit * max(1, len(self.destinations)))


def resolve_targets(
    models: list[LMModel],
    config: Config,
    fallback_parallel: Callable[[], int | None],
) -> list[CalibrationTarget]:
    """Every model the project actually uses, not just the first one that answered.

    A fleet is a set of models working together, so measuring one of them and calling the
    machine calibrated describes something the run never does.
    """
    ready = [model for model in models if model.ready and model.type == "llm"]
    if not ready:
        raise RuntimeError(
            "LM Studio is connected, but no LLM is loaded. Open Environment → "
            "Models and explicitly choose a downloaded model before calibration."
        )
    configured = configured_model(config)
    fleet_keys = (
        [item.model for item in config.fleet.models]
        if config.fleet.enabled and config.fleet.models
        else []
    )
    if fleet_keys:
        # Whatever of the fleet is up. Refusing until every configured model was loaded
        # assumed something would load the rest; loading is the user's own act now, so
        # that only meant no profile at all. The profile records which models it covers,
        # and the calibration state keeps saying "stale" until the others are measured.
        chosen = [model for model in ready if model.key in set(fleet_keys)]
        if not chosen:
            loaded = ", ".join(sorted({model.key for model in ready})) or "nothing"
            raise RuntimeError(
                f"None of the configured models ({', '.join(sorted(set(fleet_keys)))}) is "
                f"loaded in LM Studio (loaded: {loaded}). Load one in Environment → Models, "
                "or give a loaded model a job there, then measure again."
            )
        # Keep the configured order so the planner model leads the report.
        chosen.sort(key=lambda model: fleet_keys.index(model.key))
    elif configured:
        selected = select_loaded_model(ready, configured)
        # A pin naming a model nobody loaded describes a set this machine is not running.
        # Measuring what is actually up answers the question that was asked; the pin being
        # absent is reported by the setup checks, which is where it can be acted on.
        chosen = [selected] if selected else ready
    else:
        # Nothing is pinned, so every loaded LLM is part of what this machine runs.
        chosen = ready
    targets: list[CalibrationTarget] = []
    for model in chosen:
        # Asked for only when the model itself does not report a limit, so a backend
        # that cannot answer the question is never consulted needlessly.
        parallel = model.effective_parallel_limit or fallback_parallel()
        if parallel is None:
            raise RuntimeError(
                "LM Studio did not expose Max Concurrent Predictions for "
                f"{model.key}; calibration cannot infer it."
            )
        destinations = tuple(instance.id for instance in model.loaded_instances) or (model.key,)
        targets.append(
            CalibrationTarget(
                key=model.key,
                destinations=destinations,
                parallel_limit=parallel,
                model=model,
            )
        )
    return targets


class CalibrationRunner:
    def __init__(
        self,
        root: Path,
        config: Config,
        client: InferenceBackend,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.client = client
        self.progress = progress
        self.latest_queued: int | None = None

    def _report(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    async def run(
        self,
        candidates: list[int] | None = None,
        repetitions: int | None = None,
        *,
        quick: bool = False,
        max_agents: int | None = None,
    ) -> Path:
        models = await self.client.models()
        targets = resolve_targets(
            models,
            self.config,
            lambda: self.client.capabilities.safe_default_parallel_limit,
        )
        requested = candidates or self.config.calibration.concurrency
        # A run may be asked for max_agents workers, so max_agents is exactly the point the
        # profile has to have an opinion about. The desktop keeps its own agent limit, so
        # it passes the one the user actually set rather than the one in the TOML.
        ceiling = max_agents or self.config.worker.max_agents
        reps = repetitions or self.config.calibration.repetitions
        seed = self.config.calibration.seed
        fleet_capacity = sum(target.capacity for target in targets)
        combined_candidates = (
            cap_candidates(requested, fleet_capacity, ceiling) if len(targets) > 1 else []
        )
        calibration_id = f"cal-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
        directory = self.root / ".adaptea" / "calibration" / calibration_id
        directory.mkdir(parents=True, exist_ok=False)
        plans = [
            (target, cap_candidates(requested, target.capacity, ceiling)) for target in targets
        ]
        empty = [target.key for target, selected in plans if not selected]
        if empty and len(empty) == len(plans):
            raise RuntimeError("no requested concurrency is within LM Studio's parallel limit")
        manifest = {
            "calibration_id": calibration_id,
            "created_at": utc_now(),
            "seed": seed,
            "quick": quick,
            "repetitions": reps,
            "max_agents": ceiling,
            "fleet_capacity": fleet_capacity,
            "models": [
                {
                    "key": target.key,
                    "instances": len(target.destinations),
                    "parallel_limit": target.parallel_limit,
                    "capacity": target.capacity,
                    "order": interleaved_order(selected, reps, seed),
                }
                for target, selected in plans
            ],
            "combined_order": interleaved_order(combined_candidates, reps, seed)
            if combined_candidates
            else [],
            # Retained because existing readers expect a single primary model.
            "model": targets[0].key,
            "parallel_limit": targets[0].parallel_limit,
            "warmup_included_in_samples": False,
        }
        _write_json(directory / "manifest.json", manifest)

        work = workloads(quick)
        sample_path = directory / "samples.jsonl"
        telemetry_path = directory / "telemetry.jsonl"
        telemetry_path.touch()
        stop = asyncio.Event()

        async def record_telemetry(sample: TelemetrySample) -> None:
            self.latest_queued = sample.queued_predictions
            with telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(sample.model_dump_json() + "\n")

        telemetry_task = asyncio.create_task(
            poll_ps(
                self.config.lmstudio.lms_executable,
                self.config.lmstudio.telemetry_poll_seconds,
                record_telemetry,
                stop,
                self.config.lmstudio.base_url,
                self.config.lmstudio.api_token,
            )
        )
        samples: list[dict[str, Any]] = []

        def persist(batch: list[dict[str, Any]]) -> None:
            samples.extend(batch)
            with sample_path.open("a", encoding="utf-8", newline="\n") as handle:
                for sample in batch:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")

        try:
            for target, selected in plans:
                if not selected:
                    continue
                self._report(f"Warming up {target.key}…")
                await self.client.chat(
                    target.destinations[0],
                    "Calibration warm-up. Reply with exactly: warm",
                    max_output_tokens=8,
                    temperature=0,
                )
                order = interleaved_order(selected, reps, seed)
                for sequence, concurrency in enumerate(order):
                    self._report(
                        f"{target.key}: concurrency {concurrency} ({sequence + 1}/{len(order)})"
                    )
                    persist(
                        await self._batch(
                            list(target.destinations),
                            [target.key] * len(target.destinations),
                            concurrency,
                            work,
                            sequence,
                            phase="model",
                            label=target.key,
                        )
                    )
            if combined_candidates:
                # What the run actually does: several models generating at once. Measured
                # separately because a per-model curve cannot predict how they share the
                # machine's memory bandwidth.
                ring, owners = _combined_ring(targets)
                order = interleaved_order(combined_candidates, reps, seed)
                label = " + ".join(target.key for target in targets)
                for sequence, concurrency in enumerate(order):
                    self._report(
                        f"All models together: concurrency {concurrency} "
                        f"({sequence + 1}/{len(order)})"
                    )
                    persist(
                        await self._batch(
                            ring,
                            owners,
                            concurrency,
                            work,
                            sequence,
                            phase="combined",
                            label=label,
                        )
                    )
        finally:
            stop.set()
            await telemetry_task

        per_model = {
            target.key: aggregate_samples(
                [
                    row
                    for row in samples
                    if row.get("phase") == "model" and row["model"] == target.key
                ]
            )
            for target, selected in plans
            if selected
        }
        combined = aggregate_samples([row for row in samples if row.get("phase") == "combined"])
        primary = targets[0].key
        headline = combined or per_model.get(primary) or next(iter(per_model.values()), {})
        recommended, safe = recommend(headline)
        profile = build_profile(
            targets=[
                {
                    "key": target.key,
                    "instances": len(target.destinations),
                    "destinations": list(target.destinations),
                    "parallel_limit": target.parallel_limit,
                    "capacity": target.capacity,
                    "format": target.model.format,
                    "max_context_length": target.model.max_context_length,
                    "load_config": (
                        target.model.loaded_instances[0].config.model_dump(exclude_none=True)
                        if target.model.loaded_instances
                        else {}
                    ),
                    "instance_id": (
                        target.model.loaded_instances[0].id
                        if target.model.loaded_instances
                        else None
                    ),
                }
                for target, selected in plans
                if selected
            ],
            per_model=per_model,
            combined=combined,
            recommended=recommended,
            safe_max=safe,
            fleet_capacity=fleet_capacity,
            max_agents=ceiling,
        )
        result = {
            **manifest,
            "aggregate": per_model.get(primary, {}),
            "per_model": per_model,
            "combined": combined,
            "profile": profile,
        }
        _write_json(directory / "aggregate.json", result)
        _write_json(self.root / ".adaptea" / "capacity.json", profile)
        write_csv(directory / "report.csv", per_model, combined)
        write_html(directory / "report.html", per_model, combined, recommended, calibration_id)
        self._report(f"Calibration complete: {directory}")
        return directory

    async def _batch(
        self,
        destinations: list[str],
        owners: list[str],
        concurrency: int,
        work: list[Workload],
        sequence: int,
        *,
        phase: str,
        label: str,
    ) -> list[dict[str, Any]]:
        chosen = [work[(sequence + index) % len(work)] for index in range(concurrency)]
        slots = [
            (destinations[index % len(destinations)], owners[index % len(owners)])
            for index in range(concurrency)
        ]
        start = time.perf_counter()
        rows = await asyncio.gather(
            *(
                self._request(
                    destination,
                    owner,
                    item,
                    concurrency,
                    sequence,
                    index,
                    phase=phase,
                    label=label,
                )
                for index, (item, (destination, owner)) in enumerate(
                    zip(chosen, slots, strict=True)
                )
            )
        )
        wall = time.perf_counter() - start
        for row in rows:
            row["batch_wall_seconds"] = wall
        return rows

    async def _request(
        self,
        destination: str,
        owner: str,
        workload: Workload,
        concurrency: int,
        sequence: int,
        slot: int,
        *,
        phase: str,
        label: str,
    ) -> dict[str, Any]:
        attempts = 0
        while True:
            attempts += 1
            started = time.perf_counter()
            response = await self.client.chat(
                destination,
                f"{workload.prompt}\nRun nonce: {sequence}-{slot}",
                max_output_tokens=workload.max_output_tokens,
                temperature=0,
            )
            latency = time.perf_counter() - started
            output_tokens = response.stats.total_output_tokens
            threshold = int(workload.max_output_tokens * workload.minimum_output_ratio)
            valid = output_tokens is not None and output_tokens >= threshold
            row = {
                "timestamp": utc_now(),
                "phase": phase,
                "group": label,
                "model": owner,
                "destination": destination,
                "sequence": sequence,
                "slot": slot,
                "concurrency": concurrency,
                "workload": workload.name,
                "max_output_tokens": workload.max_output_tokens,
                "minimum_output_tokens": threshold,
                "input_tokens": response.stats.input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": response.stats.tokens_per_second,
                "ttft_seconds": response.stats.time_to_first_token_seconds,
                "latency_seconds": latency,
                "valid": valid,
                "attempt": attempts,
                "queued_predictions": self.latest_queued,
                "invalid_reason": None if valid else "generation shorter than workload tolerance",
            }
            if valid or attempts >= 2:
                return row


def _combined_ring(targets: list[CalibrationTarget]) -> tuple[list[str], list[str]]:
    """Interleave instances so every added worker lands on a different model first.

    Filling one model before touching the next would measure two sequential single-model
    sweeps, which is what the per-model phase already did.
    """
    ring: list[str] = []
    owners: list[str] = []
    depth = max(len(target.destinations) for target in targets)
    for index in range(depth):
        for target in targets:
            if index < len(target.destinations):
                ring.append(target.destinations[index])
                owners.append(target.key)
    return ring, owners


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, per_model: dict[str, dict[str, Any]], combined: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model, aggregate in per_model.items():
        rows.extend(
            {"model": model, "concurrency": key, **value} for key, value in aggregate.items()
        )
    rows.extend(
        {"model": "all models together", "concurrency": key, **value}
        for key, value in combined.items()
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(
    path: Path,
    per_model: dict[str, dict[str, Any]],
    combined: dict[str, Any],
    recommended: int,
    title: str,
) -> None:
    sections = [*per_model.items()]
    if combined:
        sections.append(("All models together", combined))
    maximum = max(
        (
            float(row.get("requests_per_second") or 0)
            for _name, aggregate in sections
            for row in aggregate.values()
        ),
        default=1.0,
    )
    blocks = []
    for name, aggregate in sections:
        bars = []
        for concurrency, row in aggregate.items():
            rate = float(row.get("requests_per_second") or 0)
            width = 0 if maximum == 0 else rate / maximum * 100
            marker = (
                " recommended"
                if int(concurrency) == recommended
                and (name == "All models together" or not combined)
                else ""
            )
            bars.append(
                f'<div class="row{marker}"><span>C={html.escape(concurrency)}</span>'
                f'<div class="bar" style="width:{width:.1f}%"></div><b>{rate:.3f} req/s</b></div>'
            )
        blocks.append(f"<h2>{html.escape(name)}</h2>{''.join(bars)}")
    document = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:16px system-ui;max-width:900px;margin:3rem auto;padding:0 1rem}}
h2{{font-size:1rem;margin:2rem 0 .5rem;color:#333}}
.row{{display:grid;grid-template-columns:4rem 1fr 9rem;gap:1rem;align-items:center;
margin:.7rem 0}}.bar{{height:1.6rem;background:#69b3a2;min-width:2px}}
.recommended{{background:#fff5cc;padding:.4rem}}code{{color:#555}}
</style><h1>Adaptea calibration</h1><p><code>{html.escape(title)}</code></p>
<p>Throughput by concurrency, per model and with every model generating at once.
The recommended starting concurrency is highlighted.</p>
{"".join(blocks)}
<p>Measurements are specific to this machine, model set, load configuration, and workload.</p>"""
    path.write_text(document, encoding="utf-8")
