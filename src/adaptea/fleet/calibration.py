from __future__ import annotations

import asyncio
import json
import platform
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from adaptea.config import Config
from adaptea.fleet.lifecycle import InstanceLifecycleManager
from adaptea.fleet.models import (
    FleetInventory,
    ModelInstance,
    TopologyCandidate,
    TopologyInstance,
    TopologyResult,
)
from adaptea.inference import InferenceBackend
from adaptea.models import utc_now

Benchmark = Callable[[TopologyCandidate], Awaitable[TopologyResult]]
AgentValidator = Callable[[TopologyCandidate], Awaitable[tuple[float, float, int]]]


def generate_topology_candidates(
    config: Config,
    inventory: FleetInventory,
    safe_instance_limits: Mapping[str, int] | None = None,
) -> list[TopologyCandidate]:
    limits = safe_instance_limits or {}
    configured = config.fleet.models
    if not configured:
        return []
    candidates: list[TopologyCandidate] = []
    seen: set[str] = set()
    maximum = min(config.fleet.max_loaded_instances, 3)

    def add(identifier: str, rows: list[tuple[str, str, int, int]]) -> None:
        if identifier in seen:
            return
        seen.add(identifier)
        candidates.append(_candidate(identifier, rows))

    def parallel_limit(model: Any) -> int:
        observed = max(
            (
                instance.effective_limit
                for instance in inventory.instances
                if instance.model_key == model.model
            ),
            default=model.parallel_limit or 1,
        )
        return max(1, min(observed, config.worker.max_agents))

    def worker_levels(model: Any) -> list[int]:
        parallel = parallel_limit(model)
        return sorted({1, min(2, parallel), min(4, parallel), parallel})

    if len(configured) == 1:
        model = configured[0]
        parallel = parallel_limit(model)
        for workers in worker_levels(model):
            add(f"single-1x-w{workers}", [(model.model, model.tier, 1, workers)])
        for count in range(2, maximum + 1):
            add(f"same-{count}x-w1", [(model.model, model.tier, count, 1)])
            if parallel >= 2 and count == 2:
                add("same-2x-w2", [(model.model, model.tier, 2, 2)])
    else:
        strong = next((item for item in configured if item.tier == "strong"), None)
        fast = next((item for item in configured if item.tier == "fast"), None)
        if strong:
            for workers in worker_levels(strong):
                identifier = "strong-1" if workers == 1 else f"strong-1-w{workers}"
                add(identifier, [(strong.model, "strong", 1, workers)])
            for count in range(2, maximum + 1):
                add(f"strong-{count}x-w1", [(strong.model, "strong", count, 1)])
        if fast:
            for workers in worker_levels(fast):
                identifier = "fast-1" if workers == 1 else f"fast-1-w{workers}"
                add(identifier, [(fast.model, "fast", 1, workers)])
            for count in range(2, maximum + 1):
                add(f"fast-{count}x-w1", [(fast.model, "fast", count, 1)])
        if strong and fast:
            strong_levels = worker_levels(strong)
            fast_levels = worker_levels(fast)
            for strong_count in range(1, maximum):
                for fast_count in range(1, maximum - strong_count + 1):
                    base = f"strong-{strong_count}-fast-{fast_count}"
                    variants = {(1, 1)}
                    variants.update((workers, 1) for workers in strong_levels[1:])
                    variants.update((1, workers) for workers in fast_levels[1:])
                    if strong_levels[-1] > 1 and fast_levels[-1] > 1:
                        variants.add((min(2, strong_levels[-1]), min(2, fast_levels[-1])))
                    for strong_workers, fast_workers in sorted(variants):
                        identifier = (
                            base
                            if (strong_workers, fast_workers) == (1, 1)
                            else f"{base}-sw{strong_workers}-fw{fast_workers}"
                        )
                        add(
                            identifier,
                            [
                                (strong.model, "strong", strong_count, strong_workers),
                                (fast.model, "fast", fast_count, fast_workers),
                            ],
                        )
    for candidate in candidates:
        if candidate.total_instances > config.fleet.max_loaded_instances:
            candidate.feasible = False
            candidate.feasibility_reason = "exceeds max_loaded_instances"
            continue
        if candidate.total_workers > config.worker.max_agents:
            candidate.feasible = False
            candidate.feasibility_reason = "exceeds worker.max_agents"
            continue
        for item in candidate.instances:
            safe = limits.get(item.model)
            if safe is not None and item.count > safe:
                candidate.feasible = False
                candidate.feasibility_reason = (
                    f"resource estimate permits at most {safe} instance(s) of {item.model}"
                )
                break
    return candidates


def select_topology(results: list[TopologyResult]) -> tuple[TopologyResult | None, str]:
    measured = [row for row in results if row.candidate.feasible and row.repetitions > 0]
    if not measured:
        return None, "No feasible topology produced measurements."
    pass_rates = [row.pass_rate for row in measured if row.pass_rate is not None]
    if pass_rates:
        acceptable_floor = max(pass_rates) - 0.01
        acceptable = [
            row
            for row in measured
            if row.pass_rate is not None and row.pass_rate >= acceptable_floor
        ]
    else:
        acceptable = measured
    best_reliability = max(_measurement_reliability(row) for row in acceptable)
    acceptable = [
        row for row in acceptable if _measurement_reliability(row) >= best_reliability - 0.01
    ]
    with_completion = [row for row in acceptable if row.median_completion_seconds is not None]
    if with_completion:
        selected = min(
            with_completion,
            key=lambda row: (
                row.median_completion_seconds or float("inf"),
                row.resource_pressure or 0,
            ),
        )
        return selected, (
            "Validation and measurement repeatability remained acceptable; selected the lowest "
            "measured end-to-end completion time, using resource pressure only as a tie breaker."
        )
    selected = max(
        acceptable,
        key=lambda row: (
            row.generation_tokens_per_second or 0,
            -(row.median_ttft_seconds or float("inf")),
        ),
    )
    return selected, (
        "No agent completion measurement was available; selected the best repeatable "
        "direct-inference candidate provisionally. Agent validation is still recommended."
    )


class FleetCalibrationRunner:
    def __init__(
        self,
        root: Path,
        config: Config,
        inventory: FleetInventory,
        client: InferenceBackend,
        *,
        benchmark: Benchmark | None = None,
        agent_validator: AgentValidator | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.inventory = inventory
        self.client = client
        self.benchmark = benchmark or self._direct_benchmark
        self.manage_topology = benchmark is None
        self.agent_validator = agent_validator

    async def run(
        self,
        *,
        repetitions: int = 3,
        safe_instance_limits: Mapping[str, int] | None = None,
    ) -> Path:
        candidates = generate_topology_candidates(self.config, self.inventory, safe_instance_limits)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        order = _interleaved(feasible, repetitions, self.config.calibration.seed)
        grouped: dict[str, list[TopologyResult]] = {}
        for candidate in order:
            result = await self._measure(candidate)
            grouped.setdefault(candidate.id, []).append(result)
        aggregated = [_aggregate(rows) for rows in grouped.values()]
        direct_ranked = sorted(
            [
                result
                for result in aggregated
                if result.candidate.feasible and result.generation_tokens_per_second is not None
            ],
            key=lambda row: (
                _measurement_reliability(row),
                row.generation_tokens_per_second or 0,
            ),
            reverse=True,
        )
        # A raw throughput race tends to send two near-identical batching candidates to
        # the expensive coding-agent validator. Keep the shortlist small, but require it
        # to cover batching, replicated instances, and heterogeneous fleets when present.
        direct_best = _agent_validation_shortlist(direct_ranked)
        if self.agent_validator:
            for result in direct_best:
                added: list[ModelInstance] = []
                if self.manage_topology:
                    try:
                        added = await self._prepare_topology(result.candidate)
                    except RuntimeError as exc:
                        result.candidate.feasible = False
                        result.candidate.feasibility_reason = str(exc)
                        continue
                try:
                    pass_rate, wall_seconds, retries = await self.agent_validator(result.candidate)
                finally:
                    await self._release_topology(added)
                result.pass_rate = pass_rate
                result.median_completion_seconds = wall_seconds
                result.candidate.feasibility_reason += f"; agent retries={retries}"
        selected, reason = select_topology(aggregated)
        profile = self._profile(candidates, aggregated, selected, reason)
        directory = self.root / ".adaptea" / "fleet-calibration" / f"fleet-{utc_now()[:10]}"
        suffix = 1
        while directory.exists():
            directory = directory.with_name(f"{directory.name}-{suffix}")
            suffix += 1
        directory.mkdir(parents=True)
        (directory / "profile.json").write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        target = self.root / ".adaptea" / "fleet.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        return directory

    async def _measure(self, candidate: TopologyCandidate) -> TopologyResult:
        if not self.manage_topology:
            return await self.benchmark(candidate)
        added: list[ModelInstance] = []
        try:
            added = await self._prepare_topology(candidate)
            return await self.benchmark(candidate)
        except RuntimeError as exc:
            failed = candidate.model_copy(
                update={"feasible": False, "feasibility_reason": str(exc)}
            )
            return TopologyResult(candidate=failed)
        finally:
            await self._release_topology(added)

    async def _prepare_topology(self, candidate: TopologyCandidate) -> list[ModelInstance]:
        desired_models = {row.model for row in candidate.instances}
        extras = [
            instance
            for instance in self.inventory.instances
            if instance.model_key not in desired_models
        ]
        if extras:
            raise RuntimeError(
                "candidate cannot be isolated without unloading pre-existing "
                "non-candidate instances"
            )
        manager = InstanceLifecycleManager(self.config, self.inventory)
        added: list[ModelInstance] = []
        for desired in candidate.instances:
            current = [item for item in self.inventory.instances if item.model_key == desired.model]
            if len(current) > desired.count:
                raise RuntimeError(
                    "candidate requires fewer instances than are already loaded; "
                    "refusing to unload pre-existing instances automatically"
                )
            configured = next(
                (item for item in self.config.fleet.models if item.model == desired.model), None
            )
            for index in range(len(current) + 1, desired.count + 1):
                identifier = f"adaptea-cal-{desired.tier}-{index}"
                instance = await manager.load(
                    desired.model,
                    identifier,
                    context_length=configured.context_length if configured else None,
                )
                added.append(instance)
        return added

    async def _release_topology(self, instances: list[ModelInstance]) -> None:
        manager = InstanceLifecycleManager(self.config, self.inventory)
        for instance in reversed(instances):
            await manager.unload(instance)

    async def _direct_benchmark(self, candidate: TopologyCandidate) -> TopologyResult:
        destinations: list[str] = []
        for desired in candidate.instances:
            matching = [
                instance.instance_id
                for instance in self.inventory.instances
                if instance.model_key == desired.model and instance.available
            ]
            if len(matching) < desired.count:
                return TopologyResult(
                    candidate=candidate.model_copy(
                        update={
                            "feasible": False,
                            "feasibility_reason": "required instances are not currently loaded",
                        }
                    )
                )
            for instance_id in matching[: desired.count]:
                destinations.extend([instance_id] * desired.workers_per_instance)
        started = time.perf_counter()
        responses = await asyncio.gather(
            *(
                self.client.chat(
                    destination,
                    f"Adaptea fleet calibration {candidate.id}-{index}. "
                    "Write a deterministic short Python function and one test.",
                    max_output_tokens=96,
                    temperature=0,
                )
                for index, destination in enumerate(destinations)
            )
        )
        wall = time.perf_counter() - started
        speeds = [
            response.stats.tokens_per_second
            for response in responses
            if response.stats.tokens_per_second is not None
        ]
        ttfts = [
            response.stats.time_to_first_token_seconds
            for response in responses
            if response.stats.time_to_first_token_seconds is not None
        ]
        return TopologyResult(
            candidate=candidate,
            direct_wall_seconds=wall,
            median_ttft_seconds=statistics.median(ttfts) if ttfts else None,
            generation_tokens_per_second=sum(speeds) if speeds else None,
            repetitions=1,
            successful_repetitions=1,
            measurement_success_rate=1.0,
        )

    def _profile(
        self,
        candidates: list[TopologyCandidate],
        results: list[TopologyResult],
        selected: TopologyResult | None,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "profile_version": 2,
            "models": [model.model_dump(mode="json") for model in self.inventory.downloaded],
            "loaded_instances": [
                instance.model_dump(mode="json") for instance in self.inventory.instances
            ],
            "tested_topologies": [result.model_dump(mode="json") for result in results],
            "candidate_topologies": [candidate.model_dump(mode="json") for candidate in candidates],
            "recommended_topology": (
                selected.candidate.model_dump(mode="json") if selected else {}
            ),
            "selection_reason": reason,
            "routing_profiles": {},
            "machine_signature": machine_signature(self.inventory),
            "created_at": utc_now(),
        }


def machine_signature(inventory: FleetInventory) -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "lmstudio_runtime": sorted(
            note for note in inventory.source_notes if note.startswith("lms runtime:")
        ),
        "models": sorted(
            (
                model.model_key,
                model.format,
                model.size_bytes,
                json.dumps(model.quantization, sort_keys=True),
            )
            for model in inventory.downloaded
        ),
        "instances": sorted(
            (
                item.instance_id,
                item.model_key,
                item.context_length,
                item.parallel_limit,
            )
            for item in inventory.instances
        ),
    }


def profile_is_stale(profile: dict[str, Any], inventory: FleetInventory) -> bool:
    stored = profile.get("machine_signature")
    if not isinstance(stored, dict):
        return True
    current = machine_signature(inventory)
    for key in ("system", "architecture", "lmstudio_runtime", "models"):
        if stored.get(key) != current.get(key):
            return True
    stored_instances = stored.get("instances")
    current_instances = current.get("instances")
    if not isinstance(stored_instances, list) or not stored_instances:
        return False
    if not isinstance(current_instances, list):
        return True
    stored_configs = {(row[1], row[2], row[3]) for row in stored_instances if len(row) >= 4}
    current_configs = {(row[1], row[2], row[3]) for row in current_instances if len(row) >= 4}
    return bool(stored_configs and current_configs and stored_configs != current_configs)


def _candidate(identifier: str, rows: list[tuple[str, str, int, int]]) -> TopologyCandidate:
    instances = [
        TopologyInstance(
            model=model,
            tier=tier,  # type: ignore[arg-type]
            count=count,
            workers_per_instance=workers,
        )
        for model, tier, count, workers in rows
    ]
    return TopologyCandidate(
        id=identifier,
        instances=instances,
        total_instances=sum(item.count for item in instances),
        total_workers=sum(item.count * item.workers_per_instance for item in instances),
    )


def _interleaved(
    candidates: list[TopologyCandidate], repetitions: int, seed: int
) -> list[TopologyCandidate]:
    rng = random.Random(seed)
    order: list[TopologyCandidate] = []
    for _ in range(repetitions):
        row = candidates.copy()
        rng.shuffle(row)
        order.extend(row)
    return order


def _aggregate(rows: list[TopologyResult]) -> TopologyResult:
    first = rows[0]
    attempted = len(rows)
    successful = sum(1 for row in rows if row.candidate.feasible and row.repetitions > 0)
    candidate = next(
        (row.candidate for row in rows if row.candidate.feasible and row.repetitions > 0),
        first.candidate,
    )

    def median(name: str) -> float | None:
        values = [getattr(row, name) for row in rows if getattr(row, name) is not None]
        return statistics.median(values) if values else None

    return TopologyResult(
        candidate=candidate,
        pass_rate=median("pass_rate"),
        median_completion_seconds=median("median_completion_seconds"),
        direct_wall_seconds=median("direct_wall_seconds"),
        median_ttft_seconds=median("median_ttft_seconds"),
        generation_tokens_per_second=median("generation_tokens_per_second"),
        resource_pressure=median("resource_pressure"),
        repetitions=attempted,
        successful_repetitions=successful,
        measurement_success_rate=successful / attempted if attempted else None,
    )


def _measurement_reliability(result: TopologyResult) -> float:
    if result.measurement_success_rate is not None:
        return result.measurement_success_rate
    if result.successful_repetitions:
        return result.successful_repetitions / max(1, result.repetitions)
    # Profiles written before v2 recorded only a repetition count. Treat those completed
    # rows as reliable so an upgrade does not invalidate otherwise usable measurements.
    return 1.0 if result.repetitions > 0 else 0.0


def _topology_family(candidate: TopologyCandidate) -> str:
    # Classify the scaling mechanism before the model mix. This keeps all three families
    # measurable without unloading a user's pre-existing fast/strong pair: a mixed fleet
    # can compare batching on the current instances, adding a replica, and the 1+1
    # heterogeneous baseline safely.
    if any(row.count > 1 for row in candidate.instances):
        return "replicated"
    if any(row.workers_per_instance > 1 for row in candidate.instances):
        return "continuous_batching"
    if len({row.model for row in candidate.instances}) > 1:
        return "heterogeneous"
    return "continuous_batching"


def _agent_validation_shortlist(ranked: list[TopologyResult]) -> list[TopologyResult]:
    """Keep validation bounded while comparing genuinely different topology families."""
    if not ranked:
        return []
    target = min(3, len(ranked))
    selected: list[TopologyResult] = []
    seen_families: set[str] = set()
    for result in ranked:
        family = _topology_family(result.candidate)
        if family in seen_families:
            continue
        selected.append(result)
        seen_families.add(family)
        if len(selected) == target:
            return selected
    for result in ranked:
        if result not in selected:
            selected.append(result)
        if len(selected) == target:
            break
    return selected
