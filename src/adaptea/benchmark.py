from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from adaptea.comparison import copy_local_configuration
from adaptea.config import load_config
from adaptea.git.repository import git
from adaptea.models import Plan, RunState, TaskStatus, utc_now
from adaptea.reporting.report import write_benchmark_csv, write_benchmark_html
from adaptea.runtime.controller import Orchestrator, create_run

BenchmarkMode = Literal["serial", "fixed", "naive", "adaptive"]
MODES: tuple[BenchmarkMode, ...] = ("serial", "fixed", "naive", "adaptive")


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    mode: BenchmarkMode
    repetition: int
    sequence: int


RunExecutor = Callable[[Path, Plan, BenchmarkSpec, int, int], Awaitable[RunState]]
ProgressCallback = Callable[[BenchmarkSpec, str], None]


class BenchmarkRunError(RuntimeError):
    def __init__(self, state: RunState, cause: Exception) -> None:
        super().__init__(str(cause))
        self.state = state
        self.cause = cause


def benchmark_order(repetitions: int, seed: int) -> list[BenchmarkSpec]:
    """Return a seeded, round-interleaved order with one run per mode in every round."""
    if repetitions < 3:
        raise ValueError("benchmark repetitions must be at least 3")
    rng = random.Random(seed)
    result: list[BenchmarkSpec] = []
    previous: list[BenchmarkMode] | None = None
    sequence = 0
    for repetition in range(1, repetitions + 1):
        modes = list(MODES)
        rng.shuffle(modes)
        if modes == previous:
            modes = modes[1:] + modes[:1]
        for mode in modes:
            result.append(BenchmarkSpec(mode, repetition, sequence))
            sequence += 1
        previous = modes
    return result


class BenchmarkRunner:
    def __init__(
        self,
        root: Path,
        executor: RunExecutor | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.root = root.resolve()
        self.executor = executor or execute_benchmark_run
        self.progress = progress

    async def run(
        self,
        plan: Plan,
        *,
        repetitions: int = 3,
        fixed_concurrency: int = 2,
        max_agents: int = 8,
        seed: int = 2026,
        keep_worktrees: bool = False,
    ) -> Path:
        if fixed_concurrency < 1:
            raise ValueError("fixed concurrency must be positive")
        if max_agents < 1:
            raise ValueError("max agents must be positive")
        if fixed_concurrency > max_agents:
            raise ValueError("fixed concurrency cannot exceed max agents")
        status = await git(self.root, "status", "--porcelain", "--untracked-files=all")
        if meaningful_status_lines(status.stdout):
            raise RuntimeError(
                "Benchmark requires a clean repository so every run starts at the same commit."
            )
        source_commit = (await git(self.root, "rev-parse", "HEAD")).stdout.strip()
        plan_json = plan.model_dump_json(indent=2)
        plan_sha256 = hashlib.sha256(plan_json.encode()).hexdigest()
        benchmark_id = f"bench-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
        directory = self.root / ".adaptea" / "benchmarks" / benchmark_id
        worktree_parent = directory / "worktrees"
        artifact_parent = directory / "runs"
        worktree_parent.mkdir(parents=True)
        artifact_parent.mkdir()
        order = benchmark_order(repetitions, seed)
        rows: list[dict[str, Any]] = []

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "created_at": utc_now(),
            "source_repository": str(self.root),
            "source_commit": source_commit,
            "plan_sha256": plan_sha256,
            "plan": plan.model_dump(mode="json"),
            "repetitions": repetitions,
            "seed": seed,
            "fixed_concurrency": fixed_concurrency,
            "max_agents": max_agents,
            "execution_policy": "sequential, seeded, and round-interleaved",
            "modes": {
                "serial": "fixed scheduler with concurrency and max agents set to 1",
                "fixed": f"fixed scheduler with concurrency {fixed_concurrency}",
                "naive": f"naive scheduler with max agents {max_agents}",
                "adaptive": f"adaptive scheduler with max agents {max_agents}",
            },
            "order": [
                {
                    "sequence": item.sequence,
                    "repetition": item.repetition,
                    "mode": item.mode,
                }
                for item in order
            ],
        }
        _write_json(directory / "manifest.json", manifest)

        for spec in order:
            if self.progress:
                self.progress(spec, "started")
            run_key = f"{spec.sequence + 1:02d}-{spec.mode}-r{spec.repetition}"
            worktree = worktree_parent / run_key
            result = await git(
                self.root,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                source_commit,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not prepare benchmark worktree {run_key}: {result.stderr.strip()}"
                )
            copy_local_configuration(self.root, worktree)
            started = time.perf_counter()
            state: RunState | None = None
            error: str | None = None
            try:
                state = await self.executor(worktree, plan, spec, fixed_concurrency, max_agents)
            except BenchmarkRunError as exc:
                state = exc.state
                error = f"{type(exc.cause).__name__}: {exc.cause}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            duration = time.perf_counter() - started
            row = summarize_benchmark_run(
                spec,
                state,
                duration,
                source_commit=source_commit,
                plan_sha256=plan_sha256,
                error=error,
            )
            rows.append(row)
            _write_json(artifact_parent / run_key / "result.json", row)
            if state is not None:
                _copy_run_evidence(state, artifact_parent / run_key)
            _write_json(directory / "partial-results.json", {**manifest, "runs": rows})
            if not keep_worktrees:
                await _remove_run_worktrees(self.root, worktree, state)
            if self.progress:
                self.progress(spec, "failed" if error else "finished")

        aggregate = aggregate_benchmark(rows)
        report = {**manifest, "runs": rows, **aggregate, "finished_at": utc_now()}
        json_path = directory / "benchmark.json"
        csv_path = directory / "benchmark.csv"
        html_path = directory / "benchmark.html"
        _write_json(json_path, report)
        write_benchmark_csv(csv_path, report)
        write_benchmark_html(html_path, report)
        partial = directory / "partial-results.json"
        if partial.exists():
            partial.unlink()
        return directory


async def execute_benchmark_run(
    root: Path,
    plan: Plan,
    spec: BenchmarkSpec,
    fixed_concurrency: int,
    max_agents: int,
) -> RunState:
    config = load_config(root)
    scheduler: Literal["fixed", "naive", "adaptive"]
    concurrency: int | None = None
    ceiling = max_agents
    if spec.mode == "serial":
        scheduler = "fixed"
        concurrency = 1
        ceiling = 1
    elif spec.mode == "fixed":
        scheduler = "fixed"
        concurrency = fixed_concurrency
    else:
        scheduler = spec.mode
    state = await create_run(root, config, plan, scheduler, ceiling, concurrency)
    try:
        return await Orchestrator(root, config, state).run()
    except Exception as exc:
        raise BenchmarkRunError(state, exc) from exc


def summarize_benchmark_run(
    spec: BenchmarkSpec,
    state: RunState | None,
    total_duration_seconds: float,
    *,
    source_commit: str,
    plan_sha256: str,
    error: str | None = None,
) -> dict[str, Any]:
    total = len(state.tasks) if state is not None else 0
    merged = (
        sum(task.status == TaskStatus.MERGED for task in state.tasks.values())
        if state is not None
        else 0
    )
    retries = (
        sum(max(task.attempts - 1, 0) for task in state.tasks.values()) if state is not None else 0
    )
    pressure = resource_pressure(state) if state is not None else empty_resource_pressure()
    return {
        "mode": spec.mode,
        "configuration": spec.mode,
        "repetition": spec.repetition,
        "sequence": spec.sequence,
        "source_commit": source_commit,
        "plan_sha256": plan_sha256,
        "run_id": state.run_id if state is not None else None,
        "scheduler": state.scheduler if state is not None else None,
        "final_target_concurrency": state.target_concurrency if state is not None else None,
        "parallel_limit": state.parallel_limit if state is not None else None,
        "total_duration_seconds": total_duration_seconds,
        "wall_seconds": total_duration_seconds,
        "tasks_total": total,
        "tasks_merged": merged,
        "success_rate": merged / total if total else 0.0,
        "pass_rate": merged / total if total else 0.0,
        "retry_count": retries,
        "resource_pressure": pressure,
        "error": error,
    }


def resource_pressure(state: RunState) -> dict[str, int | float | None]:
    run_dir = Path(state.repository) / ".adaptea" / "runs" / state.run_id
    telemetry = _read_jsonl(run_dir / "telemetry.jsonl")
    runtime = _read_jsonl(run_dir / "runtime-status.jsonl")
    queues = _numbers(telemetry, "queued_predictions")
    busy = [row["generating"] for row in telemetry if isinstance(row.get("generating"), bool)]
    ttft = _numbers(telemetry, "ttft_seconds")
    speed = _numbers(telemetry, "tokens_per_second")
    workers = _numbers(runtime, "running")
    parallel = max(state.parallel_limit, 1)
    return {
        "telemetry_samples": len(telemetry),
        "runtime_samples": len(runtime),
        "queue_observed_samples": len(queues),
        "queue_pressure_ratio": (
            sum(value > 0 for value in queues) / len(queues) if queues else None
        ),
        "peak_queued_predictions": max(queues) if queues else None,
        "busy_observed_samples": len(busy),
        "busy_ratio": sum(busy) / len(busy) if busy else None,
        "median_ttft_seconds": statistics.median(ttft) if ttft else None,
        "median_tokens_per_second": statistics.median(speed) if speed else None,
        "peak_workers": int(max(workers)) if workers else None,
        "median_workers": statistics.median(workers) if workers else None,
        "average_worker_utilization": (statistics.fmean(workers) / parallel if workers else None),
        "peak_worker_utilization": max(workers) / parallel if workers else None,
    }


def empty_resource_pressure() -> dict[str, int | float | None]:
    return {
        "telemetry_samples": 0,
        "runtime_samples": 0,
        "queue_observed_samples": 0,
        "queue_pressure_ratio": None,
        "peak_queued_predictions": None,
        "busy_observed_samples": 0,
        "busy_ratio": None,
        "median_ttft_seconds": None,
        "median_tokens_per_second": None,
        "peak_workers": None,
        "median_workers": None,
        "average_worker_utilization": None,
        "peak_worker_utilization": None,
    }


def aggregate_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        selected = [row for row in rows if row.get("mode") == mode]
        pressure: list[dict[str, Any]] = []
        for row in selected:
            value = row.get("resource_pressure")
            if isinstance(value, dict):
                pressure.append(value)
        medians[mode] = {
            "repetitions": len(selected),
            "median_total_duration_seconds": _median(selected, "total_duration_seconds"),
            "median_success_rate": _median(selected, "success_rate"),
            "median_retry_count": _median(selected, "retry_count"),
            "resource_pressure": {
                "telemetry_coverage_runs": sum(
                    item.get("queue_pressure_ratio") is not None for item in pressure
                ),
                "median_queue_pressure_ratio": _median(pressure, "queue_pressure_ratio"),
                "median_peak_queued_predictions": _median(pressure, "peak_queued_predictions"),
                "median_busy_ratio": _median(pressure, "busy_ratio"),
                "median_ttft_seconds": _median(pressure, "median_ttft_seconds"),
                "median_tokens_per_second": _median(pressure, "median_tokens_per_second"),
                "median_peak_workers": _median(pressure, "peak_workers"),
                "median_average_worker_utilization": _median(
                    pressure, "average_worker_utilization"
                ),
            },
        }
    commits = {str(row.get("source_commit")) for row in rows}
    plans = {str(row.get("plan_sha256")) for row in rows}
    return {
        "medians": medians,
        "formal_comparison": (
            len(commits) == 1
            and len(plans) == 1
            and all(medians[mode]["repetitions"] >= 3 for mode in MODES)
        ),
    }


def meaningful_status_lines(output: str) -> list[str]:
    """Ignore only Adaptea's own untracked artifacts when enforcing a clean source tree."""
    return [
        line
        for line in output.splitlines()
        if line.strip() and not (line.startswith("?? ") and line[3:].startswith(".adaptea/"))
    ]


def latest_benchmark(root: Path) -> tuple[Path, dict[str, Any]]:
    parent = root / ".adaptea" / "benchmarks"
    if not parent.is_dir():
        raise FileNotFoundError("no benchmark report found; run adaptea benchmark")
    candidates = [path for path in parent.iterdir() if (path / "benchmark.json").is_file()]
    if not candidates:
        raise FileNotFoundError("no benchmark report found; run adaptea benchmark")
    directory = max(candidates, key=lambda path: (path / "benchmark.json").stat().st_mtime)
    report = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"invalid benchmark report: {directory / 'benchmark.json'}")
    return directory, report


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if isinstance((value := row.get(key)), int | float) and not isinstance(value, bool)
    ]
    return statistics.median(values) if values else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(value)
        for row in rows
        if isinstance((value := row.get(key)), int | float) and not isinstance(value, bool)
    ]


def _copy_run_evidence(state: RunState, destination: Path) -> None:
    source = Path(state.repository) / ".adaptea" / "runs" / state.run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest.json",
        "plan.json",
        "state.json",
        "summary.json",
        "telemetry.jsonl",
        "runtime-status.jsonl",
        "controller-decisions.jsonl",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


async def _remove_run_worktrees(
    repository: Path, source_worktree: Path, state: RunState | None
) -> None:
    if state is not None:
        owned_root = (source_worktree / ".adaptea" / "worktrees" / state.run_id).resolve()
        listing = await git(repository, "worktree", "list", "--porcelain", check=False)
        registered = [
            Path(line.removeprefix("worktree ")).resolve()
            for line in listing.stdout.splitlines()
            if line.startswith("worktree ")
        ]
        owned = [path for path in registered if path.is_relative_to(owned_root)]
        for path in sorted(owned, key=lambda item: len(item.parts), reverse=True):
            await git(repository, "worktree", "remove", "--force", str(path), check=False)
    removed = await git(
        repository, "worktree", "remove", "--force", str(source_worktree), check=False
    )
    if removed.returncode != 0:
        raise RuntimeError(
            f"Could not clean benchmark worktree {source_worktree}: {removed.stderr.strip()}"
        )
    await git(repository, "worktree", "prune", check=False)
    if state is not None:
        branches = {
            branch
            for branch in (
                state.integration_branch,
                *(task.branch for task in state.tasks.values()),
            )
            if branch is not None and branch.startswith("adaptea-")
        }
        for branch in branches:
            exists = await git(
                repository, "show-ref", "--verify", f"refs/heads/{branch}", check=False
            )
            if exists.returncode == 0:
                await git(repository, "branch", "-D", branch, check=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
