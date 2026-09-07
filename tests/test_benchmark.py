from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from adaptea.benchmark import (
    MODES,
    BenchmarkRunner,
    BenchmarkSpec,
    aggregate_benchmark,
    benchmark_order,
    meaningful_status_lines,
    resource_pressure,
)
from adaptea.git.repository import git
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    (path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    (path / "base.txt").write_text("same starting point\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _plan() -> Plan:
    return Plan(
        goal="deterministic task",
        tasks=[TaskSpec(id="task", title="Task", description="Implement the fixture")],
    )


def test_benchmark_fixture_has_a_valid_pinned_dependency_plan() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "benchmark"
    plan = Plan.model_validate_json((fixture / "plan.json").read_text(encoding="utf-8"))
    assert [task.id for task in plan.tasks] == ["slug", "store", "search", "export", "docs"]
    assert plan.tasks[-1].depends_on == ["slug", "store", "search", "export"]
    assert (fixture / "tests" / "test_acceptance.py").is_file()


def test_benchmark_order_is_seeded_balanced_and_requires_three_repetitions() -> None:
    first = benchmark_order(3, 77)
    assert first == benchmark_order(3, 77)
    assert first != benchmark_order(3, 78)
    for offset in range(0, len(first), len(MODES)):
        assert {item.mode for item in first[offset : offset + len(MODES)]} == set(MODES)
        assert len({item.repetition for item in first[offset : offset + len(MODES)]}) == 1
    with pytest.raises(ValueError, match="at least 3"):
        benchmark_order(2, 77)


def test_clean_check_ignores_only_adaptea_artifacts() -> None:
    status = (
        "?? .adaptea/benchmarks/old/result.json\n"
        " M .adaptea/tracked.json\n M source.py\n?? notes.txt\n"
    )
    assert meaningful_status_lines(status) == [
        " M .adaptea/tracked.json",
        " M source.py",
        "?? notes.txt",
    ]


def test_resource_pressure_uses_observations_and_does_not_fill_missing_values(
    tmp_path: Path,
) -> None:
    state = RunState(
        run_id="run-pressure",
        goal="g",
        repository=str(tmp_path),
        integration_branch="integration",
        scheduler="adaptive",
        target_concurrency=2,
        user_ceiling=4,
        parallel_limit=4,
        tasks={},
    )
    run_dir = tmp_path / ".adaptea" / "runs" / state.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "telemetry.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "queued_predictions": 0,
                    "generating": False,
                    "ttft_seconds": 1.0,
                    "tokens_per_second": 20.0,
                },
                {
                    "queued_predictions": 2,
                    "generating": True,
                    "ttft_seconds": 3.0,
                    "tokens_per_second": 10.0,
                },
                {"generating": True},
            )
        )
        + "\nnot-json\n",
        encoding="utf-8",
    )
    (run_dir / "runtime-status.jsonl").write_text(
        '{"running":1}\n{"running":3}\n', encoding="utf-8"
    )

    pressure = resource_pressure(state)
    assert pressure["queue_pressure_ratio"] == 0.5
    assert pressure["peak_queued_predictions"] == 2
    assert pressure["busy_ratio"] == pytest.approx(2 / 3)
    assert pressure["median_ttft_seconds"] == 2.0
    assert pressure["median_tokens_per_second"] == 15.0
    assert pressure["peak_workers"] == 3
    assert pressure["average_worker_utilization"] == 0.5

    (run_dir / "telemetry.jsonl").write_text('{"source":"runtime"}\n', encoding="utf-8")
    missing = resource_pressure(state)
    assert missing["queue_pressure_ratio"] is None
    assert missing["median_ttft_seconds"] is None


def test_aggregate_uses_per_mode_medians_and_checks_fairness() -> None:
    rows: list[dict[str, object]] = []
    for mode_index, mode in enumerate(MODES):
        for repetition, duration in enumerate((9.0, 3.0, 6.0), 1):
            rows.append(
                {
                    "mode": mode,
                    "repetition": repetition,
                    "source_commit": "abc",
                    "plan_sha256": "plan",
                    "total_duration_seconds": duration + mode_index,
                    "success_rate": 1.0,
                    "retry_count": repetition - 1,
                    "resource_pressure": {
                        "queue_pressure_ratio": repetition / 10,
                        "peak_queued_predictions": repetition,
                        "busy_ratio": None,
                        "median_ttft_seconds": None,
                        "median_tokens_per_second": None,
                        "peak_workers": mode_index + 1,
                        "average_worker_utilization": 0.5,
                    },
                }
            )
    result = aggregate_benchmark(rows)
    assert result["formal_comparison"] is True
    assert result["medians"]["serial"]["median_total_duration_seconds"] == 6.0
    assert result["medians"]["serial"]["median_retry_count"] == 1.0
    pressure = result["medians"]["adaptive"]["resource_pressure"]
    assert pressure["median_queue_pressure_ratio"] == pytest.approx(0.2)
    assert pressure["median_busy_ratio"] is None

    rows[-1]["source_commit"] = "different"
    assert aggregate_benchmark(rows)["formal_comparison"] is False


@pytest.mark.asyncio
async def test_runner_uses_same_commit_and_plan_and_writes_all_reports(tmp_path: Path) -> None:
    expected_commit = _init_repo(tmp_path)
    seen: list[tuple[BenchmarkSpec, str, str]] = []

    async def execute(
        root: Path,
        plan: Plan,
        spec: BenchmarkSpec,
        fixed_concurrency: int,
        max_agents: int,
    ) -> RunState:
        assert fixed_concurrency == 2
        assert max_agents == 4
        commit = (await git(root, "rev-parse", "HEAD")).stdout.strip()
        seen.append((spec, commit, plan.model_dump_json()))
        scheduler = "fixed" if spec.mode in {"serial", "fixed"} else spec.mode
        task = TaskRuntime(spec=plan.tasks[0], status=TaskStatus.MERGED, attempts=spec.repetition)
        state = RunState(
            run_id=f"run-{spec.mode}-{spec.repetition}",
            goal=plan.goal,
            repository=str(root),
            integration_branch="unused",
            scheduler=scheduler,
            target_concurrency=1 if spec.mode == "serial" else 2,
            user_ceiling=1 if spec.mode == "serial" else max_agents,
            parallel_limit=4,
            tasks={"task": task},
        )
        run_dir = root / ".adaptea" / "runs" / state.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "telemetry.jsonl").write_text(
            json.dumps({"queued_predictions": spec.repetition - 1, "generating": True}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "runtime-status.jsonl").write_text(
            json.dumps({"running": 1 if spec.mode == "serial" else 2}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text('{"pass_rate":1}\n', encoding="utf-8")
        return state

    directory = await BenchmarkRunner(tmp_path, executor=execute).run(
        _plan(), repetitions=3, fixed_concurrency=2, max_agents=4, seed=11
    )

    assert len(seen) == 12
    assert {commit for _, commit, _ in seen} == {expected_commit}
    assert len({plan_json for _, _, plan_json in seen}) == 1
    report = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    assert report["formal_comparison"] is True
    assert set(report["medians"]) == set(MODES)
    assert len(report["runs"]) == 12
    assert report["source_commit"] == expected_commit
    assert (directory / "benchmark.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    with (directory / "benchmark.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert sum(row["row_type"] == "run" for row in csv_rows) == 12
    assert sum(row["row_type"] == "median" for row in csv_rows) == 4
    median_rows = [row for row in csv_rows if row["row_type"] == "median"]
    assert all(row["median_queue_pressure_ratio"] for row in median_rows)
    assert not (directory / "partial-results.json").exists()


@pytest.mark.asyncio
async def test_runner_rejects_dirty_repository(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "base.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean repository"):
        await BenchmarkRunner(tmp_path).run(_plan())
