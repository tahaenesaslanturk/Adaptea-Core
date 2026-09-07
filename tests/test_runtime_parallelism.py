from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import Config
from adaptea.inference import BackendCapabilities
from adaptea.lmstudio.models import LMModel
from adaptea.models import Plan, TaskSpec, TaskStatus
from adaptea.runtime.controller import Orchestrator, create_run
from adaptea.workers.opencode import WorkerResult


class FakeBackend:
    capabilities = BackendCapabilities(
        model_lifecycle=True,
        live_pressure_metrics=True,
        safe_default_parallel_limit=4,
    )

    async def __aenter__(self) -> FakeBackend:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def models(self) -> list[LMModel]:
        return [
            LMModel(
                type="llm",
                key="test-model",
                inference_ready=True,
                max_context_length=65536,
                parallel_limit=4,
            )
        ]


class FakeObserver:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def start(self, *args: Any, **kwargs: Any) -> bool:
        return True

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        pass


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True
    )


class ParallelTrackingWorker:
    def __init__(self) -> None:
        self.active_tasks: set[str] = set()
        self.peak_parallelism: int = 0
        self.lock = asyncio.Lock()

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        dependency_summaries: list[str],
        retry_context: str | None = None,
        progress: Any = None,
        plan_context: Any = None,
    ) -> WorkerResult:
        del goal, dependency_summaries, retry_context, progress, plan_context
        artifact_dir.mkdir(parents=True, exist_ok=True)
        async with self.lock:
            self.active_tasks.add(task.id)
            if len(self.active_tasks) > self.peak_parallelism:
                self.peak_parallelism = len(self.active_tasks)

        (worktree / f"{task.id}.txt").write_text(f"Done by {task.id}\n", encoding="utf-8")
        await asyncio.sleep(0.15)

        async with self.lock:
            self.active_tasks.remove(task.id)

        return WorkerResult(
            exit_code=0,
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            wall_seconds=0.15,
            session_id=None,
            summary=f"Finished {task.id}",
        )


class ApprovingReviewer:
    async def run(
        self, worktree: Path, artifact_dir: Path, goal: str, task: TaskSpec, validation: str
    ) -> Any:
        del worktree, artifact_dir, goal, task, validation

        class ReviewResult:
            approved = True
            reason = "LGTM"
            findings: list[str] = []
            wall_seconds = 0.05

        return ReviewResult()


async def _noop_ensure_models(*_args: Any, **_kwargs: Any) -> None:
    pass


@pytest.fixture(autouse=True)
def mock_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adaptea.runtime.controller.create_inference_backend",
        lambda *_args, **_kwargs: FakeBackend(),
    )
    monkeypatch.setattr("adaptea.runtime.controller.LogObserver", FakeObserver)
    monkeypatch.setattr(
        "adaptea.runtime.controller.Orchestrator._ensure_models_loaded",
        _noop_ensure_models,
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_independent_tasks_in_parallel_fixed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    config = Config(worker={"max_agents": 4}, lmstudio={"model": "test-model"})
    plan = Plan(
        goal="Build feature in parallel",
        tasks=[
            TaskSpec(id="task1", title="Task 1", description="Do task 1", depends_on=[]),
            TaskSpec(id="task2", title="Task 2", description="Do task 2", depends_on=[]),
            TaskSpec(id="task3", title="Task 3", description="Do task 3", depends_on=[]),
            TaskSpec(id="task4", title="Task 4", description="Do task 4", depends_on=[]),
        ],
    )

    state = await create_run(repo, config, plan, "fixed", 4, 4)
    orchestrator = Orchestrator(repo, config, state)
    worker = ParallelTrackingWorker()
    orchestrator.worker = worker  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]

    final_state = await orchestrator.run()

    assert final_state.complete
    assert all(task.status == TaskStatus.MERGED for task in final_state.tasks.values())
    assert worker.peak_parallelism == 4, (
        f"Expected peak parallelism of 4, got {worker.peak_parallelism}"
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_independent_tasks_in_parallel_naive(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    config = Config(worker={"max_agents": 4}, lmstudio={"model": "test-model"})
    plan = Plan(
        goal="Build feature in parallel",
        tasks=[
            TaskSpec(id="task1", title="Task 1", description="Do task 1", depends_on=[]),
            TaskSpec(id="task2", title="Task 2", description="Do task 2", depends_on=[]),
            TaskSpec(id="task3", title="Task 3", description="Do task 3", depends_on=[]),
        ],
    )

    state = await create_run(repo, config, plan, "naive", 4, None)
    orchestrator = Orchestrator(repo, config, state)
    worker = ParallelTrackingWorker()
    orchestrator.worker = worker  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]

    final_state = await orchestrator.run()

    assert final_state.complete
    assert all(task.status == TaskStatus.MERGED for task in final_state.tasks.values())
    assert worker.peak_parallelism == 3, (
        f"Expected peak parallelism of 3, got {worker.peak_parallelism}"
    )


@pytest.mark.asyncio
async def test_orchestrator_respects_wave_dependencies_with_parallelism(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    config = Config(worker={"max_agents": 4}, lmstudio={"model": "test-model"})
    plan = Plan(
        goal="Build feature with waves",
        tasks=[
            TaskSpec(id="base", title="Base", description="Base task", depends_on=[]),
            TaskSpec(id="branch_a", title="Branch A", description="Branch A", depends_on=["base"]),
            TaskSpec(id="branch_b", title="Branch B", description="Branch B", depends_on=["base"]),
            TaskSpec(
                id="final",
                title="Final",
                description="Final task",
                depends_on=["branch_a", "branch_b"],
            ),
        ],
    )

    state = await create_run(repo, config, plan, "fixed", 4, 4)
    orchestrator = Orchestrator(repo, config, state)
    worker = ParallelTrackingWorker()
    orchestrator.worker = worker  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]

    final_state = await orchestrator.run()

    assert final_state.complete
    assert all(task.status == TaskStatus.MERGED for task in final_state.tasks.values())
    # During branch_a and branch_b, peak parallelism should be 2
    assert worker.peak_parallelism == 2


@pytest.mark.asyncio
async def test_orchestrator_runs_independent_tasks_in_parallel_adaptive(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    config = Config(worker={"max_agents": 4}, lmstudio={"model": "test-model"})
    plan = Plan(
        goal="Build feature in parallel adaptive",
        tasks=[
            TaskSpec(id="task1", title="Task 1", description="Do task 1", depends_on=[]),
            TaskSpec(id="task2", title="Task 2", description="Do task 2", depends_on=[]),
            TaskSpec(id="task3", title="Task 3", description="Do task 3", depends_on=[]),
            TaskSpec(id="task4", title="Task 4", description="Do task 4", depends_on=[]),
        ],
    )

    state = await create_run(repo, config, plan, "adaptive", 4, 3)
    orchestrator = Orchestrator(repo, config, state)
    worker = ParallelTrackingWorker()
    orchestrator.worker = worker  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]

    final_state = await orchestrator.run()

    assert final_state.complete
    assert all(task.status == TaskStatus.MERGED for task in final_state.tasks.values())
    assert worker.peak_parallelism >= 3
