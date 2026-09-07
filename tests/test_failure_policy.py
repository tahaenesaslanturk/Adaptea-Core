from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import Config
from adaptea.models import FailureKind, Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.reviewer.opencode import ReviewResult
from adaptea.runtime.controller import Orchestrator, prepare_resume
from adaptea.runtime.failures import (
    DEFAULT_FAILURE_POLICIES,
    DEFAULT_MAX_TOTAL_RETRIES,
    classify_worker_failure,
    decide_retry,
    load_retry_policy,
    retry_policy_document,
)
from adaptea.runtime.state import StateStore
from adaptea.validation import ValidationOutcome
from adaptea.workers.opencode import WorkerResult


@pytest.mark.parametrize(
    ("failure_type", "retry", "escalate"),
    [
        (FailureKind.MODEL_ERROR, True, True),
        (FailureKind.REVIEW_REJECTION, True, True),
        (FailureKind.TIMEOUT, True, True),
        (FailureKind.VALIDATION_FAILURE, True, True),
        (FailureKind.DEPENDENCY_PROBLEM, False, False),
        (FailureKind.INFRASTRUCTURE_ERROR, True, False),
    ],
)
def test_default_retry_policy_is_bounded_and_deterministic(
    failure_type: FailureKind, retry: bool, escalate: bool
) -> None:
    first = decide_retry(failure_type, 0, 0)
    assert first.retry is retry
    assert first.escalate_to_strong is escalate
    if retry:
        exhausted = decide_retry(failure_type, 1, 1)
        assert exhausted.retry is False
        assert "budget exhausted" in exhausted.reason


def test_global_retry_cap_prevents_category_hopping() -> None:
    decision = decide_retry(
        FailureKind.MODEL_ERROR,
        retries_used_for_type=0,
        total_retries_used=DEFAULT_MAX_TOTAL_RETRIES,
    )
    assert decision.retry is False
    assert decision.reason == "task retry budget exhausted (3/3)"


@pytest.mark.parametrize(
    ("exit_code", "stderr", "stdout", "expected"),
    [
        (124, "", "", FailureKind.TIMEOUT),
        (1, "model refused the request", "", FailureKind.MODEL_ERROR),
        (127, "command missing", "", FailureKind.INFRASTRUCTURE_ERROR),
        (1, "connect ECONNREFUSED 127.0.0.1", "", FailureKind.INFRASTRUCTURE_ERROR),
        (1, "No space left on device", "", FailureKind.INFRASTRUCTURE_ERROR),
        (
            1,
            "",
            '{"type":"error","error":{"name":"APIError","data":{"message":"No models loaded."}}}',
            FailureKind.INFRASTRUCTURE_ERROR,
        ),
    ],
)
def test_worker_failure_classifier_uses_observed_signals(
    exit_code: int, stderr: str, stdout: str, expected: FailureKind
) -> None:
    assert classify_worker_failure(exit_code, stderr, stdout) == expected


def test_retry_policy_snapshot_round_trip_and_invalid_fallback() -> None:
    document = retry_policy_document()
    policies, total = load_retry_policy(document)
    assert policies == DEFAULT_FAILURE_POLICIES
    assert total == DEFAULT_MAX_TOTAL_RETRIES
    fallback, fallback_total = load_retry_policy({"categories": {}})
    assert fallback == DEFAULT_FAILURE_POLICIES
    assert fallback_total == DEFAULT_MAX_TOTAL_RETRIES
    oversized = retry_policy_document()
    oversized["max_total_retries_per_task"] = 999_999
    assert load_retry_policy(oversized) == (
        DEFAULT_FAILURE_POLICIES,
        DEFAULT_MAX_TOTAL_RETRIES,
    )


def test_failed_dependency_is_classified_once_with_no_blind_retry() -> None:
    upstream = TaskRuntime(
        spec=TaskSpec(id="upstream", title="Upstream", description="fails"),
        status=TaskStatus.FAILED,
        failure_type=FailureKind.MODEL_ERROR,
    )
    downstream = TaskRuntime(
        spec=TaskSpec(
            id="downstream",
            title="Downstream",
            description="depends",
            depends_on=["upstream"],
        )
    )
    state = make_state(Path("."), upstream, downstream)
    state.refresh_readiness()
    state.refresh_readiness()
    blocked = state.tasks["downstream"]
    assert blocked.status == TaskStatus.BLOCKED
    assert blocked.failure_type == FailureKind.DEPENDENCY_PROBLEM
    assert blocked.retry_count == 0
    assert blocked.retry_exhausted is True
    assert "upstream (model_error)" in (blocked.failure or "")
    assert len(blocked.failure_history) == 1


def test_resume_consumes_one_infrastructure_retry_then_stops() -> None:
    task = TaskRuntime(
        spec=TaskSpec(id="task", title="Task", description="work"),
        status=TaskStatus.RUNNING,
        attempts=1,
    )
    state = make_state(Path("."), task)
    prepare_resume(state)
    assert task.status == TaskStatus.READY
    assert task.failure_type == FailureKind.INFRASTRUCTURE_ERROR
    assert task.retry_count == 1
    assert task.retry_counts_by_type == {"infrastructure_error": 1}

    task.status = TaskStatus.RUNNING
    task.attempts += 1
    prepare_resume(state)
    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 1
    assert task.retry_exhausted is True
    assert "retry budget exhausted" in (task.failure or "")


def test_resume_does_not_double_count_retry_reserved_before_restart() -> None:
    task = TaskRuntime(
        spec=TaskSpec(id="task", title="Task", description="work"),
        status=TaskStatus.RETRYING,
        attempts=2,
        retry_count=1,
        retry_counts_by_type={"timeout": 1},
        retry_pending=True,
        pending_retry_context="retry timeout once",
    )
    state = make_state(Path("."), task)
    prepare_resume(state)
    assert task.status == TaskStatus.READY
    assert task.attempts == 1
    assert task.retry_count == 1
    assert task.retry_counts_by_type == {"timeout": 1}
    assert task.pending_retry_context == "retry timeout once"


def test_resume_uses_the_run_policy_snapshot(tmp_path: Path) -> None:
    task = TaskRuntime(
        spec=TaskSpec(id="task", title="Task", description="work"),
        status=TaskStatus.RUNNING,
        attempts=1,
    )
    state = make_state(tmp_path, task)
    policy = retry_policy_document()
    policy["categories"]["infrastructure_error"]["max_retries"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps({"retry_policy": policy}), encoding="utf-8")
    prepare_resume(state, tmp_path)
    assert task.status == TaskStatus.FAILED
    assert task.retry_count == 0
    assert "retry budget exhausted (0/0)" in (task.failure or "")


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    (path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


def make_state(root: Path, *tasks: TaskRuntime) -> RunState:
    return RunState(
        run_id="run-failures",
        goal="exercise failure policy",
        repository=str(root),
        integration_branch="adaptea/run-failures/integration",
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={task.spec.id: task for task in tasks},
    )


def prepare_run(root: Path) -> tuple[RunState, Config]:
    init_repo(root)
    spec = TaskSpec(id="task", title="Task", description="exercise retry")
    state = make_state(root, TaskRuntime(spec=spec))
    state.refresh_readiness()
    run_dir = root / ".adaptea" / "runs" / state.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"model": "coder", "retry_policy": retry_policy_document()}),
        encoding="utf-8",
    )
    (run_dir / "plan.json").write_text(
        Plan(goal=state.goal, tasks=[spec]).model_dump_json(), encoding="utf-8"
    )
    for name in (
        "events.jsonl",
        "telemetry.jsonl",
        "controller-decisions.jsonl",
        "routing-decisions.jsonl",
        "runtime-status.jsonl",
        "failure-decisions.jsonl",
    ):
        (run_dir / name).touch()
    StateStore(run_dir).save(state)
    config = Config()
    config.worker.executable = "unused"
    config.lmstudio.lms_executable = "missing-lms"
    config.lmstudio.telemetry_poll_seconds = 0.01
    return state, config


class SequenceWorker:
    def __init__(self, outcomes: list[int | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.retry_contexts: list[str | None] = []

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        dependency_summaries: list[str],
        retry_context: str | None = None,
        progress: Callable[[dict[str, str]], None] | None = None,
    ) -> WorkerResult:
        del goal, dependency_summaries, progress
        self.retry_contexts.append(retry_context)
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(outcome, Exception):
            raise outcome
        (artifact_dir / "stderr.log").write_text("model failed\n", encoding="utf-8")
        if outcome == 0:
            (worktree / f"{task.id}.txt").write_text("completed\n", encoding="utf-8")
        return WorkerResult(outcome, "start", "end", 0.01, None, "worker summary")


class ApprovingReviewer:
    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        validation_output: str,
    ) -> ReviewResult:
        del worktree, artifact_dir, goal, task, validation_output
        return ReviewResult(True, "approved", [], 0, "start", "end", 0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([1, 1], FailureKind.MODEL_ERROR),
        ([124, 124], FailureKind.TIMEOUT),
        (
            [OSError("connection reset"), OSError("connection reset")],
            FailureKind.INFRASTRUCTURE_ERROR,
        ),
    ],
)
async def test_worker_failures_retry_once_then_show_terminal_reason(
    tmp_path: Path,
    outcomes: list[int | Exception],
    expected: FailureKind,
) -> None:
    state, config = prepare_run(tmp_path)
    worker = SequenceWorker(outcomes)
    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker = worker  # type: ignore[assignment]
    final = await orchestrator.run()

    task = final.tasks["task"]
    assert task.status == TaskStatus.FAILED
    assert task.failure_type == expected
    assert task.attempts == 2
    assert task.retry_count == 1
    assert task.retry_counts_by_type[expected.value] == 1
    assert task.retry_exhausted is True
    assert "retry budget exhausted" in (task.failure or "")
    assert worker.calls == 2
    assert worker.retry_contexts[0] is None
    assert expected.value.replace("_", " ").split()[0].title() in worker.retry_contexts[1]
    decisions = [
        json.loads(line)
        for line in (tmp_path / ".adaptea" / "runs" / state.run_id / "failure-decisions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["decision"] for row in decisions] == ["retry", "fail"]


@pytest.mark.asyncio
async def test_validation_failure_retries_once_then_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, config = prepare_run(tmp_path)
    worker = SequenceWorker([0, 0])

    async def fail_validation(*_args: Any, **_kwargs: Any) -> ValidationOutcome:
        return ValidationOutcome(False, 1, "2 tests failed", ["pytest"])

    monkeypatch.setattr("adaptea.runtime.controller.run_validation", fail_validation)
    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker = worker  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]
    final = await orchestrator.run()

    task = final.tasks["task"]
    assert task.status == TaskStatus.FAILED
    assert task.failure_type == FailureKind.VALIDATION_FAILURE
    assert task.attempts == 2
    assert task.retry_count == 1
    assert "2 tests failed" in (task.failure or "")
    assert "Validation failure" in (worker.retry_contexts[1] or "")
