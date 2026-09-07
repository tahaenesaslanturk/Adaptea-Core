from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from adaptea.config import Config
from adaptea.git.integration import commit_worker_changes, merge_branch
from adaptea.git.repository import git, sanitize_branch
from adaptea.git.worktrees import WorktreeManager
from adaptea.models import (
    MergeConflictState,
    Plan,
    RunState,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
)
from adaptea.reviewer.opencode import ReviewResult
from adaptea.runtime.controller import Orchestrator, prepare_resume
from adaptea.runtime.state import StateStore
from adaptea.workers.opencode import WorkerResult


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    (path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    (path / "shared.txt").write_text("base\n", encoding="utf-8")
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


def test_windows_safe_branch_names() -> None:
    assert sanitize_branch("Auth API: phase/one") == "auth-api-phase-one"
    assert sanitize_branch("CON") == "task-con"
    assert "/" not in sanitize_branch("a/b")


@pytest.mark.asyncio
async def test_completed_integration_checkout_cleanup_removes_empty_run_container(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    manager = WorktreeManager(tmp_path, "run-complete")
    await manager.create_integration()

    await manager.remove(manager.integration_path)

    assert not manager.integration_path.exists()
    assert not manager.root.exists()


@pytest.mark.asyncio
async def test_worktree_merge_success_and_conflict_cleanup(tmp_path: Path) -> None:
    init_repo(tmp_path)
    manager = WorktreeManager(tmp_path, "run-test")
    integration = await manager.create_integration()
    task_path, branch = await manager.create_task("feature", 1)
    (task_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    await commit_worker_changes(task_path, "feature")
    assert (await merge_branch(integration, branch)).returncode == 0
    assert (integration / "feature.txt").read_text() == "feature\n"
    await manager.remove(task_path)

    conflict_path, conflict_branch = await manager.create_task("conflict", 1)
    (conflict_path / "shared.txt").write_text("task\n", encoding="utf-8")
    await commit_worker_changes(conflict_path, "conflict")
    (integration / "shared.txt").write_text("integration\n", encoding="utf-8")
    await git(integration, "add", "shared.txt")
    await git(
        integration,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "integration change",
    )
    result = await merge_branch(integration, conflict_branch)
    assert result.returncode != 0
    assert result.conflicting_files == ["shared.txt"]
    assert (await git(integration, "status", "--porcelain")).stdout == ""


@pytest.mark.asyncio
async def test_manual_resolution_worktree_keeps_integration_clean(tmp_path: Path) -> None:
    init_repo(tmp_path)
    manager = WorktreeManager(tmp_path, "run-resolution")
    integration = await manager.create_integration()
    task_path, task_branch = await manager.create_task("conflict", 2)
    (task_path / "shared.txt").write_text("task change\n", encoding="utf-8")
    await commit_worker_changes(task_path, "conflict")

    upstream_path, upstream_branch = await manager.create_task("upstream", 1)
    (upstream_path / "shared.txt").write_text("upstream change\n", encoding="utf-8")
    await commit_worker_changes(upstream_path, "upstream")
    assert (await merge_branch(integration, upstream_branch)).returncode == 0

    failed = await merge_branch(integration, task_branch)
    assert failed.conflicting_files == ["shared.txt"]
    assert (await git(integration, "status", "--porcelain")).stdout == ""

    resolution, resolution_branch = await manager.create_resolution("conflict", 2)
    prepared = await git(
        resolution,
        "-c",
        "user.name=Adaptea",
        "-c",
        "user.email=adaptea@localhost",
        "merge",
        "--no-ff",
        "--no-commit",
        task_branch,
        check=False,
    )
    assert prepared.returncode != 0
    unmerged = await git(resolution, "diff", "--name-only", "--diff-filter=U")
    assert unmerged.stdout.splitlines() == ["shared.txt"]
    assert "UU shared.txt" in (await git(resolution, "status", "--short")).stdout
    assert (await git(integration, "status", "--porcelain")).stdout == ""

    (resolution / "shared.txt").write_text("upstream change\ntask change\n", encoding="utf-8")
    await git(resolution, "add", "--", "shared.txt")
    await git(
        resolution,
        "-c",
        "user.name=Human",
        "-c",
        "user.email=human@example.invalid",
        "commit",
        "-m",
        "resolve conflict manually",
    )
    assert (await merge_branch(integration, resolution_branch)).returncode == 0
    assert (integration / "shared.txt").read_text() == "upstream change\ntask change\n"


@pytest.mark.asyncio
async def test_conflict_resolution_state_recovers_after_crash_and_resume(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    run_id = "run-crash-resolution"
    manager = WorktreeManager(tmp_path, run_id)
    integration = await manager.create_integration()
    task_path, task_branch = await manager.create_task("conflict", 2)
    (task_path / "shared.txt").write_text("task change\n", encoding="utf-8")
    await commit_worker_changes(task_path, "conflict")
    upstream_path, upstream_branch = await manager.create_task("upstream", 1)
    (upstream_path / "shared.txt").write_text("upstream change\n", encoding="utf-8")
    await commit_worker_changes(upstream_path, "upstream")
    assert (await merge_branch(integration, upstream_branch)).returncode == 0
    failed = await merge_branch(integration, task_branch)
    resolution_path, resolution_branch = manager.resolution_spec("conflict", 2)

    current = TaskRuntime(
        spec=TaskSpec(id="conflict", title="Conflict", description="conflicting edit"),
        status=TaskStatus.NEEDS_MANUAL_RESOLUTION,
        attempts=2,
        branch=task_branch,
        worktree=str(task_path),
        merge_conflict=MergeConflictState(
            status="preparing",
            source_branch=task_branch,
            integration_branch=manager.integration_branch,
            original_worktree=str(task_path),
            resolution_worktree=str(resolution_path),
            resolution_branch=resolution_branch,
            conflicting_files=failed.conflicting_files,
        ),
    )
    upstream = TaskRuntime(
        spec=TaskSpec(id="upstream", title="Upstream", description="upstream edit"),
        status=TaskStatus.MERGED,
        attempts=1,
        branch=upstream_branch,
    )
    state = RunState(
        run_id=run_id,
        goal="resolve safely",
        repository=str(tmp_path),
        integration_branch=manager.integration_branch,
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={"conflict": current, "upstream": upstream},
    )
    run_dir = tmp_path / ".adaptea" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"model":"coder"}\n', encoding="utf-8")
    (run_dir / "events.jsonl").touch()
    StateStore(run_dir).save(state)
    config = Config()
    config.lmstudio.lms_executable = "missing-lms"

    first = Orchestrator(tmp_path, config, state)
    await first._recover_manual_resolutions()
    conflict = state.tasks["conflict"].merge_conflict
    assert conflict is not None
    assert conflict.status == "ready"
    assert conflict.conflicting_files == ["shared.txt"]
    assert conflict.related_tasks == ["conflict", "upstream"]
    assert (await git(integration, "status", "--porcelain")).stdout == ""
    artifact = json.loads(
        (run_dir / "tasks" / "conflict" / "attempt-2" / "merge-conflict.json").read_text()
    )
    assert artifact["status"] == "ready"
    assert artifact["conflicting_files"] == ["shared.txt"]
    assert "semantic" in " ".join(artifact["manual_steps"]).lower()

    (resolution_path / "shared.txt").write_text("upstream change\ntask change\n", encoding="utf-8")
    await git(resolution_path, "add", "--", "shared.txt")
    await git(
        resolution_path,
        "-c",
        "user.name=Human",
        "-c",
        "user.email=human@example.invalid",
        "commit",
        "-m",
        "manual semantic resolution",
    )

    restored = prepare_resume(StateStore(run_dir).load(), run_dir)
    assert restored.tasks["conflict"].status == TaskStatus.NEEDS_MANUAL_RESOLUTION
    resumed = Orchestrator(tmp_path, config, restored)
    await resumed._recover_manual_resolutions()
    resolved = restored.tasks["conflict"]
    assert resolved.status == TaskStatus.MERGED
    assert resolved.merge_conflict is not None
    assert resolved.merge_conflict.status == "resolved"
    assert (integration / "shared.txt").read_text() == "upstream change\ntask change\n"


@pytest.mark.parametrize(
    ("used_retries", "expected_status", "conflict_retained"),
    [
        (0, TaskStatus.READY, False),
        (1, TaskStatus.NEEDS_MANUAL_RESOLUTION, True),
    ],
)
def test_resume_recovers_crash_immediately_after_conflict_detection(
    used_retries: int,
    expected_status: TaskStatus,
    conflict_retained: bool,
) -> None:
    task = TaskRuntime(
        spec=TaskSpec(id="conflict", title="Conflict", description="conflicting edit"),
        status=TaskStatus.REVIEWING,
        attempts=used_retries + 1,
        retry_count=used_retries,
        retry_counts_by_type={"infrastructure_error": used_retries},
        merge_conflict=MergeConflictState(
            status="detected",
            source_branch="adaptea/conflict",
            integration_branch="adaptea/integration",
            original_worktree="/tmp/task-conflict",
            resolution_worktree="/tmp/resolution-conflict",
            resolution_branch="adaptea/resolve-conflict",
            conflicting_files=["shared.txt"],
        ),
    )
    state = RunState(
        run_id="run-detected",
        goal="recover detected conflict",
        repository=".",
        integration_branch="adaptea/integration",
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={"conflict": task},
    )
    prepare_resume(state)
    assert task.status == expected_status
    assert (task.merge_conflict is not None) is conflict_retained
    assert task.failure_history[-1]["source"] == "resume_merge_conflict"
    if used_retries == 0:
        assert task.retry_count == 1
    else:
        assert "safe manual resolution worktree" in (task.failure or "")


class FakeWorker:
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
        del goal, dependency_summaries, retry_context
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress({"kind": "tool", "title": "write", "detail": f"{task.id}.txt"})
        (worktree / f"{task.id}.txt").write_text(task.title + "\n", encoding="utf-8")
        for name in ("stdout.log", "stderr.log", "worker-events.jsonl", "summary.json"):
            (artifact_dir / name).write_text("{}\n", encoding="utf-8")
        return WorkerResult(0, "start", "end", 0.01, "session", f"completed {task.id}")


class FakeReviewer:
    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        validation_output: str,
    ) -> ReviewResult:
        del worktree, artifact_dir, goal, task, validation_output
        return ReviewResult(True, "acceptance criteria satisfied", [], 0, "start", "end", 0.01)


@pytest.mark.asyncio
async def test_mocked_end_to_end_plan_worker_validate_merge_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_repo(tmp_path)
    plan = Plan(
        goal="build two features",
        tasks=[
            TaskSpec(id="a", title="A", description="feature A"),
            TaskSpec(id="b", title="B", description="feature B", depends_on=["a"]),
            TaskSpec(id="c", title="C", description="independent C"),
        ],
    )
    run_id = "run-e2e"
    manager = WorktreeManager(tmp_path, run_id)
    state = RunState(
        run_id=run_id,
        goal=plan.goal,
        repository=str(tmp_path),
        integration_branch=manager.integration_branch,
        scheduler="fixed",
        target_concurrency=2,
        user_ceiling=2,
        parallel_limit=4,
        tasks={task.id: TaskRuntime(spec=task) for task in plan.tasks},
    )
    state.refresh_readiness()
    run_dir = tmp_path / ".adaptea" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"model": "coder", "integration_branch": manager.integration_branch}),
        encoding="utf-8",
    )
    for name in ("events.jsonl", "telemetry.jsonl", "controller-decisions.jsonl"):
        (run_dir / name).touch()
    StateStore(run_dir).save(state)
    config = Config()
    config.worker.executable = "unused"
    config.lmstudio.lms_executable = "missing-lms"
    config.lmstudio.telemetry_poll_seconds = 0.01
    config.project.test_command = [
        "python",
        "-c",
        "import pathlib,sys; sys.exit(0 if pathlib.Path('.git').exists() else 1)",
    ]
    steps: list[tuple[str, dict[str, str]]] = []
    orchestrator = Orchestrator(
        tmp_path,
        config,
        state,
        activity_callback=lambda task_id, entry: steps.append((task_id, entry)),
    )
    orchestrator.worker = FakeWorker()  # type: ignore[assignment]
    orchestrator.reviewer = FakeReviewer()  # type: ignore[assignment]
    final = await orchestrator.run()
    assert all(task.status == TaskStatus.MERGED for task in final.tasks.values())
    # Each worker's steps reach the caller attributed to the task that produced them,
    # which is what lets the desktop narrate a run while it is still going.
    assert sorted(task_id for task_id, _ in steps) == ["a", "b", "c"]
    assert {entry["title"] for _, entry in steps} == {"write"}
    # Successful runs retain their integration branch but not a duplicate checkout.
    assert not manager.integration_path.exists()
    for filename in ("a.txt", "b.txt", "c.txt"):
        content = await git(tmp_path, "show", f"{manager.integration_branch}:{filename}")
        assert content.returncode == 0
    persisted = StateStore(run_dir).load()
    assert persisted.tasks["a"].attempts == 1
    assert json.loads((run_dir / "summary.json").read_text())["pass_rate"] == 1.0
