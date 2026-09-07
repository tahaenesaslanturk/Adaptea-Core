from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from adaptea.config import Config, load_config
from adaptea.desktop.protocol import DesktopCommand
from adaptea.desktop.server import DesktopBridge, _download_progress
from adaptea.diagnostics.doctor import Check
from adaptea.environment import environment_root
from adaptea.lmstudio.lms_cli import CommandResult
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.preferences import (
    ProjectPreferences,
    load_project_preferences,
    preferences_path,
    save_project_preferences,
)
from adaptea.runtime.state import StateStore
from adaptea.security.approvals import ApprovalRequest
from adaptea.services import ApplicationServices
from adaptea.setup.actions import InstallAction
from adaptea.smoke import SmokeStep, SmokeTestResult


async def responses(bridge: DesktopBridge) -> list[dict[str, Any]]:
    pending = list(bridge.requests.values())
    if pending:
        await asyncio.gather(*pending)
    return [json.loads(line) for line in bridge.output.getvalue().splitlines()]  # type: ignore[union-attr]


def test_completed_runs_apply_to_the_workspace_by_default() -> None:
    assert ProjectPreferences().completion_action == "commit"
    assert ProjectPreferences().auto_commit_completed_runs is True
    assert ProjectPreferences().auto_start_plans is False


def test_legacy_false_auto_commit_preference_migrates_to_no_commit(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    path = preferences_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"auto_commit_completed_runs": false}\n', encoding="utf-8")

    assert load_project_preferences(tmp_path).completion_action == "none"


@pytest.mark.asyncio
async def test_bridge_request_response_events_and_malformed_input(tmp_path: Path) -> None:
    class Services(ApplicationServices):
        async def diagnose(self, root: Path) -> list[Check]:
            assert root == tmp_path
            return [Check("PASS", "Git", "ready")]

    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    await bridge.accept("not-json")
    await bridge.accept(
        DesktopCommand(
            id="doctor-1", command="doctor", payload={"root": str(tmp_path)}
        ).model_dump_json()
    )
    rows = await responses(bridge)
    assert rows[0]["error"]["code"] == "invalid_request"
    assert [row.get("event") for row in rows if row["type"] == "event"] == [
        "doctor.started",
        "doctor.updated",
    ]
    response = next(row for row in rows if row.get("id") == "doctor-1")
    assert response["ok"] is True
    assert response["data"][0]["name"] == "Git"


@pytest.mark.asyncio
async def test_resumed_run_keeps_streaming_worker_activity(tmp_path: Path) -> None:
    state = RunState(
        run_id="run-resume-activity",
        goal="Retry it",
        repository=str(tmp_path),
        integration_branch="adaptea/run-resume-activity/integration",
        scheduler="adaptive",
        target_concurrency=1,
        user_ceiling=2,
        parallel_limit=2,
        tasks={
            "task-a": TaskRuntime(
                spec=TaskSpec(id="task-a", title="Task A", description="retry"),
                status=TaskStatus.MERGED,
            )
        },
    )

    class Services(ApplicationServices):
        async def resume_run(
            self,
            root: Path,
            run_id: str,
            callback: Any = None,
            activity: Any = None,
            approval: Any = None,
            *,
            retry_failed: bool = False,
        ) -> RunState:
            assert root == tmp_path and run_id == state.run_id
            assert activity is not None
            activity(
                "task-a",
                {"kind": "tool", "title": "Editing src/app.py", "detail": ""},
            )
            return state

    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    await bridge.dispatch("run.resume", {"root": str(tmp_path), "run_id": state.run_id})
    await asyncio.sleep(0)

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    activity_event = next(row for row in events if row.get("event") == "task.activity")
    assert activity_event["data"]["title"] == "Editing src/app.py"


@pytest.mark.asyncio
async def test_bridge_reuses_smoke_test_for_layered_setup_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def verify(root: Path, **kwargs: Any) -> SmokeTestResult:
        assert root == tmp_path
        step = SmokeStep(
            "LM Studio server",
            False,
            "Connection refused.",
            layer="LM Studio",
            remedy="Open LM Studio and start the local server.",
        )
        kwargs["progress"](step)
        kwargs["activity"]("Checking LM Studio…")
        return SmokeTestResult(False, [step], 0.1)

    monkeypatch.setattr("adaptea.desktop.server.run_mvp_smoke_test", verify)
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())
    await bridge.accept(
        DesktopCommand(
            id="verify-1",
            command="smoke.start",
            payload={"root": str(tmp_path)},
        ).model_dump_json()
    )
    rows = await responses(bridge)
    response = next(row for row in rows if row.get("id") == "verify-1")

    assert response["data"]["steps"][0]["layer"] == "LM Studio"
    assert response["data"]["steps"][0]["remedy"].startswith("Open LM Studio")


@pytest.mark.asyncio
async def test_bridge_cancels_a_running_calibration_request(tmp_path: Path) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Services(ApplicationServices):
        async def calibrate(
            self,
            root: Path,
            *,
            quick: bool,
            max_agents: int | None = None,
            progress: Any = None,
        ) -> Path:
            # Measuring is about the machine and its models, not a repository, so it
            # runs in the environment directory whichever project is open.
            assert root == environment_root() and quick
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return root

    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    await bridge.accept(
        DesktopCommand(
            id="calibration-1",
            command="calibration.start",
            payload={"root": str(tmp_path), "quick": True},
        ).model_dump_json()
    )
    await asyncio.wait_for(started.wait(), 1)
    await bridge.accept(
        DesktopCommand(
            id="cancel-1",
            command="command.cancel",
            payload={"request_id": "calibration-1"},
        ).model_dump_json()
    )
    await asyncio.wait_for(stopped.wait(), 1)
    rows = await responses(bridge)
    cancelled = next(row for row in rows if row.get("id") == "calibration-1")
    assert cancelled["ok"] is False
    assert cancelled["error"]["code"] == "cancelled"


@pytest.mark.asyncio
async def test_bridge_protocol_version_and_unknown_command(tmp_path: Path) -> None:
    output = io.StringIO()
    bridge = DesktopBridge(output=output)
    await bridge.accept(
        json.dumps(
            {
                "protocol": 99,
                "id": "old",
                "type": "command",
                "command": "app.ping",
                "payload": {},
            }
        )
    )
    await bridge.accept(
        DesktopCommand(
            id="unknown", command="unknown.command", payload={"root": str(tmp_path)}
        ).model_dump_json()
    )
    rows = await responses(bridge)
    assert rows[0]["error"]["code"] == "invalid_request"
    assert rows[-1]["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_desktop_can_initialize_an_opened_project_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTEA_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "README.md").write_text("# Existing project\n", encoding="utf-8")
    bridge = DesktopBridge(output=io.StringIO())
    project = await bridge.dispatch("project.initialize_git", {"root": str(tmp_path)})
    assert project.is_git is True
    assert project.git_branch == "main"
    assert (tmp_path / ".git").is_dir()


@pytest.mark.asyncio
async def test_real_sidecar_process_startup_ping_shutdown_and_restart() -> None:
    async def session() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "adaptea.desktop",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None
        ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert ready["event"] == "bridge.ready"
        process.stdin.write(
            (DesktopCommand(id="ping", command="app.ping").model_dump_json() + "\n").encode()
        )
        await process.stdin.drain()
        pong = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert pong["id"] == "ping" and pong["data"]["protocol"] == 1
        process.stdin.write(
            (DesktopCommand(id="stop", command="app.shutdown").model_dump_json() + "\n").encode()
        )
        await process.stdin.drain()
        stopped = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        assert stopped["ok"] is True
        process.stdin.close()
        await asyncio.wait_for(process.wait(), 5)
        assert process.returncode == 0

    await session()
    await session()


def sample_plan() -> Plan:
    return Plan(goal="Build it", tasks=[TaskSpec(id="task-a", title="A", description="A")])


def sample_run(root: Path, *, status: TaskStatus = TaskStatus.READY) -> RunState:
    task = sample_plan().tasks[0]
    return RunState(
        run_id="run-desktop",
        goal="Build it",
        repository=str(root),
        integration_branch="adaptea/integration/run-desktop",
        scheduler="adaptive",
        target_concurrency=2,
        user_ceiling=4,
        parallel_limit=4,
        tasks={task.id: TaskRuntime(spec=task, status=status)},
    )


@pytest.mark.asyncio
async def test_desktop_core_plan_run_resume_status_and_fleet(tmp_path: Path) -> None:
    services = ApplicationServices()
    plan = sample_plan()
    running = sample_run(tmp_path)
    complete = sample_run(tmp_path, status=TaskStatus.MERGED)
    services.plan = AsyncMock(return_value=plan)  # type: ignore[method-assign]
    services.fleet_status = AsyncMock(return_value={"enabled": True})  # type: ignore[method-assign]
    services.lmstudio_health = AsyncMock(  # type: ignore[method-assign]
        return_value={"reachable": False, "loaded_models": []}
    )
    services.unload_unconfigured_fleet_models = AsyncMock(return_value=[])  # type: ignore[method-assign]
    services.activate_configured_fleet = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
    services.create_run = AsyncMock(return_value=running)  # type: ignore[method-assign]
    services.execute_run = AsyncMock(return_value=complete)  # type: ignore[method-assign]
    services.resume_run = AsyncMock(return_value=complete)  # type: ignore[method-assign]
    output = io.StringIO()
    bridge = DesktopBridge(services=services, output=output)
    base = {"root": str(tmp_path)}

    assert await bridge.dispatch("fleet.status", base) == {"enabled": True}
    assert await bridge.dispatch("lmstudio.health", base) == {
        "reachable": False,
        "loaded_models": [],
    }
    fleet_payload = {
        "enabled": True,
        "topology": "explicit",
        "max_loaded_instances": 1,
        "models": [
            {
                "name": "local-strong",
                "model": "publisher/model",
                "tier": "strong",
                "roles": ["planner", "worker"],
                "instances": 1,
                "context_length": 32768,
                "parallel_limit": 2,
            }
        ],
    }
    fleet_result = await bridge.dispatch("fleet.configure", base | {"fleet": fleet_payload})
    # The fleet is the machine's, not this project's: it is written once, centrally, and
    # reaches a project when that project is opened.
    assert fleet_result["path"] == str(environment_root() / "adaptea.toml")
    await bridge.dispatch("environment.apply", {"root": str(tmp_path)})
    config_text = (tmp_path / "adaptea.toml").read_text(encoding="utf-8")
    assert 'model = "publisher/model"' in config_text
    assert 'roles = ["planner", "worker"]' in config_text
    assert services.activate_configured_fleet.await_count == 1
    # The desktop loads the chosen models one at a time so it can show which one it is on,
    # so a save from there writes the configuration and leaves the loading to it.
    await bridge.dispatch("fleet.configure", base | {"fleet": fleet_payload, "activate": False})
    assert services.activate_configured_fleet.await_count == 1
    saved = await bridge.dispatch(
        "fleet.combination.save", base | {"fleet": fleet_payload, "name": "Local coding"}
    )
    combination_id = saved["combination"]["id"]
    assert saved["combination"]["name"] == "Local coding"
    # Saved combinations are a library of the machine's model setups, kept once. Storing
    # them per project meant configuring — and re-measuring — the same models again for
    # every folder that used them.
    assert (environment_root() / ".adaptea" / "model-combinations.json").is_file()
    assert not (tmp_path / ".adaptea" / "model-combinations.json").exists()
    selected_combination = await bridge.dispatch(
        "fleet.combination.select", base | {"combination_id": combination_id}
    )
    assert selected_combination["combination"]["id"] == combination_id
    await bridge.dispatch("fleet.combination.delete", base | {"combination_id": combination_id})
    generated = await bridge.dispatch(
        "plan.generate", base | {"goal": "Build it", "chat": "chat-7"}
    )
    assert generated == plan
    planning_events = [
        row
        for row in (json.loads(line) for line in output.getvalue().splitlines())
        if row.get("event", "").startswith("plan.")
    ]
    assert planning_events
    assert all(row["data"]["root"] == str(tmp_path) for row in planning_events)
    # The desktop can hold several conversations about one folder, so the path alone
    # cannot say whose planning narration this is. Echoing the caller's own identifier
    # can; the core never interprets it.
    assert all(row["data"]["chat"] == "chat-7" for row in planning_events)
    created = await bridge.dispatch(
        "run.create", base | {"plan": plan.model_dump(), "mode": "adaptive", "max_agents": 4}
    )
    assert created.run_id == "run-desktop"
    run_dir = tmp_path / ".adaptea" / "runs" / running.run_id
    run_dir.mkdir(parents=True)
    running.tasks["task-a"].attempts = 1
    StateStore(run_dir).save(running)
    artifact_dir = run_dir / "tasks" / "task-a" / "attempt-1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "stdout.log").write_text("real worker output", encoding="utf-8")
    (artifact_dir / "validation.log").write_text("1 passed", encoding="utf-8")
    (artifact_dir / "failure-decisions.jsonl").write_text(
        '{"type":"validation_failure","decision":"retry"}\n', encoding="utf-8"
    )
    assert (await bridge.dispatch("run.status", base | {"run_id": running.run_id})).run_id
    artifacts = await bridge.dispatch(
        "task.artifacts", base | {"run_id": running.run_id, "task_id": "task-a"}
    )
    assert artifacts["output"] == "real worker output"
    assert artifacts["validation"] == "1 passed"
    assert '"decision":"retry"' in artifacts["failure_decisions"]
    assert (await bridge.dispatch("run.start", base | {"run_id": running.run_id})).complete
    assert (await bridge.dispatch("run.resume", base | {"run_id": running.run_id})).complete


@pytest.mark.asyncio
async def test_apply_run_preserves_unrelated_changes_and_requires_a_fully_merged_run(
    tmp_path: Path,
) -> None:
    def local_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    local_git("init", "-b", "main")
    (tmp_path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    local_git("add", ".")
    local_git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    state = sample_run(tmp_path, status=TaskStatus.MERGED)
    state.source_branch = "main"
    state.source_commit = local_git("rev-parse", "HEAD").stdout.strip()
    local_git("checkout", "-b", state.integration_branch)
    (tmp_path / "result.txt").write_text("completed\n", encoding="utf-8")
    local_git("add", "result.txt")
    local_git(
        "-c",
        "user.name=Adaptea",
        "-c",
        "user.email=adaptea@localhost",
        "commit",
        "-m",
        "completed run",
    )
    (tmp_path / "second-result.txt").write_text("also completed\n", encoding="utf-8")
    local_git("add", "second-result.txt")
    local_git(
        "-c",
        "user.name=Adaptea",
        "-c",
        "user.email=adaptea@localhost",
        "commit",
        "-m",
        "completed another task",
    )
    local_git("checkout", "main")
    run_dir = tmp_path / ".adaptea" / "runs" / state.run_id
    StateStore(run_dir).save(state)
    bridge = DesktopBridge(output=io.StringIO())

    # A path that the reviewed branch would overwrite remains fail-closed through Git.
    (tmp_path / "result.txt").write_text("local collision\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not apply the completed run"):
        await bridge.dispatch("run.apply", {"root": str(tmp_path), "run_id": state.run_id})
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "local collision\n"
    (tmp_path / "result.txt").unlink()

    # Unrelated local work is not a reason to strand the completed run on its internal
    # integration branch; Git preserves it during the fast-forward.
    (tmp_path / "local.txt").write_text("uncommitted\n", encoding="utf-8")

    local_git("checkout", "-b", "another-branch")
    with pytest.raises(ValueError, match="started on main"):
        await bridge._apply_run(tmp_path, state, expected_branch="main")
    local_git("checkout", "main")

    result = await bridge.dispatch("run.apply", {"root": str(tmp_path), "run_id": state.run_id})
    assert result["applied"] is True
    assert result["branch"] == "main"
    assert result["commit"] != local_git("rev-parse", state.integration_branch).stdout.strip()
    assert local_git("rev-list", "--count", f"{state.source_commit}..HEAD").stdout.strip() == "1"
    assert local_git("show", "-s", "--format=%an <%ae>", "HEAD").stdout.strip() == (
        "Adaptea <322406691+adaptea[bot]@users.noreply.github.com>"
    )
    assert (
        local_git("show", "-s", "--format=%s", "HEAD")
        .stdout.strip()
        .startswith("adaptea: Build it")
    )
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "completed\n"
    assert (tmp_path / "second-result.txt").read_text(encoding="utf-8") == "also completed\n"
    assert (tmp_path / "local.txt").read_text(encoding="utf-8") == "uncommitted\n"
    persisted = StateStore(run_dir).load()
    assert persisted.applied_branch == "main"
    assert persisted.applied_commit == result["commit"]
    assert persisted.applied_at is not None

    incomplete = sample_run(tmp_path, status=TaskStatus.FAILED)
    incomplete.run_id = "run-incomplete"
    StateStore(tmp_path / ".adaptea" / "runs" / incomplete.run_id).save(incomplete)
    with pytest.raises(ValueError, match="every task merged"):
        await bridge.dispatch("run.apply", {"root": str(tmp_path), "run_id": incomplete.run_id})


@pytest.mark.asyncio
async def test_project_preference_automatically_applies_a_successful_run(tmp_path: Path) -> None:
    def local_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    local_git("init", "-b", "main")
    (tmp_path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    local_git("add", ".")
    local_git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    state = sample_run(tmp_path, status=TaskStatus.MERGED)
    state.source_branch = "main"
    state.source_commit = local_git("rev-parse", "HEAD").stdout.strip()
    local_git("checkout", "-b", state.integration_branch)
    (tmp_path / "automatic.txt").write_text("applied\n", encoding="utf-8")
    local_git("add", "automatic.txt")
    local_git(
        "-c",
        "user.name=Adaptea",
        "-c",
        "user.email=adaptea@localhost",
        "commit",
        "-m",
        "automatic result",
    )
    integration_commit = local_git("rev-parse", "HEAD").stdout.strip()
    local_git("checkout", "main")
    StateStore(tmp_path / ".adaptea" / "runs" / state.run_id).save(state)
    output = io.StringIO()
    bridge = DesktopBridge(output=output)

    saved = await bridge.dispatch(
        "project.preferences.update",
        {
            "root": str(tmp_path),
            "auto_commit_completed_runs": True,
            "auto_start_plans": True,
        },
    )
    # Project settings can be changed independently; updating one must preserve the other.
    saved = await bridge.dispatch(
        "project.preferences.update",
        {"root": str(tmp_path), "auto_commit_completed_runs": True},
    )
    loaded = await bridge.dispatch("project.preferences.get", {"root": str(tmp_path)})
    final = await bridge._finalize_run(tmp_path, state)

    assert saved.auto_commit_completed_runs is True
    assert saved.auto_start_plans is True
    assert loaded.auto_commit_completed_runs is True
    assert loaded.auto_start_plans is True
    assert final.applied_branch == "main"
    assert final.applied_commit != integration_commit
    assert local_git("rev-parse", "HEAD").stdout.strip() == final.applied_commit
    assert local_git("rev-list", "--count", f"{state.source_commit}..HEAD").stdout.strip() == "1"
    assert (tmp_path / "automatic.txt").read_text(encoding="utf-8") == "applied\n"
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(row.get("event") == "run.auto_committed" for row in events)


@pytest.mark.asyncio
async def test_no_commit_completion_action_keeps_the_reviewed_branch_only(
    tmp_path: Path,
) -> None:
    state = sample_run(tmp_path, status=TaskStatus.MERGED)
    output = io.StringIO()
    bridge = DesktopBridge(output=output)
    saved = await bridge.dispatch(
        "project.preferences.update",
        {"root": str(tmp_path), "completion_action": "none"},
    )

    final = await bridge._finalize_run(tmp_path, state)

    assert saved.completion_action == "none"
    assert final.applied_commit is None
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert not any(row.get("event") == "run.auto_commit_failed" for row in events)
    assert any(row.get("event") == "run.completed" for row in events)


@pytest.mark.asyncio
async def test_push_completion_action_pushes_exact_local_commit_to_origin(
    tmp_path: Path,
) -> None:
    def local_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    local_git("init", "-b", "main")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    local_git("add", "base.txt")
    local_git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    remote = tmp_path / "origin.git"
    local_git("init", "--bare", str(remote))
    local_git("remote", "add", "origin", str(remote))
    commit = local_git("rev-parse", "HEAD").stdout.strip()
    state = sample_run(tmp_path, status=TaskStatus.MERGED)
    state.applied_branch = "main"
    state.applied_commit = commit
    StateStore(tmp_path / ".adaptea" / "runs" / state.run_id).save(state)

    pushed = await DesktopBridge(output=io.StringIO())._push_applied_run(tmp_path, state)

    assert pushed == {"remote": "origin", "branch": "main", "commit": commit}
    assert (
        local_git("--git-dir", str(remote), "rev-parse", "refs/heads/main").stdout.strip() == commit
    )
    persisted = StateStore(tmp_path / ".adaptea" / "runs" / state.run_id).load()
    assert persisted.pushed_remote == "origin"
    assert persisted.pushed_branch == "main"
    assert persisted.pushed_at is not None


@pytest.mark.asyncio
async def test_automatic_apply_failure_keeps_the_reviewed_branch_safe(tmp_path: Path) -> None:
    state = sample_run(tmp_path, status=TaskStatus.MERGED)
    state.source_branch = "main"
    output = io.StringIO()
    bridge = DesktopBridge(output=output)
    await bridge.dispatch(
        "project.preferences.update",
        {"root": str(tmp_path), "auto_commit_completed_runs": True},
    )

    final = await bridge._finalize_run(tmp_path, state)

    assert final.applied_commit is None
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    failed = next(row for row in events if row.get("event") == "run.auto_commit_failed")
    assert failed["data"]["integration_branch"] == state.integration_branch
    assert "Git status could not be read" in failed["data"]["message"]


@pytest.mark.asyncio
async def test_desktop_core_doctor_setup_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = ApplicationServices()
    services.diagnose = AsyncMock(return_value=[Check("PASS", "Git", "ready")])  # type: ignore[method-assign]
    services.calibrate = AsyncMock(  # type: ignore[method-assign]
        return_value=tmp_path / ".adaptea" / "calibration" / "one"
    )
    services.calibrate_fleet = AsyncMock(  # type: ignore[method-assign]
        return_value=tmp_path / ".adaptea" / "fleet-calibration" / "one"
    )

    class FakeSetupManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def diagnose(self) -> dict[str, bool]:
            return {"required_ready": True}

    monkeypatch.setattr("adaptea.desktop.server.SetupManager", FakeSetupManager)
    bridge = DesktopBridge(services=services, output=io.StringIO())
    base = {"root": str(tmp_path)}
    assert (await bridge.dispatch("doctor", base))[0].name == "Git"
    assert (await bridge.dispatch("setup.diagnose", base))["required_ready"] is True
    assert "directory" in await bridge.dispatch("calibration.start", base | {"quick": True})
    assert "directory" in await bridge.dispatch(
        "fleet.calibration.start", base | {"repetitions": 1, "agent_validation": False}
    )


@pytest.mark.asyncio
async def test_desktop_reads_and_changes_inference_backend_without_losing_lmstudio(
    tmp_path: Path,
) -> None:
    environment = environment_root()
    (environment / "adaptea.toml").write_text(
        '[lmstudio]\nmodel = "legacy"\nbase_url = "http://lm.test:1234"\n',
        encoding="utf-8",
    )
    bridge = DesktopBridge(output=io.StringIO())
    # The backend belongs to the machine, so these commands answer for the environment
    # whether or not a project names itself in the payload.
    base = {"root": str(tmp_path)}

    before = await bridge.dispatch("inference.config", base)
    changed = await bridge.dispatch("inference.configure", base | {"backend": "ollama"})
    after = await bridge.dispatch("inference.config", {})

    assert before == {
        "backend": "lmstudio",
        "base_url": "http://lm.test:1234",
        "model": "legacy",
    }
    assert changed["backend"] == "ollama"
    assert after["backend"] == "ollama"
    assert after["base_url"] == "http://127.0.0.1:11434"
    # Switching backend keeps the other backend's remembered model.
    assert 'model = "legacy"' in (environment / "adaptea.toml").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_desktop_project_config_never_inherits_a_parent_projects_model(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / "adaptea.toml").write_text(
        '[lmstudio]\nmodel = "parents-model"\n', encoding="utf-8"
    )

    result = await DesktopBridge(output=io.StringIO()).dispatch(
        "inference.config", {"root": str(child)}
    )

    assert result["backend"] == "lmstudio"
    assert result["model"] is None


@pytest.mark.asyncio
async def test_desktop_downloads_lmstudio_models_itself_rather_than_through_lms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lms get` only asks LM Studio's service to download, and that service ignores us.

    Adaptea fills LM Studio's models folder itself so that stopping the download is a
    matter of stopping this process, not of asking a daemon nicely.
    """
    commands: list[tuple[str, ...]] = []
    fetched: list[tuple[str, Path, str | None]] = []

    class Runner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, *args: str, **_kwargs: object) -> CommandResult:
            commands.append(args)
            return CommandResult(0, "downloaded", "")

    async def fake_download(
        repo_id: str, root: Path, *, quantization: str | None = None, **_kwargs: object
    ) -> Path:
        fetched.append((repo_id, root, quantization))
        return root / repo_id

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config()

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            return {"downloaded_models": [{"model_key": "publisher/coder"}]}

    monkeypatch.setattr("adaptea.desktop.server.SetupCommandRunner", Runner)
    monkeypatch.setattr("adaptea.desktop.server.download_repository", fake_download)
    monkeypatch.setattr("adaptea.desktop.server.lmstudio_models_root", lambda: tmp_path / "models")
    result = await DesktopBridge(services=Services(), output=io.StringIO()).dispatch(
        "model.download", {"root": str(tmp_path), "model": "publisher/coder-GGUF:Q4_K_M"}
    )

    assert commands == []
    assert fetched == [("publisher/coder-GGUF", tmp_path / "models", "Q4_K_M")]
    assert result["status"]["downloaded_models"][0]["model_key"] == "publisher/coder"


@pytest.mark.asyncio
async def test_desktop_confirms_a_cancelled_download_only_once_it_has_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interface says "Paused" on the strength of this answer, so it has to be true."""
    stopped = asyncio.Event()
    started = asyncio.Event()

    async def never_ending(*_args: object, **_kwargs: object) -> Path:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stopped.set()
            raise
        raise AssertionError("the download should have been cancelled")

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config()

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            return {"downloaded_models": []}

    monkeypatch.setattr("adaptea.desktop.server.download_repository", never_ending)
    monkeypatch.setattr("adaptea.desktop.server.lmstudio_models_root", lambda: tmp_path / "models")
    bridge = DesktopBridge(services=Services(), output=io.StringIO())
    download = asyncio.create_task(
        bridge.dispatch("model.download", {"root": str(tmp_path), "model": "publisher/coder"})
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    answer = await asyncio.wait_for(
        bridge.dispatch(
            "model.download.cancel", {"root": str(tmp_path), "model": "publisher/coder"}
        ),
        timeout=5,
    )

    assert answer == {"cancelled": True, "models": ["publisher/coder"]}
    # The confirmation arrived after the transfer unwound, not merely after cancel() was
    # requested — which is the difference between a true "Paused" and a false one.
    assert stopped.is_set()
    with pytest.raises(asyncio.CancelledError):
        await download


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        (
            "llamacpp",
            "publisher/coder-GGUF:Q5_K_M",
            (
                "hf",
                "download",
                "publisher/coder-GGUF",
                "--include",
                "*Q5_K_M*.gguf",
                "--cache-dir",
                "CACHE",
            ),
        ),
        (
            "vllm",
            "publisher/coder",
            ("hf", "download", "publisher/coder", "--cache-dir", "CACHE"),
        ),
    ],
)
async def test_desktop_downloads_huggingface_models_for_native_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    model: str,
    expected: tuple[str, ...],
) -> None:
    commands: list[tuple[str, ...]] = []
    cached = tmp_path / "cache" / "models--publisher--coder"
    snapshot = cached / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model-Q5_K_M.gguf").write_text("weights", encoding="utf-8")

    class Runner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, *args: str, **_kwargs: object) -> CommandResult:
            commands.append(args)
            return CommandResult(0, str(cached), "")

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config(inference={"backend": backend})

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            return {"downloaded_models": [{"model_key": "publisher/coder"}]}

    monkeypatch.setattr("adaptea.desktop.server.SetupCommandRunner", Runner)
    monkeypatch.setattr("adaptea.desktop.server.huggingface_repo_path", lambda *_args: cached)
    monkeypatch.setattr(
        "adaptea.desktop.server.huggingface_cache_root", lambda *_args: tmp_path / "cache"
    )
    monkeypatch.setattr("adaptea.desktop.server._huggingface_cli", lambda *_args: "hf")
    result = await DesktopBridge(services=Services(), output=io.StringIO()).dispatch(
        "model.download", {"root": str(tmp_path), "model": model}
    )

    assert commands == [
        tuple(str(tmp_path / "cache") if item == "CACHE" else item for item in expected)
    ]
    assert result["status"]["downloaded_models"]


def test_download_progress_normalizes_cli_percentages() -> None:
    assert _download_progress("pulling manifest · 42.6% · 18 MB/s") == {
        "detail": "pulling manifest · 42.6% · 18 MB/s",
        "percent": 42.6,
    }
    assert _download_progress("Process active · no failure reported")["percent"] is None


@pytest.mark.asyncio
async def test_desktop_deletes_an_unused_ollama_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    class Runner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, *args: str, **_kwargs: object) -> CommandResult:
            commands.append(args)
            return CommandResult(0, "deleted", "")

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config(inference={"backend": "ollama"})

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            return {"configured_models": [], "loaded_instances": [], "downloaded_models": []}

    monkeypatch.setattr("adaptea.desktop.server.SetupCommandRunner", Runner)
    result = await DesktopBridge(services=Services(), output=io.StringIO()).dispatch(
        "model.delete", {"root": str(tmp_path), "model": "qwen3:8b"}
    )

    assert commands == [("ollama", "rm", "qwen3:8b")]
    assert result["model"] == "qwen3:8b"


@pytest.mark.asyncio
async def test_desktop_moves_an_unused_huggingface_repo_to_recoverable_trash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hub" / "models--publisher--coder"
    source.mkdir(parents=True)

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config(inference={"backend": "vllm"})

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            return {
                "configured_models": [],
                "loaded_instances": [],
                "combinations": [],
                "downloaded_models": [{"model_key": "publisher/coder", "local_path": str(source)}],
            }

    trash_root = tmp_path / "environment"
    monkeypatch.setattr("adaptea.desktop.server.environment_root", lambda: trash_root)
    monkeypatch.setattr(
        "adaptea.desktop.server.resolve_huggingface_model_path",
        lambda *_args: str(source),
    )

    result = await DesktopBridge(services=Services(), output=io.StringIO()).dispatch(
        "model.delete", {"root": str(tmp_path), "model": "publisher/coder"}
    )

    assert result["model"] == "publisher/coder"
    assert not source.exists()
    assert len(list((trash_root / ".trash" / "models").iterdir())) == 1


@pytest.mark.asyncio
async def test_safe_repair_only_approves_explicit_supported_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approvals: list[bool] = []
    constructor_kwargs: list[dict[str, Any]] = []

    class FakeSetupManager:
        def __init__(self, *_args: object, **kwargs: Any) -> None:
            confirm_install = kwargs["confirm_install"]
            assert callable(confirm_install)
            approvals.extend(
                [
                    confirm_install(
                        InstallAction("OpenCode", ("brew",), "brew", "official", "install")
                    ),
                    confirm_install(InstallAction("Git", ("brew",), "brew", "official", "install")),
                ]
            )
            constructor_kwargs.append(kwargs)

        async def fix_all(self, *, automatic: bool = False) -> dict[str, object]:
            assert automatic is True
            return {"snapshot": {"required_ready": True}, "messages": [], "failures": []}

    monkeypatch.setattr("adaptea.desktop.server.SetupManager", FakeSetupManager)
    bridge = DesktopBridge(output=io.StringIO())
    await bridge.dispatch(
        "setup.repair_safe",
        {
            "root": str(tmp_path),
            "approved_components": ["OpenCode", "Anything else"],
            "selected_model": "publisher/chosen",
        },
    )
    assert approvals == [True, False]
    # Model assignment and loading belong to Environment > Models, so safe repair never
    # chooses one: any stray selected_model in the payload is ignored.
    assert "select_model" not in constructor_kwargs[0]


async def test_setup_repair_safe_accepts_lms_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approvals: list[bool] = []

    class FakeSetupManager:
        def __init__(self, *_args: object, **kwargs: Any) -> None:
            confirm_install = kwargs["confirm_install"]
            confirm = kwargs["confirm"]
            approvals.extend(
                [
                    confirm_install(
                        InstallAction("LM Studio llmster", ("curl",), "curl", "official", "install")
                    ),
                    confirm_install(InstallAction("lms", ("lms",), "lms", "official", "bootstrap")),
                    confirm("Install headless LM Studio llmster / lms CLI runtime?"),
                ]
            )

        async def fix_all(self, *, automatic: bool = False) -> dict[str, object]:
            return {"snapshot": {"required_ready": True}, "messages": [], "failures": []}

    monkeypatch.setattr("adaptea.desktop.server.SetupManager", FakeSetupManager)
    bridge = DesktopBridge(output=io.StringIO())
    await bridge.dispatch(
        "setup.repair_safe",
        {
            "root": str(tmp_path),
            "approved_components": ["lms"],
        },
    )
    assert approvals == [True, True, True]


async def test_environment_is_shared_and_reaches_every_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTEA_STATE_DIR", str(tmp_path / "state"))
    services = ApplicationServices()
    services.fleet_status = AsyncMock(return_value={"enabled": True})  # type: ignore[method-assign]
    services.unload_unconfigured_fleet_models = AsyncMock(return_value=[])  # type: ignore[method-assign]
    services.activate_configured_fleet = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
    bridge = DesktopBridge(services=services, output=io.StringIO())

    status = await bridge.dispatch("environment.status", {})
    assert status["configured"] is False
    assert status["root"].endswith("environment")

    # Configuring the environment needs no project open at all: with no root, the core
    # falls back to the environment's own directory.
    await bridge.dispatch("inference.configure", {"backend": "ollama"})
    await bridge.dispatch(
        "fleet.configure",
        {
            "fleet": {
                "enabled": True,
                "models": [
                    {
                        "name": "strong-1",
                        "model": "qwen3-coder:30b",
                        "tier": "strong",
                        "roles": ["planner", "worker"],
                    }
                ],
            },
            "activate": False,
        },
    )
    assert (await bridge.dispatch("environment.status", {}))["configured"] is True

    # Opening a project writes that one answer into it, so no project has to be
    # configured a second time and none can drift into a stale model selection.
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for project in (first, second):
        applied = await bridge.dispatch("environment.apply", {"root": str(project)})
        assert applied["written"]
        config = load_config(project, explicit=project / "adaptea.toml")
        assert config.inference.backend == "ollama"
        assert [model.model for model in config.fleet.models] == ["qwen3-coder:30b"]

    # Applying again changes nothing, so re-opening a project does not keep touching it.
    assert (await bridge.dispatch("environment.apply", {"root": str(first)}))["written"] == []


async def test_a_project_configured_before_the_shared_environment_seeds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTEA_STATE_DIR", str(tmp_path / "state"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "adaptea.toml").write_text(
        '[inference]\nbackend = "ollama"\n\n[ollama]\nmodel = "qwen3-coder:30b"\n',
        encoding="utf-8",
    )
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())

    # Upgrading must not replace a fleet someone already configured with empty defaults.
    applied = await bridge.dispatch("environment.apply", {"root": str(project)})

    assert applied["adopted"] is True
    assert load_config(project, explicit=project / "adaptea.toml").ollama.model == (
        "qwen3-coder:30b"
    )
    assert (await bridge.dispatch("environment.status", {}))["configured"] is True


async def test_inference_server_management_commands(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "adaptea.toml").write_text('[inference]\nbackend = "lmstudio"\n', encoding="utf-8")
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())

    status = await bridge.dispatch("inference.server.status", {"root": str(project)})
    assert "reachable" in status
    assert status["backend"] == "lmstudio"


async def test_a_project_can_name_the_combination_it_runs_on(tmp_path: Path) -> None:
    services = ApplicationServices()
    services.fleet_status = AsyncMock(return_value={"enabled": True})  # type: ignore[method-assign]
    services.unload_unconfigured_fleet_models = AsyncMock(return_value=[])  # type: ignore[method-assign]
    services.activate_configured_fleet = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
    bridge = DesktopBridge(services=services, output=io.StringIO())

    def fleet(model: str) -> dict[str, Any]:
        return {
            "enabled": True,
            "models": [
                {"name": model, "model": model, "tier": "strong", "roles": ["planner", "worker"]}
            ],
        }

    strong = await bridge.dispatch(
        "fleet.combination.save", {"fleet": fleet("strong-30b"), "name": "Strong"}
    )
    fast = await bridge.dispatch(
        "fleet.combination.save", {"fleet": fleet("fast-8b"), "name": "Fast"}
    )
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()

    # Two projects, two choices, one library — and choosing for one must not move the
    # other, which is why a project's choice reads a combination rather than selecting it.
    await bridge.dispatch(
        "project.combination.set",
        {"root": str(first), "combination_id": strong["combination"]["id"]},
    )
    await bridge.dispatch(
        "project.combination.set",
        {"root": str(second), "combination_id": fast["combination"]["id"]},
    )

    assert load_project_preferences(first).model_combination_id == strong["combination"]["id"]
    assert [
        model.model for model in load_config(first, explicit=first / "adaptea.toml").fleet.models
    ] == ["strong-30b"]
    assert [
        model.model for model in load_config(second, explicit=second / "adaptea.toml").fleet.models
    ] == ["fast-8b"]

    # Re-opening a project applies the combination it named, not the global default.
    await bridge.dispatch("environment.apply", {"root": str(first)})
    assert [
        model.model for model in load_config(first, explicit=first / "adaptea.toml").fleet.models
    ] == ["strong-30b"]

    # A combination that is deleted stops being the project's; it falls back rather than
    # leaving the project pointing at nothing.
    await bridge.dispatch(
        "fleet.combination.delete", {"combination_id": strong["combination"]["id"]}
    )
    applied = await bridge.dispatch("environment.apply", {"root": str(first)})
    assert applied["combination_id"] is None


@pytest.mark.asyncio
async def test_bridge_run_abort_and_models_stop(tmp_path: Path) -> None:
    unloaded_called: list[str] = []

    class Services(ApplicationServices):
        def abort_run(self, run_id: str) -> bool:
            return True

        async def stop_models(self, root: Path, *, force: bool = False) -> list[str]:
            unloaded_called.append("stopped-model-1")
            return ["stopped-model-1"]

    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    # Default abort does not unload models
    await bridge.accept(
        json.dumps(
            {
                "id": "req-abort-default",
                "command": "run.abort",
                "payload": {"root": str(tmp_path), "run_id": "run-123"},
            }
        )
    )
    # Explicit abort with stop_models: true
    await bridge.accept(
        json.dumps(
            {
                "id": "req-abort-with-models",
                "command": "run.abort",
                "payload": {"root": str(tmp_path), "run_id": "run-123", "stop_models": True},
            }
        )
    )
    await bridge.accept(
        json.dumps(
            {
                "id": "req-stop-models",
                "command": "models.stop",
                "payload": {"root": str(tmp_path)},
            }
        )
    )
    items = await responses(bridge)
    abort_default_resp = next(r for r in items if r.get("id") == "req-abort-default")
    assert abort_default_resp["ok"] is True
    assert abort_default_resp["data"]["changed"] is True
    assert abort_default_resp["data"]["unloaded"] == []

    abort_models_resp = next(r for r in items if r.get("id") == "req-abort-with-models")
    assert abort_models_resp["ok"] is True
    assert abort_models_resp["data"]["changed"] is True
    assert abort_models_resp["data"]["unloaded"] == ["stopped-model-1"]

    stop_resp = next(r for r in items if r.get("id") == "req-stop-models")
    assert stop_resp["ok"] is True
    assert stop_resp["data"]["unloaded"] == ["stopped-model-1"]
    assert len(unloaded_called) == 2
    # Stopping a run reports that the models are still there, because that is the whole
    # difference between stopping the work and stopping the machine.
    assert abort_default_resp["data"]["models_kept_loaded"] is True
    assert abort_models_resp["data"]["models_kept_loaded"] is False


@pytest.mark.asyncio
async def test_bridge_abort_unloads_models_only_when_the_project_asks_for_it(
    tmp_path: Path,
) -> None:
    class Services(ApplicationServices):
        def abort_run(self, run_id: str) -> bool:
            return True

        async def stop_models(self, root: Path, *, force: bool = False) -> list[str]:
            return ["stopped-model-1"]

    save_project_preferences(tmp_path, ProjectPreferences(stop_models_on_stop=True))
    bridge = DesktopBridge(services=Services(), output=io.StringIO())

    result = await bridge.dispatch("run.abort", {"root": str(tmp_path), "run_id": "run-123"})

    assert result["unloaded"] == ["stopped-model-1"]
    assert result["models_kept_loaded"] is False


@pytest.mark.asyncio
async def test_bridge_asks_about_a_refused_command_and_relays_the_answer(tmp_path: Path) -> None:
    request = ApprovalRequest(
        request_id="req-1",
        command="npm install zod",
        task_id="task-a",
        run_id="run-123",
        requested_at="2026-01-01T00:00:00+00:00",
        reason="no explicit project-local user approval matched",
    )
    answered: list[tuple[str, str]] = []

    class Services(ApplicationServices):
        def pending_command_approvals(self, run_id: str | None = None) -> list[dict[str, Any]]:
            return [request.as_dict()]

        def resolve_command_approval(
            self, request_id: str, decision: Any, run_id: str | None = None
        ) -> dict[str, Any]:
            answered.append((request_id, decision))
            return {
                "request": request.as_dict(),
                "decision": decision,
                "approved": True,
                "remembered": decision == "always",
                "config_path": str(tmp_path / "adaptea.toml"),
            }

    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    # The question reaches the desktop as an event while the run keeps going.
    bridge._announce_command_approval(request)
    await asyncio.sleep(0)

    listed = await bridge.dispatch("command.approvals.list", {"root": str(tmp_path)})
    assert [item["command"] for item in listed["approvals"]] == ["npm install zod"]

    resolved = await bridge.dispatch(
        "command.approval.resolve",
        {"root": str(tmp_path), "request_id": "req-1", "decision": "always"},
    )
    assert resolved["approved"] is True
    assert answered == [("req-1", "always")]

    with pytest.raises(ValueError):
        await bridge.dispatch(
            "command.approval.resolve",
            {"root": str(tmp_path), "request_id": "req-1", "decision": "maybe"},
        )

    events = [
        json.loads(line)
        for line in output.getvalue().splitlines()
        if json.loads(line).get("type") == "event"
    ]
    assert [row["event"] for row in events] == [
        "command.approval_requested",
        "command.approval_resolved",
    ]
    assert events[0]["data"]["command"] == "npm install zod"


@pytest.mark.asyncio
async def test_bridge_model_download_cancel() -> None:
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())
    terminated = []

    class FakeRunner:
        def terminate(self) -> None:
            terminated.append(True)

    async def fake_download() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(fake_download())
    runner = FakeRunner()
    bridge.requests["req-dl"] = task
    bridge.active_downloads["my/model"] = (task, runner)  # type: ignore[assignment]

    result = await bridge.dispatch("model.download.cancel", {"model": "my/model"})
    assert result == {"cancelled": True, "models": ["my/model"]}
    assert terminated == [True]
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    assert "my/model" not in bridge.active_downloads


@pytest.mark.asyncio
async def test_bridge_command_cancel_cancels_active_download() -> None:
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())
    terminated = []

    class FakeRunner:
        def terminate(self) -> None:
            terminated.append(True)

    async def fake_download() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(fake_download())
    runner = FakeRunner()
    bridge.requests["req-dl"] = task
    bridge.active_downloads["my/model"] = (task, runner)  # type: ignore[assignment]

    result = await bridge.dispatch("command.cancel", {"request_id": "req-dl"})
    assert result == {"cancelled": True}
    assert terminated == [True]
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    assert "my/model" not in bridge.active_downloads


@pytest.mark.asyncio
async def test_cancelled_download_cannot_race_to_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped = asyncio.Event()

    class Runner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, *_args: str, **_kwargs: object) -> CommandResult:
            await stopped.wait()
            # Some CLIs exit zero after handling Ctrl-C. That must still be cancellation.
            return CommandResult(0, "interrupted", "")

        async def terminate(self) -> None:
            stopped.set()

    class Services(ApplicationServices):
        def load_config(self, _root: Path) -> Config:
            return Config()

        async def fleet_status(self, _root: Path) -> dict[str, object]:  # type: ignore[override]
            raise AssertionError("A cancelled download must not refresh as completed")

    monkeypatch.setattr("adaptea.desktop.server.SetupCommandRunner", Runner)
    output = io.StringIO()
    bridge = DesktopBridge(services=Services(), output=output)
    request = DesktopCommand(
        id="download-race",
        command="model.download",
        payload={"root": str(tmp_path), "model": "publisher/coder"},
    )
    await bridge.accept(request.model_dump_json())
    for _ in range(100):
        if "publisher/coder" in bridge.active_downloads:
            break
        await asyncio.sleep(0.01)
    assert "publisher/coder" in bridge.active_downloads
    task = bridge.requests["download-race"]

    result = await bridge.dispatch("model.download.cancel", {"model": "publisher/coder"})
    await task

    assert result["models"] == ["publisher/coder"]
    events = [
        json.loads(line).get("event")
        for line in output.getvalue().splitlines()
        if json.loads(line).get("type") == "event"
    ]
    assert "model.download.cancelled" in events
    assert "model.download.completed" not in events


@pytest.mark.asyncio
async def test_bridge_cancel_downloads_all() -> None:
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())

    async def fake_download() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass

    task1 = asyncio.create_task(fake_download())
    task2 = asyncio.create_task(fake_download())
    terminated1 = []
    terminated2 = []

    class FakeRunner1:
        def terminate(self) -> None:
            terminated1.append(True)

    class FakeRunner2:
        def terminate(self) -> None:
            terminated2.append(True)

    bridge.active_downloads["model-1"] = (task1, FakeRunner1())  # type: ignore[assignment]
    bridge.active_downloads["model-2"] = (task2, FakeRunner2())  # type: ignore[assignment]

    cancelled = await bridge.cancel_downloads()
    assert set(cancelled) == {"model-1", "model-2"}
    assert bridge.active_downloads == {}
    await asyncio.sleep(0)
    assert task1.cancelled() or task1.done()
    assert task2.cancelled() or task2.done()
    assert terminated1 == [True]
    assert terminated2 == [True]


@pytest.mark.asyncio
async def test_bridge_eof_terminates_downloads_before_core_exit() -> None:
    bridge = DesktopBridge(services=ApplicationServices(), output=io.StringIO())
    terminated: list[bool] = []

    class FakeRunner:
        async def terminate(self) -> None:
            terminated.append(True)

    async def fake_download() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(fake_download())
    bridge.active_downloads["my/model"] = (task, FakeRunner())  # type: ignore[assignment]

    await bridge.serve(io.StringIO(""))
    await asyncio.gather(task, return_exceptions=True)

    assert terminated == [True]
    assert task.cancelled()
    assert bridge.active_downloads == {}


def test_setup_progress_tracker_emits_structured_stages() -> None:
    from adaptea.desktop.server import SetupProgressTracker

    events: list[dict[str, object]] = []

    def fake_emit(event: str, **data: object) -> None:
        events.append({"event": event, **data})

    tracker = SetupProgressTracker({"OpenCode"}, fake_emit)
    tracker.emit()
    assert events[-1]["component"] == "OpenCode"
    assert events[-1]["stage"] == "preparing"
    assert events[-1]["percent"] == 5.0

    tracker.on_notice("OpenCode is missing; checking for a supported installation path…")
    assert events[-1]["component"] == "OpenCode"
    assert events[-1]["stage"] == "checking"

    tracker.on_notice(
        "Downloading OpenCode… GitHub download speed can make this take a few minutes."
    )
    assert events[-1]["stage"] == "downloading"
    assert events[-1]["percent"] == 50.0

    tracker.on_runner_output("==> Downloading https://github.com/anomalyco/opencode 45.0%")
    assert events[-1]["detail"] == "==> Downloading https://github.com/anomalyco/opencode 45.0%"
    assert events[-1]["percent"] == 51.5

    tracker.finish()
    assert events[-1]["stage"] == "completed"
    assert events[-1]["percent"] == 100.0
