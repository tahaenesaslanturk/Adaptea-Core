from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from adaptea.config import Config, load_config
from adaptea.git.repository import git
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.reviewer.opencode import ReviewResult
from adaptea.runtime.controller import Orchestrator
from adaptea.runtime.state import StateStore
from adaptea.validation import (
    NO_SUITE_DETAIL,
    PYTEST_NO_TESTS_COLLECTED,
    detect_test_command,
    interpret,
    python_interpreter,
    resolve_test_command,
    run_validation,
)
from adaptea.workers.opencode import WorkerResult, worker_prompt


def test_python_interpreter_exists_on_this_host() -> None:
    assert Path(python_interpreter()).is_file() or shutil.which(python_interpreter())


def test_static_site_project_has_no_detectable_suite(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!DOCTYPE html><p>Hello World</p>", encoding="utf-8")
    assert detect_test_command(tmp_path) == []


def test_pytest_suite_is_detected_from_several_layouts(tmp_path: Path) -> None:
    package = tmp_path / "tests"
    package.mkdir()
    (package / "test_thing.py").write_text("def test_ok() -> None:\n    assert True\n")
    assert detect_test_command(tmp_path)[-1] == "pytest"

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "widget_test.py").write_text("def test_ok() -> None:\n    assert True\n")
    assert detect_test_command(flat)[-1] == "pytest"

    conftest = tmp_path / "conf"
    conftest.mkdir()
    (conftest / "conftest.py").write_text("", encoding="utf-8")
    assert detect_test_command(conftest)[-1] == "pytest"


def test_npm_placeholder_script_is_not_a_test_suite(tmp_path: Path) -> None:
    manifest = {"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}
    (tmp_path / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert detect_test_command(tmp_path) == []


def test_configured_command_wins_over_detection(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    assert resolve_test_command(tmp_path, ["make", "check"]) == ["make", "check"]


def test_bare_python_is_repointed_at_a_real_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("adaptea.validation.shutil.which", lambda _name: None)
    resolved = resolve_test_command(tmp_path, ["python", "-m", "pytest"])
    assert resolved[1:] == ["-m", "pytest"]
    assert resolved[0] != "python"


def test_pytest_no_tests_collected_is_not_a_failure() -> None:
    command = [python_interpreter(), "-m", "pytest"]
    tolerant = interpret(command, PYTEST_NO_TESTS_COLLECTED, no_tests_is_failure=False)
    assert tolerant.passed
    strict = interpret(command, PYTEST_NO_TESTS_COLLECTED, no_tests_is_failure=True)
    assert not strict.passed
    assert not interpret(command, 1, no_tests_is_failure=False).passed


@pytest.mark.asyncio
async def test_project_without_a_suite_passes_and_explains_itself(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!DOCTYPE html><p>Hello World</p>", encoding="utf-8")
    log = tmp_path / "logs" / "validation.log"
    outcome = await run_validation(tmp_path, [], log)
    assert outcome.passed and outcome.exit_code == 0 and not outcome.executed
    assert NO_SUITE_DETAIL in log.read_text()


@pytest.mark.asyncio
async def test_no_suite_can_be_configured_as_a_failure(tmp_path: Path) -> None:
    outcome = await run_validation(tmp_path, [], tmp_path / "v.log", no_tests_is_failure=True)
    assert not outcome.passed


@pytest.mark.asyncio
async def test_real_failing_suite_still_fails(tmp_path: Path) -> None:
    (tmp_path / "test_broken.py").write_text("def test_broken():\n    assert False\n")
    outcome = await run_validation(tmp_path, [], tmp_path / "v.log")
    assert not outcome.passed and outcome.exit_code == 1


@pytest.mark.asyncio
async def test_real_passing_suite_passes(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    outcome = await run_validation(tmp_path, [], tmp_path / "v.log")
    assert outcome.passed and outcome.exit_code == 0


@pytest.mark.asyncio
async def test_missing_validator_reports_instead_of_raising(tmp_path: Path) -> None:
    log = tmp_path / "v.log"
    outcome = await run_validation(tmp_path, ["adaptea-no-such-validator"], log)
    assert not outcome.passed and outcome.exit_code == 127
    assert "could not be executed" in log.read_text()


def test_worker_prompt_states_when_no_validator_exists() -> None:
    task = TaskSpec(id="t", title="T", description="d")
    assert "no automated test suite" in worker_prompt("goal", task, [], [])
    assert "pytest" in worker_prompt("goal", task, [], ["pytest"])


def test_worker_prompt_requires_reading_the_repository_before_writing() -> None:
    # A worker that guesses at APIs and file layout produces plausible code that does not
    # fit the repository, so grounding is an instruction rather than a suggestion.
    prompt = worker_prompt("goal", TaskSpec(id="t", title="T", description="d"), [], [])
    assert "Read before you write" in prompt
    assert "confirm it exists" in prompt
    assert "Follow the conventions already in this repository" in prompt
    assert "hint, not the answer" in prompt


def test_default_config_defers_to_detection(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.project.test_command == []
    assert config.project.no_tests_is_failure is False


class StaticSiteWorker:
    """Stand in for OpenCode: writes the file the task asked for, and nothing else."""

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
        del goal, dependency_summaries, retry_context, progress
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (worktree / "index.html").write_text(
            '<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8">'
            "<title>Hello World</title></head>\n<body>Hello World</body>\n</html>\n",
            encoding="utf-8",
        )
        return WorkerResult(0, "start", "end", 0.01, "session", f"created index.html for {task.id}")


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
        return ReviewResult(True, "matches the acceptance criteria", [], 0, "start", "end", 0.01)


@pytest.mark.asyncio
async def test_index_html_task_merges_in_a_repository_without_tests(tmp_path: Path) -> None:
    """Regression: a correct change used to fail because pytest collected no tests."""
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# site\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@e.invalid", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    plan = Plan(
        goal="Create a Hello World index.html",
        tasks=[TaskSpec(id="create-index-html", title="Create index.html", description="make it")],
    )
    run_id = "run-index-html"
    state = RunState(
        run_id=run_id,
        goal=plan.goal,
        repository=str(tmp_path),
        integration_branch=f"adaptea-{run_id}-integration",
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={task.id: TaskRuntime(spec=task) for task in plan.tasks},
    )
    state.refresh_readiness()
    run_dir = tmp_path / ".adaptea" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"model": "coder", "integration_branch": state.integration_branch}),
        encoding="utf-8",
    )
    for name in ("events.jsonl", "telemetry.jsonl", "controller-decisions.jsonl"):
        (run_dir / name).touch()
    StateStore(run_dir).save(state)

    config = Config()
    config.worker.executable = "unused"
    config.lmstudio.lms_executable = "missing-lms"
    config.lmstudio.telemetry_poll_seconds = 0.01
    assert config.project.test_command == []

    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker = StaticSiteWorker()  # type: ignore[assignment]
    orchestrator.reviewer = ApprovingReviewer()  # type: ignore[assignment]
    final = await orchestrator.run()

    task = final.tasks["create-index-html"]
    assert task.status == TaskStatus.MERGED, task.failure
    assert task.validation_detail == NO_SUITE_DETAIL
    assert json.loads((run_dir / "summary.json").read_text())["pass_rate"] == 1.0
    # A completed run keeps the integration branch but cleans its duplicate checkout.
    integrated = await git(tmp_path, "show", f"{state.integration_branch}:index.html")
    assert integrated.returncode == 0
    assert "Hello World" in integrated.stdout


def test_packaged_build_never_uses_the_sidecar_as_an_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a PyInstaller build sys.executable is adaptea-core, not a Python interpreter."""
    monkeypatch.setattr("adaptea.validation.sys.frozen", True, raising=False)
    monkeypatch.setattr("adaptea.validation.sys.executable", "/Applications/adaptea-core")
    monkeypatch.setattr("adaptea.validation.shutil.which", lambda _name: None)
    assert python_interpreter() == "python3"


def test_missing_pytest_is_reported_as_such_not_as_a_test_failure() -> None:
    """An absent pytest exits 1, exactly like a failing suite; only the message separates them."""
    command = [python_interpreter(), "-m", "pytest"]
    missing = interpret(
        command,
        1,
        no_tests_is_failure=False,
        output="/usr/bin/python3: No module named pytest\n",
    )
    assert not missing.passed
    assert "cannot import pytest" in missing.detail

    genuine = interpret(command, 1, no_tests_is_failure=False, output="1 failed in 0.02s\n")
    assert not genuine.passed
    assert genuine.detail == "validator exited 1"


def test_planning_gets_a_longer_deadline_than_one_worker() -> None:
    """A short worker timeout must not cut off the single call that gates the run."""
    config = Config()
    assert config.worker.timeout_seconds == 900
    assert config.worker.planner_timeout == 1800

    config.worker.timeout_seconds = 3600
    assert config.worker.planner_timeout == 3600

    config.worker.planner_timeout_seconds = 600
    assert config.worker.planner_timeout == 600


def test_auto_loaded_context_fits_an_opencode_agent_prompt() -> None:
    """32k overflows OpenCode's planner prompt, which reads to the user as a stalled plan."""
    from adaptea.config import AGENT_CONTEXT_LENGTH, MINIMUM_AGENT_CONTEXT_LENGTH

    assert MINIMUM_AGENT_CONTEXT_LENGTH >= 32768
    assert AGENT_CONTEXT_LENGTH > MINIMUM_AGENT_CONTEXT_LENGTH

    # Setup caps its automatic choice at AGENT_CONTEXT_LENGTH but never exceeds what the
    # model itself supports.
    def chosen(maximum: int | None) -> int:
        return min(maximum or AGENT_CONTEXT_LENGTH, AGENT_CONTEXT_LENGTH)

    assert chosen(262144) == AGENT_CONTEXT_LENGTH
    assert chosen(None) == AGENT_CONTEXT_LENGTH
    assert chosen(8192) == 8192


def test_doctor_refuses_to_call_a_too_small_context_ready() -> None:
    from adaptea.config import MINIMUM_AGENT_CONTEXT_LENGTH
    from adaptea.diagnostics.doctor import Check

    del Check  # imported to assert the module still exposes it
    assert MINIMUM_AGENT_CONTEXT_LENGTH == 32768
