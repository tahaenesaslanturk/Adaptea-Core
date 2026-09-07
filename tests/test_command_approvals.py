"""A denied command has to be answerable, not just recorded.

These cover the path from a worker being refused to the user's answer reaching the next
attempt: the request itself, what is never asked about, what an approval changes in the
live policy and on disk, and the orchestrator wiring that carries it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import CommandSecurityConfig, Config
from adaptea.models import RunState, TaskRuntime, TaskSpec
from adaptea.runtime.controller import Orchestrator
from adaptea.security.approvals import ApprovalRequest, CommandApprovalBroker
from adaptea.security.commands import decide_command, policy_document
from adaptea.setup.configuration import merge_approved_commands
from adaptea.workers.activity import summarize_event


def _broker(root: Path, **kwargs: Any) -> CommandApprovalBroker:
    (root / "adaptea.toml").write_text('[worker]\nexecutable = "opencode"\n', encoding="utf-8")
    return CommandApprovalBroker(
        root=root, security=CommandSecurityConfig(), run_id="run-approvals", **kwargs
    )


def test_a_denied_command_becomes_a_question_with_the_exact_command(tmp_path: Path) -> None:
    asked: list[ApprovalRequest] = []
    broker = _broker(tmp_path, notify=asked.append)

    request = broker.request("npm   install   zod", task_id="task-a")

    assert request is not None
    assert request.command == "npm install zod"
    assert request.task_id == "task-a"
    assert request.run_id == "run-approvals"
    assert asked == [request]
    assert broker.snapshot() == [request]


def test_nothing_is_asked_about_a_safe_a_blocked_or_an_already_pending_command(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path)
    pending = broker.request("npm install zod", task_id="task-a")

    assert pending is not None
    # A safe default needs no answer, a blocked command can never be given one, and the
    # same command asked twice is one question, not two.
    assert broker.request("git status --short", task_id="task-a") is None
    assert broker.request("sudo rm -rf /", task_id="task-a") is None
    assert broker.request("npm install zod", task_id="task-b") is None
    assert broker.request("   ", task_id="task-a") is None
    assert broker.snapshot() == [pending]


def test_approving_for_this_run_changes_the_live_policy_without_writing_the_project(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path)
    request = broker.request("npm install zod", task_id="task-a")
    assert request is not None

    outcome = broker.resolve(request.request_id, "once")

    assert outcome.approved is True
    assert outcome.remembered is False
    assert outcome.config_path is None
    assert decide_command("npm install zod", broker.security).action == "allow"
    assert "approved_commands" not in (tmp_path / "adaptea.toml").read_text(encoding="utf-8")
    # The next attempt has to be told the command is available to it now.
    assert any("npm install zod" in note for note in broker.context_for("task-a"))
    assert broker.snapshot() == []


def test_approving_always_records_the_command_in_the_project_configuration(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path)
    request = broker.request("npm install zod", task_id="task-a")
    assert request is not None

    outcome = broker.resolve(request.request_id, "always")

    assert outcome.remembered is True
    assert outcome.config_path == str(tmp_path / "adaptea.toml")
    stored = Config.model_validate(
        tomllib.loads((tmp_path / "adaptea.toml").read_text(encoding="utf-8"))
    )
    assert stored.worker.command_security.approved_commands == ["npm install zod"]


def test_denying_leaves_the_policy_exactly_as_it_was(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    request = broker.request("rm -rf build", task_id="task-a")
    assert request is not None

    outcome = broker.resolve(request.request_id, "deny")

    assert outcome.approved is False
    assert broker.security.approved_commands == []
    assert broker.context_for("task-a") == []
    assert decide_command("rm -rf build", broker.security).action == "deny"


def test_answering_a_request_twice_is_refused_rather_than_silently_reapproved(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path)
    request = broker.request("npm install zod", task_id="task-a")
    assert request is not None
    broker.resolve(request.request_id, "once")

    with pytest.raises(KeyError):
        broker.resolve(request.request_id, "always")


def test_an_approval_preserves_the_approvals_the_project_already_had(tmp_path: Path) -> None:
    path = tmp_path / "adaptea.toml"
    path.write_text(
        '[worker.command_security]\napproved_commands = ["rm generated.txt"]\n', encoding="utf-8"
    )

    merge_approved_commands(path, ["npm install zod"])

    stored = tomllib.loads(path.read_text(encoding="utf-8"))
    assert stored["worker"]["command_security"]["approved_commands"] == [
        "rm generated.txt",
        "npm install zod",
    ]
    # Writing the same approval again is not a change, so the file is left alone.
    assert merge_approved_commands(path, ["npm install zod"]) is None


def test_a_blocked_tool_call_reports_the_command_it_was_refused() -> None:
    activity = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "error",
                    "input": {"command": "npm  install  zod"},
                    "error": "The user has specified a rule which prevents this tool call.",
                },
            }
        }
    )

    assert activity is not None
    assert activity.kind == "tool-blocked"
    # The display text is shortened; the approval needs the command itself.
    assert activity.command == "npm install zod"


def _orchestrator(tmp_path: Path, announced: list[ApprovalRequest]) -> Orchestrator:
    state = RunState(
        run_id="run-approvals",
        goal="approve a command",
        repository=str(tmp_path),
        integration_branch="adaptea/run-approvals/integration",
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={
            "task-a": TaskRuntime(
                spec=TaskSpec(id="task-a", title="Task A", description="install a dependency"),
                attempts=1,
            )
        },
    )
    run_dir = tmp_path / ".adaptea" / "runs" / state.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"model": "coder", "command_security": policy_document(CommandSecurityConfig())}
        ),
        encoding="utf-8",
    )
    for name in ("events.jsonl", "runtime-status.jsonl", "command-security-decisions.jsonl"):
        (run_dir / name).touch()
    (tmp_path / "adaptea.toml").write_text('[worker]\nexecutable = "opencode"\n', encoding="utf-8")
    return Orchestrator(tmp_path, Config(), state, approval_callback=announced.append)


def test_the_run_announces_a_refused_command_and_applies_the_answer(tmp_path: Path) -> None:
    announced: list[ApprovalRequest] = []
    orchestrator = _orchestrator(tmp_path, announced)

    orchestrator._request_command_approval("task-a", "npm install zod")

    assert [request.command for request in announced] == ["npm install zod"]
    assert orchestrator.pending_command_approvals() == announced

    outcome = orchestrator.resolve_command_approval(announced[0].request_id, "once")

    assert outcome.approved is True
    # The workers the orchestrator launches read this very object, so the next attempt
    # runs the command instead of being refused it again.
    assert orchestrator.config.worker.command_security.approved_commands == ["npm install zod"]
    events = [
        json.loads(line)
        for line in (tmp_path / ".adaptea" / "runs" / "run-approvals" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in events] == [
        "command_approval_requested",
        "command_approval_resolved",
    ]
