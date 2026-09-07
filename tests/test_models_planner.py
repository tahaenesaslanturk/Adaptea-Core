from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from adaptea.config import Config
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.planner.opencode import OpenCodePlanner, extract_json
from adaptea.runtime.controller import prepare_resume


def test_plan_dag_and_dependency_readiness() -> None:
    plan = Plan(
        goal="goal",
        tasks=[
            TaskSpec(id="a", title="A", description="A"),
            TaskSpec(id="b", title="B", description="B", depends_on=["a"]),
            TaskSpec(id="c", title="C", description="C"),
        ],
    )
    state = RunState(
        run_id="run-x",
        goal="goal",
        repository=".",
        integration_branch="integration",
        scheduler="fixed",
        target_concurrency=2,
        user_ceiling=2,
        parallel_limit=4,
        tasks={task.id: TaskRuntime(spec=task) for task in plan.tasks},
    )
    state.refresh_readiness()
    assert state.tasks["a"].status == TaskStatus.READY
    assert state.tasks["b"].status == TaskStatus.PENDING
    assert state.tasks["c"].status == TaskStatus.READY
    state.tasks["a"].status = TaskStatus.MERGED
    state.refresh_readiness()
    assert state.tasks["b"].status == TaskStatus.READY
    state.tasks["a"].status = TaskStatus.FAILED
    state.refresh_readiness()
    assert state.tasks["b"].status == TaskStatus.BLOCKED


def test_plan_rejects_cycles_and_extracts_event_json() -> None:
    with pytest.raises(ValidationError):
        Plan(
            goal="bad",
            tasks=[
                TaskSpec(id="a", title="A", description="A", depends_on=["b"]),
                TaskSpec(id="b", title="B", description="B", depends_on=["a"]),
            ],
        )
    value = extract_json('{"type":"text","text":"{\\"goal\\":\\"g\\",\\"tasks\\":[]}"}')
    assert value["goal"] == "g"
    split = extract_json(
        '{"part":{"text":"{\\"goal\\":\\"g\\","}}\n{"part":{"text":"\\"tasks\\":[]}"}}'
    )
    assert split == {"goal": "g", "tasks": []}


@pytest.mark.asyncio
async def test_planner_retries_malformed_output(tmp_path: Path) -> None:
    planner = OpenCodePlanner(tmp_path, Config(), "coder")
    planner._invoke = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("not json", "", 0),
            (
                '{"goal":"g","tasks":[{"id":"a","title":"A","description":"do A",'
                '"depends_on":[],"acceptance_criteria":[],"files_hint":[],"risk":"low"}]}',
                "",
                0,
            ),
        ]
    )
    result = await planner.plan("g", retries=1)
    assert result.tasks[0].id == "a"
    assert planner._invoke.await_count == 2


def test_resume_preserves_merged_and_recovers_running() -> None:
    first = TaskSpec(id="a", title="A", description="A")
    second = TaskSpec(id="b", title="B", description="B", depends_on=["a"])
    state = RunState(
        run_id="r",
        goal="g",
        repository=".",
        integration_branch="i",
        scheduler="fixed",
        target_concurrency=1,
        user_ceiling=1,
        parallel_limit=1,
        tasks={
            "a": TaskRuntime(spec=first, status=TaskStatus.MERGED, attempts=1),
            "b": TaskRuntime(spec=second, status=TaskStatus.RUNNING, attempts=1),
        },
    )
    prepare_resume(state)
    assert state.tasks["a"].status == TaskStatus.MERGED
    assert state.tasks["b"].status == TaskStatus.READY


def test_resume_round_trip_preserves_fleet_identity_and_pinning_history() -> None:
    task = TaskSpec(id="a", title="A", description="A", complexity="low")
    state = RunState(
        run_id="fleet-resume",
        goal="g",
        repository=".",
        integration_branch="i",
        scheduler="adaptive",
        target_concurrency=2,
        user_ceiling=4,
        parallel_limit=4,
        tasks={
            "a": TaskRuntime(
                spec=task,
                status=TaskStatus.RUNNING,
                attempts=1,
                assigned_tier="fast",
                assigned_model="fast-model",
                assigned_instance="fast-2",
                routing_history=[{"instance": "fast-2", "tier": "fast"}],
            )
        },
        fleet_enabled=True,
        planner_model="strong-1",
        fleet_topology={"instances": [{"instance_id": "fast-2"}]},
    )
    restored = RunState.model_validate_json(state.model_dump_json())
    prepare_resume(restored)
    assert restored.fleet_enabled is True
    assert restored.planner_model == "strong-1"
    assert restored.fleet_topology == state.fleet_topology
    assert restored.tasks["a"].assigned_instance == "fast-2"
    assert restored.tasks["a"].routing_history == [{"instance": "fast-2", "tier": "fast"}]
    assert restored.tasks["a"].status == TaskStatus.READY


def test_a_follow_up_message_revises_the_plan_instead_of_replacing_it() -> None:
    from adaptea.planner.prompts import planner_prompt

    plan = Plan(
        goal="Add appointment reminders",
        tasks=[
            TaskSpec(id="t1", title="Reminder model", description="Store reminders."),
            TaskSpec(id="t2", title="Send job", description="Send them.", depends_on=["t1"]),
        ],
    )
    fresh = planner_prompt("Add appointment reminders")
    revision = planner_prompt("Also send SMS", previous=plan)

    assert "This is a revision" not in fresh
    assert "Reminder model" not in fresh
    # The planner has to see the plan the user is looking at, or "also send SMS" is an
    # instruction to build a whole application that only sends SMS.
    assert "This is a revision" in revision
    assert "Reminder model" in revision
    assert "keep their IDs stable" in revision


def _failed_run() -> RunState:
    first = TaskSpec(id="a", title="A", description="A")
    second = TaskSpec(id="b", title="B", description="B", depends_on=["a"])
    third = TaskSpec(id="c", title="C", description="C")
    return RunState(
        run_id="r",
        goal="g",
        repository=".",
        integration_branch="i",
        scheduler="adaptive",
        target_concurrency=1,
        user_ceiling=4,
        parallel_limit=4,
        tasks={
            "a": TaskRuntime(
                spec=first,
                status=TaskStatus.FAILED,
                attempts=3,
                retry_count=2,
                retry_exhausted=True,
                failure="validation failed",
                failure_reason="tests did not pass",
            ),
            "b": TaskRuntime(spec=second, status=TaskStatus.BLOCKED, retry_exhausted=True),
            "c": TaskRuntime(spec=third, status=TaskStatus.MERGED, attempts=1),
        },
    )


def test_an_automatic_resume_leaves_a_spent_retry_budget_alone() -> None:
    state = _failed_run()
    prepare_resume(state)
    # Clearing this on every resume would loop forever on a task that always fails.
    assert state.tasks["a"].status == TaskStatus.FAILED
    assert state.tasks["b"].status == TaskStatus.BLOCKED


def test_a_requested_retry_reopens_failed_work_and_what_it_blocked() -> None:
    state = _failed_run()
    prepare_resume(state, retry_failed=True)

    # The failed task runs again, and the task it blocked stops being blocked with it.
    assert state.tasks["a"].status == TaskStatus.READY
    assert state.tasks["b"].status == TaskStatus.PENDING
    assert state.tasks["a"].retry_count == 0
    assert state.tasks["a"].retry_exhausted is False
    assert state.tasks["b"].retry_exhausted is False
    # Merged work is never redone.
    assert state.tasks["c"].status == TaskStatus.MERGED
    assert state.tasks["c"].attempts == 1
    # What failed is still on the record, and the next attempt is told about it.
    assert any(row.get("source") == "user_retry" for row in state.tasks["a"].failure_history)
    assert "tests did not pass" in (state.tasks["a"].pending_retry_context or "")


@pytest.mark.asyncio
async def test_planner_invoke_passes_opencode_config_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from unittest.mock import MagicMock

    planner = OpenCodePlanner(tmp_path, Config(), "test-model")
    captured_env: dict[str, str] = {}

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_env
        captured_env = kwargs.get("env", {})
        proc = MagicMock()
        proc.returncode = 0
        return proc

    async def fake_communicate(
        process: Any, timeout: Any = None, on_activity: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        res = MagicMock()
        res.stdout = b'{"goal":"g","tasks":[]}'
        res.stderr = b""
        res.timed_out = False
        return res

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("adaptea.planner.opencode.communicate_with_activity", fake_communicate)

    await planner._invoke("plan prompt")
    assert "OPENCODE_CONFIG_CONTENT" in captured_env
    config_data = json.loads(captured_env["OPENCODE_CONFIG_CONTENT"])
    assert "provider" in config_data or "providers" in config_data
