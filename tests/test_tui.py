from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import RichLog, TextArea

from adaptea.diagnostics.doctor import Check
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.projects import RecentProjects, initialize_repository
from adaptea.smoke import SmokeStep, SmokeTestResult
from adaptea.tui.app import (
    ActiveRunScreen,
    AdapteaApp,
    CalibrationScreen,
    CompletedRunScreen,
    DoctorScreen,
    HomeScreen,
    NewRunScreen,
    PlanReviewScreen,
    ProjectScreen,
    RunsScreen,
    SettingsScreen,
    SetupScreen,
    SmokeTestScreen,
)


class FakeServices:
    def __init__(self) -> None:
        self.diagnose_started = asyncio.Event()
        self.release_diagnose = asyncio.Event()
        self.release_diagnose.set()

    async def diagnose(self, root: Path) -> list[Check]:
        del root
        self.diagnose_started.set()
        await self.release_diagnose.wait()
        return [
            Check("PASS", "LM Studio server", "Running"),
            Check("PASS", "Selected model", "qwen-coder"),
            Check("PASS", "OpenCode", "Ready"),
            Check("PASS", "OpenCode → LM Studio", "configured"),
            Check("WARN", "Capacity profile", "run quick calibration"),
            Check("PASS", "Git repository", "fixture"),
        ]

    async def plan(self, root: Path, goal: str) -> Plan:
        del root
        return Plan(goal=goal, tasks=[TaskSpec(id="task-a", title="A", description="A")])

    async def available_models(self, root: Path) -> tuple[list[object], str | None]:
        del root
        return [], None

    async def calibrate(self, root: Path, **kwargs: Any) -> Path:
        del kwargs
        path = root / ".adaptea" / "calibration" / "fake"
        path.mkdir(parents=True)
        return path

    def save_plan(self, root: Path, plan: Plan) -> Path:
        path = root / "plan.json"
        path.write_text(plan.model_dump_json(), encoding="utf-8")
        return path

    async def create_run(self, root: Path, plan: Plan, *_args: Any) -> RunState:
        return make_state(root, plan)

    async def execute_run(self, root: Path, state: RunState, callback: Any = None) -> RunState:
        del root
        if callback:
            callback(state, None, 1)
        await asyncio.sleep(10)
        return state

    def set_admission(self, run_id: str, *, enabled: bool) -> bool:
        del run_id, enabled
        return True

    def abort_run(self, run_id: str) -> bool:
        del run_id
        return True


class CompletingServices(FakeServices):
    async def execute_run(self, root: Path, state: RunState, callback: Any = None) -> RunState:
        del root
        if callback:
            callback(state, None, 1)
        for task in state.tasks.values():
            task.status = TaskStatus.MERGED
            task.validation_exit_code = 0
            task.attempts = 1
        return state


def make_state(root: Path, plan: Plan | None = None) -> RunState:
    selected = plan or Plan(
        goal="Build fixture",
        tasks=[
            TaskSpec(id="a", title="A", description="A"),
            TaskSpec(id="b", title="B", description="B", depends_on=["a"]),
        ],
    )
    state = RunState(
        run_id="run-tui",
        goal=selected.goal,
        repository=str(root),
        integration_branch="adaptea/run-tui/integration",
        scheduler="adaptive",
        target_concurrency=3,
        user_ceiling=8,
        parallel_limit=8,
        tasks={task.id: TaskRuntime(spec=task) for task in selected.tasks},
    )
    state.refresh_readiness()
    return state


def make_app(tmp_path: Path, services: FakeServices | None = None) -> AdapteaApp:
    return AdapteaApp(
        tmp_path,
        services=services or FakeServices(),  # type: ignore[arg-type]
        recents=RecentProjects(tmp_path / "recent.json"),
    )


@pytest.mark.asyncio
async def test_tui_starts_on_home_with_real_state_labels(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert len(app.screen.query("#new-project")) == 1
        assert len(app.screen.query("#open-project")) == 1
        environment = app.screen.query_one("#environment").content
        assert "LM Studio server" in str(environment)
        assert "run quick calibration" in str(environment)


@pytest.mark.asyncio
async def test_long_diagnostics_do_not_freeze_navigation(tmp_path: Path) -> None:
    services = FakeServices()
    services.release_diagnose.clear()
    app = make_app(tmp_path, services)
    async with app.run_test(size=(120, 42)) as pilot:
        await services.diagnose_started.wait()
        await pilot.press("?")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "HelpScreen"
        services.release_diagnose.set()


@pytest.mark.asyncio
async def test_core_screens_mount_in_mocked_environment(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    plan = Plan(goal="Build it", tasks=[TaskSpec(id="a", title="A", description="A")])
    screens = [
        ProjectScreen(),
        SetupScreen(),
        DoctorScreen(),
        CalibrationScreen(),
        NewRunScreen(),
        PlanReviewScreen(plan),
        RunsScreen(),
        SettingsScreen(),
        SmokeTestScreen(),
    ]
    async with app.run_test(size=(140, 46)) as pilot:
        await pilot.pause()
        for screen in screens:
            app.push_screen(screen)
            await pilot.pause()
            assert app.screen is screen
            app.pop_screen()
            await pilot.pause()


@pytest.mark.asyncio
async def test_plan_editing_and_active_run_capacity_render(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    plan = Plan(
        goal="Build it",
        tasks=[
            TaskSpec(id="a", title="A", description="A"),
            TaskSpec(id="b", title="B", description="B", depends_on=["a"]),
        ],
    )
    review = PlanReviewScreen(plan)
    async with app.run_test(size=(140, 46)) as pilot:
        app.push_screen(review)
        await pilot.pause()
        assert review.query_one("#plan-table").row_count == 2
        edited = TaskSpec(
            id="a", title="Edited A", description="Updated", acceptance_criteria=["passes"]
        )
        review._task_edited(edited)
        assert review.plan.tasks[0].title == "Edited A"
        app.pop_screen()
        active = ActiveRunScreen(make_state(tmp_path, plan))
        app.push_screen(active)
        await pilot.pause()
        capacity = str(active.query_one("#capacity").content)
        assert "LM Studio ceiling       8" in capacity
        assert "Adaptea target          3" in capacity
        assert "Ready tasks" in capacity
        assert active.query_one("#task-table").row_count == 2


@pytest.mark.asyncio
async def test_keyboard_shortcuts_open_runs_and_setup(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert isinstance(app.screen, RunsScreen)
        await pilot.press("escape")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)


@pytest.mark.asyncio
async def test_leaving_active_run_does_not_cancel_background_worker(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    active = ActiveRunScreen(make_state(tmp_path))
    async with app.run_test(size=(120, 42)) as pilot:
        app.push_screen(active)
        await pilot.pause()
        worker = app.background_runs[active.state.run_id]
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not active
        assert not worker.is_cancelled
        assert not worker.is_finished


@pytest.mark.asyncio
async def test_full_mocked_new_run_plan_and_completion_workflow(tmp_path: Path) -> None:
    await initialize_repository(tmp_path)
    app = make_app(tmp_path, CompletingServices())
    async with app.run_test(size=(140, 46)) as pilot:
        new_run = NewRunScreen()
        app.push_screen(new_run)
        await pilot.pause()
        new_run.query_one("#goal", TextArea).text = "Build a tested feature"
        worker = new_run.generate_plan()
        await worker.wait()
        await pilot.pause()
        assert isinstance(app.screen, PlanReviewScreen)
        review = app.screen
        start_worker = review.start_run()
        await start_worker.wait()
        for _ in range(5):
            await pilot.pause()
            if isinstance(app.screen, CompletedRunScreen):
                break
        assert isinstance(app.screen, CompletedRunScreen)
        assert all(task.status == TaskStatus.MERGED for task in app.screen.state.tasks.values())


@pytest.mark.asyncio
async def test_mocked_mvp_smoke_test_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_smoke(*_args: Any, **kwargs: Any) -> SmokeTestResult:
        step = SmokeStep("Final test suite", True, "passed")
        progress = kwargs.get("progress")
        if progress:
            progress(step)
        return SmokeTestResult(True, [step], 0.1)

    monkeypatch.setattr("adaptea.tui.app.run_mvp_smoke_test", fake_smoke)
    app = make_app(tmp_path)
    screen = SmokeTestScreen()
    async with app.run_test(size=(120, 42)) as pilot:
        app.push_screen(screen)
        await pilot.pause()
        worker = screen.run_smoke()
        await worker.wait()
        assert "ADAPTEA MVP IS WORKING" in str(screen.query_one(RichLog).lines)
