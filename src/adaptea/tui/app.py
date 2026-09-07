from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import Worker

from adaptea.comparison import comparison_summary, execute_comparison
from adaptea.config import load_config
from adaptea.diagnostics.doctor import Check
from adaptea.history import RunRecord, list_runs
from adaptea.models import Plan, RunState, TaskSpec, TaskStatus, TelemetrySample
from adaptea.projects import (
    RecentProjects,
    create_project,
    filesystem_roots,
    initialize_repository,
    inspect_project,
)
from adaptea.services import ApplicationServices, read_jsonl
from adaptea.setup.configuration import merge_adaptea_config
from adaptea.setup.manager import SetupManager, create_setup_logger
from adaptea.smoke import SmokeStep, run_mvp_smoke_test

LOGO = "A⟨EA⟩  ADAPTEA"
TAGLINE = "adaptive local agents"


class AdapteaApp(App[None]):
    TITLE = "ADAPTEA"
    SUB_TITLE = TAGLINE
    CSS = """
    Screen { background: #0b0f14; color: #d7e0ea; }
    Header { background: #111923; color: #8ce0c2; }
    Footer { background: #111923; color: #93a4b7; }
    .page { padding: 1 3; }
    .brand { color: #79e0b3; text-style: bold; margin-bottom: 1; }
    .muted { color: #8291a3; }
    .success { color: #79e0b3; }
    .warning { color: #f0c674; }
    .danger { color: #ff7b72; }
    .section { color: #9fb4ca; text-style: bold; margin-top: 1; }
    .card { background: #111923; padding: 1 2; margin: 1 0; border-left: solid #26394c; }
    .actions { height: auto; margin: 1 0; }
    .actions Button { margin-right: 1; min-width: 18; }
    Button.-primary { background: #19795a; color: white; }
    Button:focus { text-style: bold; }
    Input, TextArea, Select, RadioSet, OptionList, DataTable, RichLog, DirectoryTree {
        background: #0e151e; border: tall #243548;
    }
    TextArea { height: 8; }
    #home-body, #project-body { max-width: 110; width: 100%; align-horizontal: center; }
    #environment { min-height: 9; }
    #recent-runs { height: 9; }
    #folder-layout { grid-size: 2; grid-columns: 2fr 1fr; height: 1fr; }
    #folder-tree { height: 1fr; }
    #folder-side { padding-left: 1; }
    #plan-table, #runs-table, #task-table { height: 1fr; min-height: 12; }
    #run-grid { grid-size: 2; grid-columns: 2fr 1fr; height: 1fr; }
    #controller-events { height: 1fr; min-height: 14; }
    #capacity { height: 8; }
    #setup-list, #doctor-list, #smoke-log, #calibration-log { height: 1fr; min-height: 15; }
    #modal { width: 80%; max-width: 90; height: auto; max-height: 90%; padding: 1 2;
             background: #111923; border: round #3d5a73; }
    FolderBrowserScreen #modal { height: 90%; }
    .field-label { color: #9fb4ca; margin-top: 1; }
    .metric { color: #d7e0ea; }
    .status-line { height: auto; }
    TabbedContent { height: 1fr; }
    """
    BINDINGS = [
        Binding("ctrl+n", "new_run", "New Run", show=True),
        Binding("ctrl+r", "runs", "Runs", show=True),
        Binding("ctrl+s", "setup", "Setup", show=True),
        Binding("c", "calibration", "Calibration", show=True),
        Binding("?", "help", "Help", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        root: Path | None = None,
        *,
        services: ApplicationServices | None = None,
        recents: RecentProjects | None = None,
    ) -> None:
        super().__init__()
        initial = (root or Path.cwd()).expanduser().resolve()
        self.active_project: Path | None = initial if initial.is_dir() else None
        self.services = services or ApplicationServices()
        self.recents = recents or RecentProjects()
        self.background_runs: dict[str, Worker[RunState | None]] = {}

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def set_active_project(self, path: Path) -> None:
        self.active_project = path.expanduser().resolve()
        self.recents.remember(self.active_project)

    def require_project(self) -> Path | None:
        if self.active_project and self.active_project.is_dir():
            return self.active_project
        self.active_project = None
        self.notify("Select or create a project first.", severity="warning")
        self.push_screen(FolderBrowserScreen(), self._folder_selected)
        return None

    def _folder_selected(self, selected: Path | None) -> None:
        if selected:
            self.set_active_project(selected)
            self.push_screen(ProjectScreen())

    def action_home(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        if not isinstance(self.screen, HomeScreen):
            self.push_screen(HomeScreen())

    def action_new_run(self) -> None:
        if self.require_project():
            self.push_screen(NewRunScreen())

    def action_runs(self) -> None:
        self.push_screen(RunsScreen())

    def action_setup(self) -> None:
        if self.require_project():
            self.push_screen(SetupScreen())

    def action_calibration(self) -> None:
        if self.require_project():
            self.push_screen(CalibrationScreen())

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


class AdapteaScreen(Screen[None]):
    BINDINGS = [Binding("escape", "back", "Back", show=False)]

    @property
    def adaptea(self) -> AdapteaApp:
        return cast(AdapteaApp, self.app)

    def action_back(self) -> None:
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def project(self) -> Path:
        root = self.adaptea.active_project
        if root is None or not root.is_dir():
            raise RuntimeError("No active project is selected.")
        return root

    def concise_error(self, exc: Exception) -> str:
        return (str(exc).strip() or exc.__class__.__name__).splitlines()[0][:400]


class HomeScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="home-body", classes="page"):
            yield Static(f"{LOGO}\n{TAGLINE}", classes="brand")
            yield Static("CURRENT PROJECT", classes="section")
            yield Static("Loading…", id="current-project", classes="card")
            yield Static("ENVIRONMENT", classes="section")
            yield Static("Checking real services…", id="environment", classes="card")
            with Horizontal(classes="actions"):
                yield Button("New Coding Run", id="new-run", variant="primary")
                yield Button("New Project", id="new-project")
                yield Button("Open Existing Project", id="open-project")
                yield Button("Recent Projects", id="recent-projects")
            with Horizontal(classes="actions"):
                yield Button("Compare Serial vs Adaptive", id="compare")
                yield Button("MVP Smoke Test", id="smoke-test")
            with Horizontal(classes="actions"):
                yield Button("Setup", id="setup")
                yield Button("Doctor", id="doctor")
                yield Button("Calibration", id="calibration")
                yield Button("Runs", id="runs")
                yield Button("Settings", id="settings")
            yield Static("RECENT RUNS", classes="section")
            yield Static("No runs found.", id="recent-runs", classes="card")
        yield Footer()

    def on_mount(self) -> None:
        root = self.adaptea.active_project
        if root and root.is_dir():
            self.query_one("#current-project", Static).update(f"{root.name}\n{root}")
            self.refresh_environment()
        else:
            self.adaptea.active_project = None
            self.query_one("#current-project", Static).update("No project selected")
            self.query_one("#environment", Static).update("Select a project to run diagnostics.")
        self._render_recent_runs()

    def _render_recent_runs(self) -> None:
        projects = [Path(row.path) for row in self.adaptea.recents.load()]
        if self.adaptea.active_project:
            projects.insert(0, self.adaptea.active_project)
        rows = list_runs(list(dict.fromkeys(projects)))[:5]
        text = "\n".join(
            f"{row.run_id:<28} {row.scheduler:<9} {row.status:<10} {row.merged}/{row.tasks}"
            for row in rows
        )
        self.query_one("#recent-runs", Static).update(text or "No runs found.")

    @work(exclusive=True)
    async def refresh_environment(self) -> None:
        try:
            checks = await self.adaptea.services.diagnose(self.project())
        except Exception as exc:
            self.query_one("#environment", Static).update(
                f"[red]Diagnostics unavailable:[/] {self.concise_error(exc)}\nOpen Setup to repair."
            )
            return
        wanted = {
            "LM Studio server",
            "Selected model",
            "OpenCode",
            "OpenCode → LM Studio",
            "Capacity profile",
            "Git repository",
        }
        lines = []
        for check in checks:
            if check.name in wanted:
                marker = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[check.level]
                color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[check.level]
                lines.append(f"[{color}]{marker}[/] {check.name:<24} {check.detail}")
        self.query_one("#environment", Static).update("\n".join(lines))

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        actions: dict[str, Any] = {
            "new-run": self.adaptea.action_new_run,
            "new-project": lambda: self.app.push_screen(NewProjectScreen()),
            "open-project": lambda: self.app.push_screen(
                FolderBrowserScreen(), self._select_project
            ),
            "recent-projects": lambda: self.app.push_screen(RecentProjectsScreen()),
            "compare": lambda: self._project_screen(ComparisonScreen()),
            "smoke-test": lambda: self._project_screen(SmokeTestScreen()),
            "setup": self.adaptea.action_setup,
            "doctor": lambda: self._project_screen(DoctorScreen()),
            "calibration": self.adaptea.action_calibration,
            "runs": self.adaptea.action_runs,
            "settings": lambda: self._project_screen(SettingsScreen()),
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def _select_project(self, selected: Path | None) -> None:
        if selected:
            self.adaptea.set_active_project(selected)
            self.app.push_screen(ProjectScreen())

    def _project_screen(self, screen: Screen[None]) -> None:
        if self.adaptea.require_project():
            self.app.push_screen(screen)


class FolderBrowserScreen(ModalScreen[Path | None]):
    def __init__(self, start: Path | None = None) -> None:
        super().__init__()
        self.selected = (start or Path.cwd()).expanduser().resolve()
        self.recent_paths: list[Path] = []

    def compose(self) -> ComposeResult:
        roots = [(str(path), str(path)) for path in filesystem_roots()]
        with Vertical(id="modal"):
            yield Static("OPEN EXISTING PROJECT", classes="brand")
            yield Label("Drive / filesystem root", classes="field-label")
            yield Select(roots, value=str(Path(self.selected.anchor or self.selected)), id="roots")
            yield Label("Enter a path manually", classes="field-label")
            with Horizontal():
                yield Input(value=str(self.selected), id="manual-path")
                yield Button("Go", id="go")
                yield Button("Up", id="up")
            with Grid(id="folder-layout"):
                yield DirectoryTree(str(self.selected), id="folder-tree")
                with Vertical(id="folder-side"):
                    yield Static("SELECTED", classes="section")
                    yield Static(str(self.selected), id="selected-path", classes="card")
                    yield Static("RECENT FOLDERS", classes="section")
                    yield ListView(id="folder-recents")
            with Horizontal(classes="actions"):
                yield Button("Use This Folder", id="confirm", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        recent = RecentProjects().load()
        self.recent_paths = [Path(row.path) for row in recent]
        view = self.query_one("#folder-recents", ListView)
        for row in recent:
            view.append(ListItem(Label(row.path), id=f"recent-{len(view.children)}"))

    def _navigate(self, path: Path) -> None:
        candidate = path.expanduser()
        if not candidate.is_dir():
            self.notify("That folder does not exist.", severity="error")
            return
        self.selected = candidate.resolve()
        self.query_one("#manual-path", Input).value = str(self.selected)
        self.query_one("#selected-path", Static).update(str(self.selected))
        tree = self.query_one("#folder-tree", DirectoryTree)
        tree.path = str(self.selected)
        tree.reload()

    @on(DirectoryTree.DirectorySelected)
    def directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.selected = Path(event.path).resolve()
        self.query_one("#manual-path", Input).value = str(self.selected)
        self.query_one("#selected-path", Static).update(str(self.selected))

    @on(Select.Changed, "#roots")
    def root_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._navigate(Path(str(event.value)))

    @on(ListView.Selected, "#folder-recents")
    def recent_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and 0 <= index < len(self.recent_paths):
            self._navigate(self.recent_paths[index])

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "go":
            self._navigate(Path(self.query_one("#manual-path", Input).value))
        elif button_id == "up":
            self._navigate(self.selected.parent)
        elif button_id == "confirm":
            self.dismiss(self.selected)
        elif button_id == "cancel":
            self.dismiss(None)


class RecentProjectsScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("RECENT PROJECTS", classes="brand")
            yield OptionList(id="project-options")
            with Horizontal(classes="actions"):
                yield Button("Open", id="open", variant="primary")
                yield Button("Browse…", id="browse")
        yield Footer()

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        for row in self.adaptea.recents.load():
            options.add_option(row.path)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            self.app.push_screen(FolderBrowserScreen(), self._selected)
        elif event.button.id == "open":
            options = self.query_one(OptionList)
            if options.highlighted is not None:
                option = options.get_option_at_index(options.highlighted)
                self._selected(Path(str(option.prompt)))

    def _selected(self, path: Path | None) -> None:
        if path:
            self.adaptea.set_active_project(path)
            self.app.push_screen(ProjectScreen())


class NewProjectScreen(AdapteaScreen):
    def __init__(self) -> None:
        super().__init__()
        self.parent_folder = Path.cwd().resolve()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static("NEW PROJECT", classes="brand")
            yield Label("Parent folder", classes="field-label")
            with Horizontal():
                yield Input(value=str(self.parent_folder), id="parent")
                yield Button("Browse…", id="browse")
            yield Label("Project name", classes="field-label")
            yield Input(placeholder="hospital-opl-platform", id="project-name")
            yield Label("What do you want to build?", classes="field-label")
            yield TextArea(id="new-project-goal")
            yield Static(
                "Adaptea will create the folder, initialize Git, make an initial commit, and "
                "generate a bootstrap-aware plan.",
                classes="card muted",
            )
            yield Static("", id="new-project-status")
            with Horizontal(classes="actions"):
                yield Button("Create & Generate Plan", id="create", variant="primary")
                yield Button("Back", id="back")
        yield Footer()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            self.app.push_screen(FolderBrowserScreen(self.parent_folder), self._parent_selected)
        elif event.button.id == "create":
            self.create_and_plan()
        elif event.button.id == "back":
            self.action_back()

    def _parent_selected(self, path: Path | None) -> None:
        if path:
            self.parent_folder = path
            self.query_one("#parent", Input).value = str(path)

    @work(exclusive=True)
    async def create_and_plan(self) -> None:
        status = self.query_one("#new-project-status", Static)
        try:
            parent = Path(self.query_one("#parent", Input).value)
            name = self.query_one("#project-name", Input).value
            goal = self.query_one("#new-project-goal", TextArea).text.strip()
            if not goal:
                raise ValueError("Describe what you want to build.")
            status.update("Creating folder and initializing Git…")
            info = await create_project(parent, name)
            self.adaptea.set_active_project(info.path)
            status.update("Generating a bootstrap-aware plan…")
            plan = await self.adaptea.services.plan(info.path, goal)
        except Exception as exc:
            status.update(f"[red]{self.concise_error(exc)}[/]")
            return
        self.app.push_screen(PlanReviewScreen(plan))


class ProjectScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="project-body", classes="page"):
            yield Static("PROJECT", classes="brand")
            yield Static("Inspecting…", id="project-info", classes="card")
            with Horizontal(classes="actions"):
                yield Button("New Coding Run", id="new-run", variant="primary")
                yield Button("Open Runs", id="runs")
                yield Button("Project Settings", id="settings")
                yield Button("Change Project", id="change")
            with Horizontal(classes="actions"):
                yield Button("Initialize Git Repository", id="init-git")
                yield Button("Home", id="home")
        yield Footer()

    def on_mount(self) -> None:
        self.inspect()

    @work(exclusive=True)
    async def inspect(self) -> None:
        try:
            info = await inspect_project(self.project())
            config = load_config(info.path)
            capacity_path = info.path / ".adaptea" / "capacity.json"
            capacity = (
                json.loads(capacity_path.read_text(encoding="utf-8"))
                if capacity_path.is_file()
                else {}
            )
            calibration = capacity.get("recommended_starting_concurrency", "not calibrated")
            git_line = (
                f"✓ {info.git_status} ({info.git_branch})" if info.is_git else "✗ Not initialized"
            )
            text = (
                f"[b]{info.name}[/]\n{info.path}\n\n"
                f"Git             {git_line}\n"
                f"Project type    {info.project_type}\n"
                f"Model           {config.lmstudio.model or 'not selected'}\n"
                f"Calibration     C*={calibration}\n"
                f"Config          {'loaded' if info.has_config else 'defaults / not saved'}"
            )
            self.query_one("#project-info", Static).update(text)
            self.query_one("#init-git", Button).display = not info.is_git
        except Exception as exc:
            self.query_one("#project-info", Static).update(f"[red]{self.concise_error(exc)}[/]")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "new-run":
            self.app.push_screen(NewRunScreen())
        elif button_id == "runs":
            self.app.push_screen(RunsScreen())
        elif button_id == "settings":
            self.app.push_screen(SettingsScreen())
        elif button_id == "change":
            self.app.push_screen(FolderBrowserScreen(), self._changed)
        elif button_id == "init-git":
            self.app.push_screen(
                ConfirmScreen(
                    "Initialize Git Repository?",
                    "This creates .git, .gitignore, and an initial commit in the selected folder.",
                ),
                self._confirmed_init,
            )
        elif button_id == "home":
            self.adaptea.action_home()

    def _changed(self, path: Path | None) -> None:
        if path:
            self.adaptea.set_active_project(path)
            self.inspect()

    def _confirmed_init(self, confirmed: bool | None) -> None:
        if confirmed:
            self.initialize_git()

    @work(exclusive=True)
    async def initialize_git(self) -> None:
        try:
            await initialize_repository(self.project())
        except Exception as exc:
            self.notify(self.concise_error(exc), severity="error")
        else:
            self.notify("Git repository initialized.")
            self.inspect()


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(self.dialog_title, classes="brand")
            yield Static(self.message, classes="card")
            with Horizontal(classes="actions"):
                yield Button("Confirm", id="yes", variant="error")
                yield Button("Cancel", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


def render_checks(checks: list[Check]) -> str:
    colors = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    markers = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    return "\n".join(
        f"[{colors[row.level]}]{markers[row.level]} {row.level:<4}[/]  {row.name:<28} {row.detail}"
        for row in checks
    )


class DoctorScreen(AdapteaScreen):
    def __init__(self) -> None:
        super().__init__()
        self.check_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("DOCTOR", classes="brand")
            yield Static(
                "System · Inference · Coding Agent · Repository · Adaptea", classes="muted"
            )
            yield RichLog(id="doctor-list", markup=True, wrap=True)
            with Horizontal(classes="actions"):
                yield Button("Re-check", id="recheck", variant="primary")
                yield Button("Fix in Setup", id="setup")
                yield Button("Copy Diagnostics", id="copy")
                yield Button("MVP Smoke Test", id="smoke")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_checks()

    @work(exclusive=True)
    async def refresh_checks(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        log.write("Running diagnostics…")
        try:
            checks = await self.adaptea.services.diagnose(self.project())
        except Exception as exc:
            log.write(f"[red]Diagnostics failed:[/] {self.concise_error(exc)}")
            return
        log.clear()
        self.check_text = render_checks(checks)
        log.write(self.check_text)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recheck":
            self.refresh_checks()
        elif event.button.id == "setup":
            self.app.push_screen(SetupScreen())
        elif event.button.id == "smoke":
            self.app.push_screen(SmokeTestScreen())
        elif event.button.id == "copy":
            self.app.copy_to_clipboard(self.check_text)
            self.notify("Diagnostics copied.")


class SetupScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("SETUP", classes="brand")
            yield RichLog(id="setup-list", markup=True, wrap=True)
            yield Static("", id="setup-message")
            with Horizontal(classes="actions"):
                yield Button("Re-check", id="recheck", variant="primary")
                yield Button("Apply Safe Repairs", id="fix")
                yield Button("Select Model", id="model")
                yield Button("Quick Calibration", id="calibrate")
        yield Footer()

    def on_mount(self) -> None:
        self.diagnose()

    @work(exclusive=True)
    async def diagnose(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        log.write("Inspecting setup state…")
        try:
            checks = await self.adaptea.services.diagnose(self.project())
        except Exception as exc:
            log.write(f"[red]{self.concise_error(exc)}[/]")
            return
        log.clear()
        log.write(render_checks(checks))

    @work(exclusive=True)
    async def fix_safe(self) -> None:
        message = self.query_one("#setup-message", Static)
        message.update("Applying safe, local setup actions…")
        try:
            config = load_config(self.project())
            logger, log_path = create_setup_logger(self.project())
            manager = SetupManager(self.project(), config, logger=logger)
            outcome = await manager.fix_all(automatic=True)
            lines = [*outcome.messages, *outcome.failures]
            message.update("\n".join(lines[-5:]) + f"\nLog: {log_path}")
        except Exception as exc:
            message.update(f"[red]{self.concise_error(exc)}[/]")
        self.diagnose()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recheck":
            self.diagnose()
        elif event.button.id == "fix":
            self.fix_safe()
        elif event.button.id == "model":
            self.app.push_screen(ModelSelectionScreen())
        elif event.button.id == "calibrate":
            self.app.push_screen(CalibrationScreen())


class ModelSelectionScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("MODEL SELECTION", classes="brand")
            yield Static(
                "Loading locally available LM Studio models…", id="model-note", classes="card"
            )
            yield OptionList(id="models")
            with Horizontal(classes="actions"):
                yield Button("Use Selected Model", id="save", variant="primary")
                yield Button("Refresh", id="refresh")
        yield Footer()

    def on_mount(self) -> None:
        self.load_models()

    @work(exclusive=True)
    async def load_models(self) -> None:
        options = self.query_one("#models", OptionList)
        options.clear_options()
        try:
            models, configured = await self.adaptea.services.available_models(self.project())
        except Exception as exc:
            self.query_one("#model-note", Static).update(f"[red]{self.concise_error(exc)}[/]")
            return
        for model in models:
            key = str(getattr(model, "key", "unknown"))
            loaded = bool(getattr(model, "loaded", False))
            parallel = None
            instances = getattr(model, "loaded_instances", [])
            if instances:
                parallel = getattr(instances[0].config, "parallel", None)
            options.add_option(
                f"{'●' if loaded else '○'} {key}"
                f"{'  [configured]' if key == configured else ''}  parallel={parallel or 'unknown'}"
            )
        self.query_one("#model-note", Static).update(
            "Changing the model invalidates incompatible calibration profiles. "
            "Recalibration is recommended."
        )

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh":
            self.load_models()
        elif event.button.id == "save":
            options = self.query_one("#models", OptionList)
            if options.highlighted is None:
                return
            prompt = str(options.get_option_at_index(options.highlighted).prompt)
            key = prompt.split(" ", 1)[1].split("  ", 1)[0]
            config = load_config(self.project())
            merge_adaptea_config(
                self.project() / "adaptea.toml",
                base_url=config.lmstudio.base_url,
                model=key,
                lms_executable=config.lmstudio.lms_executable,
                opencode_executable=config.worker.executable,
                max_agents=config.worker.max_agents,
                scheduler=config.worker.default_scheduler,
            )
            self.notify("Model selected. Run calibration before Adaptive mode.")


class CalibrationScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("CALIBRATION", classes="brand")
            yield Static("Loading current capacity profile…", id="capacity-profile", classes="card")
            yield RichLog(id="calibration-log", markup=True, wrap=True)
            with Horizontal(classes="actions"):
                yield Button("Quick Calibration", id="quick", variant="primary")
                yield Button("Full Calibration", id="full")
                yield Button("Advanced: edit adaptea.toml", id="advanced")
        yield Footer()

    def on_mount(self) -> None:
        self.render_profile()

    def render_profile(self) -> None:
        config = load_config(self.project())
        profile_path = self.project() / ".adaptea" / "capacity.json"
        if profile_path.is_file():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except ValueError:
                profile = {}
            text = (
                f"Model: {config.lmstudio.model or 'not selected'}\n"
                f"Recommended starting concurrency: "
                f"{profile.get('recommended_starting_concurrency', 'unknown')}\n"
                f"LM Studio ceiling: {profile.get('safe_max_concurrency', 'unknown')}"
            )
        else:
            text = (
                f"Model: {config.lmstudio.model or 'not selected'}\n"
                "Adaptive scheduler has no calibration prior. Run Quick Calibration."
            )
        self.query_one("#capacity-profile", Static).update(text)

    @work(exclusive=True)
    async def run_calibration(self, quick: bool) -> None:
        log = self.query_one("#calibration-log", RichLog)
        log.clear()
        log.write("Quick calibration started…" if quick else "Full calibration started…")

        def progress(message: str) -> None:
            log.write(message)

        try:
            path = await self.adaptea.services.calibrate(
                self.project(), quick=quick, progress=progress
            )
        except Exception as exc:
            log.write(f"[red]Calibration failed:[/] {self.concise_error(exc)}")
            return
        log.write(f"[green]Calibration complete:[/] {path}")
        self.render_profile()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quick":
            self.run_calibration(True)
        elif event.button.id == "full":
            self.run_calibration(False)
        elif event.button.id == "advanced":
            self.app.push_screen(SettingsScreen())


class NewRunScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        config = load_config(self.project())
        with VerticalScroll(classes="page"):
            yield Static("NEW CODING RUN", classes="brand")
            yield Label("Repository", classes="field-label")
            with Horizontal():
                yield Input(value=str(self.project()), id="repository")
                yield Button("Browse…", id="browse")
            yield Label("What do you want Adaptea to build?", classes="field-label")
            yield TextArea(id="goal")
            yield Label("Run Mode", classes="field-label")
            with RadioSet(id="run-mode"):
                yield RadioButton("Adaptive", id="adaptive", value=True)
                yield RadioButton("Serial / Single Agent", id="serial")
                yield RadioButton("Fixed Concurrency", id="fixed")
                yield RadioButton("Naive Parallel", id="naive")
            with Horizontal():
                yield Label("Max agents: ")
                yield Input(value=str(config.worker.max_agents), type="integer", id="max-agents")
                yield Label(" Fixed concurrency: ")
                yield Input(value="2", type="integer", id="fixed-concurrency")
            yield Static("", id="new-run-status")
            with Horizontal(classes="actions"):
                yield Button("Generate Plan", id="generate", variant="primary")
                yield Button("Back", id="back")
        yield Footer()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            self.app.push_screen(FolderBrowserScreen(self.project()), self._repository_selected)
        elif event.button.id == "generate":
            self.generate_plan()
        elif event.button.id == "back":
            self.action_back()

    def _repository_selected(self, path: Path | None) -> None:
        if path:
            self.query_one("#repository", Input).value = str(path)

    def selected_mode(self) -> tuple[str, int | None]:
        pressed = self.query_one("#run-mode", RadioSet).pressed_button
        selected = pressed.id if pressed else "adaptive"
        if selected == "serial":
            return "fixed", 1
        if selected == "fixed":
            return "fixed", int(self.query_one("#fixed-concurrency", Input).value)
        return selected or "adaptive", None

    @work(exclusive=True)
    async def generate_plan(self) -> None:
        status = self.query_one("#new-run-status", Static)
        try:
            root = Path(self.query_one("#repository", Input).value).expanduser().resolve()
            info = await inspect_project(root)
            if not info.is_git:
                raise ValueError(
                    "The selected folder is not a Git repository. Initialize Git first."
                )
            goal = self.query_one("#goal", TextArea).text.strip()
            if not goal:
                raise ValueError("Enter a development goal.")
            maximum = int(self.query_one("#max-agents", Input).value)
            if maximum < 1:
                raise ValueError("Max agents must be at least 1.")
            mode, fixed = self.selected_mode()
            self.adaptea.set_active_project(root)
            status.update("Generating a validated dependency plan through OpenCode…")
            plan = await self.adaptea.services.plan(root, goal)
        except Exception as exc:
            status.update(f"[red]{self.concise_error(exc)}[/]")
            return
        self.app.push_screen(PlanReviewScreen(plan, mode, maximum, fixed))


class PlanReviewScreen(AdapteaScreen):
    def __init__(
        self,
        plan: Plan,
        mode: str = "adaptive",
        max_agents: int = 8,
        fixed_concurrency: int | None = None,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.mode = mode
        self.max_agents = max_agents
        self.fixed_concurrency = fixed_concurrency

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("PLAN REVIEW", classes="brand")
            yield Static(self.plan.goal, classes="card")
            yield DataTable(id="plan-table", cursor_type="row")
            yield Static("", id="plan-error", classes="danger")
            with Horizontal(classes="actions"):
                yield Button("Start Run", id="start", variant="primary")
                yield Button("Edit Task", id="edit")
                yield Button("Add Task", id="add")
                yield Button("Delete Task", id="delete")
                yield Button("Regenerate", id="regenerate")
                yield Button("Save Plan", id="save")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("State", "Task", "Title", "Dependencies", "Acceptance")
        self.render_tasks()

    def render_tasks(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for task in self.plan.tasks:
            table.add_row(
                "READY" if not task.depends_on else "DEPENDENT",
                task.id,
                task.title,
                ", ".join(task.depends_on) or "—",
                "; ".join(task.acceptance_criteria) or "—",
                key=task.id,
            )

    def selected_task(self) -> TaskSpec | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        task_id = str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        return next((task for task in self.plan.tasks if task.id == task_id), None)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "start":
            self.start_run()
        elif button_id == "edit":
            task = self.selected_task()
            if task:
                self.app.push_screen(TaskEditorScreen(task), self._task_edited)
        elif button_id == "add":
            self.app.push_screen(TaskEditorScreen(), self._task_edited)
        elif button_id == "delete":
            task = self.selected_task()
            if task and len(self.plan.tasks) > 1:
                try:
                    self.plan = Plan(
                        goal=self.plan.goal,
                        tasks=[item for item in self.plan.tasks if item.id != task.id],
                    )
                except ValidationError as exc:
                    self.query_one("#plan-error", Static).update(
                        "Remove dependencies on this task before deleting it. "
                        + str(exc).splitlines()[0]
                    )
                else:
                    self.render_tasks()
        elif button_id == "regenerate":
            self.regenerate()
        elif button_id == "save":
            path = self.adaptea.services.save_plan(self.project(), self.plan)
            self.notify(f"Plan saved: {path}")
        elif button_id == "back":
            self.action_back()

    def _task_edited(self, task: TaskSpec | None) -> None:
        if task is None:
            return
        tasks = [task if existing.id == task.id else existing for existing in self.plan.tasks]
        if not any(existing.id == task.id for existing in self.plan.tasks):
            tasks.append(task)
        try:
            self.plan = Plan(goal=self.plan.goal, tasks=tasks)
        except ValidationError as exc:
            self.query_one("#plan-error", Static).update(str(exc).splitlines()[0])
            return
        self.query_one("#plan-error", Static).update("")
        self.render_tasks()

    @work(exclusive=True)
    async def regenerate(self) -> None:
        try:
            self.plan = await self.adaptea.services.plan(self.project(), self.plan.goal)
        except Exception as exc:
            self.query_one("#plan-error", Static).update(self.concise_error(exc))
        else:
            self.render_tasks()

    @work(exclusive=True)
    async def start_run(self) -> None:
        try:
            state = await self.adaptea.services.create_run(
                self.project(),
                self.plan,
                cast(Any, self.mode),
                self.max_agents,
                self.fixed_concurrency,
            )
        except Exception as exc:
            self.query_one("#plan-error", Static).update(self.concise_error(exc))
            return
        self.app.push_screen(ActiveRunScreen(state))


class TaskEditorScreen(ModalScreen[TaskSpec | None]):
    def __init__(self, task: TaskSpec | None = None) -> None:
        super().__init__()
        self.task_spec = task

    def compose(self) -> ComposeResult:
        task = self.task_spec
        with VerticalScroll(id="modal"):
            yield Static("EDIT TASK" if task else "ADD TASK", classes="brand")
            yield Label("ID", classes="field-label")
            yield Input(value=task.id if task else "", id="task-id", disabled=task is not None)
            yield Label("Title", classes="field-label")
            yield Input(value=task.title if task else "", id="task-title")
            yield Label("Description", classes="field-label")
            yield TextArea(task.description if task else "", id="task-description")
            yield Label("Acceptance criteria (one per line)", classes="field-label")
            yield TextArea(
                "\n".join(task.acceptance_criteria) if task else "", id="task-acceptance"
            )
            yield Label("Depends on (comma separated IDs)", classes="field-label")
            yield Input(value=", ".join(task.depends_on) if task else "", id="task-dependencies")
            yield Static("", id="editor-error", classes="danger")
            with Horizontal(classes="actions"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            task = TaskSpec(
                id=self.query_one("#task-id", Input).value,
                title=self.query_one("#task-title", Input).value,
                description=self.query_one("#task-description", TextArea).text,
                acceptance_criteria=[
                    line.strip()
                    for line in self.query_one("#task-acceptance", TextArea).text.splitlines()
                    if line.strip()
                ],
                depends_on=[
                    item.strip()
                    for item in self.query_one("#task-dependencies", Input).value.split(",")
                    if item.strip()
                ],
                files_hint=self.task_spec.files_hint if self.task_spec else [],
                risk=self.task_spec.risk if self.task_spec else "medium",
            )
        except ValidationError as exc:
            self.query_one("#editor-error", Static).update(str(exc).splitlines()[0])
            return
        self.dismiss(task)


class ActiveRunScreen(AdapteaScreen):
    BINDINGS = [Binding("escape", "leave", "Leave screen", show=True)]

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state
        self.started = time.monotonic()
        self.last_sample: TelemetrySample | None = None
        self.running_workers = 0
        self.admission_paused = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(classes="page"):
            yield Static(f"ADAPTEA RUN  {self.state.run_id}", classes="brand")
            yield Static(
                f"Goal: {self.state.goal}\nScheduler: {self.state.scheduler.upper()}",
                classes="card",
            )
            with Grid(id="run-grid"):
                with Vertical():
                    yield Static("CAPACITY", classes="section")
                    yield Static("", id="capacity", classes="card")
                    yield Static("TASKS", classes="section")
                    yield DataTable(id="task-table", cursor_type="row")
                with Vertical():
                    yield Static("ADMISSION CONTROLLER", classes="section")
                    yield RichLog(id="controller-events", markup=True, wrap=True)
            with Horizontal(classes="actions"):
                yield Button("Task Detail", id="detail")
                yield Button("Leave Screen (run continues)", id="leave")
                yield Button("Stop Admitting", id="stop-admitting")
                yield Button("Abort Entire Run", id="abort", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("Task", "Status", "Worker / branch", "Attempts")
        self.render_state(self.state, None, 0)
        self.background_runs_start()

    def background_runs_start(self) -> None:
        existing = self.adaptea.background_runs.get(self.state.run_id)
        if existing is not None and not existing.is_finished:
            return
        worker = self.adaptea.run_worker(
            self.execute(),
            name=f"run-{self.state.run_id}",
            group="coding-runs",
            exclusive=False,
            exit_on_error=False,
        )
        self.adaptea.background_runs[self.state.run_id] = worker

    def render_state(self, state: RunState, sample: TelemetrySample | None, running: int) -> None:
        self.state = state
        self.last_sample = sample
        self.running_workers = running
        if not self.is_mounted:
            return
        ready = sum(task.status == TaskStatus.READY for task in state.tasks.values())
        merged = sum(task.status == TaskStatus.MERGED for task in state.tasks.values())
        queued = (
            sample.queued_predictions
            if sample and sample.queued_predictions is not None
            else "unknown"
        )
        speed = (
            f"{sample.tokens_per_second:.1f} tok/s"
            if sample and sample.tokens_per_second is not None
            else "unknown"
        )
        self.query_one("#capacity", Static).update(
            f"LM Studio ceiling       {state.parallel_limit}\n"
            f"Adaptea target          {state.target_concurrency}\n"
            f"Workers running         {running}\n"
            f"Ready tasks             {ready}\n"
            f"Queued predictions      {queued}\n"
            f"Recent speed            {speed}\n"
            f"Merged                  {merged}/{len(state.tasks)}"
        )
        table = self.query_one("#task-table", DataTable)
        table.clear()
        for task_id, task in state.tasks.items():
            table.add_row(
                task_id,
                task.status.value.upper(),
                task.assigned_instance or task.branch or "—",
                str(task.attempts),
                key=task_id,
            )
        events = read_jsonl(
            self.project() / ".adaptea" / "runs" / state.run_id / "controller-decisions.jsonl"
        )
        log = self.query_one("#controller-events", RichLog)
        log.clear()
        if events:
            for row in events[-20:]:
                stamp = str(row.get("timestamp", ""))[11:19]
                log.write(
                    f"[b]{stamp}  Target {row.get('old_target')} → {row.get('new_target')}[/]\n"
                    f"Reason: {row.get('reason')}"
                )
        else:
            log.write("No target change yet. Decisions appear here live.")

    async def execute(self) -> RunState | None:
        try:
            final = await self.adaptea.services.execute_run(
                self.project(), self.state, self.render_state
            )
        except Exception as exc:
            self.adaptea.notify(
                f"Run failed: {self.concise_error(exc)}", severity="error", timeout=8
            )
            if self.is_mounted:
                self.app.push_screen(CompletedRunScreen(self.state, error=self.concise_error(exc)))
            return None
        self.adaptea.notify(f"Run {final.run_id} finished.")
        if self.is_mounted:
            self.app.push_screen(CompletedRunScreen(final))
        return final

    def selected_task_id(self) -> str | None:
        table = self.query_one("#task-table", DataTable)
        if table.row_count == 0:
            return None
        return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)

    @on(DataTable.RowSelected, "#task-table")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(WorkerDetailScreen(self.state, str(event.row_key.value)))

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "detail":
            task_id = self.selected_task_id()
            if task_id:
                self.app.push_screen(WorkerDetailScreen(self.state, task_id))
        elif event.button.id == "leave":
            self.action_leave()
        elif event.button.id == "stop-admitting":
            self.admission_paused = not self.admission_paused
            changed = self.adaptea.services.set_admission(
                self.state.run_id, enabled=not self.admission_paused
            )
            if changed:
                event.button.label = (
                    "Resume Admitting" if self.admission_paused else "Stop Admitting"
                )
                self.notify(
                    "New admissions paused; running workers remain active."
                    if self.admission_paused
                    else "New admissions resumed."
                )
            else:
                self.notify("The run controller is no longer active.", severity="warning")
        elif event.button.id == "abort":
            self.app.push_screen(
                ConfirmScreen(
                    "Abort Entire Run?",
                    "This terminates active coding workers and marks their tasks failed. "
                    "Healthy workers are never stopped by merely leaving this screen.",
                ),
                self._confirm_abort,
            )

    def _confirm_abort(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        if self.adaptea.services.abort_run(self.state.run_id):
            self.notify("Abort requested. Worker processes are shutting down.", severity="warning")
        else:
            self.notify("The run controller is no longer active.", severity="warning")

    def action_leave(self) -> None:
        self.notify("The run continues in the background. Open Runs to monitor it.")
        self.action_back()


class WorkerDetailScreen(AdapteaScreen):
    def __init__(self, state: RunState, task_id: str) -> None:
        super().__init__()
        self.state = state
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        task = self.state.tasks[self.task_id]
        yield Header()
        with Vertical(classes="page"):
            yield Static(f"TASK  {self.task_id}", classes="brand")
            yield Static(
                f"Status: {task.status.value.upper()}\nBranch: {task.branch or '—'}\n"
                f"Worktree: {task.worktree or '—'}\nStarted: {task.started_at or '—'}\n"
                f"Model: {task.assigned_model or 'legacy default'}\n"
                f"Instance: {task.assigned_instance or 'legacy default'}\n"
                f"Tier: {task.assigned_tier or 'single-model'}\n"
                f"Routing: {task.routing_reason or 'legacy single-model routing'}\n"
                f"Dependencies: {', '.join(task.spec.depends_on) or 'none'}\n"
                f"Acceptance: {'; '.join(task.spec.acceptance_criteria) or 'not specified'}",
                classes="card",
            )
            with TabbedContent():
                with TabPane("Summary"):
                    yield Static(task.summary or task.failure or "No summary yet.")
                with TabPane("Live Output"):
                    yield RichLog(id="worker-output", wrap=True)
                with TabPane("Validation"):
                    yield Static(
                        "Not run yet"
                        if task.validation_passed is None
                        else (
                            f"{'Passed' if task.validation_passed else 'Failed'} — "
                            f"{task.validation_detail or 'no detail'}\n"
                            f"Validator: {' '.join(task.validation_command) or 'none detected'}\n"
                            f"Exit code: {task.validation_exit_code}"
                        )
                    )
                with TabPane("Git"):
                    yield Static(
                        f"Branch: {task.branch or '—'}\nWorktree: {task.worktree or '—'}\n"
                        f"Merge state: {task.status.value}"
                    )
        yield Footer()

    def on_mount(self) -> None:
        task = self.state.tasks[self.task_id]
        log = self.query_one("#worker-output", RichLog)
        if task.attempts:
            path = (
                self.project()
                / ".adaptea"
                / "runs"
                / self.state.run_id
                / "tasks"
                / self.task_id
                / f"attempt-{task.attempts}"
                / "stdout.log"
            )
            if path.is_file():
                log.write(path.read_text(encoding="utf-8", errors="replace")[-12000:])
                return
        log.write("Captured OpenCode output will appear here when available.")


class CompletedRunScreen(AdapteaScreen):
    def __init__(self, state: RunState, *, error: str | None = None) -> None:
        super().__init__()
        self.state = state
        self.error = error

    def compose(self) -> ComposeResult:
        merged = sum(task.status == TaskStatus.MERGED for task in self.state.tasks.values())
        passed = sum(task.validation_passed is True for task in self.state.tasks.values())
        failed = len(self.state.tasks) - merged
        retries = sum(max(0, task.attempts - 1) for task in self.state.tasks.values())
        honest_status = (
            "RUN COMPLETE" if failed == 0 and not self.error else "RUN REQUIRES ATTENTION"
        )
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static(honest_status, classes="brand")
            yield Static(
                f"Run                    {self.state.run_id}\n"
                f"Tasks                  {len(self.state.tasks)}\n"
                f"Merged                 {merged}\n"
                f"Validation passed      {passed}/{len(self.state.tasks)}\n"
                f"Retries                {retries}\n"
                f"Scheduler              {self.state.scheduler}\n"
                f"Final integration      {self.state.integration_branch}\n"
                + (f"\nError: {self.error}" if self.error else ""),
                classes="card",
            )
            with Horizontal(classes="actions"):
                yield Button("View Tasks", id="tasks")
                yield Button("View Report", id="report")
                yield Button("Open Repository", id="repository")
                yield Button("Start Another Run", id="new", variant="primary")
        yield Footer()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tasks":
            first = next(iter(self.state.tasks), None)
            if first:
                self.app.push_screen(WorkerDetailScreen(self.state, first))
        elif event.button.id == "report":
            self.app.push_screen(ReportsScreen())
        elif event.button.id == "repository":
            self.notify(str(self.project()))
        elif event.button.id == "new":
            self.app.push_screen(NewRunScreen())


class RunsScreen(AdapteaScreen):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[RunRecord] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("RUN HISTORY", classes="brand")
            yield DataTable(id="runs-table", cursor_type="row")
            yield Static("", id="runs-message")
            with Horizontal(classes="actions"):
                yield Button("Open", id="open", variant="primary")
                yield Button("Resume", id="resume")
                yield Button("View Report", id="report")
                yield Button("Delete Run Metadata", id="delete", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Run ID", "Date", "Repository", "Goal", "Scheduler", "Status", "Tasks")
        self.reload()

    def reload(self) -> None:
        projects = [Path(row.path) for row in self.adaptea.recents.load()]
        if self.adaptea.active_project:
            projects.insert(0, self.adaptea.active_project)
        self.records = list_runs(list(dict.fromkeys(projects)))
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for row in self.records:
            table.add_row(
                row.run_id,
                row.created_at[:19],
                row.repository.name,
                row.goal[:40],
                row.scheduler,
                row.status,
                f"{row.merged}/{row.tasks}",
                key=f"{row.repository}:{row.run_id}",
            )

    def selected(self) -> RunRecord | None:
        table = self.query_one("#runs-table", DataTable)
        if table.row_count == 0:
            return None
        index = table.cursor_row
        return self.records[index] if 0 <= index < len(self.records) else None

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        record = self.selected()
        if record is None:
            return
        if event.button.id == "open":
            state = RunState.model_validate_json(
                (record.run_dir / "state.json").read_text(encoding="utf-8")
            )
            self.adaptea.set_active_project(record.repository)
            self.app.push_screen(CompletedRunScreen(state))
        elif event.button.id == "resume":
            if not record.resumable:
                self.notify("This run is already complete.")
                return
            self.resume(record)
        elif event.button.id == "report":
            self.app.push_screen(ReportsScreen())
        elif event.button.id == "delete":
            self.app.push_screen(
                ConfirmScreen(
                    "Delete Run Metadata?",
                    "Only metadata files are removed. Git branches and worktrees are not deleted.",
                ),
                lambda confirmed: self.delete_metadata(record) if confirmed else None,
            )

    @work(exclusive=True)
    async def resume(self, record: RunRecord) -> None:
        self.adaptea.set_active_project(record.repository)
        try:
            final = await self.adaptea.services.resume_run(record.repository, record.run_id)
        except Exception as exc:
            self.query_one("#runs-message", Static).update(f"[red]{self.concise_error(exc)}[/]")
            return
        self.app.push_screen(CompletedRunScreen(final))

    def delete_metadata(self, record: RunRecord) -> None:
        from adaptea.history import delete_run_metadata

        delete_run_metadata(record)
        self.notify("Run metadata deleted. Branches and worktrees were left untouched.")
        self.reload()


class ReportsScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static("REPORTS & BENCHMARKS", classes="brand")
            yield Markdown(id="report-content")
        yield Footer()

    def on_mount(self) -> None:
        rows = list_runs([self.project()])[:10]
        lines = [
            "| Run | Scheduler | Status | Tasks | Pass rate | Duration |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            duration = f"{row.duration_seconds:.1f}s" if row.duration_seconds is not None else "—"
            pass_rate = row.merged / row.tasks if row.tasks else 0
            lines.append(
                f"| {row.run_id} | {row.scheduler} | {row.status} | {row.tasks} | "
                f"{pass_rate:.0%} | {duration} |"
            )
        lines.append("\nMetrics not measured by a run are intentionally shown as —.")
        self.query_one(Markdown).update("\n".join(lines))


class SettingsScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        config = load_config(self.project())
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static("PROJECT SETTINGS", classes="brand")
            for label, widget_id, value in (
                ("LM Studio URL", "lm-url", config.lmstudio.base_url),
                ("Selected model", "selected-model", config.lmstudio.model or ""),
                ("OpenCode executable", "opencode", config.worker.executable),
                ("Default max agents", "default-max", str(config.worker.max_agents)),
                ("Default scheduler", "default-scheduler", config.worker.default_scheduler),
                ("Telemetry poll interval", "poll", str(config.lmstudio.telemetry_poll_seconds)),
            ):
                yield Label(label, classes="field-label")
                yield Input(value=value, id=widget_id)
            yield Static(
                "Advanced controller thresholds remain in adaptea.toml.", classes="card muted"
            )
            yield Static("", id="settings-message")
            with Horizontal(classes="actions"):
                yield Button("Save Settings", id="save", variant="primary")
                yield Button("Open Config File", id="config-path")
                yield Button("Select Model", id="model")
        yield Footer()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.save()
        elif event.button.id == "config-path":
            self.notify(str(self.project() / "adaptea.toml"))
        elif event.button.id == "model":
            self.app.push_screen(ModelSelectionScreen())

    def save(self) -> None:
        message = self.query_one("#settings-message", Static)
        try:
            maximum = int(self.query_one("#default-max", Input).value)
            scheduler = self.query_one("#default-scheduler", Input).value
            poll = float(self.query_one("#poll", Input).value)
            if maximum < 1 or poll < 0.25:
                raise ValueError("Max agents must be ≥1 and poll interval must be ≥0.25s.")
            if scheduler not in {"adaptive", "fixed", "naive"}:
                raise ValueError("Default scheduler must be adaptive, fixed, or naive.")
            merge_adaptea_config(
                self.project() / "adaptea.toml",
                base_url=self.query_one("#lm-url", Input).value,
                model=self.query_one("#selected-model", Input).value,
                lms_executable=load_config(self.project()).lmstudio.lms_executable,
                opencode_executable=self.query_one("#opencode", Input).value,
                max_agents=maximum,
                scheduler=scheduler,
            )
            _merge_poll_interval(self.project() / "adaptea.toml", poll)
        except (OSError, ValueError) as exc:
            message.update(f"[red]{self.concise_error(exc)}[/]")
            return
        message.update(
            "[green]Settings saved.[/] A model or parallel-limit change requires recalibration."
        )


class SmokeTestScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="page"):
            yield Static("MVP SMOKE TEST", classes="brand")
            yield Static(
                "Runs real environment, direct inference, OpenCode, worktree, worker, validation, "
                "merge, dependency-unlock, and final checks in a temporary fixture.",
                classes="card",
            )
            yield RichLog(id="smoke-log", markup=True, wrap=True)
            with Horizontal(classes="actions"):
                yield Button("Run MVP Smoke Test", id="run", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#run")
    def pressed(self) -> None:
        self.run_smoke()

    @work(exclusive=True)
    async def run_smoke(self) -> None:
        log = self.query_one("#smoke-log", RichLog)
        log.clear()
        log.write("Starting non-destructive smoke test…")

        def progress(step: SmokeStep) -> None:
            marker = "[green]✓[/]" if step.success else "[red]✗[/]"
            log.write(f"{marker} {step.layer} › {step.name}\n   {step.detail}")
            if step.remedy:
                log.write(f"   [yellow]Fix:[/] {step.remedy}")

        result = await run_mvp_smoke_test(self.project(), progress=progress)
        if result.success:
            log.write("\n[bold green]ADAPTEA MVP IS WORKING[/]")
        else:
            log.write("\n[bold red]SMOKE TEST REQUIRES ATTENTION[/]")


class ComparisonScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        config = load_config(self.project())
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static("COMPARE SERIAL VS ADAPTIVE", classes="brand")
            yield Static(
                "A — Serial / C=1 and B — Adaptive start from two isolated worktrees at the "
                "same clean Git commit. B never runs on top of A's changes.",
                classes="card",
            )
            yield Label("Goal", classes="field-label")
            yield TextArea(id="comparison-goal")
            yield Label("Max agents", classes="field-label")
            yield Input(value=str(config.worker.max_agents), type="integer", id="comparison-max")
            yield RichLog(id="comparison-log", markup=True, wrap=True)
            with Horizontal(classes="actions"):
                yield Button("Prepare & Run Comparison", id="run", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#run")
    def pressed(self) -> None:
        self.run_comparison()

    @work(exclusive=True)
    async def run_comparison(self) -> None:
        log = self.query_one("#comparison-log", RichLog)
        log.clear()
        try:
            goal = self.query_one("#comparison-goal", TextArea).text.strip()
            if not goal:
                raise ValueError("Enter a comparison goal.")
            maximum = int(self.query_one("#comparison-max", Input).value)
            log.write("Generating one shared plan…")
            plan = await self.adaptea.services.plan(self.project(), goal)
            log.write("Preparing equivalent clean starting states…")
            result = await execute_comparison(self.project(), plan, maximum)
            summary = comparison_summary(result)
        except Exception as exc:
            log.write(f"[red]{self.concise_error(exc)}[/]")
            return
        serial = cast(dict[str, object], summary["serial"])
        adaptive = cast(dict[str, object], summary["adaptive"])
        serial_pass_rate = float(cast(Any, serial["pass_rate"]))
        adaptive_pass_rate = float(cast(Any, adaptive["pass_rate"]))
        materially_different = abs(serial_pass_rate - adaptive_pass_rate) > 0.01
        log.write(
            "\n[b]SERIAL                         ADAPTIVE[/]\n"
            f"Duration   {float(cast(Any, serial['duration_seconds'])):.1f}s"
            f"             {float(cast(Any, adaptive['duration_seconds'])):.1f}s\n"
            f"Pass rate  {serial_pass_rate:.0%}             "
            f"{adaptive_pass_rate:.0%}\n"
            f"Tasks      {serial['merged']}/{serial['tasks']}               "
            f"{adaptive['merged']}/{adaptive['tasks']}\n"
            f"Retries    {serial['retries']}                 {adaptive['retries']}\n"
            f"Target     1                 {adaptive['final_target']}"
        )
        log.write(
            "\n[b]Adaptive parallelism[/]\n"
            f"Peak workers       {adaptive['peak_workers']}\n"
            f"Average workers    {float(cast(Any, adaptive['average_workers'])):.2f}\n"
            f"Target range       {adaptive['target_min']} → {adaptive['target_max']}"
        )
        if materially_different:
            log.write("\n[yellow]No winner declared because pass rates differ materially.[/]")


class HelpScreen(AdapteaScreen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="page"):
            yield Static("HELP", classes="brand")
            yield Markdown(
                """
### Normal workflow

Setup → Doctor → Select model → Quick calibration → Choose/create project → New run →
Review plan → Start → Watch capacity and workers → Completion report.

### Keys

- `Ctrl+N` — new coding run
- `Ctrl+R` — run history
- `Ctrl+S` — setup
- `C` — calibration (Ctrl+C remains the terminal interrupt key)
- `Esc` — back or close
- `Enter` — activate focused control
- `Q` — quit

Leaving an active-run screen does not terminate healthy workers. Adaptea is non-preemptive.
                """
            )
        yield Footer()


def _merge_poll_interval(path: Path, value: float) -> None:
    """Small focused TOML update that preserves the rest of the user's config."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    header = "[lmstudio]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        lines.extend(([""] if lines else []) + [header, f"telemetry_poll_seconds = {value}"])
    else:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].strip().startswith("[")
            ),
            len(lines),
        )
        match = next(
            (
                index
                for index in range(start + 1, end)
                if lines[index].strip().startswith("telemetry_poll_seconds")
            ),
            None,
        )
        if match is None:
            lines.insert(end, f"telemetry_poll_seconds = {value}")
        else:
            lines[match] = f"telemetry_poll_seconds = {value}"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_tui(root: Path | None = None) -> None:
    AdapteaApp(root).run()
