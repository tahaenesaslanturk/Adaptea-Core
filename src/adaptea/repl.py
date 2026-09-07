from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from adaptea.calibration.runner import CalibrationRunner
from adaptea.cli_ui import (
    ASCII_LOGO,
    COMPACT_WORDMARK,
    TAGLINE,
    create_panel,
    create_table,
    format_path,
)
from adaptea.comparison import comparison_summary, execute_comparison
from adaptea.config import load_config
from adaptea.diagnostics.doctor import run_doctor
from adaptea.history import list_runs
from adaptea.inference import create_inference_backend
from adaptea.models import RunState, TaskStatus, TelemetrySample
from adaptea.projects import RecentProjects, inspect_project, user_state_directory
from adaptea.runtime.controller import Orchestrator, create_run, prepare_resume
from adaptea.runtime.state import StateStore, latest_run
from adaptea.services import ApplicationServices
from adaptea.setup.ui import run_setup_center
from adaptea.smoke import SmokeStep, run_mvp_smoke_test

COMMANDS: dict[str, str] = {
    "/run": "Plan and execute an adaptive coding task",
    "/doctor": "Diagnose local LLM engine, model, and git setup",
    "/setup": "Interactive configuration and environment setup",
    "/calibrate": "Measure hardware throughput and create capacity profile",
    "/fleet": "Inspect local multi-model fleet, loaded tiers, and slots",
    "/compare": "Benchmark serial vs. adaptive execution on a plan",
    "/smoke": "Run real MVP pipeline smoke test in temp fixture",
    "/project": "Switch active project directory or view recent projects",
    "/history": "View recent coding runs and their statuses",
    "/resume": "Resume an interrupted or partial run",
    "/config": "Inspect active project configuration",
    "/policy": "Show worker command security rules and approvals",
    "/clear": "Clear the terminal screen",
    "/help": "Show available commands and usage guide",
    "/exit": "Exit Adaptea",
}


class SlashCommandCompleter(Completer):
    """Provides auto-completion for slash commands with formatted descriptions."""

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Any:
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        word = text.split(maxsplit=1)[0]
        for cmd, desc in COMMANDS.items():
            if cmd.startswith(word):
                yield Completion(
                    cmd,
                    start_position=-len(word),
                    display=cmd,
                    display_meta=desc,
                )


class InteractiveRepl:
    """Modern keyboard-driven interactive CLI REPL for Adaptea."""

    def __init__(self, root: Path | None = None, console: Console | None = None) -> None:
        self.root = (root or Path.cwd()).expanduser().resolve()
        self.console = console or Console()
        self.recents = RecentProjects()
        self.recents.remember(self.root)
        self.services = ApplicationServices()

        history_dir = user_state_directory()
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = history_dir / "repl_history"

        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(self.history_file)),
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            style=Style.from_dict(
                {
                    "prompt": "#79e0b3 bold",
                    "path": "#38bdf8",
                    "arrow": "#64748b",
                }
            ),
        )

    def _print_welcome(self) -> None:
        """Render the welcome banner and current environment status."""
        self.console.clear()
        self.console.print(ASCII_LOGO)
        self.console.print(f" {COMPACT_WORDMARK} [dim]•[/] {TAGLINE}\n")

        try:
            project_info = asyncio.run(inspect_project(self.root))
            branch = (
                f" [dim](git:[/] [cyan]{project_info.git_branch}[/][dim])[/]"
                if project_info.git_branch
                else ""
            )
        except Exception:
            branch = ""
        self.console.print(f" [dim]Workspace:[/]  [bold cyan]{format_path(self.root)}[/]{branch}")

        try:
            config = load_config(self.root)
            backend = config.inference.backend
            if backend == "ollama":
                engine_name = "Ollama"
                model_name = config.ollama.model or "auto"
                host_info = config.ollama.base_url
            elif backend == "llamacpp":
                engine_name = "llama.cpp"
                model_name = config.llamacpp.model or "auto"
                host_info = config.llamacpp.base_url
            elif backend == "vllm":
                engine_name = "vLLM"
                model_name = config.vllm.model or "auto"
                host_info = config.vllm.base_url
            else:
                engine_name = "LM Studio"
                model_name = config.lmstudio.model or "auto"
                host_info = config.lmstudio.base_url

            self.console.print(
                f" [dim]Engine:[/]     [green]{engine_name}[/] @ {host_info} [dim]({model_name})[/]"
            )
        except Exception:
            self.console.print(" [dim]Engine:[/]     [yellow]Configuration uninitialized[/]")

        self.console.print(
            "\n [dim]Type a task prompt or[/] [bold cyan]/[/][dim] for commands "
            "([/][bold yellow]?[/][dim] for help, [/][bold cyan]/exit[/][dim] to quit).[/]\n"
        )

    def run(self) -> None:
        """Main REPL event loop."""
        self._print_welcome()

        while True:
            try:
                rel_path = self.root.name
                prompt_text = HTML(
                    f"<prompt>adaptea</prompt> <path>{rel_path}</path><arrow>&gt; </arrow>"
                )
                user_input = self.session.prompt(prompt_text).strip()
            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                self.console.print("\n[dim]Goodbye![/]")
                break

            if not user_input:
                continue

            if user_input in {"exit", "quit", ":q"}:
                self.console.print("[dim]Goodbye![/]")
                break

            try:
                self._dispatch(user_input)
            except Exception as exc:
                self.console.print(f"[bold red]✗ Error:[/] {exc}\n")

    def _dispatch(self, line: str) -> None:
        """Parse and route user commands or natural language tasks."""
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in {"/exit", "/quit"}:
                sys.exit(0)
            elif cmd in {"/help", "/?"}:
                self._cmd_help()
            elif cmd == "/clear":
                self._print_welcome()
            elif cmd == "/doctor":
                self._cmd_doctor()
            elif cmd == "/setup":
                self._cmd_setup()
            elif cmd == "/calibrate":
                self._cmd_calibrate(arg)
            elif cmd == "/fleet":
                self._cmd_fleet()
            elif cmd == "/compare":
                self._cmd_compare(arg)
            elif cmd == "/smoke":
                self._cmd_smoke()
            elif cmd in {"/project", "/cd"}:
                self._cmd_project(arg)
            elif cmd in {"/history", "/runs"}:
                self._cmd_history()
            elif cmd == "/resume":
                self._cmd_resume(arg)
            elif cmd in {"/config", "/settings"}:
                self._cmd_config()
            elif cmd == "/policy":
                self._cmd_policy()
            elif cmd == "/run":
                if not arg:
                    self.console.print("[yellow]Usage:[/] /run <coding goal or prompt>")
                    return
                self._cmd_run(arg)
            else:
                self.console.print(
                    f"[red]Unknown command:[/] {cmd}. "
                    "Type [bold cyan]/help[/] to see available commands."
                )
        else:
            self._cmd_run(line)

    def _cmd_help(self) -> None:
        """Display slash command reference table."""
        table = create_table(
            ("Command", {"style": "bold cyan", "width": 18}),
            ("Description", {"style": "default"}),
            title="Available Commands",
        )
        for cmd, desc in COMMANDS.items():
            table.add_row(cmd, desc)
        self.console.print(table)
        self.console.print(
            "[dim]Tip: You can also type any prompt directly without '/' "
            "to start an adaptive run.[/]\n"
        )

    def _cmd_doctor(self) -> None:
        """Run system diagnostics inline."""
        self.console.print("[dim]Running diagnostics…[/]")
        try:
            config = load_config(self.root)
            checks = asyncio.run(run_doctor(self.root, config))
        except Exception as exc:
            self.console.print(f"[bold red]✗ Diagnostics failed:[/] {exc}\n")
            return

        table = create_table(
            ("Status", {"no_wrap": True, "justify": "center"}),
            ("Stage", {"style": "dim", "no_wrap": True}),
            ("Check", {"style": "bold"}),
            ("Detail", {"style": "default"}),
            title="Diagnostic Health Checks",
        )

        pass_count = sum(1 for c in checks if c.level == "PASS")
        warn_count = sum(1 for c in checks if c.level == "WARN")
        fail_count = sum(1 for c in checks if c.level == "FAIL")

        for check in checks:
            status_text = (
                "[bold green]✓ PASS[/]"
                if check.level == "PASS"
                else "[bold yellow]▲ WARN[/]"
                if check.level == "WARN"
                else "[bold red]✗ FAIL[/]"
            )
            detail = format_path(check.detail, self.root)
            table.add_row(status_text, check.stage, check.name, detail)

        self.console.print(table)
        summary_parts = [f"[bold green]● {pass_count} passed[/]"]
        if warn_count:
            summary_parts.append(
                f"[bold yellow]▲ {warn_count} warning{'s' if warn_count != 1 else ''}[/]"
            )
        if fail_count:
            summary_parts.append(f"[bold red]✗ {fail_count} failed[/]")
        self.console.print("   [dim]•[/]   ".join(summary_parts) + "\n")

    def _cmd_setup(self) -> None:
        """Run interactive setup center."""
        try:
            config = load_config(self.root)
            asyncio.run(run_setup_center(self.root, config, automatic=False, console=self.console))
        except Exception as exc:
            self.console.print(f"[bold red]✗ Setup encountered an error:[/] {exc}\n")

    def _cmd_calibrate(self, arg: str) -> None:
        """Run calibration."""
        self.console.print("[dim]Starting calibration…[/]")
        try:
            config = load_config(self.root)
            candidates = [int(x.strip()) for x in arg.split(",")] if arg else None

            async def _exec() -> Path:
                async with create_inference_backend(config, timeout=600) as client:
                    return await CalibrationRunner(
                        self.root,
                        config,
                        client,
                        lambda msg: self.console.print(f"[dim]{msg}[/]"),
                    ).run(candidates, None, quick=False)

            path = asyncio.run(_exec())
            self.console.print(
                f"[bold green]✓ Calibration complete:[/] {format_path(path, self.root)}\n"
            )
        except Exception as exc:
            self.console.print(f"[bold red]✗ Calibration failed:[/] {exc}\n")

    def _cmd_fleet(self) -> None:
        """Show fleet status and inventory."""
        from adaptea.fleet.discovery import discover_fleet

        try:
            config = load_config(self.root)
            inventory = asyncio.run(discover_fleet(self.root, config))
        except Exception as exc:
            self.console.print(f"[bold red]✗ Fleet discovery failed:[/] {exc}\n")
            return

        table = create_table(
            ("Status", {"no_wrap": True}),
            ("Model Key", {"style": "bold cyan"}),
            ("Instance ID", {"style": "default"}),
            ("Tier", {"no_wrap": True}),
            ("Roles", {"style": "default"}),
            ("Slots", {"justify": "right"}),
            title="Fleet Inventory & Models",
        )
        loaded = {i.model_key for i in inventory.instances}
        for m in inventory.downloaded:
            if m.model_key not in loaded:
                table.add_row(
                    "[dim]○ downloaded[/]",
                    m.model_key,
                    "[dim]—[/]",
                    "[dim]—[/]",
                    "[dim]unassigned[/]",
                    "[dim]—[/]",
                )
        for i in inventory.instances:
            tier_style = (
                "[bold cyan]STRONG[/]"
                if i.capability_tier.lower() == "strong"
                else "[bold blue]FAST[/]"
            )
            roles_styled = (
                ", ".join(
                    f"[bold yellow]{r}[/]"
                    if r == "planner"
                    else f"[bold green]{r}[/]"
                    if r == "worker"
                    else f"[bold magenta]{r}[/]"
                    for r in i.roles
                )
                if i.roles
                else "[dim]unassigned[/]"
            )
            table.add_row(
                "[bold green]● loaded[/]",
                i.model_key,
                i.instance_id,
                tier_style,
                roles_styled,
                str(i.parallel_limit or "auto"),
            )
        self.console.print(table)
        self.console.print()

    def _cmd_compare(self, arg: str) -> None:
        """Run comparison benchmark between serial and adaptive execution."""
        goal = arg or self.session.prompt("Goal for comparison benchmark: ").strip()
        if not goal:
            return

        self.console.print(f"[dim]Planning and comparing execution for:[/] {goal}")
        try:
            config = load_config(self.root)
            plan = asyncio.run(self.services.plan(self.root, goal))
            result = asyncio.run(
                execute_comparison(self.root, plan, max_agents=config.worker.max_agents)
            )
            summary = comparison_summary(result)
            self.console.print(
                f"[bold green]✓ Comparison finished![/] Saved to: {result.directory}"
            )
            self.console.print(create_panel(str(summary), title="Comparison Summary"))
            self.console.print()
        except Exception as exc:
            self.console.print(f"[bold red]✗ Comparison failed:[/] {exc}\n")

    def _cmd_smoke(self) -> None:
        """Run MVP pipeline smoke test."""
        self.console.print("[dim]Executing MVP smoke test…[/]")

        def progress(step: SmokeStep) -> None:
            marker = "[bold green]✓[/]" if step.success else "[bold red]✗[/]"
            self.console.print(
                f" {marker} [bold cyan]{step.layer}[/] › [bold]{step.name}[/]: {step.detail}"
            )

        try:
            result = asyncio.run(run_mvp_smoke_test(self.root, progress=progress))
            if result.success:
                self.console.print("[bold green]✓ MVP smoke test passed![/]\n")
            else:
                self.console.print("[bold red]✗ MVP smoke test failed.[/]\n")
        except Exception as exc:
            self.console.print(f"[bold red]✗ Smoke test error:[/] {exc}\n")

    def _cmd_project(self, path_str: str) -> None:
        """Switch active project directory."""
        if not path_str:
            recents = self.recents.load()
            if not recents:
                self.console.print("[dim]No recent projects recorded.[/]")
                return
            table = create_table(
                ("Index", {"justify": "center", "width": 8}),
                ("Project Path", {"style": "bold cyan"}),
                title="Recent Projects",
            )
            for idx, item in enumerate(recents, 1):
                table.add_row(str(idx), item.path)
            self.console.print(table)
            choice = self.session.prompt("Select project index or enter directory path: ").strip()
            if not choice:
                return
            if choice.isdigit() and 1 <= int(choice) <= len(recents):
                target = Path(recents[int(choice) - 1].path)
            else:
                target = Path(choice).expanduser().resolve()
        else:
            target = Path(path_str).expanduser().resolve()

        if not target.is_dir():
            self.console.print(f"[bold red]✗ Directory not found:[/] {target}\n")
            return

        self.root = target
        self.recents.remember(self.root)
        self.console.print(f"[bold green]✓ Switched project to:[/] {format_path(self.root)}\n")

    def _cmd_history(self) -> None:
        """Display recent runs in a rich table."""
        runs = list_runs([self.root])[:10]
        if not runs:
            self.console.print("[dim]No runs found for this project.[/]\n")
            return

        table = create_table(
            ("Run ID", {"style": "bold cyan"}),
            ("Status", {"no_wrap": True}),
            ("Scheduler", {"style": "dim"}),
            ("Tasks", {"justify": "center"}),
            ("Duration", {"justify": "right"}),
            ("Goal", {"style": "default"}),
            title="Recent Runs",
        )
        for r in runs:
            status_badge = (
                "[bold green]COMPLETED[/]"
                if r.status == "COMPLETED"
                else "[bold red]FAILED[/]"
                if r.status == "FAILED"
                else f"[yellow]{r.status}[/]"
            )
            duration_str = f"{r.duration_seconds:.1f}s" if r.duration_seconds else "—"
            table.add_row(
                r.run_id,
                status_badge,
                r.scheduler,
                f"{r.merged}/{r.tasks}",
                duration_str,
                r.goal[:50],
            )
        self.console.print(table)
        self.console.print()

    def _cmd_resume(self, run_id: str) -> None:
        """Resume an interrupted run."""
        directory = self.root / ".adaptea" / "runs" / run_id if run_id else latest_run(self.root)
        if not directory or not (directory / "state.json").exists():
            self.console.print("[bold red]✗ No resumable run found.[/]\n")
            return

        try:
            store = StateStore(directory)
            state = prepare_resume(store.load(), directory)
            store.save(state)
            config = load_config(self.root)
            self.console.print(f"[dim]Resuming run {state.run_id}…[/]")
            final = asyncio.run(Orchestrator(self.root, config, state).run())
            status_desc = "completed" if final.complete else "partial"
            self.console.print(f"[bold green]✓ Run resumed ({status_desc}).[/]\n")
        except Exception as exc:
            self.console.print(f"[bold red]✗ Resume failed:[/] {exc}\n")

    def _cmd_config(self) -> None:
        """Display active configuration."""
        try:
            config = load_config(self.root)
            backend = config.inference.backend
            if backend == "ollama":
                endpoint, model = config.ollama.base_url, config.ollama.model
            elif backend == "llamacpp":
                endpoint, model = config.llamacpp.base_url, config.llamacpp.model
            elif backend == "vllm":
                endpoint, model = config.vllm.base_url, config.vllm.model
            else:
                endpoint, model = config.lmstudio.base_url, config.lmstudio.model
            panel_text = (
                f"[bold]Inference Backend:[/] {backend} @ {endpoint}\n"
                f"[bold]Model:[/] {model or 'auto'}\n"
                f"[bold]Default Scheduler:[/] {config.worker.default_scheduler}\n"
                f"[bold]Max Agents:[/] {config.worker.max_agents}\n"
                f"[bold]Fleet Mode:[/] {'Enabled' if config.fleet.enabled else 'Disabled'}\n"
                f"[bold]Worker Timeout:[/] {config.worker.timeout_seconds}s"
            )
            self.console.print(
                create_panel(
                    panel_text,
                    title=f"Configuration • {format_path(self.root / 'adaptea.toml', self.root)}",
                )
            )
            self.console.print()
        except Exception as exc:
            self.console.print(f"[bold red]✗ Could not read configuration:[/] {exc}\n")

    def _cmd_policy(self) -> None:
        """Display worker command security policies."""
        from adaptea.security.commands import policy_document

        try:
            policy = policy_document(load_config(self.root).worker.command_security)
            categories = policy["categories"]
            table = create_table(
                ("Category", {"style": "bold"}),
                ("Action", {"style": "cyan"}),
                ("Examples / approvals", {"style": "default"}),
                title="Command Security Policy",
            )
            safe = categories["safe_default"]["patterns"]
            approved = categories["approval_required"]["approved_patterns"]
            blocked = categories["blocked"]["patterns"]
            table.add_row("[bold green]Safe default[/]", "allow", ", ".join(safe[:6]) + ", …")
            table.add_row(
                "[bold yellow]Approval required[/]",
                "deny unless approved",
                ", ".join(approved) if approved else "none",
            )
            table.add_row("[bold red]Blocked[/]", "always deny", ", ".join(blocked[:6]) + ", …")
            self.console.print(table)
            self.console.print()
        except Exception as exc:
            self.console.print(f"[bold red]✗ Policy inspect error:[/] {exc}\n")

    def _cmd_run(self, goal: str) -> None:
        """Execute a coding task with the adaptive orchestrator."""
        self.console.print(f"\n[bold cyan]⚡ ADAPTEA RUN[/] [dim]›[/] [bold white]{goal}[/]")
        try:
            config = load_config(self.root)
        except Exception as exc:
            self.console.print(f"[bold red]✗ Configuration error:[/] {exc}\n")
            return

        self.console.print("[dim]Generating task plan with admission control…[/]")
        try:
            plan = asyncio.run(self.services.plan(self.root, goal))
            state = asyncio.run(
                create_run(
                    self.root,
                    config,
                    plan,
                    config.worker.default_scheduler,
                    max_agents=None,
                    concurrency=None,
                )
            )
        except Exception as exc:
            self.console.print(f"[bold red]✗ Planning failed:[/] {exc}\n")
            return

        plan_table = create_table(
            ("Task ID", {"style": "cyan", "width": 14}),
            ("Title", {"style": "bold"}),
            ("Dependencies", {"style": "dim"}),
            title=f"Plan • {len(plan.tasks)} tasks",
        )
        for task in plan.tasks:
            deps = ", ".join(task.depends_on) if task.depends_on else "[dim]none[/]"
            plan_table.add_row(task.id, task.title, deps)
        self.console.print(plan_table)
        self.console.print(f"[dim]Starting {state.scheduler} execution across local workers…[/]\n")

        def status_panel(curr: RunState, sample: TelemetrySample | None, running: int) -> Panel:
            merged = sum(t.status == TaskStatus.MERGED for t in curr.tasks.values())
            lines = [
                f"[bold cyan]⚡ {curr.run_id}[/]  [dim]•[/]  "
                f"Slots: [cyan]{running}/{curr.target_concurrency}[/] active  [dim]•[/]  "
                f"Merged: [bold green]{merged}/{len(curr.tasks)}[/]"
            ]
            if sample and sample.tokens_per_second:
                lines.append(
                    f"[dim]Speed:[/] {sample.tokens_per_second:.1f} tok/s  "
                    f"[dim]TTFT:[/] {sample.ttft_seconds or 0:.2f}s"
                )
            return Panel("\n".join(lines), box=box.ROUNDED, border_style="#3b82f6")

        async def _execute() -> RunState:
            with Live(console=self.console, refresh_per_second=4) as live:
                controller = Orchestrator(
                    self.root,
                    config,
                    state,
                    status_callback=lambda c, s, r: live.update(status_panel(c, s, r)),
                )
                return await controller.run()

        try:
            final_state = asyncio.run(_execute())
        except Exception as exc:
            self.console.print(f"[bold red]✗ Run failed during execution:[/] {exc}\n")
            return

        summary_table = create_table(
            ("Task", {"style": "bold cyan", "width": 14}),
            ("Status", {"width": 14}),
            ("Attempts", {"justify": "center", "width": 10}),
            ("Detail", {"style": "default"}),
            title=f"Run Summary • {final_state.run_id}",
        )
        for task_id, task_runtime in final_state.tasks.items():
            status_badge = (
                "[bold green]✓ MERGED[/]"
                if task_runtime.status == TaskStatus.MERGED
                else "[bold red]✗ FAILED[/]"
                if task_runtime.status == TaskStatus.FAILED
                else f"[yellow]{task_runtime.status.value.upper()}[/]"
            )
            detail = task_runtime.failure or task_runtime.summary or ""
            summary_table.add_row(task_id, status_badge, str(task_runtime.attempts), detail[:80])

        self.console.print(summary_table)
        all_merged = all(t.status == TaskStatus.MERGED for t in final_state.tasks.values())
        if all_merged:
            self.console.print(
                f"[bold green]✓ All {len(final_state.tasks)} tasks "
                "completed and merged successfully![/]\n"
            )
        else:
            self.console.print(
                "[bold yellow]▲ Some tasks require attention or manual resolution.[/]\n"
            )


def start_repl(root: Path | None = None) -> None:
    """Entry point to launch the interactive Adaptea REPL."""
    repl = InteractiveRepl(root)
    repl.run()
