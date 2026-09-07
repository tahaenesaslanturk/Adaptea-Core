from __future__ import annotations

import asyncio
import json
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from adaptea import __version__
from adaptea.benchmark import BenchmarkRunner, BenchmarkSpec
from adaptea.calibration.runner import CalibrationRunner
from adaptea.cli_ui import create_panel, create_table, format_path, print_banner
from adaptea.config import Config, load_config
from adaptea.diagnostics.doctor import run_doctor
from adaptea.inference import create_inference_backend
from adaptea.models import Plan, RunState, TaskStatus, TelemetrySample, utc_now
from adaptea.reporting.report import (
    benchmark_summary,
    read_report,
    write_benchmark_csv,
    write_benchmark_html,
)
from adaptea.runtime.controller import Orchestrator, create_run, prepare_resume
from adaptea.runtime.state import StateStore, latest_run
from adaptea.services import ApplicationServices
from adaptea.setup.manager import create_setup_logger
from adaptea.setup.ui import run_setup_center

app = typer.Typer(
    name="adaptea",
    help="Adaptive admission control for local OpenCode workers served by LM Studio & Ollama.",
    no_args_is_help=False,
    invoke_without_command=True,
)
fleet_app = typer.Typer(help="Discover, configure, inspect, and calibrate the local model fleet.")
app.add_typer(fleet_app, name="fleet")
console = Console()


def _root() -> Path:
    return Path.cwd().resolve()


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the version and exit.", is_eager=True),
    ] = False,
    dashboard: Annotated[
        bool,
        typer.Option("--dashboard", "--tui", help="Launch the full-screen widget TUI dashboard."),
    ] = False,
) -> None:
    if version:
        console.print(f"adaptea {__version__}")
        raise typer.Exit()
    if dashboard:
        from adaptea.tui import run_tui

        run_tui(_root())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from adaptea.repl import start_repl

        start_repl(_root())


@app.command()
def setup(
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help=(
                "Perform safe setup actions automatically; installs and model downloads "
                "still confirm."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show command output in addition to the setup log.")
    ] = False,
) -> None:
    """Open the interactive Adaptea Setup Center."""
    root = _root()
    try:
        config = load_config(root)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        logger, log_path = create_setup_logger(root, verbose)
        logger.exception("could not load adaptea.toml")
        console.print(
            create_panel(
                "[bold red]What failed:[/] Adaptea could not read the existing adaptea.toml.\n"
                f"[bold red]Why:[/] {exc}\n"
                "[bold cyan]What Adaptea tried:[/] parsed the nearest project configuration.\n"
                "[bold green]Next:[/] correct the reported TOML value, then re-run adaptea setup.\n\n"
                f"[dim]Log:[/] {log_path}",
                title="Adaptea Setup Error",
                border_style="#ef4444",
            )
        )
        raise typer.Exit(1) from exc
    ready = _run(run_setup_center(root, config, automatic=auto, verbose=verbose, console=console))
    if ready is not True:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Inspect host tools, LM Studio APIs, model configuration, and capacity state."""
    root = _root()
    print_banner(console, root, compact=True, command_name="doctor")
    checks = _run(run_doctor(root, load_config(root)))
    assert isinstance(checks, list)

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
        if check.level == "PASS":
            status_text = "[bold green]✓ PASS[/]"
        elif check.level == "WARN":
            status_text = "[bold yellow]▲ WARN[/]"
        else:
            status_text = "[bold red]✗ FAIL[/]"

        detail = format_path(check.detail, root)
        table.add_row(status_text, check.stage, check.name, detail)

    console.print(table)

    summary_parts = [f"[bold green]● {pass_count} passed[/]"]
    if warn_count:
        summary_parts.append(
            f"[bold yellow]▲ {warn_count} warning{'s' if warn_count != 1 else ''}[/]"
        )
    if fail_count:
        summary_parts.append(f"[bold red]✗ {fail_count} failed[/]")

    summary_str = "   [dim]•[/]   ".join(summary_parts)
    border = "#22c55e" if fail_count == 0 else "#ef4444"
    console.print(create_panel(summary_str, border_style=border, padding=(0, 2)))

    if fail_count > 0:
        failures = [c for c in checks if c.level == "FAIL" and c.remedy]
        if failures:
            console.print("\n[bold red]Suggested Actions:[/]")
            for f in failures:
                console.print(f"  • [bold]{f.name}[/]: {f.remedy}")
        raise typer.Exit(1)


@app.command()
def calibrate(
    concurrency: Annotated[
        str | None, typer.Option(help="Comma-separated candidates, e.g. 1,2,3,4,6,8.")
    ] = None,
    repetitions: Annotated[int | None, typer.Option(min=1)] = None,
    quick: Annotated[bool, typer.Option(help="Short development calibration.")] = False,
) -> None:
    """Measure useful local LM Studio concurrency and write a starting profile."""
    root = _root()
    print_banner(console, root, compact=True, command_name="calibrate")
    config = load_config(root)
    try:
        candidates = [int(item.strip()) for item in concurrency.split(",")] if concurrency else None
    except ValueError as exc:
        raise typer.BadParameter("concurrency must be comma-separated positive integers") from exc

    async def execute() -> Path:
        async with create_inference_backend(config, timeout=600) as client:
            return await CalibrationRunner(
                root, config, client, lambda message: console.print(f"[dim]{message}[/]")
            ).run(candidates, repetitions, quick=quick)

    try:
        path = cast(Path, _run(execute()))
    except Exception as exc:
        console.print(f"[bold red]✗ Calibration failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]✓ Calibration complete:[/] {format_path(path, root)}")
    console.print(f"Capacity profile: {format_path(root / '.adaptea' / 'capacity.json', root)}")


async def _make_plan(root: Path, config: Config, goal: str) -> Plan:
    from adaptea.services import ApplicationServices

    del config
    return await ApplicationServices().plan(root, goal)


@app.command("repl")
def repl_command() -> None:
    """Launch the interactive Adaptea command REPL."""
    from adaptea.repl import start_repl

    start_repl(_root())


@app.command("tui")
@app.command("dashboard")
def tui_command() -> None:
    """Launch the legacy full-screen widget dashboard."""
    from adaptea.tui import run_tui

    run_tui(_root())


@app.command("smoke-test")
def smoke_test_command(
    keep_fixture: Annotated[
        bool,
        typer.Option(help="Keep the temporary fixture and print its path for inspection."),
    ] = False,
) -> None:
    """Exercise the real Adaptea MVP pipeline in a deterministic temporary repository."""
    from adaptea.smoke import SmokeStep, SmokeTestResult, run_mvp_smoke_test

    root = _root()
    print_banner(console, root, compact=True, command_name="smoke-test")

    def progress(step: SmokeStep) -> None:
        marker = "[bold green]✓[/]" if step.success else "[bold red]✗[/]"
        console.print(f" {marker} [bold cyan]{step.layer}[/] › [bold]{step.name}[/]: {step.detail}")
        if step.remedy:
            console.print(f"    [yellow]Fix:[/] {step.remedy}")

    result = cast(
        SmokeTestResult,
        _run(run_mvp_smoke_test(root, progress=progress, keep_fixture=keep_fixture)),
    )
    if result.artifact_directory:
        console.print(f"[dim]Fixture directory:[/] {result.artifact_directory}")
    if not result.success:
        console.print(
            create_panel(
                "[bold red]ADAPTEA MVP SMOKE TEST REQUIRES ATTENTION[/]", border_style="#ef4444"
            )
        )
        raise typer.Exit(1)
    console.print(create_panel("[bold green]ADAPTEA MVP IS WORKING[/]", border_style="#22c55e"))


@app.command("command-policy")
def command_policy_command() -> None:
    """Show coding-worker command categories and explicit project approvals."""
    from adaptea.security.commands import policy_document

    root = _root()
    print_banner(console, root, compact=True, command_name="command-policy")
    policy = policy_document(load_config(root).worker.command_security)
    categories = cast(dict[str, dict[str, object]], policy["categories"])
    table = create_table(
        ("Category", {"style": "bold"}),
        ("Effective action", {"style": "cyan"}),
        ("Examples / approvals", {"style": "default"}),
        title="Worker Command Security Policy",
    )
    safe = cast(list[str], categories["safe_default"]["patterns"])
    approved = cast(list[str], categories["approval_required"]["approved_patterns"])
    blocked = cast(list[str], categories["blocked"]["patterns"])
    table.add_row("[bold green]Safe default[/]", "allow", ", ".join(safe[:6]) + ", …")
    table.add_row(
        "[bold yellow]Approval required[/]",
        "deny unless explicitly approved",
        ", ".join(approved) if approved else "no project approvals",
    )
    table.add_row("[bold red]Blocked[/]", "always deny", ", ".join(blocked[:6]) + ", …")
    console.print(table)
    console.print(
        "\n[dim]Approve an exact scoped pattern with [/][bold cyan]adaptea approve-command[/]"
        "[dim], or by hand under [/][bold cyan][worker.command_security] approved_commands[/]"
        "[dim] in adaptea.toml.[/]"
    )


@app.command("approve-command")
def approve_command_command(
    command: str = typer.Argument(..., help="The exact command to allow, e.g. 'npm install zod'"),
) -> None:
    """Approve one command for this project's coding workers.

    The desktop asks this question for you while a run is in flight. This is the same
    answer for a terminal run, where there is nobody to ask.
    """
    from adaptea.security.approvals import remember_approval
    from adaptea.security.commands import decide_command

    root = _root()
    print_banner(console, root, compact=True, command_name="approve-command")
    decision = decide_command(command, load_config(root).worker.command_security)
    if decision.category == "blocked":
        console.print(
            f"[bold red]Refused:[/] [bold]{decision.command}[/] matches the blocked pattern "
            f"[bold]{decision.matched_pattern}[/]. Blocked commands cannot be approved."
        )
        raise typer.Exit(code=1)
    if decision.action == "allow" and not decision.explicit_user_approval:
        console.print(
            f"[bold green]Already allowed:[/] [bold]{decision.command}[/] is a safe default; "
            "no approval is needed."
        )
        return
    written = remember_approval(root, command)
    console.print(
        f"[bold green]Approved:[/] [bold]{decision.command}[/]"
        + (f"\n[dim]Recorded in {written}[/]" if written else "")
    )


@fleet_app.command("list")
def fleet_list() -> None:
    """List downloaded models and first-class loaded model instances."""
    from adaptea.fleet.discovery import discover_fleet
    from adaptea.fleet.models import FleetInventory

    root = _root()
    print_banner(console, root, compact=True, command_name="fleet list")
    try:
        inventory = cast(FleetInventory, _run(discover_fleet(root, load_config(root))))
    except Exception as exc:
        console.print(f"[bold red]✗ Fleet discovery failed:[/] {exc}")
        raise typer.Exit(1) from exc

    table = create_table(
        ("Status", {"style": "default", "no_wrap": True}),
        ("Model Key", {"style": "bold cyan"}),
        ("Instance ID", {"style": "default"}),
        ("Tier", {"style": "default", "no_wrap": True}),
        ("Roles", {"style": "default"}),
        ("Slots", {"justify": "right", "no_wrap": True}),
        title="Fleet Models & Loaded Instances",
    )
    loaded_models = {instance.model_key for instance in inventory.instances}
    for model in inventory.downloaded:
        if model.model_key not in loaded_models:
            table.add_row(
                "[dim]○ downloaded[/]",
                model.model_key,
                "[dim]—[/]",
                "[dim]—[/]",
                "[dim]unassigned[/]",
                "[dim]—[/]",
            )
    for instance in inventory.instances:
        tier_style = (
            "[bold cyan]STRONG[/]"
            if instance.capability_tier.lower() == "strong"
            else "[bold blue]FAST[/]"
        )
        roles_styled = (
            ", ".join(
                f"[bold yellow]{r}[/]"
                if r == "planner"
                else f"[bold green]{r}[/]"
                if r == "worker"
                else f"[bold magenta]{r}[/]"
                for r in instance.roles
            )
            if instance.roles
            else "[dim]unassigned[/]"
        )
        table.add_row(
            "[bold green]● loaded[/]",
            instance.model_key,
            instance.instance_id,
            tier_style,
            roles_styled,
            f"[bold white]{instance.parallel_limit or 'auto'}[/]",
        )
    console.print(table)


@fleet_app.command("status")
def fleet_status() -> None:
    """Show configured pools, loaded instances, and calibration freshness."""
    from adaptea.fleet.calibration import profile_is_stale
    from adaptea.fleet.discovery import discover_fleet
    from adaptea.fleet.models import FleetInventory

    root = _root()
    print_banner(console, root, compact=True, command_name="fleet status")
    config = load_config(root)
    try:
        inventory = cast(FleetInventory, _run(discover_fleet(root, config)))
    except Exception as exc:
        console.print(f"[bold red]✗ Fleet discovery failed:[/] {exc}")
        raise typer.Exit(1) from exc

    status_lines = []
    status_lines.append(
        f"[bold]Fleet Mode:[/] {'[bold green]Enabled (Multi-Model Fleet)[/]' if config.fleet.enabled else '[dim]Disabled (Legacy single-model mode)[/]'}"
    )
    planner = next((item for item in inventory.instances if "planner" in item.roles), None)
    reviewer = next((item for item in inventory.instances if "reviewer" in item.roles), planner)
    status_lines.append(
        f"[bold]Planner:[/] {f'[cyan]{planner.instance_id}[/]' if planner else '[yellow]not configured/loaded[/]'}"
    )
    rev_str = f"[cyan]{reviewer.instance_id}[/]" if reviewer else "[yellow]not configured/loaded[/]"
    if reviewer is planner and reviewer is not None:
        rev_str += " [dim](planner fallback)[/]"
    status_lines.append(f"[bold]Reviewer:[/] {rev_str}")

    profile_path = root / ".adaptea" / "fleet.json"
    if profile_path.is_file():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            stale = profile_is_stale(profile, inventory)
        except (OSError, ValueError):
            stale = True
        status_lines.append(
            f"[bold]Fleet Profile:[/] {'[bold yellow]stale[/]' if stale else '[bold green]calibrated/current[/]'} [dim]({format_path(profile_path, root)})[/]"
        )
    else:
        status_lines.append(
            "[bold]Fleet Profile:[/] [yellow]not calibrated[/] [dim](run 'adaptea fleet calibrate')[/]"
        )

    console.print(
        create_panel("\n".join(status_lines), title="Fleet Overview", border_style="#334155")
    )

    for tier in ("FAST", "STRONG"):
        tier_table = create_table(
            ("Instance", {"style": "cyan"}),
            ("Model", {"style": "bold"}),
            ("Active Workers", {"justify": "right"}),
            ("Target Slots", {"justify": "right"}),
            title=f"{tier} Worker Pool",
        )
        rows = [item for item in inventory.instances if item.capability_tier == tier.lower()]
        if rows:
            for item in rows:
                tier_table.add_row(
                    item.instance_id,
                    item.model_key,
                    str(item.running_workers),
                    str(item.effective_limit),
                )
            console.print(tier_table)
        else:
            console.print(f"[dim]  No {tier.lower()} instances configured/loaded[/]\n")


@fleet_app.command("configure")
def fleet_configure(
    mode: Annotated[
        str | None, typer.Option(help="simple or advanced; prompts when omitted.")
    ] = None,
    planner: Annotated[str | None, typer.Option(help="Planner model key.")] = None,
    reviewer: Annotated[str | None, typer.Option(help="Reviewer model key.")] = None,
    strong: Annotated[str | None, typer.Option(help="Strong worker model key.")] = None,
    fast: Annotated[str | None, typer.Option(help="Optional fast worker model key.")] = None,
    instances: Annotated[str, typer.Option(help="auto or an explicit positive count.")] = "auto",
) -> None:
    """Assign explicit user-defined model tiers and roles without ranking model sizes."""
    from adaptea.config import FleetConfig, FleetModelConfig
    from adaptea.fleet.discovery import discover_fleet
    from adaptea.fleet.models import FleetInventory
    from adaptea.setup.configuration import merge_fleet_config

    root = _root()
    print_banner(console, root, compact=True, command_name="fleet configure")
    config = load_config(root)
    try:
        inventory = cast(FleetInventory, _run(discover_fleet(root, config)))
    except Exception as exc:
        console.print(f"[bold red]Could not discover local models:[/] {exc}")
        raise typer.Exit(1) from exc
    keys = [model.model_key for model in inventory.downloaded]
    if not keys:
        console.print("[bold red]No downloaded LM Studio LLMs were discovered.[/]")
        raise typer.Exit(1)
    selected_mode = mode or typer.prompt("Mode [simple/advanced]", default="simple")
    if selected_mode not in {"simple", "advanced"}:
        raise typer.BadParameter("mode must be simple or advanced")
    count: int | Literal["auto"]
    if instances == "auto":
        count = "auto"
    else:
        try:
            count = int(instances)
        except ValueError as exc:
            raise typer.BadParameter("instances must be auto or a positive integer") from exc
        if count < 1:
            raise typer.BadParameter("instances must be positive")

    def choose(label: str, current: str | None) -> str:
        if current:
            if current not in keys:
                raise typer.BadParameter(f"{current} is not a discovered model key")
            return current
        for index, key in enumerate(keys, 1):
            console.print(f" {index}. [cyan]{key}[/]")
        selected = cast(int, typer.prompt(label, type=int))
        if selected < 1 or selected > len(keys):
            raise typer.BadParameter("model selection is out of range")
        return keys[selected - 1]

    if selected_mode == "simple":
        model = choose("Model for planner, reviewer, and workers", planner or reviewer or strong)
        models = [
            FleetModelConfig(
                name="single",
                model=model,
                tier="strong",
                roles=["planner", "worker", "reviewer"],
                instances=count,
            )
        ]
    else:
        planner_model = choose("Planner model", planner)
        reviewer_model = choose("Reviewer model", reviewer)
        strong_model = choose("Strong worker model", strong or planner_model)
        fast_model = choose("Fast worker model", fast) if fast or len(keys) > 1 else None
        models = []
        if planner_model == strong_model:
            models.append(
                FleetModelConfig(
                    name="strong",
                    model=strong_model,
                    tier="strong",
                    roles=["planner", "worker"],
                    instances=count,
                )
            )
        else:
            models.extend(
                [
                    FleetModelConfig(
                        name="planner",
                        model=planner_model,
                        tier="strong",
                        roles=["planner"],
                        instances=1,
                    ),
                    FleetModelConfig(
                        name="strong",
                        model=strong_model,
                        tier="strong",
                        roles=["worker"],
                        instances=count,
                    ),
                ]
            )
        if fast_model and fast_model != strong_model:
            models.append(
                FleetModelConfig(
                    name="fast",
                    model=fast_model,
                    tier="fast",
                    roles=["worker"],
                    instances=count,
                )
            )
        reviewer_entry = next((item for item in models if item.model == reviewer_model), None)
        if reviewer_entry:
            if "reviewer" not in reviewer_entry.roles:
                reviewer_entry.roles.append("reviewer")
        else:
            models.append(
                FleetModelConfig(
                    name="reviewer",
                    model=reviewer_model,
                    tier="strong",
                    roles=["reviewer"],
                    instances=1,
                )
            )
    fleet = FleetConfig(
        enabled=True,
        topology="auto" if count == "auto" else "explicit",
        max_loaded_instances=config.fleet.max_loaded_instances,
        models=models,
        routing=config.fleet.routing,
    )
    backup = merge_fleet_config(root / "adaptea.toml", fleet)
    console.print(f"[bold green]✓ Fleet configured:[/] {format_path(root / 'adaptea.toml', root)}")
    if backup:
        console.print(f"[dim]Backup:[/] {format_path(backup, root)}")
    console.print("[dim]Model tiers came from your choices, not parameter-count inference.[/]")


@fleet_app.command("calibrate")
def fleet_calibrate(
    repetitions: Annotated[int, typer.Option(min=1, max=10)] = 3,
    agent_validation: Annotated[
        bool,
        typer.Option(
            "--agent-validation/--direct-only",
            help=(
                "Validate the best direct candidates through the real deterministic agent fixture."
            ),
        ),
    ] = True,
) -> None:
    """Measure bounded loaded topologies and write .adaptea/fleet.json."""
    root = _root()
    print_banner(console, root, compact=True, command_name="fleet calibrate")
    config = load_config(root)
    if not config.fleet.enabled:
        console.print("[bold red]Fleet is not configured. Run adaptea fleet configure first.[/]")
        raise typer.Exit(1)

    try:
        directory = cast(
            Path,
            _run(
                ApplicationServices().calibrate_fleet(
                    root,
                    repetitions=repetitions,
                    agent_validation=agent_validation,
                )
            ),
        )
    except Exception as exc:
        console.print(f"[bold red]✗ Fleet calibration failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]✓ Fleet calibration complete:[/] {format_path(directory, root)}")
    console.print(f"Profile: {format_path(root / '.adaptea' / 'fleet.json', root)}")


@app.command("plan")
def plan_command(goal: Annotated[str, typer.Argument(help="Large development goal.")]) -> None:
    """Ask OpenCode (using LM Studio) for a validated dependency plan."""
    root = _root()
    print_banner(console, root, compact=True, command_name="plan")
    config = load_config(root)
    try:
        result = _run(_make_plan(root, config, goal))
        assert isinstance(result, Plan)
    except Exception as exc:
        console.print(f"[bold red]✗ Planning failed:[/] {exc}")
        raise typer.Exit(1) from exc
    plan_id = f"plan-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
    directory = root / ".adaptea" / "runs" / plan_id
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "plan.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(result.model_dump_json(indent=2))
    console.print(f"[bold green]✓ Saved plan:[/] {format_path(path, root)}")


@app.command("run")
def run_command(
    goal: Annotated[str | None, typer.Argument(help="Large development goal.")] = None,
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Use an edited plan JSON file.")
    ] = None,
    scheduler: Annotated[
        str | None,
        typer.Option(help="adaptive, fixed, or naive; defaults to adaptea.toml."),
    ] = None,
    max_agents: Annotated[int | None, typer.Option("--max-agents", min=1)] = None,
    concurrency: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Plan and execute workers with admission control, validation, and serialized merge."""
    root = _root()
    print_banner(console, root, compact=True, command_name="run")
    config = load_config(root)
    scheduler = scheduler or config.worker.default_scheduler
    if scheduler not in {"adaptive", "fixed", "naive"}:
        raise typer.BadParameter("scheduler must be adaptive, fixed, or naive")
    if (goal is None) == (plan is None):
        raise typer.BadParameter("provide exactly one of GOAL or --plan")
    if scheduler == "fixed" and concurrency is None:
        raise typer.BadParameter("--concurrency is required with fixed scheduler")
    try:
        if plan:
            task_plan = Plan.model_validate_json(plan.read_text(encoding="utf-8"))
        else:
            assert goal is not None
            task_plan = cast(Plan, _run(_make_plan(root, config, goal)))
        state = _run(
            create_run(
                root,
                config,
                task_plan,
                scheduler,  # type: ignore[arg-type]
                max_agents,
                concurrency,
            )
        )
        assert isinstance(state, RunState)
        console.print(
            f"[bold cyan]⚡ ADAPTEA RUN[/] [bold]{state.run_id}[/]  "
            f"[dim]scheduler=[/][cyan]{scheduler}[/]  "
            f"[dim]target slots=[/][cyan]{state.target_concurrency}[/]"
        )

        async def execute() -> RunState:
            with Live(console=console, refresh_per_second=4) as live:
                controller = Orchestrator(
                    root,
                    config,
                    state,
                    status_callback=lambda current, sample, running: live.update(
                        _status_panel(current, sample, running)
                    ),
                )
                return await controller.run()

        final = _run(execute())
        assert isinstance(final, RunState)
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        console.print(f"[bold red]✗ Run failed:[/] {exc}")
        raise typer.Exit(1) from exc
    _print_state(final, root)
    if any(task.status != TaskStatus.MERGED for task in final.tasks.values()):
        raise typer.Exit(1)


@app.command()
def status(run_id: Annotated[str | None, typer.Argument()] = None) -> None:
    """Show persisted status for a run (latest when omitted)."""
    root = _root()
    print_banner(console, root, compact=True, command_name="status")
    directory = root / ".adaptea" / "runs" / run_id if run_id else latest_run(root)
    if not directory or not (directory / "state.json").exists():
        console.print("[bold red]✗ Run not found.[/]")
        raise typer.Exit(1)
    _print_state(StateStore(directory).load(), root)


@app.command()
def resume(run_id: Annotated[str, typer.Argument()]) -> None:
    """Safely resume an interrupted run without rerunning merged tasks."""
    root = _root()
    print_banner(console, root, compact=True, command_name="resume")
    directory = root / ".adaptea" / "runs" / run_id
    store = StateStore(directory)
    if not store.path.exists():
        console.print("[bold red]✗ Run not found.[/]")
        raise typer.Exit(1)
    state = prepare_resume(store.load(), directory)
    store.save(state)
    try:
        final = _run(Orchestrator(root, load_config(root), state).run())
        assert isinstance(final, RunState)
    except Exception as exc:
        console.print(f"[bold red]✗ Resume failed:[/] {exc}")
        raise typer.Exit(1) from exc
    _print_state(final, root)


@app.command()
def report(
    calibration_dir: Annotated[
        Path | None, typer.Option("--calibration-dir", help="Calibration artifact directory.")
    ] = None,
    benchmark_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-jsonl",
            help=(
                "Aggregate JSONL rows with configuration, wall_seconds, pass_rate, "
                "tasks_per_second."
            ),
        ),
    ] = None,
) -> None:
    """Print the latest calibration aggregation and artifact paths."""
    root = _root()
    print_banner(console, root, compact=True, command_name="report")
    if benchmark_jsonl:
        try:
            rows = [
                json.loads(line)
                for line in benchmark_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            aggregate = benchmark_summary(rows)
            output = benchmark_jsonl.parent / "benchmark-aggregate.json"
            output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
            write_benchmark_csv(output.with_suffix(".csv"), aggregate)
            write_benchmark_html(output.with_suffix(".html"), aggregate)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            console.print(f"[bold red]✗ Benchmark report unavailable:[/] {exc}")
            raise typer.Exit(1) from exc
        console.print_json(json.dumps(aggregate))
        console.print(f"JSON: {format_path(output, root)}")
        console.print(f"CSV: {format_path(output.with_suffix('.csv'), root)}")
        console.print(f"HTML: {format_path(output.with_suffix('.html'), root)}")
        return
    try:
        directory, value = read_report(root, calibration_dir)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]✗ Report unavailable:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(value))
    console.print(f"CSV: {format_path(directory / 'report.csv', root)}")
    console.print(f"HTML: {format_path(directory / 'report.html', root)}")


@app.command("benchmark")
def benchmark_command(
    plan: Annotated[
        Path, typer.Option("--plan", help="Validated plan JSON used unchanged for every run.")
    ],
    repetitions: Annotated[int, typer.Option(min=3)] = 3,
    fixed_concurrency: Annotated[int, typer.Option("--fixed-concurrency", min=1)] = 2,
    max_agents: Annotated[int, typer.Option("--max-agents", min=1)] = 8,
    seed: Annotated[int, typer.Option(help="Seed for reproducible round ordering.")] = 2026,
    keep_worktrees: Annotated[
        bool, typer.Option(help="Keep isolated source worktrees after collecting evidence.")
    ] = False,
) -> None:
    """Fairly compare Serial, Fixed, Naive, and Adaptive on one commit and plan."""
    root = _root()
    print_banner(console, root, compact=True, command_name="benchmark")
    try:
        task_plan = Plan.model_validate_json(plan.read_text(encoding="utf-8"))

        def progress(spec: BenchmarkSpec, status: str) -> None:
            marker = {"started": "→", "finished": "✓", "failed": "✗"}[status]
            color = {"started": "cyan", "finished": "green", "failed": "red"}[status]
            console.print(
                f" [{color}]{marker}[/] {spec.mode.title()} repetition "
                f"{spec.repetition}/{repetitions} [{color}]{status}[/]"
            )

        directory = cast(
            Path,
            _run(
                BenchmarkRunner(root, progress=progress).run(
                    task_plan,
                    repetitions=repetitions,
                    fixed_concurrency=fixed_concurrency,
                    max_agents=max_agents,
                    seed=seed,
                    keep_worktrees=keep_worktrees,
                )
            ),
        )
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        console.print(f"[bold red]✗ Benchmark failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[bold green]✓ Benchmark complete:[/] {format_path(directory, root)}")
    console.print(f"JSON: {format_path(directory / 'benchmark.json', root)}")
    console.print(f"CSV: {format_path(directory / 'benchmark.csv', root)}")
    console.print(f"HTML: {format_path(directory / 'benchmark.html', root)}")


def _print_state(state: RunState, root: Path | None = None) -> None:
    table = create_table(
        ("Task", {"style": "bold cyan", "width": 14}),
        ("Status", {"width": 14}),
        ("Attempts", {"width": 10, "justify": "center"}),
        ("Detail", {"style": "default"}),
        title=f"Run Summary • {state.run_id}",
    )
    for task_id, task in state.tasks.items():
        detail = task.failure or task.summary or ""
        destination = task.assigned_instance or ""
        if destination:
            detail = f"{destination}: {detail}"
        if task.status == TaskStatus.MERGED:
            status_badge = "[bold green]✓ MERGED[/]"
        elif task.status in {TaskStatus.RUNNING, TaskStatus.VALIDATING}:
            status_badge = "[bold cyan]● RUNNING[/]"
        elif task.status == TaskStatus.RETRYING:
            status_badge = "[bold yellow]▲ RETRYING[/]"
        elif task.status == TaskStatus.FAILED:
            status_badge = "[bold red]✗ FAILED[/]"
        else:
            status_badge = f"[dim]○ {task.status.value.upper()}[/]"
        table.add_row(task_id, status_badge, str(task.attempts), detail[:100])
    console.print(
        f"[bold]Model backend:[/] LM Studio   "
        f"[bold]Scheduler:[/] [cyan]{state.scheduler}[/]   "
        f"[bold]Target:[/] [cyan]{state.target_concurrency}[/]   "
        f"[bold]Parallel ceiling:[/] [cyan]{state.parallel_limit}[/]"
    )
    console.print(table)


def _status_panel(state: RunState, sample: TelemetrySample | None, running: int) -> Panel:
    ready = sum(task.status == TaskStatus.READY for task in state.tasks.values())
    merged = sum(task.status == TaskStatus.MERGED for task in state.tasks.values())
    total = len(state.tasks)
    telemetry: list[str] = []
    if sample is not None:
        queued = sample.queued_predictions
        speed = sample.tokens_per_second
        ttft = sample.ttft_seconds
        generating = sample.generating
        if generating is not None:
            telemetry.append(f"generating={'yes' if generating else 'no'}")
        if queued is not None and queued > 0:
            telemetry.append(f"queued={queued}")
        if speed is not None:
            telemetry.append(f"speed={speed:.1f} tok/s")
        if ttft is not None:
            telemetry.append(f"TTFT={ttft:.2f}s")
    lines = [
        f"[bold cyan]⚡ ADAPTEA RUN[/] [bold white]{state.run_id}[/]   [dim]•[/]   "
        f"Scheduler: [cyan]{state.scheduler}[/]   [dim]•[/]   "
        f"Slots: [cyan]{running}/{state.target_concurrency}[/] active   [dim]•[/]   "
        f"Merged: [bold green]{merged}/{total}[/] ([dim]{ready} ready[/])",
    ]
    if state.fleet_enabled:
        assignments: dict[str, int] = {}
        for task in state.tasks.values():
            if task.assigned_instance and task.status in {
                TaskStatus.RUNNING,
                TaskStatus.VALIDATING,
                TaskStatus.RETRYING,
            }:
                assignments[task.assigned_instance] = assignments.get(task.assigned_instance, 0) + 1
        topology = state.fleet_topology.get("instances", [])
        if isinstance(topology, list):
            lines.append("Fleet destinations:")
            for row in topology:
                if not isinstance(row, dict):
                    continue
                instance = str(row.get("instance_id", "unknown"))
                tier = str(row.get("capability_tier", "unknown")).upper()
                target = row.get("admission_target") or row.get("parallel_limit") or "unknown"
                lines.append(
                    f"  {tier:<6} {instance:<28} running "
                    f"{assignments.get(instance, 0)} / target {target}"
                )
    if telemetry:
        lines.append("[dim]Telemetry:[/] " + "   ".join(telemetry))
    return Panel(
        "\n".join(lines),
        title=f"[bold cyan]ADAPTEA {state.run_id}[/]",
        box=box.ROUNDED,
        border_style="#3b82f6",
    )


def discover_opencode() -> str | None:
    """Return the first supported OpenCode executable available on PATH."""
    return shutil.which("opencode") or shutil.which("opencode2")


if __name__ == "__main__":
    app()
