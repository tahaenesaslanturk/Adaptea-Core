from __future__ import annotations

import platform
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from adaptea.config import Config, FleetConfig, FleetModelConfig, load_config
from adaptea.diagnostics.system import DiagnosticSnapshot, LocalModel, SystemDiagnostics
from adaptea.setup.actions import InstallAction, SetupCommandRunner
from adaptea.setup.configuration import merge_fleet_config
from adaptea.setup.manager import SetupManager, SetupOutcome, create_setup_logger


def setup_table(snapshot: DiagnosticSnapshot) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=2)
    table.add_column(style="bold", min_width=24)
    table.add_column()

    def row(ready: bool, label: str, state: str, warning: bool = False) -> None:
        if ready:
            table.add_row("[green]✓[/]", label, f"[green]{state}[/]")
        elif warning:
            table.add_row("[yellow]![/]", label, f"[yellow]{state}[/]")
        else:
            table.add_row("[red]✗[/]", label, f"[red]{state}[/]")

    row(snapshot.python_ready, "Python", snapshot.python_version)
    row(bool(snapshot.git_executable), "Git", snapshot.git_version or "Missing")
    row(bool(snapshot.opencode_executable), "OpenCode", snapshot.opencode_version or "Missing")
    row(
        bool(snapshot.lms_executable),
        "LM Studio / llmster",
        snapshot.lms_version
        or (
            "Desktop detected; lms bootstrap available"
            if snapshot.lms_bootstrap_executable
            else (
                "Desktop detected; lms CLI needs repair"
                if snapshot.lmstudio_desktop
                else "Not ready"
            )
        ),
        warning=snapshot.lmstudio_desktop is not None,
    )
    server_ready = (
        snapshot.server_reachable and snapshot.native_api_usable and snapshot.openai_api_usable
    )
    server_state = "Running"
    if not snapshot.server_reachable:
        server_state = "Stopped"
    elif not snapshot.openai_api_usable:
        server_state = "Running; OpenAI /v1 API unavailable"
    row(
        server_ready,
        "LM Studio server",
        server_state,
    )
    selected = snapshot.selected_model
    model_state = "Not selected"
    if selected:
        model_state = f"{selected.key} ({'loaded' if selected.loaded else 'not loaded'})"
    elif snapshot.models:
        model_state = f"{len(snapshot.models)} local model(s); select one"
    row(bool(selected and selected.loaded), "Model", model_state, warning=selected is not None)
    row(
        snapshot.opencode_configured,
        "OpenCode → LM Studio",
        "Configured" if snapshot.opencode_configured else "Not configured",
        warning=True,
    )
    row(
        snapshot.adaptea_config,
        "Adaptea configuration",
        "Ready" if snapshot.adaptea_config else "Missing",
        warning=True,
    )
    row(
        snapshot.capacity_profile,
        "Calibration",
        "Profile available" if snapshot.capacity_profile else "Not run (optional now)",
        warning=True,
    )
    return table


def render_setup(console: Console, snapshot: DiagnosticSnapshot) -> None:
    console.print(
        Panel(
            setup_table(snapshot),
            title="[bold cyan]⚡ ADAPTEA SETUP[/]",
            box=box.ROUNDED,
            border_style="#334155",
            padding=(1, 2),
        )
    )


def configure_fleet(
    console: Console, root: Path, snapshot: DiagnosticSnapshot, config: Config
) -> None:
    mode = Prompt.ask(
        "Model mode",
        choices=["simple", "fleet", "cancel"],
        default="simple",
        console=console,
    )
    if mode == "cancel":
        return
    if mode == "simple":
        merge_fleet_config(root / "adaptea.toml", FleetConfig(enabled=False))
        console.print(
            "[green]Simple mode selected.[/] One configured LM Studio model remains the planner "
            "and worker model."
        )
        return
    if not snapshot.models:
        console.print("[yellow]No downloaded local LLMs were discovered.[/]")
        return
    planner = _select_model(console, snapshot.models)
    if planner is None:
        return
    strong = _select_model(console, snapshot.models) or planner
    use_fast = Confirm.ask(
        "Configure a separate fast worker model? Tiers are your choice, not inferred from size.",
        default=len(snapshot.models) > 1,
        console=console,
    )
    fast = _select_model(console, snapshot.models) if use_fast else None
    models: list[FleetModelConfig] = []
    if planner.key == strong.key:
        models.append(
            FleetModelConfig(
                name="strong",
                model=strong.key,
                tier="strong",
                roles=["planner", "worker", "reviewer"],
            )
        )
    else:
        models.extend(
            [
                FleetModelConfig(
                    name="planner",
                    model=planner.key,
                    tier="strong",
                    roles=["planner", "reviewer"],
                    instances=1,
                ),
                FleetModelConfig(
                    name="strong",
                    model=strong.key,
                    tier="strong",
                    roles=["worker"],
                ),
            ]
        )
    if fast and fast.key not in {planner.key, strong.key}:
        models.append(FleetModelConfig(name="fast", model=fast.key, tier="fast", roles=["worker"]))
    fleet = FleetConfig(
        enabled=True,
        topology="auto",
        max_loaded_instances=config.fleet.max_loaded_instances,
        models=models,
        routing=config.fleet.routing,
    )
    merge_fleet_config(root / "adaptea.toml", fleet)
    console.print(
        "[green]Adaptive Fleet configured.[/] Load the selected instances, then run "
        "[bold]adaptea fleet calibrate[/]."
    )


def _confirm_install(console: Console, action: InstallAction) -> bool:
    warning = (
        "\n[bold yellow]This executes a remote installation script.[/]"
        if action.remote_script
        else ""
    )
    console.print(
        Panel(
            f"{action.explanation}\n\n[bold]Exact command[/]\n{action.display_command}\n\n"
            f"[bold]Official source[/]\n{action.source}{warning}",
            title=f"Install {action.component}?",
            border_style="yellow",
        )
    )
    return Confirm.ask("Allow this installation?", default=False, console=console)


def _select_model(console: Console, models: list[LocalModel]) -> LocalModel | None:
    table = Table("#", "Model", "Architecture", "Size", "Context", "State")
    for index, model in enumerate(models, 1):
        size = f"{model.size_bytes / 1024**3:.1f} GB" if model.size_bytes else "unknown"
        table.add_row(
            str(index),
            model.display_name or model.key,
            model.architecture or "unknown",
            size,
            str(model.max_context_length or "unknown"),
            "loaded" if model.loaded else "local",
        )
    console.print(table)
    choice = IntPrompt.ask("Select local coding model (0 to cancel)", default=1, console=console)
    return models[choice - 1] if 1 <= choice <= len(models) else None


def _context_length(console: Console, model: LocalModel) -> int | None:
    maximum = model.max_context_length
    default = min(maximum or 8192, 8192)
    value = IntPrompt.ask(
        f"Context length (model maximum: {maximum or 'unknown'})",
        default=default,
        console=console,
    )
    if maximum and value > maximum:
        console.print(f"[yellow]Using model maximum {maximum}.[/]")
        return maximum
    return value


def manual_guidance(console: Console, system: str) -> None:
    if system == "Windows":
        llmster = "irm https://lmstudio.ai/install.ps1 | iex"
        ollama = "irm https://ollama.com/install.ps1 | iex  OR  winget install Ollama.Ollama"
        opencode = "winget install SST.opencode  OR  npm install -g opencode-ai  OR  choco install opencode"
        git = "winget install Git.Git  OR  https://git-scm.com/download/win"
    else:
        llmster = "curl -fsSL https://lmstudio.ai/install.sh | bash"
        ollama = "curl -fsSL https://ollama.com/install.sh | sh"
        opencode = "brew install anomalyco/tap/opencode  OR  npm install -g opencode-ai"
        git = "brew install git  OR  xcode-select --install"
    console.print(
        Panel(
            "LM Studio: https://lmstudio.ai/download\n"
            f"Headless llmster: {llmster}\n"
            f"Ollama: https://ollama.com/download ({ollama})\n"
            "llama.cpp: https://github.com/ggerganov/llama.cpp\n"
            "vLLM: pip install vllm\n"
            f"OpenCode: {opencode}\n"
            f"Git: {git}\n\n"
            "After installation, restart the terminal if new executables are not visible, "
            "then choose Re-check.",
            title="Manual setup",
        )
    )


def _show_outcome(console: Console, outcome: SetupOutcome) -> None:
    for message in outcome.messages:
        console.print(f"[cyan]•[/] {message}")
    for failure in outcome.failures:
        console.print(Panel(failure, title="Setup action failed", border_style="red"))


async def run_setup_center(
    root: Path,
    config: Config,
    *,
    automatic: bool = False,
    verbose: bool = False,
    console: Console | None = None,
) -> bool:
    terminal = console or Console()
    logger, log_path = create_setup_logger(root, verbose)
    diagnostics = SystemDiagnostics(root, config)
    runner = SetupCommandRunner(
        logger,
        lambda line: terminal.print(line, markup=False),
        verbose,
    )
    manager = SetupManager(
        root,
        config,
        diagnostics=diagnostics,
        runner=runner,
        logger=logger,
        confirm_install=lambda action: _confirm_install(terminal, action),
        confirm=lambda message: Confirm.ask(message, default=True, console=terminal),
        select_model=lambda models: _select_model(terminal, models),
        request_model_identifier=lambda: (
            Prompt.ask(
                "LM Studio model identifier (leave empty to skip)", default="", console=terminal
            ).strip()
            or None
        ),
        request_context_length=(
            (lambda _model: None) if automatic else (lambda model: _context_length(terminal, model))
        ),
        request_max_agents=(
            (lambda current: current)
            if automatic
            else (
                lambda current: IntPrompt.ask(
                    "Maximum OpenCode workers (hard ceiling, not immediate fan-out)",
                    default=current,
                    console=terminal,
                )
            )
        ),
        notice=lambda message: terminal.print(f"[cyan]{message}[/]"),
        snapshot_updated=lambda snapshot: render_setup(terminal, snapshot),
    )
    try:
        snapshot = await manager.diagnose()
        if automatic:
            outcome = await manager.fix_all(automatic=True)
            _show_outcome(terminal, outcome)
            snapshot = outcome.snapshot
        else:
            while True:
                choice = Prompt.ask(
                    "Action",
                    choices=["fix", "fleet", "manual", "check", "smoke", "exit"],
                    default="fix",
                    console=terminal,
                )
                if choice == "exit":
                    break
                if choice == "manual":
                    manual_guidance(terminal, platform.system())
                    continue
                if choice == "fleet":
                    configure_fleet(terminal, root, snapshot, config)
                    config = load_config(root)
                    manager.config = config
                    manager.diagnostics.config = config
                    continue
                if choice == "smoke":
                    result = await manager.smoke_test(snapshot)
                    style = "green" if result.success else "red"
                    terminal.print(
                        f"[{style}]{result.detail} Latency: {result.latency_seconds:.2f}s[/]"
                    )
                    continue
                if choice == "fix":
                    outcome = await manager.fix_all()
                    _show_outcome(terminal, outcome)
                    snapshot = outcome.snapshot
                else:
                    snapshot = await manager.diagnose()
                if snapshot.required_ready:
                    break
        snapshot = await manager.diagnose()
        if snapshot.required_ready:
            terminal.print(
                Panel(
                    "[bold green]ADAPTEA IS READY[/]\n\n"
                    "All required components passed shared doctor checks.\n"
                    "Not calibrated yet.\n\nRecommended next step:\n"
                    "    adaptea calibrate --quick",
                    border_style="green",
                )
            )
            return True
        terminal.print(
            Panel(
                "Setup is not complete yet. Re-run [bold]adaptea setup[/] after resolving the "
                f"remaining items.\n\nDebug log: {log_path}",
                border_style="yellow",
            )
        )
        return False
    except (OSError, ValueError) as exc:
        logger.exception("expected setup failure")
        terminal.print(
            Panel(
                f"What failed: setup could not complete an action.\n"
                f"Why: {exc}\n"
                "What Adaptea tried: see the debug log.\n"
                f"Next: fix the reported dependency and re-run setup.\n\nLog: {log_path}",
                title="Adaptea setup",
                border_style="red",
            )
        )
        return False
