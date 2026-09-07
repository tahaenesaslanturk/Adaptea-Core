from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from adaptea.config import AGENT_CONTEXT_LENGTH, Config, FleetModelConfig
from adaptea.diagnostics.system import (
    DiagnosticSnapshot,
    LocalModel,
    SystemDiagnostics,
    find_lms_executable,
    find_ollama_executable,
    project_opencode_path,
)
from adaptea.inference import (
    configured_model,
    create_inference_backend,
    inference_connection,
    set_configured_model,
)
from adaptea.lmstudio.client import LMStudioClient, LMStudioError
from adaptea.lmstudio.lms_cli import CommandResult
from adaptea.planner.opencode import opencode_config
from adaptea.setup.actions import (
    InstallAction,
    SetupCommandRunner,
    git_install_action,
    llamacpp_install_action,
    llmster_install_action,
    ollama_install_action,
    opencode_install_action,
    vllm_install_action,
)
from adaptea.setup.configuration import (
    merge_adaptea_config,
    merge_fleet_config,
    merge_opencode_config,
)


@dataclass(slots=True)
class SetupOutcome:
    snapshot: DiagnosticSnapshot
    messages: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SmokeResult:
    success: bool
    latency_seconds: float
    detail: str


def command_failure_reason(result: CommandResult) -> str:
    lines = (result.stderr or result.stdout).strip().splitlines()
    for line in lines:
        if "permission denied" in line.lower():
            return line.strip()
    for line in lines:
        if "error" in line.lower():
            return line.strip()
    return lines[-1].strip() if lines else f"exit code {result.returncode}"


#: Setup runs many times per session, and each run wrote a log that was never removed.
#: Keeping a bounded window preserves the recent history a failure is diagnosed from
#: without growing `.adaptea/logs` without limit.
SETUP_LOG_RETENTION = 10


def prune_setup_logs(directory: Path, keep: int = SETUP_LOG_RETENTION) -> list[Path]:
    """Delete all but the newest ``keep`` setup logs, returning what was removed."""
    logs = sorted(directory.glob("setup-*.log"), key=lambda path: path.name, reverse=True)
    removed: list[Path] = []
    for path in logs[keep:]:
        try:
            path.unlink()
        except OSError:  # A log held open elsewhere is not worth failing setup over.
            continue
        removed.append(path)
    return removed


def create_setup_logger(root: Path, verbose: bool = False) -> tuple[logging.Logger, Path]:
    directory = root / ".adaptea" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    # Prune before opening the new handler so this session's log is never a candidate.
    prune_setup_logs(directory, SETUP_LOG_RETENTION - 1)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = directory / f"setup-{timestamp}.log"
    logger = logging.getLogger(f"adaptea.setup.{timestamp}.{id(root)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.debug("setup started; verbose=%s", verbose)
    return logger, path


class SetupManager:
    def __init__(
        self,
        root: Path,
        config: Config,
        *,
        diagnostics: SystemDiagnostics | None = None,
        runner: SetupCommandRunner | None = None,
        logger: logging.Logger | None = None,
        confirm_install: Callable[[InstallAction], bool] | None = None,
        confirm: Callable[[str], bool] | None = None,
        select_model: Callable[[list[LocalModel]], LocalModel | None] | None = None,
        request_model_identifier: Callable[[], str | None] | None = None,
        request_context_length: Callable[[LocalModel], int | None] | None = None,
        request_max_agents: Callable[[int], int] | None = None,
        notice: Callable[[str], None] | None = None,
        snapshot_updated: Callable[[DiagnosticSnapshot], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.diagnostics = diagnostics or SystemDiagnostics(root, config)
        self.logger = logger or logging.getLogger("adaptea.setup")
        self.notice = notice or (lambda _message: None)
        self.snapshot_updated = snapshot_updated or (lambda _snapshot: None)
        self.runner = runner or SetupCommandRunner(self.logger, self.notice)
        self.confirm_install = confirm_install or (lambda _action: False)
        self.confirm = confirm or (lambda _message: False)
        self.select_model = select_model or self._default_select_model
        self.request_model_identifier = request_model_identifier or (lambda: None)
        self.request_context_length = request_context_length or (lambda _model: None)
        self.request_max_agents = request_max_agents or (lambda current: current)

    async def diagnose(self) -> DiagnosticSnapshot:
        snapshot = await self.diagnostics.collect()
        self.logger.debug("diagnostic snapshot: %s", snapshot)
        self.snapshot_updated(snapshot)
        return snapshot

    async def fix_all(self, *, automatic: bool = False) -> SetupOutcome:
        messages: list[str] = []
        failures: list[str] = []
        explicit_model_key: str | None = None
        self.notice("Inspecting the local toolchain and project configuration…")
        snapshot = await self.diagnose()
        is_lmstudio = self.config.inference.backend == "lmstudio"

        if not snapshot.git_executable:
            self.notice("Git is missing; checking for a safe installation path…")
            action = git_install_action(self.diagnostics.system, self.diagnostics.which)
            await self._install_or_explain(
                action,
                "Git is required. Install it from https://git-scm.com/downloads and re-run setup.",
                messages,
                failures,
            )
            snapshot = await self.diagnose()

        if is_lmstudio and not snapshot.lms_executable:
            self.notice("LM Studio CLI is unavailable; checking the installed desktop runtime…")
            if snapshot.lmstudio_desktop:
                if snapshot.lms_bootstrap_executable and self.confirm(
                    "Run LM Studio Desktop's bundled 'lms bootstrap' repair now?"
                ):
                    result = await self.runner.run(
                        snapshot.lms_bootstrap_executable,
                        "bootstrap",
                        "-y",
                        timeout=60,
                    )
                    if result.returncode == 0:
                        messages.append("Bootstrapped LM Studio's bundled lms CLI.")
                    else:
                        reason = command_failure_reason(result)
                        failures.append(
                            "LM Studio CLI bootstrap failed: "
                            f"{reason}. Open LM Studio once and retry setup."
                        )
                elif not snapshot.lms_bootstrap_executable and self.confirm(
                    "Launch LM Studio Desktop once to initialize its bundled lms CLI?"
                ):
                    launch = (
                        ("open", str(snapshot.lmstudio_desktop))
                        if self.diagnostics.system == "Darwin"
                        else (str(snapshot.lmstudio_desktop),)
                    )
                    await self.runner.run(*launch, timeout=30)
            else:
                action = llmster_install_action(self.diagnostics.system)
                await self._install_or_explain(
                    action,
                    "Install LM Studio Desktop from https://lmstudio.ai/download "
                    "or download the official lms CLI (headless).",
                    messages,
                    failures,
                )
            snapshot = await self.diagnose()
            if not snapshot.lms_executable:
                if snapshot.lmstudio_desktop:
                    messages.append(
                        "LM Studio Desktop is installed, but lms is still unavailable. "
                        "Open LM Studio once, then restart this terminal and choose Re-check."
                    )
                else:
                    messages.append(
                        "lms CLI is still unavailable. Download the official lms CLI via "
                        "Safe Repairs or install LM Studio Desktop from https://lmstudio.ai/download."
                    )

        backend_kind = snapshot.backend_kind
        if (
            backend_kind == "ollama"
            and not snapshot.backend_executable
            and not snapshot.server_reachable
        ):
            self.notice("Ollama is unavailable; installing the selected local runtime…")
            await self._install_or_explain(
                ollama_install_action(self.diagnostics.system),
                "Install Ollama from https://ollama.com/download, then re-run setup.",
                messages,
                failures,
            )
            snapshot = await self.diagnose()
        elif (
            backend_kind == "llamacpp"
            and not snapshot.backend_executable
            and not snapshot.server_reachable
        ):
            self.notice("llama.cpp is unavailable; checking for a supported installation path…")
            await self._install_or_explain(
                llamacpp_install_action(self.diagnostics.system, self.diagnostics.which),
                "Install llama.cpp or llama-server from https://github.com/ggerganov/llama.cpp, then re-run setup.",
                messages,
                failures,
            )
            snapshot = await self.diagnose()
        elif (
            backend_kind == "vllm"
            and not snapshot.backend_executable
            and not snapshot.server_reachable
        ):
            self.notice("vLLM is unavailable; installing the vllm Python package…")
            await self._install_or_explain(
                vllm_install_action(self.diagnostics.system, self.diagnostics.which),
                "Install vLLM (e.g. 'pip install vllm'), then re-run setup.",
                messages,
                failures,
            )
            snapshot = await self.diagnose()

        if is_lmstudio and snapshot.lms_executable and not snapshot.server_reachable:
            self.notice("Starting the LM Studio server on the configured local endpoint…")
            if await self.start_server(snapshot.lms_executable):
                messages.append("LM Studio server started on the configured loopback endpoint.")
            else:
                failures.append(
                    "LM Studio server did not start. Adaptea tried lms server start, "
                    "then lms daemon up "
                    "and retried. Open LM Studio or inspect the setup log."
                )
            snapshot = await self.diagnose()
        elif (
            backend_kind == "ollama"
            and snapshot.backend_executable
            and not snapshot.server_reachable
        ):
            self.notice("Starting the Ollama server on the configured local endpoint…")
            if await self.start_ollama_server(snapshot.backend_executable):
                messages.append("Ollama server started on the configured loopback endpoint.")
            else:
                failures.append(
                    "Ollama server did not start or respond. Start Ollama using the desktop app "
                    "or 'ollama serve'."
                )
            snapshot = await self.diagnose()
        elif backend_kind in {"llamacpp", "vllm"} and not snapshot.server_reachable:
            self.notice(f"Checking for a running {snapshot.backend_display_name} server…")
            if await self.start_generic_server():
                messages.append(f"{snapshot.backend_display_name} server is reachable.")
            else:
                failures.append(
                    f"{snapshot.backend_display_name} server is not reachable on the configured endpoint. "
                    f"Start {snapshot.backend_display_name} server and confirm it is listening."
                )
            snapshot = await self.diagnose()

        selected = self._configured_or_single_model(snapshot, allow_ready_fallback=not automatic)
        backend_executable = snapshot.lms_executable if is_lmstudio else snapshot.backend_executable
        if not snapshot.models and backend_executable:
            self.notice(
                f"No downloaded {snapshot.backend_display_name} model was found; "
                "a model choice is required."
            )
            identifier = self.request_model_identifier()
            if identifier:
                explicit_model_key = identifier
                action = InstallAction(
                    component=f"model {identifier}",
                    command=(
                        (backend_executable, "get", identifier, "--gguf")
                        if is_lmstudio
                        else (backend_executable, "pull", identifier)
                    ),
                    display_command=(
                        f'lms get "{identifier}" --gguf'
                        if is_lmstudio
                        else f'ollama pull "{identifier}"'
                    ),
                    source=(
                        "LM Studio model catalog / Hugging Face"
                        if is_lmstudio
                        else "Ollama model library"
                    ),
                    explanation=(
                        "Searches for and downloads the selected GGUF model. "
                        "Model downloads can be "
                        "many gigabytes."
                    ),
                )
                await self._install_or_explain(
                    action,
                    f"No model was downloaded; enter a model identifier for "
                    f"{snapshot.backend_display_name}.",
                    messages,
                    failures,
                )
                snapshot = await self.diagnose()
                selected = next(
                    (
                        model
                        for model in snapshot.models
                        if identifier in {model.key, model.instance_id}
                    ),
                    None,
                )
            else:
                messages.append(
                    "No local LLM is installed. Use 'lms get --gguf' in LM Studio or "
                    "'ollama pull <model>' in Ollama, then re-check."
                )
        if selected is None and snapshot.models and not automatic:
            selected = self.select_model(snapshot.models)

        if is_lmstudio and selected and not selected.loaded and snapshot.server_reachable:
            # Automatic repair covers service/configuration fixes, not choosing a model for
            # the user. The desktop assigns and loads models in Environment > Models, so an
            # automatic pass only loads the model it just downloaded on request; otherwise
            # loading requires an interactive confirmation.
            should_load = (
                bool(explicit_model_key)
                and explicit_model_key in {selected.key, selected.instance_id}
                if automatic
                else self.confirm(f"Load local model {selected.key} now?")
            )
            if should_load:
                self.notice(f"Loading existing local model: {selected.key}")
                context = self.request_context_length(selected)
                if context is None and automatic:
                    context = min(
                        selected.max_context_length or AGENT_CONTEXT_LENGTH,
                        AGENT_CONTEXT_LENGTH,
                    )
                try:
                    await self.load_model(selected.key, context)
                    messages.append(f"Loaded {selected.key}.")
                except (LMStudioError, ValueError) as exc:
                    failures.append(
                        f"Model load failed: {exc}. Adaptea used LM Studio's native model-load API."
                    )
                snapshot = await self.diagnose()
                selected = next(
                    (model for model in snapshot.models if model.key == selected.key), selected
                )

        if not snapshot.opencode_executable:
            self.notice("OpenCode is missing; checking for a supported installation path…")
            action = opencode_install_action(self.diagnostics.system, self.diagnostics.which)
            await self._install_or_explain(
                action,
                "Install OpenCode from https://opencode.ai/docs/ and re-run setup.",
                messages,
                failures,
            )
            snapshot = await self.diagnose()

        selected_key = selected.key if selected else configured_model(self.config)
        planner = self.config.fleet.planner() if self.config.fleet.enabled else None
        model_configuration_changed = bool(
            selected_key
            and (
                configured_model(self.config) != selected_key
                or (
                    self.config.fleet.enabled and (planner is None or planner.model != selected_key)
                )
            )
        )
        configuration_needed = (
            not snapshot.opencode_configured
            or not snapshot.adaptea_config
            or model_configuration_changed
        )
        if (
            configuration_needed
            and selected_key
            and snapshot.opencode_executable
            and (snapshot.lms_executable if is_lmstudio else snapshot.server_reachable)
        ):
            self.notice("Writing project-local Adaptea and OpenCode configuration…")
            set_configured_model(self.config, selected_key)
            self.config.worker.executable = snapshot.opencode_executable
            if snapshot.lms_executable:
                self.config.lmstudio.lms_executable = snapshot.lms_executable
            maximum = max(1, self.request_max_agents(self.config.worker.max_agents))
            self.config.worker.max_agents = maximum
            fleet_changed = self._align_fleet_planner(selected) if selected else False
            try:
                connection = inference_connection(self.config)
                backend_base_url = connection.openai_base_url.removesuffix("/v1")
                opencode_path, opencode_backup = merge_opencode_config(
                    self.root,
                    snapshot.opencode_executable,
                    selected_key,
                    backend_base_url,
                    backend=self.config.inference.backend,
                )
                adaptea_backup = merge_adaptea_config(
                    self.root / "adaptea.toml",
                    base_url=backend_base_url,
                    model=selected_key,
                    lms_executable=(snapshot.lms_executable or self.config.lmstudio.lms_executable),
                    opencode_executable=snapshot.opencode_executable,
                    max_agents=maximum,
                    scheduler=self.config.worker.default_scheduler,
                    backend=self.config.inference.backend,
                    ollama_executable=self.config.ollama.executable,
                )
                fleet_backup = (
                    merge_fleet_config(self.root / "adaptea.toml", self.config.fleet)
                    if fleet_changed
                    else None
                )
                messages.append(f"Configured OpenCode locally at {opencode_path}.")
                messages.append(f"Updated {self.root / 'adaptea.toml'}.")
                if opencode_backup:
                    messages.append(f"OpenCode backup: {opencode_backup}")
                if adaptea_backup:
                    messages.append(f"Adaptea config backup: {adaptea_backup}")
                if fleet_backup:
                    messages.append(f"Fleet config backup: {fleet_backup}")
            except (OSError, ValueError) as exc:
                self.logger.exception("configuration update failed")
                failures.append(
                    f"Configuration update failed: {exc}. "
                    "Existing unrelated settings were not replaced."
                )
            snapshot = await self.diagnose()

        if snapshot.selected_model and is_lmstudio:
            messages.append(
                "Model concurrency is managed automatically within LM Studio's reported ceiling."
            )

        if automatic and not snapshot.required_ready and not failures:
            messages.append(
                "Automatic setup completed every safe available action. "
                "A choice or confirmed software/model "
                "download is still required."
            )
        if failures:
            self.notice(f"Safe repair finished with {len(failures)} item(s) needing attention.")
        elif snapshot.required_ready:
            self.notice("Safe repair finished; all required checks are ready.")
        else:
            self.notice("Safe repair finished; a user choice is still required.")
        return SetupOutcome(snapshot=snapshot, messages=messages, failures=failures)

    async def _install_or_explain(
        self,
        action: InstallAction | None,
        fallback: str,
        messages: list[str],
        failures: list[str],
    ) -> None:
        if action is None:
            messages.append(fallback)
            return
        if not self.confirm_install(action):
            messages.append(f"Cancelled {action.component} installation. {fallback}")
            return
        if action.component == "OpenCode":
            self.notice(
                "Downloading OpenCode… GitHub download speed can make this take a few minutes."
            )
        self.notice(f"Running: {action.display_command}")
        result = await self.runner.run(*action.command)
        if result.returncode == 0:
            messages.append(f"{action.component} installer completed.")
        else:
            reason = command_failure_reason(result)
            failures.append(
                f"{action.component} installation failed: {reason}. Adaptea tried: "
                f"{action.display_command}. Try the official source: {action.source}"
            )

    def lms_executable(self, override: str | None = None) -> str | None:
        """The ``lms`` CLI this machine actually has, PATH or not."""
        return override or find_lms_executable(
            self.config.lmstudio.lms_executable,
            self.diagnostics.system,
            self.diagnostics.which,
        )

    def ollama_executable(self, override: str | None = None) -> str | None:
        return override or find_ollama_executable(
            self.config.ollama.executable,
            self.diagnostics.system,
            self.diagnostics.which,
        )

    async def _wait_until_reachable(self, attempts: int, delay: float = 1.0) -> bool:
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(delay)
            try:
                snapshot = await self.diagnostics.collect()
                if snapshot.server_reachable:
                    return True
            except Exception:
                pass
            try:
                async with create_inference_backend(self.config, timeout=1.0) as client:
                    await client.models()
                    return True
            except Exception:
                continue
        return False

    async def _wait_until_stopped(self, attempts: int, delay: float = 0.5) -> bool:
        for _ in range(attempts):
            try:
                snapshot = await self.diagnostics.collect()
                if not snapshot.server_reachable:
                    return True
            except Exception:
                pass
            try:
                async with create_inference_backend(self.config, timeout=1.0) as client:
                    await client.models()
            except Exception:
                return True
            await asyncio.sleep(delay)
        return False

    async def start_backend_server(self, executable: str | None = None) -> bool:
        """Start whichever local inference service this project is configured for."""
        backend_kind = self.config.inference.backend
        if backend_kind == "lmstudio":
            exe = self.lms_executable(executable)
            return await self.start_server(exe) if exe else False
        if backend_kind == "ollama":
            return await self.start_ollama_server(executable)
        return await self.start_generic_server()

    async def start_server(self, lms_executable: str | None = None) -> bool:
        exe = self.lms_executable(lms_executable)
        if not exe:
            return False
        parsed = urlparse(self.config.lmstudio.base_url)
        port = parsed.port or 1234
        result = await self.runner.run(
            exe,
            "server",
            "start",
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            timeout=60,
        )
        # Exit zero only means the CLI accepted the request. Whether the endpoint actually
        # answers is what "started" has to mean here, so the daemon retry still runs when
        # a nominally successful start left nothing listening.
        if result.returncode == 0 and await self._wait_until_reachable(attempts=5):
            return True
        daemon = await self.runner.run(exe, "daemon", "up", timeout=60)
        if daemon.returncode != 0:
            return False
        retry = await self.runner.run(
            exe,
            "server",
            "start",
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            timeout=60,
        )
        return retry.returncode == 0 and await self._wait_until_reachable(attempts=5)

    async def start_ollama_server(self, ollama_executable: str | None = None) -> bool:
        exe = self.ollama_executable(ollama_executable)
        launched = False
        if self.diagnostics.system == "Darwin":
            for app_path in [
                Path("/Applications/Ollama.app"),
                Path.home() / "Applications" / "Ollama.app",
            ]:
                if app_path.exists():
                    opened = await self.runner.run("open", "-a", str(app_path), timeout=10)
                    launched = opened.returncode == 0
                    break
        # Without the desktop app -- and on every other platform -- the CLI's own daemon is
        # what serves the API. Nothing used to launch it there, so "Start server" only ever
        # waited and reported failure. `ollama serve` never exits, so it is spawned detached
        # rather than run to completion.
        if not launched and exe:
            launched = await self.runner.spawn(exe, "serve")
        if not launched:
            return False
        return await self._wait_until_reachable(attempts=10)

    async def start_generic_server(self) -> bool:
        return await self._wait_until_reachable(attempts=5)

    async def stop_server(self, executable: str | None = None) -> bool:
        backend_kind = self.config.inference.backend
        if backend_kind == "lmstudio":
            exe = self.lms_executable(executable)
            if not exe:
                return False
            await self.runner.run(exe, "server", "stop", timeout=30)
            await self.runner.run(exe, "daemon", "down", timeout=30)
        elif backend_kind == "ollama":
            if self.diagnostics.system == "Darwin":
                # The menu-bar app supervises the serving helper and restarts it, so the app
                # has to go first. Its own process is named "Ollama", which a case-sensitive
                # pattern never matched -- that is why the endpoint stayed up after "Stop".
                await self.runner.run("osascript", "-e", 'quit app "Ollama"', timeout=10)
                await self.runner.run("pkill", "-if", "Ollama.app", timeout=10)
                await self.runner.run("pkill", "-ix", "ollama", timeout=10)
            elif self.diagnostics.system == "Windows":
                await self.runner.run("taskkill", "/F", "/IM", "ollama.exe", "/T", timeout=10)
                await self.runner.run("taskkill", "/F", "/IM", "ollama app.exe", "/T", timeout=10)
            else:
                await self.runner.run("pkill", "-if", "ollama", timeout=10)
        elif backend_kind in {"llamacpp", "vllm"}:
            proc_name = "llama-server" if backend_kind == "llamacpp" else "vllm"
            if self.diagnostics.system == "Windows":
                await self.runner.run("taskkill", "/F", "/IM", f"{proc_name}.exe", "/T", timeout=10)
            else:
                await self.runner.run("pkill", "-if", proc_name, timeout=10)
        else:
            return False
        # Whether the endpoint stopped answering is the only honest answer here; the kill
        # commands exit zero even when they matched nothing.
        return await self._wait_until_stopped(attempts=8)

    async def restart_server(self, executable: str | None = None) -> bool:
        await self.stop_server(executable)
        await asyncio.sleep(1.0)
        return await self.start_backend_server(executable)

    async def server_status(self) -> dict[str, Any]:
        backend_kind = self.config.inference.backend
        base_url = (
            self.config.lmstudio.base_url
            if backend_kind == "lmstudio"
            else self.config.ollama.base_url
            if backend_kind == "ollama"
            else self.config.llamacpp.base_url
            if backend_kind == "llamacpp"
            else self.config.vllm.base_url
        )
        reachable = False
        error = None
        try:
            async with create_inference_backend(self.config, timeout=1.5) as client:
                await client.models()
                reachable = True
        except Exception as exc:
            error = str(exc)
        return {
            "backend": backend_kind,
            "base_url": base_url,
            "reachable": reachable,
            "error": error if not reachable else None,
        }

    async def load_model(self, model: str, context_length: int | None = None) -> None:
        config: dict[str, int] = {}
        if context_length is not None:
            if context_length <= 0:
                raise ValueError("context length must be positive")
            config["context_length"] = context_length
        async with LMStudioClient(
            self.config.lmstudio.base_url, self.config.lmstudio.api_token, timeout=600
        ) as client:
            await client.load(model, **config)

    async def smoke_test(self, snapshot: DiagnosticSnapshot | None = None) -> SmokeResult:
        current = snapshot or await self.diagnose()
        selected = current.selected_model
        if not current.opencode_executable or not selected or not selected.ready:
            return SmokeResult(False, 0, "OpenCode and an inference-ready model are required.")
        command = [
            current.opencode_executable,
            "models",
            "adaptea",
        ]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="adaptea-smoke-") as temporary:
            smoke_root = Path(temporary)
            source_config = project_opencode_path(self.root)
            target_config = smoke_root / source_config.name
            target_config.parent.mkdir(parents=True, exist_ok=True)
            target_config.write_text(
                json.dumps(
                    opencode_config(
                        self.config,
                        selected.instance_id or selected.key,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            result = await self.runner.run(*command, cwd=smoke_root, timeout=30)
        latency = time.perf_counter() - started
        output = result.stdout + "\n" + result.stderr
        destination = selected.instance_id or selected.key
        success = result.returncode == 0 and (
            f"adaptea/{destination}" in output or destination in output
        )
        detail = (
            f"OpenCode recognizes the selected project-local {current.backend_display_name} model."
            if success
            else (
                "OpenCode did not list the project-local model within 30 seconds."
                if result.returncode == -1
                else (
                    f"OpenCode model-catalog check failed with exit code {result.returncode}; "
                    "see the setup log."
                )
            )
        )
        return SmokeResult(success, latency, detail)

    @staticmethod
    def _default_select_model(models: list[LocalModel]) -> LocalModel | None:
        return models[0] if len(models) == 1 else None

    def _configured_or_single_model(
        self,
        snapshot: DiagnosticSnapshot,
        *,
        allow_ready_fallback: bool = True,
    ) -> LocalModel | None:
        planner = self.config.fleet.planner() if self.config.fleet.enabled else None
        configured_key = planner.model if planner else configured_model(self.config)
        configured = next(
            (
                model
                for model in snapshot.models
                if configured_key in {model.key, model.instance_id}
            ),
            None,
        )
        if configured and configured.ready:
            return configured
        if allow_ready_fallback:
            ready = [model for model in snapshot.models if model.ready]
            if len(ready) == 1:
                return ready[0]
        if configured:
            return configured
        return self._default_select_model(snapshot.models) if allow_ready_fallback else None

    def _align_fleet_planner(self, selected: LocalModel) -> bool:
        """Keep Fleet's effective planner in sync with Setup's selected model."""
        if not self.config.fleet.enabled:
            return False

        changed = False
        selected_entry = next(
            (item for item in self.config.fleet.models if item.model == selected.key), None
        )
        for item in self.config.fleet.models:
            if item is selected_entry or "planner" not in item.roles:
                continue
            item.roles = [role for role in item.roles if role != "planner"] or ["worker"]
            changed = True

        if selected_entry is None:
            used_names = {item.name for item in self.config.fleet.models}
            base_name = "setup-strong"
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}-{suffix}"
                suffix += 1
            self.config.fleet.models.append(
                FleetModelConfig(
                    name=name,
                    model=selected.key,
                    tier="strong",
                    roles=["planner", "worker", "reviewer"],
                    instances=1,
                    context_length=selected.context_length,
                    parallel_limit=selected.parallel,
                )
            )
            return True

        roles = list(selected_entry.roles)
        if "planner" not in roles:
            roles.insert(0, "planner")
        if "worker" not in roles:
            roles.append("worker")
        if "reviewer" not in roles:
            roles.append("reviewer")
        if roles != selected_entry.roles:
            selected_entry.roles = roles
            changed = True
        if selected_entry.tier != "strong":
            selected_entry.tier = "strong"
            changed = True
        return changed
