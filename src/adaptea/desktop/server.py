from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from pydantic import BaseModel, ValidationError

from adaptea import __version__
from adaptea.config import FleetConfig
from adaptea.desktop import PROTOCOL_VERSION
from adaptea.desktop.protocol import DesktopCommand, DesktopError, DesktopEvent, DesktopResponse
from adaptea.downloads import (
    DownloadError,
    DownloadProgress,
    download_repository,
    human_bytes,
    lmstudio_models_root,
    parse_repository,
    pull_ollama_model,
)
from adaptea.environment import (
    adopt_from_project,
    apply_to_project,
    environment_root,
    is_configured,
)
from adaptea.fleet.combinations import (
    active_report,
    capture_active_capacity,
    clear_active_combination,
    combination_inputs,
    combinations_path,
    delete_combination,
    save_combination,
    select_combination,
)
from adaptea.fleet.discovery import (
    huggingface_cache_root,
    huggingface_repo_path,
    resolve_huggingface_model_path,
    resolve_lmstudio_model_path,
)
from adaptea.history import delete_run_metadata, list_runs
from adaptea.lmstudio.lms_cli import resolve_executable
from adaptea.models import Plan, RunState, TaskStatus, utc_now
from adaptea.preferences import (
    ProjectPreferences,
    load_project_preferences,
    save_project_preferences,
)
from adaptea.projects import RecentProjects, create_project, initialize_repository, inspect_project
from adaptea.reporting.report import read_report
from adaptea.runtime.state import StateStore
from adaptea.security.approvals import ApprovalDecision, ApprovalRequest
from adaptea.services import ApplicationServices
from adaptea.setup.actions import SetupCommandRunner
from adaptea.setup.configuration import (
    merge_fleet_config,
    merge_inference_model,
    merge_inference_selection,
    merge_opencode_models_config,
)
from adaptea.setup.manager import SetupManager, create_setup_logger
from adaptea.smoke import run_mvp_smoke_test
from adaptea.workers.activity import Activity


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {key: json_value(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    return value


class DesktopBridge:
    """Thin protocol adapter; all product behavior remains in Python application services."""

    def __init__(
        self,
        *,
        services: ApplicationServices | None = None,
        output: TextIO | None = None,
        demo: bool = False,
    ) -> None:
        # A desktop project is an exact folder selection. It must never inherit a model,
        # worker limit, or backend from an ancestor repository.
        self.services = services or ApplicationServices(project_scoped_config=True)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        self.output = output or sys.stdout
        self.demo = demo
        self.write_lock = asyncio.Lock()
        self.requests: dict[str, asyncio.Task[None]] = {}
        # A download is a task, and only *some* downloads also have a subprocess. The
        # ones Adaptea performs itself have nothing to terminate: cancelling the task is
        # what stops the transfer, because the task is the transfer.
        self.active_downloads: dict[str, tuple[asyncio.Task[Any], SetupCommandRunner | None]] = {}
        self.cancelled_downloads: set[str] = set()
        self.last_task_status: dict[str, dict[str, str]] = {}
        self.last_target: dict[str, int] = {}
        self.closing = False

    async def cancel_downloads(self, model_key: str | None = None) -> list[str]:
        """Stop active model downloads, and do not return until they have stopped.

        The interface reports "Paused" on the strength of this answer, so returning as
        soon as `cancel()` has been *requested* is how the app came to claim a download
        had stopped while it was still running. Waiting for the task to unwind is what
        makes the confirmation mean something.
        """
        cancelled: list[str] = []
        entries = (
            [(model_key, self.active_downloads.pop(model_key, None))]
            if model_key
            else [
                (key, self.active_downloads.pop(key, None)) for key in list(self.active_downloads)
            ]
        )
        awaiting: list[asyncio.Task[Any]] = []
        current = asyncio.current_task()
        for key, entry in entries:
            if entry is None:
                continue
            task, runner = entry
            cancelled.append(key)
            self.cancelled_downloads.add(key)
            if runner is not None:
                res = runner.terminate()
                if inspect.isawaitable(res):
                    await res
            task.cancel()
            # A download cancelling itself cannot wait for itself to finish.
            if task is not current and not task.done():
                awaiting.append(task)
        if awaiting:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*awaiting, return_exceptions=True), timeout=10.0
                )
        return cancelled

    async def _verify_huggingface_cache(
        self, repo_id: str, backend: Literal["llamacpp", "vllm"], quantization: str | None
    ) -> None:
        """Confirm the Hub CLI left a usable model behind rather than an empty cache."""
        cached_repo = huggingface_repo_path(repo_id, backend)
        if not cached_repo.is_dir():
            raise DownloadError(
                f"Hugging Face finished without creating a cache entry for {repo_id}."
            )
        if backend != "llamacpp":
            return
        wanted = (quantization or "Q4_K_M").lower()
        if not any(
            wanted in path.name.lower() for path in (cached_repo / "snapshots").rglob("*.gguf")
        ):
            raise DownloadError(
                f"No GGUF files matching {quantization or 'Q4_K_M'} were found in {repo_id}. "
                "Specify another quantization after a colon."
            )

    async def emit(self, event_name: str, **data: Any) -> None:
        await self._write(DesktopEvent(event=event_name, data=json_value(data)))

    def emit_nowait(self, event_name: str, **data: Any) -> None:
        asyncio.create_task(self.emit(event_name, **data))

    async def _write(self, message: BaseModel) -> None:
        line = message.model_dump_json(exclude_none=True) + "\n"
        async with self.write_lock:
            try:
                if hasattr(self.output, "buffer"):
                    self.output.buffer.write(line.encode("utf-8", errors="replace"))
                    self.output.buffer.flush()
                else:
                    self.output.write(line)
                    self.output.flush()
            except (BrokenPipeError, OSError):
                # The desktop can close while a durable run continues in this process.
                pass

    async def accept(self, raw: str) -> None:
        request_id = "invalid"
        try:
            value = json.loads(raw)
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                request_id = value["id"]
            command = DesktopCommand.model_validate(value)
            if command.protocol != PROTOCOL_VERSION:
                raise ValueError(
                    f"Unsupported protocol {command.protocol}; expected {PROTOCOL_VERSION}."
                )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            await self._write(
                DesktopResponse(
                    id=request_id,
                    ok=False,
                    error=DesktopError(code="invalid_request", message=str(exc)),
                )
            )
            return
        if command.id in self.requests:
            await self._write(
                DesktopResponse(
                    id=command.id,
                    ok=False,
                    error=DesktopError(
                        code="duplicate_request", message="Request id is already active."
                    ),
                )
            )
            return
        task = asyncio.create_task(self._execute(command))
        self.requests[command.id] = task

        def remove_request(_task: asyncio.Task[None], key: str = command.id) -> None:
            self.requests.pop(key, None)

        task.add_done_callback(remove_request)

    async def _execute(self, request: DesktopCommand) -> None:
        try:
            data = await self.dispatch(request.command, request.payload)
            response = DesktopResponse(id=request.id, ok=True, data=json_value(data))
        except asyncio.CancelledError:
            response = DesktopResponse(
                id=request.id,
                ok=False,
                error=DesktopError(code="cancelled", message="Command cancelled.", retryable=True),
            )
        except Exception as exc:
            response = DesktopResponse(
                id=request.id,
                ok=False,
                error=DesktopError(
                    code=_error_code(exc),
                    message=_concise_error(exc),
                    retryable=isinstance(exc, (ConnectionError, TimeoutError)),
                ),
            )
        await self._write(response)

    async def dispatch(self, command: str, payload: dict[str, Any]) -> Any:
        if command == "app.ping":
            return {
                "protocol": PROTOCOL_VERSION,
                "version": __version__,
                "pid": os.getpid(),
                "demo": self.demo,
            }
        if command == "app.shutdown":
            if self.services.active_orchestrators:
                raise RuntimeError(
                    "An Adaptea run is active. The core will remain alive so workers can finish."
                )
            self.closing = True
            return {"closing": True}
        if command == "command.cancel":
            target = _string(payload, "request_id")
            task = self.requests.get(target)
            if task is None:
                return {"cancelled": False}
            for _m, (d_task, runner) in list(self.active_downloads.items()):
                if d_task == task:
                    self.active_downloads.pop(_m, None)
                    self.cancelled_downloads.add(_m)
                    if runner is not None:
                        res = runner.terminate()
                        if inspect.isawaitable(res):
                            await res
            task.cancel()
            return {"cancelled": True}
        if command == "model.download.cancel":
            raw_model = payload.get("model")
            target_model = (
                _string(payload, "model").strip()
                if isinstance(raw_model, str) and raw_model.strip()
                else None
            )
            cancelled_models = await self.cancel_downloads(target_model)
            return {"cancelled": True, "models": cancelled_models}
        if command == "projects.recent":
            return RecentProjects().load()
        if command == "project.forget":
            recents = RecentProjects()
            removed = recents.forget(Path(_string(payload, "path")))
            projects = recents.load()
            await self.emit("project.forgotten", path=_string(payload, "path"))
            return {"removed": removed, "projects": projects}
        if command == "project.create":
            project = await create_project(
                Path(_string(payload, "parent")), _string(payload, "name")
            )
            RecentProjects().remember(project.path)
            await self.emit("project.changed", project=json_value(project))
            return project
        # Configuring the environment is not about any one project, so these commands
        # fall back to the environment's own directory. It holds an ``adaptea.toml`` and
        # nothing else, which is all the fleet, model and diagnostic services ever needed
        # from a "root".
        # The inference environment is one machine-wide answer, so the commands that read
        # or change it ignore whichever project happens to be open and always work on the
        # environment's own directory. Configuring the same models, and measuring them
        # again, once per project was the thing that made no sense.
        root = (
            environment_root()
            if command in GLOBAL_ENVIRONMENT_COMMANDS
            else _root(payload)
            if payload.get("root")
            else environment_root()
            if command in ENVIRONMENT_COMMANDS
            else _root(payload)
        )
        if command == "environment.status":
            return {
                "root": str(environment_root()),
                "configured": is_configured(),
            }
        if command == "environment.apply":
            inheriting = _root(payload)
            adopted = adopt_from_project(inheriting)
            # A project may name its own combination; most name none and take whichever
            # one the environment has selected. A combination that has since been deleted
            # falls back rather than leaving the project with no fleet at all.
            chosen = load_project_preferences(inheriting).model_combination_id
            fleet = capacity = None
            if chosen:
                try:
                    fleet, capacity = combination_inputs(environment_root(), chosen)
                except KeyError:
                    chosen = None
            written = apply_to_project(inheriting, fleet=fleet, capacity=capacity)
            if written:
                await self.emit("environment.applied", root=str(inheriting))
            return {
                "root": str(inheriting),
                "combination_id": chosen,
                "adopted": adopted is not None,
                "written": [str(item) for item in written],
            }
        if command == "project.inspect":
            project = await inspect_project(root)
            RecentProjects().remember(root)
            return project
        if command == "project.preferences.get":
            return load_project_preferences(root)
        if command == "project.combination.set":
            # Which combination a project runs on is the project's choice; the
            # combinations and their measurements stay global. Applying immediately is
            # what makes the choice mean something without a second confirming step.
            raw = payload.get("combination_id")
            chosen = raw if isinstance(raw, str) and raw else None
            if chosen is not None:
                # Fail before writing a preference that points at nothing.
                combination_inputs(environment_root(), chosen)
            existing = load_project_preferences(root)
            preferences = existing.model_copy(update={"model_combination_id": chosen})
            save_project_preferences(root, preferences)
            applied = await self.dispatch("environment.apply", {"root": str(root)})
            await self.emit("project.combination_changed", root=str(root), combination_id=chosen)
            return {"combination_id": chosen, "applied": applied}
        if command == "project.preferences.update":
            current_preferences = load_project_preferences(root)
            completion_action = payload.get(
                "completion_action", current_preferences.completion_action
            )
            # Accept the 0.5.x desktop protocol during rolling upgrades. New clients use
            # the three-state action and can distinguish local commit from origin push.
            if "completion_action" not in payload and "auto_commit_completed_runs" in payload:
                legacy = payload["auto_commit_completed_runs"]
                if not isinstance(legacy, bool):
                    raise ValueError("auto_commit_completed_runs must be true or false")
                completion_action = "commit" if legacy else "none"
            auto_start = payload.get("auto_start_plans", current_preferences.auto_start_plans)
            stop_models_on_stop = payload.get(
                "stop_models_on_stop", current_preferences.stop_models_on_stop
            )
            stop_models_on_finish = payload.get(
                "stop_models_on_finish", current_preferences.stop_models_on_finish
            )
            if completion_action not in {"none", "commit", "commit_and_push"}:
                raise ValueError("completion_action must be none, commit, or commit_and_push")
            if not isinstance(auto_start, bool):
                raise ValueError("auto_start_plans must be true or false")
            preferences = ProjectPreferences(
                completion_action=completion_action,
                auto_start_plans=auto_start,
                stop_models_on_stop=bool(stop_models_on_stop),
                stop_models_on_finish=bool(stop_models_on_finish),
            )
            path = save_project_preferences(root, preferences)
            await self.emit(
                "project.preferences_changed",
                path=str(path),
                preferences=json_value(preferences),
            )
            return preferences
        if command == "project.initialize_git":
            await initialize_repository(root)
            project = await inspect_project(root)
            RecentProjects().remember(root)
            await self.emit("project.changed", project=json_value(project))
            return project
        if command == "doctor":
            await self.emit("doctor.started", root=str(root))
            checks = await self.services.diagnose(root)
            await self.emit("doctor.updated", checks=json_value(checks))
            return checks
        if command in {"lmstudio.health", "inference.health"}:
            return await self.services.lmstudio_health(root)
        if command == "inference.config":
            config = self.services.load_config(root)
            selected = (
                config.lmstudio.model
                if config.inference.backend == "lmstudio"
                else config.ollama.model
                if config.inference.backend == "ollama"
                else config.llamacpp.model
                if config.inference.backend == "llamacpp"
                else config.vllm.model
            )
            base_url = (
                config.lmstudio.base_url
                if config.inference.backend == "lmstudio"
                else config.ollama.base_url
                if config.inference.backend == "ollama"
                else config.llamacpp.base_url
                if config.inference.backend == "llamacpp"
                else config.vllm.base_url
            )
            return {
                "backend": config.inference.backend,
                "base_url": base_url,
                "model": selected,
            }
        if command == "inference.configure":
            if self.services.active_orchestrators:
                raise RuntimeError("A run is active. Stop it before changing inference backend.")
            backend = _string(payload, "backend")
            if backend not in {"lmstudio", "ollama", "llamacpp", "vllm"}:
                raise ValueError("backend must be 'lmstudio', 'ollama', 'llamacpp', or 'vllm'")
            path = root / "adaptea.toml"
            backup_path = merge_inference_selection(path, backend)
            project_root = payload.get("root")
            if project_root:
                p_path = Path(str(project_root)) / "adaptea.toml"
                if p_path.exists():
                    merge_inference_selection(p_path, backend)
            await self.emit("inference.configured", backend=backend, path=str(path))
            return {
                "backend": backend,
                "path": str(path),
                "backup": str(backup_path) if backup_path else None,
            }
        if command == "inference.server.status":
            manager = SetupManager(root, self.services.load_config(root))
            return await manager.server_status()
        if command == "inference.server.start":
            manager = SetupManager(root, self.services.load_config(root))
            ok = await manager.start_backend_server()
            status = await manager.server_status()
            await self.emit("inference.server.changed", status=status)
            return {"ok": ok, "status": status}
        if command == "inference.server.stop":
            manager = SetupManager(root, self.services.load_config(root))
            ok = await manager.stop_server()
            status = await manager.server_status()
            await self.emit("inference.server.changed", status=status)
            return {"ok": ok, "status": status}
        if command == "inference.server.restart":
            manager = SetupManager(root, self.services.load_config(root))
            ok = await manager.restart_server()
            status = await manager.server_status()
            await self.emit("inference.server.changed", status=status)
            return {"ok": ok, "status": status}
        if command == "setup.diagnose":
            manager = SetupManager(root, self.services.load_config(root))
            return await manager.diagnose()
        if command == "setup.repair_safe":
            if self.services.active_orchestrators:
                raise RuntimeError("A run is active. Stop it before repairing the environment.")
            logger, log_path = create_setup_logger(root)
            raw_approved = payload.get("approved_components", [])
            lms_aliases = {
                "lms",
                "LM Studio",
                "LM Studio CLI",
                "LM Studio llmster",
                "llmster",
                "lms CLI",
                "LM Studio Desktop",
            }
            approved_components = set()
            if isinstance(raw_approved, list):
                for item in raw_approved:
                    if not isinstance(item, str):
                        continue
                    if item in lms_aliases:
                        approved_components.update(
                            {"LM Studio llmster", "lms", "lms CLI", "LM Studio Desktop"}
                        )
                    elif item in {"Git", "OpenCode", "Ollama"}:
                        approved_components.add(item)
            tracker = SetupProgressTracker(approved_components, self.emit_nowait)
            tracker.emit()
            manager = SetupManager(
                root,
                self.services.load_config(root),
                logger=logger,
                runner=SetupCommandRunner(
                    logger,
                    tracker.on_runner_output,
                    verbose=True,
                ),
                notice=tracker.on_notice,
                confirm_install=lambda action: action.component in approved_components,
                confirm=lambda _message: any(item in approved_components for item in lms_aliases),
            )
            outcome = await manager.fix_all(automatic=True)
            tracker.finish()
            await self.emit("setup.completed", outcome=json_value(outcome), log_path=str(log_path))
            return outcome
        if command == "fleet.status":
            return await self.services.fleet_status(root)
        if command == "fleet.combination.save":
            fleet = FleetConfig.model_validate(payload.get("fleet"))
            raw_name = payload.get("name")
            raw_id = payload.get("combination_id")
            combination = save_combination(
                root,
                fleet,
                name=raw_name if isinstance(raw_name, str) else None,
                combination_id=raw_id if isinstance(raw_id, str) and raw_id else None,
            )
            status = await self.services.fleet_status(root)
            await self.emit("fleet.combination.saved", combination=combination)
            return {"combination": combination, "status": status}
        if command == "fleet.combination.select":
            combination_id = _string(payload, "combination_id")
            fleet, combination = select_combination(root, combination_id)
            configured = await self.dispatch(
                "fleet.configure",
                {"root": str(root), "fleet": fleet.model_dump(mode="json"), "activate": False},
            )
            await self.emit("fleet.combination.selected", combination=combination)
            return {**configured, "combination": combination}
        if command == "fleet.combination.delete":
            combination_id = _string(payload, "combination_id")
            delete_combination(root, combination_id)
            status = await self.services.fleet_status(root)
            await self.emit("fleet.combination.deleted", combination_id=combination_id)
            return {"deleted": combination_id, "status": status}
        if command == "fleet.combination.unload":
            combination_id = _string(payload, "combination_id")
            unloaded = await self.services.unload_combination_models(root, combination_id)
            status = await self.services.fleet_status(root)
            await self.emit("fleet.changed", reason="combination.unload", unloaded=unloaded)
            return {"unloaded": unloaded, "status": status}
        if command == "model.download":
            if self.services.active_orchestrators:
                raise RuntimeError("A run is active. Stop it before downloading a model.")
            model_key = _string(payload, "model").strip()
            if not model_key or model_key.startswith("-") or len(model_key) > 300:
                raise ValueError("Enter a valid model name, without command-line options.")
            config = self.services.load_config(root)
            backend = config.inference.backend
            logger, log_path = create_setup_logger(root)
            last_emitted = 0.0

            def announce(message: str, percent: float | None = None) -> None:
                self.emit_nowait("setup.progress", message=message)
                detail: dict[str, object] = (
                    {"detail": message, "percent": percent}
                    if percent is not None
                    else _download_progress(message)
                )
                self.emit_nowait(
                    "model.download.progress", backend=backend, model=model_key, **detail
                )

            def report_bytes(progress: DownloadProgress) -> None:
                # A byte-accurate stream reports thousands of times a second. Throttling
                # here keeps the pipe to the interface readable without coarsening the
                # measurement itself.
                nonlocal last_emitted
                now = time.monotonic()
                finished = progress.total is not None and progress.completed >= progress.total
                if now - last_emitted < 0.25 and not finished:
                    return
                last_emitted = now
                detail = progress.detail
                if progress.total:
                    detail = (
                        f"{progress.detail} · {human_bytes(progress.completed)}"
                        f" of {human_bytes(progress.total)}"
                    )
                announce(detail, progress.percent)

            current_task = asyncio.current_task()
            if current_task is not None:
                self.active_downloads[model_key] = (current_task, None)
            await self.emit("model.download.started", backend=backend, model=model_key)
            try:
                if backend == "ollama":
                    # Held open by this process, so cancelling this task closes the
                    # connection and Ollama abandons the pull with it.
                    await pull_ollama_model(
                        model_key,
                        base_url=config.ollama.base_url,
                        on_progress=report_bytes,
                    )
                elif backend == "lmstudio":
                    # `lms get` only asks LM Studio's service to download; the service
                    # then keeps going whatever happens to the CLI, and to Adaptea. The
                    # models folder is a plain mirror of Hugging Face repositories, so
                    # filling it ourselves gets the same result under our own control.
                    repo_id, quantization = parse_repository(model_key)
                    model_directory = await download_repository(
                        repo_id,
                        lmstudio_models_root(),
                        quantization=quantization,
                        on_progress=report_bytes,
                    )
                    logger.info("downloaded %s into %s", repo_id, model_directory)
                else:
                    # The Hub CLI downloads in its own process rather than handing the
                    # job to a daemon, so terminating it does stop the transfer.
                    hf_backend = backend
                    downloaded_repo, llama_quantization = _huggingface_model_spec(
                        model_key, hf_backend
                    )
                    cache_directory = str(huggingface_cache_root(hf_backend))
                    hf_executable = _huggingface_cli(config.vllm.executable)
                    download_args: tuple[str, ...] = (
                        (
                            hf_executable,
                            "download",
                            downloaded_repo,
                            "--include",
                            f"*{llama_quantization or 'Q4_K_M'}*.gguf",
                            "--cache-dir",
                            cache_directory,
                        )
                        if backend == "llamacpp"
                        else (
                            hf_executable,
                            "download",
                            downloaded_repo,
                            "--cache-dir",
                            cache_directory,
                        )
                    )
                    runner = SetupCommandRunner(logger, announce, verbose=True)
                    if current_task is not None:
                        self.active_downloads[model_key] = (current_task, runner)
                    download_result = await runner.run(
                        *download_args, cwd=root, timeout=6 * 60 * 60
                    )
                    if download_result.returncode != 0:
                        detail_lines = (
                            (download_result.stderr or download_result.stdout).strip().splitlines()
                        )
                        raise DownloadError(
                            detail_lines[-1]
                            if detail_lines
                            else f"{download_args[0]} exited with {download_result.returncode}"
                        )
                    await self._verify_huggingface_cache(
                        downloaded_repo, hf_backend, llama_quantization
                    )
                if model_key in self.cancelled_downloads:
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                await self.emit("model.download.cancelled", backend=backend, model=model_key)
                raise
            except (DownloadError, ValueError) as exc:
                await self.emit(
                    "model.download.failed",
                    backend=backend,
                    model=model_key,
                    message=str(exc),
                )
                raise RuntimeError(f"Could not download {model_key}: {exc}") from exc
            finally:
                self.active_downloads.pop(model_key, None)
                self.cancelled_downloads.discard(model_key)
            status = await self.services.fleet_status(root)
            await self.emit(
                "model.download.completed",
                backend=backend,
                model=model_key,
                log_path=str(log_path),
            )
            return {"model": model_key, "status": status, "log_path": str(log_path)}
        if command == "model.delete":
            if self.services.active_orchestrators:
                raise RuntimeError("A run is active. Stop it before deleting a model.")
            model_key = _string(payload, "model").strip()
            status = await self.services.fleet_status(root)
            configured_rows = status.get("configured_models")
            loaded_rows = status.get("loaded_instances")
            combination_rows = status.get("combinations")
            configured_model_keys = (
                {item.get("model") for item in configured_rows if isinstance(item, dict)}
                if isinstance(configured_rows, list)
                else set()
            )
            loaded_model_keys = (
                {item.get("model_key") for item in loaded_rows if isinstance(item, dict)}
                if isinstance(loaded_rows, list)
                else set()
            )
            saved_model_keys: set[object] = set()
            if isinstance(combination_rows, list):
                for combination in combination_rows:
                    models = combination.get("models") if isinstance(combination, dict) else None
                    if isinstance(models, list):
                        saved_model_keys.update(
                            model.get("model") for model in models if isinstance(model, dict)
                        )
            if (
                model_key in configured_model_keys
                or model_key in loaded_model_keys
                or model_key in saved_model_keys
            ):
                raise RuntimeError(
                    "Remove this model from every active combination and unload it "
                    "before deleting its files."
                )
            config = self.services.load_config(root)
            if config.inference.backend == "ollama":
                result = await SetupCommandRunner(create_setup_logger(root)[0]).run(
                    config.ollama.executable, "rm", model_key, cwd=root, timeout=120
                )
                if result.returncode != 0:
                    delete_detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        f"Could not delete {model_key}: {delete_detail or 'ollama rm failed'}"
                    )
            elif config.inference.backend == "lmstudio":
                downloaded_rows = status.get("downloaded_models")
                row = (
                    next(
                        (
                            item
                            for item in downloaded_rows
                            if isinstance(item, dict) and item.get("model_key") == model_key
                        ),
                        None,
                    )
                    if isinstance(downloaded_rows, list)
                    else None
                )
                verified_path = resolve_lmstudio_model_path(
                    str(row.get("local_path")) if row and row.get("local_path") else None
                )
                source = Path(verified_path) if verified_path else None
                if source is None or not source.exists():
                    raise RuntimeError(
                        "LM Studio did not expose a verified local path for this model. Delete it from LM Studio's My Models screen."
                    )
                trash = environment_root() / ".trash" / "models"
                trash.mkdir(parents=True, exist_ok=True)
                destination = trash / f"{uuid.uuid4().hex[:10]}-{source.name}"
                shutil.move(str(source), destination)
            elif config.inference.backend in {"llamacpp", "vllm"}:
                downloaded_rows = status.get("downloaded_models")
                row = (
                    next(
                        (
                            item
                            for item in downloaded_rows
                            if isinstance(item, dict) and item.get("model_key") == model_key
                        ),
                        None,
                    )
                    if isinstance(downloaded_rows, list)
                    else None
                )
                verified_path = resolve_huggingface_model_path(
                    str(row.get("local_path")) if row and row.get("local_path") else None,
                    config.inference.backend,
                )
                source = Path(verified_path) if verified_path else None
                if source is None or not source.exists():
                    raise RuntimeError(
                        "Hugging Face did not expose a verified cache entry for this model."
                    )
                trash = environment_root() / ".trash" / "models"
                trash.mkdir(parents=True, exist_ok=True)
                destination = trash / f"{uuid.uuid4().hex[:10]}-{source.name}"
                shutil.move(str(source), destination)
            else:
                raise RuntimeError(
                    f"Deleting downloaded files is not managed for the {config.inference.backend} backend."
                )
            refreshed = await self.services.fleet_status(root)
            await self.emit("model.deleted", backend=config.inference.backend, model=model_key)
            return {"model": model_key, "status": refreshed}
        if command == "model.load":
            model_key = _string(payload, "model")
            raw_context = payload.get("context_length")
            context_length = int(raw_context) if isinstance(raw_context, int | float) else None
            instance = await self.services.load_model(root, model_key, context_length)
            status = await self.services.fleet_status(root)
            await self.emit("fleet.changed", reason="model.load", model=model_key)
            return {**instance, "status": status}
        if command == "model.unload":
            instance_id = _string(payload, "instance_id")
            await self.services.unload_model(root, instance_id)
            status = await self.services.fleet_status(root)
            await self.emit("fleet.changed", reason="model.unload", instance_id=instance_id)
            return {"instance_id": instance_id, "status": status}
        if command == "fleet.configure":
            fleet = FleetConfig.model_validate(payload.get("fleet"))
            unloaded = await self.services.unload_unconfigured_fleet_models(root, fleet)
            current = self.services.load_config(root)
            path = root / "adaptea.toml"
            backup_path = merge_fleet_config(path, fleet)
            planner = fleet.planner()
            if planner is not None:
                merge_inference_model(path, current.inference.backend, planner.model)
            elif not fleet.models:
                # An emptied fleet selects nothing at all. Leaving the previous planner
                # named here pointed the next run at a model this same save unloaded.
                merge_inference_model(path, current.inference.backend, None)
            opencode_path = None
            opencode_backup = None
            if fleet.models:
                opencode_path, opencode_backup = merge_opencode_models_config(
                    root,
                    current.worker.executable,
                    [item.model for item in fleet.models],
                    current.lmstudio.base_url,
                )
            # The desktop loads the chosen models itself, one at a time, so it can show
            # which one it is on. Activating here as well would load the whole set inside
            # a single silent call and leave the page with nothing to report.
            activate = payload.get("activate") is not False
            loaded, load_error = (
                await self.services.activate_configured_fleet(root) if activate else (False, None)
            )
            status = await self.services.fleet_status(root)
            await self.emit(
                "fleet.configured",
                path=str(path),
                backup=str(backup_path) if backup_path else None,
                loaded=loaded,
                load_error=load_error,
            )
            return {
                "path": str(path),
                "backup": str(backup_path) if backup_path else None,
                "fleet": fleet,
                "unloaded": unloaded,
                "opencode_path": str(opencode_path) if opencode_path else None,
                "opencode_backup": str(opencode_backup) if opencode_backup else None,
                "loaded": loaded,
                "load_error": load_error,
                "status": status,
            }
        if command == "fleet.reset":
            if self.services.active_orchestrators:
                raise RuntimeError("A run is active. Stop it before resetting the model fleet.")
            current = self.services.load_config(root)
            unloaded = await self.services.unload_selected_models(root)
            empty = FleetConfig(enabled=False)
            path = root / "adaptea.toml"
            backup_path = merge_fleet_config(path, empty)
            merge_inference_model(path, current.inference.backend, None)
            clear_active_combination(root)
            status = await self.services.fleet_status(root)
            await self.emit("fleet.configured", reason="reset", unloaded=unloaded)
            return {
                "path": str(path),
                "backup": str(backup_path) if backup_path else None,
                "unloaded": unloaded,
                "status": status,
            }
        if command == "plan.generate":
            goal = _string(payload, "goal")
            # Which conversation asked. The desktop app can hold several chats on the
            # same folder, so the project path alone cannot say whose narration this is;
            # echoing the caller's own identifier can. It is opaque to the core.
            chat = payload.get("chat")
            chat_id = chat if isinstance(chat, str) and chat else None
            # A follow-up message revises the plan on screen instead of discarding it.
            raw_previous = payload.get("previous")
            previous = Plan.model_validate(raw_previous) if isinstance(raw_previous, dict) else None
            await self.emit(
                "plan.started",
                root=str(root),
                chat=chat_id,
                goal=goal,
                revision=previous is not None,
            )
            await self.emit(
                "plan.progress",
                root=str(root),
                chat=chat_id,
                message="Preparing the selected planner model…",
            )
            # The planner reports each tool call it makes. Elapsed time alone said only
            # that the process had not exited, which the user could already see.
            observed: list[str] = []
            loop = asyncio.get_running_loop()

            def report_activity(activity: Activity) -> None:
                observed.append(activity.text)
                loop.create_task(
                    self.emit(
                        "plan.progress",
                        root=str(root),
                        chat=chat_id,
                        message=activity.text,
                        kind=activity.kind,
                        tool=activity.tool,
                        step=len(observed),
                        total_tokens=activity.total_tokens,
                    )
                )

            planning = asyncio.create_task(
                self.services.plan(root, goal, report_activity, previous)
            )
            elapsed = 0
            try:
                while not planning.done():
                    done, _pending = await asyncio.wait({planning}, timeout=8)
                    if done:
                        break
                    elapsed += 8
                    # Only fall back to elapsed time while the planner has reported nothing;
                    # otherwise it would overwrite the activity the user is reading.
                    if not observed:
                        await self.emit(
                            "plan.progress",
                            root=str(root),
                            chat=chat_id,
                            message=f"Planner is still working · {elapsed}s elapsed",
                        )
                plan = await planning
            finally:
                if not planning.done():
                    planning.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await planning
            await self.emit("plan.completed", root=str(root), chat=chat_id, plan=json_value(plan))
            return plan
        if command == "plan.save":
            plan = Plan.model_validate(payload.get("plan"))
            return {"path": str(self.services.save_plan(root, plan))}
        if command == "run.create":
            plan = Plan.model_validate(payload.get("plan"))
            mode = _mode(payload)
            maximum = _integer(payload, "max_agents", minimum=1, default=8)
            fixed = 1 if payload.get("mode") == "serial" else payload.get("concurrency")
            concurrency = int(fixed) if isinstance(fixed, int) else None
            state = await self.services.create_run(root, plan, mode, maximum, concurrency)
            await self.emit("run.created", state=json_value(state))
            return state
        if command == "run.start":
            state = _load_run(root, _string(payload, "run_id"))
            return await self._execute_run(root, state)
        if command == "run.resume":
            run_id = _string(payload, "run_id")
            # Resuming an interrupted run recovers work that was in flight. Retrying a
            # finished one is a different request the user has to make deliberately: it
            # renews a retry budget that a policy decided was spent.
            retry_failed = bool(payload.get("retry_failed", False))

            def callback(state: RunState, sample: object, running: int) -> None:
                self._runtime_event(state, sample, running)

            def activity(task_id: str, entry: dict[str, str]) -> None:
                self.emit_nowait(
                    "task.activity",
                    run_id=run_id,
                    task_id=task_id,
                    kind=entry.get("kind", "event"),
                    title=entry.get("title", "step"),
                    detail=entry.get("detail", ""),
                )

            await self.emit("run.resuming", run_id=run_id, retry_failed=retry_failed)
            state = await self.services.resume_run(
                root,
                run_id,
                callback,
                activity,
                self._announce_command_approval,
                retry_failed=retry_failed,
            )
            return await self._finalize_run(root, state)
        if command == "run.status":
            return _load_run(root, _string(payload, "run_id"))
        if command == "task.artifacts":
            state = _load_run(root, _string(payload, "run_id"))
            task_id = _string(payload, "task_id")
            runtime = state.tasks.get(task_id)
            if runtime is None:
                raise ValueError(f"Task not found: {task_id}")
            if runtime.attempts < 1:
                return {"attempt": 0, "output": "", "stderr": "", "validation": ""}
            artifact_dir = (
                root
                / ".adaptea"
                / "runs"
                / state.run_id
                / "tasks"
                / task_id
                / f"attempt-{runtime.attempts}"
            )
            return {
                "attempt": runtime.attempts,
                "output": _tail_file(artifact_dir / "stdout.log"),
                "stderr": _tail_file(artifact_dir / "stderr.log"),
                "validation": _tail_file(artifact_dir / "validation.log"),
                "merge_conflict": _tail_file(artifact_dir / "merge-conflict.log"),
                "failure_decisions": _tail_file(artifact_dir / "failure-decisions.jsonl"),
            }
        if command == "run.pause_admissions":
            return {
                "changed": self.services.set_admission(_string(payload, "run_id"), enabled=False)
            }
        if command == "run.resume_admissions":
            return {
                "changed": self.services.set_admission(_string(payload, "run_id"), enabled=True)
            }
        if command == "command.approvals.list":
            requested = payload.get("run_id")
            return {
                "approvals": self.services.pending_command_approvals(
                    requested if isinstance(requested, str) and requested else None
                )
            }
        if command == "command.approval.resolve":
            request_id = _string(payload, "request_id")
            decision = payload.get("decision")
            if decision not in ("once", "always", "deny"):
                raise ValueError("decision must be one of: once, always, deny")
            requested_run = payload.get("run_id")
            answer = self.services.resolve_command_approval(
                request_id,
                cast(ApprovalDecision, decision),
                requested_run if isinstance(requested_run, str) and requested_run else None,
            )
            await self.emit("command.approval_resolved", **answer)
            return answer
        if command in ("models.stop", "models.unload"):
            unloaded = await self.services.stop_models(root, force=True)
            await self.emit("models.stopped", unloaded=unloaded)
            return {"unloaded": unloaded}
        if command == "run.abort":
            run_id = _string(payload, "run_id")
            # Stopping a run stops the work, not the models. Reloading a local model
            # costs minutes and the user almost always stops in order to start again,
            # so unloading happens only when it is asked for — by this call or by the
            # project's own preference.
            requested = payload.get("stop_models")
            stop_models_requested = (
                bool(requested)
                if requested is not None
                else load_project_preferences(root).stop_models_on_stop
            )
            changed = self.services.abort_run(run_id)
            unloaded_models: list[str] = []
            if stop_models_requested:
                unloaded_models = await self.services.stop_models(root, force=True)
                if unloaded_models:
                    await self.emit("models.stopped", unloaded=unloaded_models)
            await self.emit(
                "run.aborted",
                run_id=run_id,
                changed=changed,
                models_unloaded=unloaded_models,
            )
            return {
                "changed": changed,
                "unloaded": unloaded_models,
                "models_kept_loaded": not stop_models_requested,
            }
        if command == "runs.list":
            paths = [
                Path(item) for item in payload.get("projects", [str(root)]) if isinstance(item, str)
            ]
            return list_runs(paths)
        if command == "run.delete_metadata":
            run_id = _string(payload, "run_id")
            record = next((item for item in list_runs([root]) if item.run_id == run_id), None)
            if record is None:
                raise ValueError(f"Run not found: {run_id}")
            delete_run_metadata(record)
            return {"deleted": run_id, "git_cleanup": False}
        if command == "run.apply":
            run_id = _string(payload, "run_id")
            state = _load_run(root, run_id)
            return await self._apply_run(root, state)
        if command == "calibration.start":
            quick = bool(payload.get("quick", False))
            # The desktop keeps its own agent limit, so the sweep measures the ceiling the
            # user actually set rather than the one written in adaptea.toml.
            requested = payload.get("max_agents")
            ceiling = int(requested) if isinstance(requested, int) and requested > 0 else None
            await self.emit("calibration.started", quick=quick, max_agents=ceiling)
            directory = await self.services.calibrate(
                root,
                quick=quick,
                max_agents=ceiling,
                progress=lambda message: self.emit_nowait("calibration.sample", message=message),
            )
            if active_report(root) is not None:
                _directory, completed_report = read_report(root, directory)
                capture_active_capacity(root, completed_report)
            await self.emit("calibration.completed", directory=str(directory))
            return {"directory": str(directory)}
        if command == "fleet.calibration.start":
            await self.emit("calibration.started", kind="fleet")
            directory = await self.services.calibrate_fleet(
                root,
                repetitions=_integer(payload, "repetitions", minimum=1, default=3),
                agent_validation=bool(payload.get("agent_validation", True)),
                progress=lambda message: self.emit_nowait("calibration.sample", message=message),
            )
            await self.emit("calibration.completed", kind="fleet", directory=str(directory))
            return {"directory": str(directory)}
        if command == "smoke.start":
            smoke_result = await run_mvp_smoke_test(
                root,
                progress=lambda step: self.emit_nowait("smoke.step", step=json_value(step)),
                activity=lambda message: self.emit_nowait("smoke.progress", message=message),
            )
            return smoke_result
        if command == "report.get":
            raw_directory = payload.get("directory")
            if not isinstance(raw_directory, str):
                active_combination_report = active_report(root)
                if active_combination_report is not None:
                    combination_id, combination_report = active_combination_report
                    if combination_report is None:
                        raise FileNotFoundError(
                            "the selected model combination has not been measured yet"
                        )
                    return {
                        "directory": str(combinations_path(root)),
                        "combination_id": combination_id,
                        "report": combination_report,
                    }
            directory, value = read_report(
                root, Path(raw_directory) if isinstance(raw_directory, str) else None
            )
            return {"directory": str(directory), "report": value}
        if command == "settings.get":
            return self.services.load_config(root)
        raise ValueError(f"Unsupported desktop command: {command}")

    def _announce_command_approval(self, request: ApprovalRequest) -> None:
        """Ask the desktop about a command the worker policy refused.

        The run keeps going while the question is open. An answer applies to the task's
        next attempt, which is what makes an approval worth asking for at all.
        """
        self.emit_nowait("command.approval_requested", **request.as_dict())

    async def _execute_run(self, root: Path, state: RunState) -> RunState:
        await self.emit("run.started", state=json_value(state))

        def callback(current: RunState, sample: object, running: int) -> None:
            self._runtime_event(current, sample, running)

        def activity(task_id: str, entry: dict[str, str]) -> None:
            self.emit_nowait(
                "task.activity",
                run_id=state.run_id,
                task_id=task_id,
                kind=entry.get("kind", "event"),
                title=entry.get("title", "step"),
                detail=entry.get("detail", ""),
            )

        final = await self.services.execute_run(
            root, state, callback, activity, self._announce_command_approval
        )
        return await self._finalize_run(root, final)

    async def _finalize_run(self, root: Path, final: RunState) -> RunState:
        all_merged = bool(final.tasks) and all(
            task.status == TaskStatus.MERGED for task in final.tasks.values()
        )
        preferences = load_project_preferences(root)
        if all_merged and preferences.completion_action != "none":
            try:
                if not final.source_branch:
                    raise ValueError(
                        "This run does not record its starting branch. Apply it manually after "
                        "reviewing the current branch."
                    )
                applied = await self._apply_run(root, final, expected_branch=final.source_branch)
            except ValueError as exc:
                await self.emit(
                    "run.auto_commit_failed",
                    run_id=final.run_id,
                    integration_branch=final.integration_branch,
                    message=str(exc),
                )
            else:
                await self.emit("run.auto_committed", run_id=final.run_id, **applied)
                if preferences.completion_action == "commit_and_push":
                    try:
                        pushed = await self._push_applied_run(root, final)
                    except ValueError as exc:
                        await self.emit(
                            "run.auto_push_failed",
                            run_id=final.run_id,
                            branch=final.applied_branch,
                            commit=final.applied_commit,
                            message=str(exc),
                        )
                    else:
                        await self.emit("run.auto_pushed", run_id=final.run_id, **pushed)
        if preferences.stop_models_on_finish:
            unloaded = await self.services.stop_models(root)
            if unloaded:
                await self.emit("models.stopped", unloaded=unloaded)
        name = "run.completed" if all_merged else "run.failed"
        await self.emit(name, state=json_value(final))
        return final

    async def _apply_run(
        self, root: Path, state: RunState, *, expected_branch: str | None = None
    ) -> dict[str, Any]:
        from adaptea.git.integration import create_squashed_commit
        from adaptea.git.repository import git

        if not state.tasks or any(
            task.status != TaskStatus.MERGED for task in state.tasks.values()
        ):
            raise ValueError("Only a run whose every task merged successfully can be applied.")
        # Read status to distinguish a missing/broken repository from a merge failure, but
        # do not reject every dirty workspace. Git's fast-forward merge safely preserves
        # unrelated tracked/untracked files and refuses paths that the run would overwrite.
        # The blanket clean-tree requirement stranded reviewed code on an internal branch
        # because harmless files such as .DS_Store or a new package-lock.json were present.
        workspace_status = await git(root, "status", "--porcelain", check=False)
        if workspace_status.returncode != 0:
            raise ValueError("The current workspace Git status could not be read.")
        branch = await git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
        if branch.returncode != 0:
            raise ValueError(
                "The current workspace is on a detached HEAD; switch to a branch first."
            )
        branch_name = branch.stdout.strip()
        if expected_branch is not None and branch_name != expected_branch:
            raise ValueError(
                f"This run started on {expected_branch}, but the workspace is now on "
                f"{branch_name}. Switch back or apply the integration branch manually."
            )
        parent_commit = (await git(root, "rev-parse", "HEAD")).stdout.strip()
        goal = " ".join(state.goal.split())
        subject_limit = 72 - len("adaptea: ")
        if len(goal) > subject_limit:
            goal = goal[: subject_limit - 3].rstrip() + "..."
        message = f"adaptea: {goal}\n\nRun: {state.run_id}\nTasks: {len(state.tasks)} completed"
        try:
            completed_commit = await create_squashed_commit(
                root,
                state.integration_branch,
                parent_commit,
                message,
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        result = await git(root, "merge", "--ff-only", completed_commit, check=False)
        if result.returncode != 0:
            git_detail = (result.stderr or result.stdout).strip()
            raise ValueError(
                f"Could not apply the completed run to {branch_name}. "
                "Local changes that do not overlap the run are preserved; commit, move, or "
                "stash any paths Git reports as overlapping, then try again. "
                f"Git output: {git_detail}"
            )
        commit = (await git(root, "rev-parse", "HEAD")).stdout.strip()
        state.applied_branch = branch_name
        state.applied_commit = commit
        state.applied_at = utc_now()
        StateStore(root / ".adaptea" / "runs" / state.run_id).save(state)
        return {"applied": True, "branch": branch_name, "commit": commit}

    async def _push_applied_run(self, root: Path, state: RunState) -> dict[str, Any]:
        """Push exactly the reviewed local commit to ``origin`` without force.

        The refspec names the immutable commit rather than an ambient checkout. A remote
        branch that advanced independently rejects the push normally, leaving the local
        commit and the remote untouched for the user to reconcile.
        """
        from adaptea.git.repository import git

        if not state.applied_branch or not state.applied_commit:
            raise ValueError("The completed run has not been committed locally yet.")
        remote = await git(root, "remote", "get-url", "origin", check=False)
        if remote.returncode != 0 or not remote.stdout.strip():
            raise ValueError(
                "No origin remote is configured. The completed work is committed locally; "
                "add an origin remote and push it when ready."
            )
        result = await git(
            root,
            "push",
            "origin",
            f"{state.applied_commit}:refs/heads/{state.applied_branch}",
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ValueError(
                "Origin rejected the non-force push. The completed work remains committed "
                f"locally. Git output: {detail}"
            )
        state.pushed_remote = "origin"
        state.pushed_branch = state.applied_branch
        state.pushed_at = utc_now()
        StateStore(root / ".adaptea" / "runs" / state.run_id).save(state)
        return {
            "remote": state.pushed_remote,
            "branch": state.pushed_branch,
            "commit": state.applied_commit,
        }

    def _runtime_event(self, state: RunState, sample: object, running: int) -> None:
        previous_tasks = self.last_task_status.setdefault(state.run_id, {})
        for task_id, task in state.tasks.items():
            current = str(task.status)
            previous = previous_tasks.get(task_id)
            if previous is not None and previous != current:
                self.emit_nowait(
                    "task.status_changed",
                    run_id=state.run_id,
                    task_id=task_id,
                    previous=previous,
                    status=current,
                    task=json_value(task),
                )
            previous_tasks[task_id] = current
        previous_target = self.last_target.get(state.run_id)
        if previous_target is not None and previous_target != state.target_concurrency:
            decision = _latest_jsonl(
                Path(state.repository)
                / ".adaptea"
                / "runs"
                / state.run_id
                / "controller-decisions.jsonl"
            )
            self.emit_nowait(
                "controller.target_changed",
                run_id=state.run_id,
                previous=previous_target,
                target=state.target_concurrency,
                reason=decision.get("reason", "Controller target adjusted"),
                decision=decision,
            )
        self.last_target[state.run_id] = state.target_concurrency
        self.emit_nowait(
            "run.updated",
            state=json_value(state),
            telemetry=json_value(sample),
            running=running,
        )

    async def serve(self, input_stream: TextIO | None = None) -> None:
        source = input_stream or sys.stdin
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[str | None] = asyncio.Queue()
        installed_signals: list[signal.Signals] = []

        def request_shutdown() -> None:
            # Tauri closes the core with SIGTERM. Turn that into the same orderly path as
            # stdin EOF so active download runners get interrupted before Python exits.
            self.closing = True
            lines.put_nowait(None)

        for shutdown_signal in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(shutdown_signal, request_shutdown)
                installed_signals.append(shutdown_signal)
            except (NotImplementedError, RuntimeError):
                pass

        def read_input() -> None:
            while True:
                line = source.readline()
                loop.call_soon_threadsafe(lines.put_nowait, line or None)
                if not line:
                    return

        threading.Thread(target=read_input, name="adaptea-desktop-input", daemon=True).start()
        await self.emit("bridge.ready", pid=os.getpid(), version=__version__)
        try:
            while not self.closing:
                try:
                    line = await asyncio.wait_for(lines.get(), timeout=0.25)
                except TimeoutError:
                    continue
                if line is None:
                    break
                if line.strip():
                    await self.accept(line)
        finally:
            self.services.abort_all()
            await self.cancel_downloads()
            pending = [task for task in self.requests.values() if not task.done()]
            if pending:
                for task in pending:
                    task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=3.0,
                    )
                except TimeoutError:
                    pass
            for shutdown_signal in installed_signals:
                loop.remove_signal_handler(shutdown_signal)


#: Commands that describe the machine rather than a repository, and so may arrive with
#: no project open at all.
#: Commands whose subject is the machine's inference environment itself. They always
#: run against the environment directory, whatever project is open, so one set of models
#: and one measurement serve every project.
_PERCENT = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")


class SetupProgressTracker:
    """Tracks setup and installation steps with live percentages and command details."""

    def __init__(
        self,
        approved_components: set[str],
        on_emit: Callable[..., None],
    ) -> None:
        lms_names = {
            "lms",
            "LM Studio",
            "LM Studio CLI",
            "LM Studio llmster",
            "llmster",
            "lms CLI",
            "LM Studio Desktop",
        }
        self.approved_components = approved_components
        self.on_emit = on_emit
        is_lms_only = bool(approved_components) and approved_components.issubset(lms_names)
        self.is_single_component = len(approved_components) == 1 or is_lms_only
        if is_lms_only:
            single_name = "lms CLI"
        elif self.is_single_component:
            single_name = next(iter(approved_components))
        else:
            single_name = "System"
        self.current_component = single_name
        self.current_stage = "preparing"
        self.step = 1
        self.total_steps = 4 if self.is_single_component else 5
        self.percent: float | None = 5.0
        self.last_message = (
            f"Starting {single_name} setup…"
            if self.is_single_component
            else "Starting environment setup…"
        )
        self.last_detail = ""

    def on_notice(self, message: str) -> None:
        self.last_message = message
        lower = message.lower()
        if "inspecting" in lower:
            self.current_component = "Toolchain"
            self.current_stage = "checking"
            self.step = 1
            self.percent = 10.0
        elif re.search(r"\bgit\b", lower):
            self.current_component = "Git"
            self.current_stage = "checking" if "checking" in lower else "installing"
            self.step = 2 if not self.is_single_component else 1
            self.percent = 20.0
        elif any(k in lower for k in ("lms", "llmster", "lm studio")):
            self.current_component = (
                "lms CLI" if any(k in lower for k in ("lms", "llmster")) else "LM Studio"
            )
            if any(k in lower for k in ("downloading", "running:", "installing", "bootstrap")):
                self.current_stage = "downloading"
                self.percent = 50.0
            else:
                self.current_stage = "checking"
                self.percent = 35.0
            self.step = 2 if self.is_single_component else 3
        elif any(k in lower for k in ("ollama", "llama.cpp", "vllm", "runtime")):
            self.current_component = "Runtime"
            self.current_stage = "starting" if "starting" in lower else "checking"
            self.step = 3 if not self.is_single_component else 2
            self.percent = 40.0
        elif "opencode" in lower:
            self.current_component = "OpenCode"
            if any(k in lower for k in ("downloading", "running:", "installing")):
                self.current_stage = "downloading"
                self.percent = 50.0
            else:
                self.current_stage = "checking"
                self.percent = 35.0
            self.step = 2 if self.is_single_component else 4
        elif any(k in lower for k in ("writing project-local", "configuration", "configured")):
            self.current_component = "Configuration"
            self.current_stage = "configuring"
            self.step = 3 if self.is_single_component else 5
            self.percent = 85.0
        elif any(k in lower for k in ("finished", "completed", "ready")):
            self.current_stage = "completed"
            self.step = self.total_steps
            self.percent = 100.0
        self.emit()

    def on_runner_output(self, line: str) -> None:
        self.last_detail = line
        match = _PERCENT.search(line)
        if match:
            parsed = float(match.group(1))
            if self.is_single_component:
                self.percent = min(90.0, max(20.0, 20.0 + (parsed * 0.7)))
            else:
                self.percent = min(85.0, max(45.0, 45.0 + (parsed * 0.4)))
        else:
            lower = line.lower()
            if any(k in lower for k in ("downloading", "fetch", "http", "curl")):
                if self.percent is not None and self.percent < 65.0:
                    self.percent = min(65.0, self.percent + 2.0)
            elif any(
                k in lower for k in ("installing", "unpacking", "linking", "pour", "brew", "npm")
            ):
                if self.percent is not None and self.percent < 85.0:
                    self.percent = min(85.0, max(65.0, self.percent + 2.0))
        self.emit()

    def finish(self) -> None:
        self.current_stage = "completed"
        self.step = self.total_steps
        self.percent = 100.0
        self.last_message = "Setup complete."
        self.emit()

    def emit(self) -> None:
        self.on_emit(
            "setup.progress",
            message=self.last_message,
            detail=self.last_detail or self.last_message,
            percent=round(self.percent, 1) if self.percent is not None else None,
            component=self.current_component,
            stage=self.current_stage,
            step=self.step,
            total_steps=self.total_steps,
        )


def _download_progress(message: str) -> dict[str, object]:
    """Turn the stable part of CLI output into provider-neutral download progress."""
    match = _PERCENT.search(message)
    percent = min(100.0, max(0.0, float(match.group(1)))) if match else None
    return {"detail": message, "percent": percent}


_HF_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_QUANTIZATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _huggingface_model_spec(
    value: str, backend: Literal["llamacpp", "vllm"]
) -> tuple[str, str | None]:
    """Validate a Hub repo and llama.cpp's optional `:quantization` suffix."""
    repo_id = value
    quantization: str | None = None
    if backend == "llamacpp" and ":" in value:
        repo_id, quantization = value.rsplit(":", 1)
    if not _HF_REPO.fullmatch(repo_id):
        raise ValueError("Enter a Hugging Face model ID in owner/repository format.")
    if quantization is not None and not _HF_QUANTIZATION.fullmatch(quantization):
        raise ValueError("Enter a quantization such as Q4_K_M after the model ID.")
    return repo_id, quantization


def _huggingface_cli(vllm_executable: str) -> str:
    """Find the current or legacy Hub CLI, including one beside a vLLM executable."""
    resolved_vllm = Path(resolve_executable(vllm_executable))
    suffix = ".exe" if os.name == "nt" else ""
    if resolved_vllm.is_file():
        for name in (f"hf{suffix}", f"huggingface-cli{suffix}"):
            sibling = resolved_vllm.with_name(name)
            if sibling.is_file():
                return str(sibling)
    for name in ("hf", "huggingface-cli"):
        resolved = resolve_executable(name)
        if Path(resolved).is_file() or shutil.which(resolved):
            return resolved
    raise RuntimeError(
        "Hugging Face CLI (`hf`) is required for llama.cpp and vLLM model downloads. "
        "Install the `huggingface_hub` package, then retry."
    )


GLOBAL_ENVIRONMENT_COMMANDS = frozenset(
    {
        "calibration.start",
        "fleet.combination.delete",
        "fleet.combination.save",
        "fleet.combination.select",
        "fleet.combination.unload",
        "fleet.configure",
        "fleet.reset",
        "fleet.status",
        "inference.config",
        "inference.configure",
        "inference.health",
        "lmstudio.health",
        "model.download",
        "model.download.cancel",
        "model.delete",
        "model.load",
        "model.unload",
        "report.get",
    }
)

ENVIRONMENT_COMMANDS = frozenset(
    {
        # `environment.apply` names the project it is writing into and resolves that root
        # itself; it is listed so the shared resolution above does not reject it first.
        "environment.apply",
        "environment.status",
        "doctor",
        "fleet.status",
        "fleet.configure",
        "fleet.combination.delete",
        "fleet.combination.save",
        "fleet.combination.select",
        "fleet.combination.unload",
        "fleet.reset",
        "inference.config",
        "inference.configure",
        "inference.health",
        "lmstudio.health",
        "model.download",
        "model.download.cancel",
        "model.delete",
        "model.load",
        "model.unload",
        "report.get",
        "setup.diagnose",
        "setup.repair_safe",
        "calibration.start",
        # A command approval belongs to a live run, which already knows its project. The
        # desktop asks for these before any folder is open — on reconnect, to recover a
        # question a durable run is still waiting on — so they must not require a root.
        "command.approvals.list",
        "command.approval.resolve",
    }
)


def _root(payload: dict[str, Any]) -> Path:
    value = payload.get("root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A project root is required.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Project folder does not exist: {root}")
    return root


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required.")
    return value.strip()


def _integer(payload: dict[str, Any], key: str, *, minimum: int, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}.")
    return value


def _mode(payload: dict[str, Any]) -> Literal["adaptive", "fixed", "naive"]:
    value = payload.get("mode", "adaptive")
    if value == "serial":
        return "fixed"
    if value not in {"adaptive", "fixed", "naive"}:
        raise ValueError("mode must be adaptive, serial, fixed, or naive.")
    return cast(Literal["adaptive", "fixed", "naive"], value)


def _load_run(root: Path, run_id: str) -> RunState:
    path = root / ".adaptea" / "runs" / run_id
    store = StateStore(path)
    if not store.path.is_file():
        raise ValueError(f"Run not found: {run_id}")
    return store.load()


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_error"
    if isinstance(exc, ValueError):
        return "invalid_argument"
    return "core_error"


def _concise_error(exc: Exception) -> str:
    return (str(exc).strip() or exc.__class__.__name__).splitlines()[0][:500]


def _tail_file(path: Path, limit: int = 120_000) -> str:
    """Read a bounded UTF-8 tail from a known run artifact path."""
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read(limit).decode(errors="replace")


def _latest_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    demo = "--demo" in sys.argv[1:]
    asyncio.run(DesktopBridge(demo=demo).serve())


if __name__ == "__main__":
    main()
