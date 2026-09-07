from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import Config
from adaptea.diagnostics.system import find_lms_executable
from adaptea.lmstudio.lms_cli import CommandResult
from adaptea.setup import manager as manager_module
from adaptea.setup.manager import SetupManager


class RecordingRunner:
    """Captures what the manager asked the machine to do, run or spawn."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.spawned: list[tuple[str, ...]] = []

    async def run(self, *args: str, **_kwargs: Any) -> CommandResult:
        self.commands.append(args)
        return CommandResult(0, "", "")

    async def spawn(self, *args: str, **_kwargs: Any) -> bool:
        self.spawned.append(args)
        return True


class StubDiagnostics:
    """A machine where nothing the setup controls needs is on ``PATH``."""

    def __init__(self, system: str, path_entries: dict[str, str] | None = None) -> None:
        self.system = system
        self._path = path_entries or {}

    def which(self, name: str) -> str | None:
        return self._path.get(name)


def manager_for(
    tmp_path: Path,
    system: str,
    *,
    backend: str = "lmstudio",
    path_entries: dict[str, str] | None = None,
) -> tuple[SetupManager, RecordingRunner]:
    config = Config()
    config.inference.backend = backend  # type: ignore[assignment]
    runner = RecordingRunner()
    manager = SetupManager(
        tmp_path,
        config,
        diagnostics=StubDiagnostics(system, path_entries),  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
        logger=logging.getLogger(f"test.server-control.{id(tmp_path)}"),
    )
    return manager, runner


class StubBackendClient:
    def __init__(self, reachable: Callable[[], bool]) -> None:
        self._reachable = reachable

    async def __aenter__(self) -> StubBackendClient:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def models(self) -> list[str]:
        if not self._reachable():
            raise RuntimeError("connection refused")
        return []


def patch_endpoint(monkeypatch: pytest.MonkeyPatch, reachable: Callable[[], bool]) -> None:
    monkeypatch.setattr(
        manager_module,
        "create_inference_backend",
        lambda _config, timeout=None: StubBackendClient(reachable),
    )

    async def instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(manager_module.asyncio, "sleep", instant)


def test_find_lms_executable_falls_back_to_the_bundled_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / ".lmstudio" / "bin" / "lms"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_lms_executable("lms", "Darwin", lambda _name: None) == str(bundled)


def test_find_lms_executable_prefers_the_path_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    assert find_lms_executable("lms", "Darwin", lambda _name: "/usr/local/bin/lms") == (
        "/usr/local/bin/lms"
    )


@pytest.mark.asyncio
async def test_stop_server_uses_the_lms_cli_that_is_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Finder-launched app has no ``~/.lmstudio/bin`` on PATH; Stop still has to work."""
    bundled = tmp_path / ".lmstudio" / "bin" / "lms"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    manager, runner = manager_for(tmp_path, "Darwin")
    patch_endpoint(monkeypatch, lambda: False)

    assert await manager.stop_server() is True
    assert (str(bundled), "server", "stop") in runner.commands
    assert (str(bundled), "daemon", "down") in runner.commands


@pytest.mark.asyncio
async def test_stop_server_reports_failure_while_the_endpoint_still_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _runner = manager_for(
        tmp_path, "Darwin", backend="ollama", path_entries={"ollama": "/usr/local/bin/ollama"}
    )
    patch_endpoint(monkeypatch, lambda: True)

    assert await manager.stop_server() is False


@pytest.mark.asyncio
async def test_stop_server_quits_the_ollama_app_before_killing_its_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The menu-bar app respawns the server, and its process name is capitalised."""
    manager, runner = manager_for(
        tmp_path, "Darwin", backend="ollama", path_entries={"ollama": "/usr/local/bin/ollama"}
    )
    patch_endpoint(monkeypatch, lambda: False)

    assert await manager.stop_server() is True
    joined = [" ".join(command) for command in runner.commands]
    assert any("quit app" in command for command in joined)
    assert any(command.startswith("pkill -ix ollama") for command in joined)


@pytest.mark.asyncio
async def test_start_ollama_server_launches_the_daemon_without_a_desktop_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux nothing used to launch anything; Start only waited and then failed."""
    manager, runner = manager_for(
        tmp_path, "Linux", backend="ollama", path_entries={"ollama": "/usr/bin/ollama"}
    )
    patch_endpoint(monkeypatch, lambda: True)

    assert await manager.start_ollama_server() is True
    assert runner.spawned == [("/usr/bin/ollama", "serve")]


@pytest.mark.asyncio
async def test_start_backend_server_dispatches_on_the_configured_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / ".lmstudio" / "bin" / "lms"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    manager, runner = manager_for(tmp_path, "Darwin")
    patch_endpoint(monkeypatch, lambda: True)

    assert await manager.start_backend_server() is True
    assert runner.commands[0][:3] == (str(bundled), "server", "start")


@pytest.mark.asyncio
async def test_restart_server_stops_then_starts_the_same_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / ".lmstudio" / "bin" / "lms"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    manager, runner = manager_for(tmp_path, "Darwin")
    reachable = [False]
    patch_endpoint(monkeypatch, lambda: reachable[0])

    assert await manager.restart_server() is False  # endpoint never comes back
    joined = [" ".join(command) for command in runner.commands]
    assert joined.index(f"{bundled} server stop") < joined.index(
        f"{bundled} server start --port 1234 --bind 127.0.0.1"
    )
