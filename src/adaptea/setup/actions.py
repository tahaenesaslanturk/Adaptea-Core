from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shutil
import signal
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptea.lmstudio.lms_cli import CommandResult, normalize_subprocess_command

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True, slots=True)
class InstallAction:
    component: str
    command: tuple[str, ...]
    display_command: str
    source: str
    explanation: str
    remote_script: bool = False


class SetupCommandRunner:
    def __init__(
        self,
        logger: logging.Logger,
        output: Callable[[str], None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.logger = logger
        self.output = output or (lambda _line: None)
        self.verbose = verbose
        self._process: asyncio.subprocess.Process | None = None

    async def terminate(self) -> None:
        """Terminate the running subprocess and its process tree, if active."""
        if self._process is not None:
            await self._stop_process(self._process)

    @staticmethod
    async def _stop_process(
        process: asyncio.subprocess.Process, grace_seconds: float = 3.0
    ) -> None:
        if process.returncode is not None:
            return
        # Download clients use Ctrl-C/SIGINT to close their streaming request cleanly.
        # Going straight to SIGTERM can kill only the CLI monitor while a daemon-backed
        # transfer (notably LM Studio or Ollama) keeps running in the background.
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGINT)
                except (ProcessLookupError, PermissionError):
                    process.send_signal(signal.SIGINT)
            elif platform.system() == "Windows":
                process.send_signal(vars(signal).get("CTRL_BREAK_EVENT", signal.SIGTERM))
            else:
                process.send_signal(signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return

        try:
            await asyncio.wait_for(process.wait(), grace_seconds)
            return
        except (TimeoutError, asyncio.CancelledError):
            pass

        # A client that does not honor the interrupt gets a short TERM window before the
        # final hard kill. Every stage targets the whole group/tree, not just its parent.
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            elif platform.system() == "Windows":
                with suppress(Exception):
                    import subprocess as win_subprocess

                    win_subprocess.run(
                        ["taskkill", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                        check=False,
                    )
                process.terminate()
            else:
                process.terminate()
            await asyncio.wait_for(process.wait(), 1.0)
            return
        except (ProcessLookupError, PermissionError):
            return
        except (TimeoutError, asyncio.CancelledError):
            with suppress(Exception):
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                else:
                    process.kill()
            with suppress(Exception):
                await asyncio.wait_for(process.wait(), 1.0)

    async def spawn(self, *args: str, cwd: Path | None = None) -> bool:
        """Launch a long-running server and return as soon as it is running.

        ``run`` waits for the child to exit, which a server never does; calling it for
        ``ollama serve`` blocked until the timeout and then killed the daemon it had just
        started. The child is detached so it keeps serving after this process goes away.
        """
        display = format_command(args)
        self.logger.info("spawning: %s", display)
        command = normalize_subprocess_command(args)
        spawn_kwargs: dict[str, Any] = {}
        if platform.system() == "Windows":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            spawn_kwargs["creationflags"] = 0x00000008 | 0x00000200
        else:
            spawn_kwargs["start_new_session"] = True
        try:
            await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                **spawn_kwargs,
            )
        except OSError as exc:
            self.logger.error("could not spawn %s: %s", display, exc)
            return False
        return True

    async def run(self, *args: str, cwd: Path | None = None, timeout: float = 900) -> CommandResult:
        display = format_command(args)
        self.logger.info("running: %s", display)
        command = normalize_subprocess_command(args)
        spawn_kwargs: dict[str, Any] = {}
        if platform.system() == "Windows":
            spawn_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            spawn_kwargs["start_new_session"] = True
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
        except OSError as exc:
            self.logger.exception("could not start command")
            return CommandResult(127, "", str(exc))
        self._process = process
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(
            self._read_stream(process.stdout, "stdout", stdout_chunks)
        )
        stderr_task = asyncio.create_task(
            self._read_stream(process.stderr, "stderr", stderr_chunks)
        )
        heartbeat_task = asyncio.create_task(self._heartbeat(process, display))
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            timed_out = True
            await self._stop_process(process)
            self.logger.error("command timed out: %s", display)
            stderr_chunks.append("command timed out")
        except asyncio.CancelledError:
            await self._stop_process(process)
            raise
        finally:
            self._process = None
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            stdout_task.cancel()
            stderr_task.cancel()
            with suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, return_exceptions=True), timeout=1.0
                )
        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        returncode = (
            -1 if timed_out else (process.returncode if process.returncode is not None else -1)
        )
        self.logger.info("exit=%s\nstdout:\n%s\nstderr:\n%s", returncode, stdout_text, stderr_text)
        return CommandResult(returncode, stdout_text, stderr_text)

    async def _read_stream(
        self,
        stream: asyncio.StreamReader,
        label: str,
        chunks: list[str],
    ) -> None:
        pending = ""
        while chunk := await stream.read(1024):
            text = chunk.decode(errors="replace")
            chunks.append(text)
            pending += ANSI_ESCAPE.sub("", text)
            # Download CLIs redraw a single terminal line with carriage returns. Reading
            # only to `\n` held every percentage update until the transfer had finished.
            parts = re.split(r"[\r\n]", pending)
            pending = parts.pop()
            for visible_line in parts:
                if not visible_line.strip():
                    continue
                self.logger.info("%s: %s", label, visible_line.strip())
                if self.verbose:
                    self.output(visible_line.strip())
        if pending.strip():
            self.logger.info("%s: %s", label, pending.strip())
            if self.verbose:
                self.output(pending.strip())

    async def _heartbeat(
        self, process: asyncio.subprocess.Process, display: str, interval: float = 8
    ) -> None:
        started = time.monotonic()
        while process.returncode is None:
            await asyncio.sleep(interval)
            if process.returncode is None:
                elapsed = int(time.monotonic() - started)
                self.logger.debug("process active after %ss: %s", elapsed, display)
                self.output(f"Process active · {elapsed}s elapsed · no failure reported")


def format_command(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(
        f'"{item}"' if any(char.isspace() for char in item) else item for item in command
    )


def llmster_install_action(system: str | None = None) -> InstallAction | None:
    current = system or platform.system()
    if current in ("Darwin", "Linux"):
        script = "curl -fsSL https://lmstudio.ai/install.sh | bash"
        return InstallAction(
            "LM Studio llmster",
            ("/bin/bash", "-lc", script),
            script,
            "https://lmstudio.ai/install.sh",
            "Downloads and installs LM Studio's official headless llmster runtime and lms CLI.",
            remote_script=True,
        )
    if current == "Windows":
        script = "irm https://lmstudio.ai/install.ps1 | iex"
        return InstallAction(
            "LM Studio llmster",
            ("powershell.exe", "-NoProfile", "-Command", script),
            script,
            "https://lmstudio.ai/install.ps1",
            "Downloads and installs LM Studio's official headless llmster runtime and lms CLI.",
            remote_script=True,
        )
    return None


lms_install_action = llmster_install_action


def lmstudio_desktop_install_action(
    system: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> InstallAction | None:
    """Return an installation action for the full LM Studio Desktop application if supported."""
    current = system or platform.system()
    if current == "Darwin":
        brew = which("brew")
        if brew:
            command: tuple[str, ...] = (brew, "install", "--cask", "lm-studio")
            return InstallAction(
                "LM Studio Desktop",
                command,
                format_command(command),
                "https://lmstudio.ai/download",
                "Installs LM Studio Desktop via Homebrew Cask.",
            )
    return None


def ollama_install_action(system: str | None = None) -> InstallAction | None:
    """Return Ollama's official unattended installer for supported desktop hosts."""
    current = system or platform.system()
    if current == "Darwin":
        script = "curl -fsSL https://ollama.com/install.sh | sh"
        return InstallAction(
            "Ollama",
            ("/bin/bash", "-lc", script),
            script,
            "https://ollama.com/download",
            "Downloads and installs Ollama using its official macOS installer.",
            remote_script=True,
        )
    if current == "Windows":
        script = "irm https://ollama.com/install.ps1 | iex"
        return InstallAction(
            "Ollama",
            ("powershell.exe", "-NoProfile", "-Command", script),
            script,
            "https://ollama.com/download/windows",
            "Downloads and installs Ollama using its official Windows installer.",
            remote_script=True,
        )
    return None


def llamacpp_install_action(
    system: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> InstallAction | None:
    current = system or platform.system()
    if current == "Darwin":
        brew = which("brew")
        if brew:
            command: tuple[str, ...] = (brew, "install", "llama.cpp")
            return InstallAction(
                "llama.cpp",
                command,
                format_command(command),
                "https://github.com/ggerganov/llama.cpp",
                "Installs llama.cpp via Homebrew.",
            )
    if current == "Windows":
        if which("winget"):
            command = (
                "winget",
                "install",
                "--id",
                "ggerganov.llama.cpp",
                "-e",
                "--source",
                "winget",
            )
            return InstallAction(
                "llama.cpp",
                command,
                format_command(command),
                "https://github.com/ggerganov/llama.cpp",
                "Installs llama.cpp via Windows Package Manager (winget).",
            )
        if which("choco"):
            command = ("choco", "install", "llama.cpp", "-y")
            return InstallAction(
                "llama.cpp",
                command,
                format_command(command),
                "https://github.com/ggerganov/llama.cpp",
                "Installs llama.cpp via Chocolatey.",
            )
    return None


def vllm_install_action(
    system: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> InstallAction | None:
    python = which("python3") or which("python") or "python"
    command: tuple[str, ...] = (python, "-m", "pip", "install", "vllm")
    return InstallAction(
        "vLLM",
        command,
        format_command(command),
        "https://docs.vllm.ai",
        "Installs vLLM Python package using pip.",
    )


def opencode_install_action(
    system: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> InstallAction | None:
    current = system or platform.system()
    if current == "Darwin":
        brew = which("brew")
        if not brew and which is shutil.which:
            brew = next(
                (
                    str(path)
                    for path in (
                        Path("/opt/homebrew/bin/brew"),
                        Path("/usr/local/bin/brew"),
                    )
                    if path.is_file()
                ),
                None,
            )
        if brew:
            command: tuple[str, ...] = (brew, "install", "anomalyco/tap/opencode")
            return InstallAction(
                "OpenCode",
                command,
                format_command(command),
                "https://opencode.ai/docs/",
                "Installs stable OpenCode from its official Homebrew tap.",
            )
        if which("npm"):
            command = ("npm", "install", "-g", "opencode-ai")
            return InstallAction(
                "OpenCode",
                command,
                format_command(command),
                "https://www.npmjs.com/package/opencode-ai",
                "Installs stable OpenCode globally with the already-installed npm.",
            )
        if which("curl"):
            script = (
                "set -o pipefail; curl -fsSL --connect-timeout 15 --retry 3 "
                "https://opencode.ai/install | bash -s -- --no-modify-path"
            )
            return InstallAction(
                "OpenCode",
                ("/bin/bash", "-lc", script),
                script,
                "https://opencode.ai/install",
                "Downloads and executes OpenCode's official installer.",
                remote_script=True,
            )
    if current == "Windows":
        candidates = [
            (
                "winget",
                ("winget", "install", "--id", "SST.opencode", "-e", "--source", "winget"),
                "Installs stable OpenCode through Windows Package Manager (winget).",
            ),
            (
                "choco",
                ("choco", "install", "opencode", "-y"),
                "Installs stable OpenCode with Chocolatey.",
            ),
            (
                "scoop",
                ("scoop", "install", "opencode"),
                "Installs stable OpenCode with Scoop.",
            ),
            (
                "npm",
                ("npm", "install", "-g", "opencode-ai"),
                "Installs stable OpenCode globally with npm.",
            ),
        ]
        for executable, command, explanation in candidates:
            if which(executable):
                return InstallAction(
                    "OpenCode",
                    command,
                    format_command(command),
                    "https://opencode.ai/docs/",
                    explanation,
                )
    return None


def git_install_action(
    system: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> InstallAction | None:
    current = system or platform.system()
    if current == "Darwin" and which("brew"):
        command: tuple[str, ...] = ("brew", "install", "git")
        return InstallAction(
            "Git",
            command,
            format_command(command),
            "https://git-scm.com/download/mac",
            "Installs Git using the existing Homebrew package manager.",
        )
    if current == "Windows":
        if which("winget"):
            command = ("winget", "install", "--id", "Git.Git", "-e", "--source", "winget")
            return InstallAction(
                "Git",
                command,
                format_command(command),
                "https://git-scm.com/download/win",
                "Installs Git for Windows through Windows Package Manager.",
            )
        if which("choco"):
            command = ("choco", "install", "git", "-y")
            return InstallAction(
                "Git",
                command,
                format_command(command),
                "https://git-scm.com/download/win",
                "Installs Git using the existing Chocolatey package manager.",
            )
        if which("scoop"):
            command = ("scoop", "install", "git")
            return InstallAction(
                "Git",
                command,
                format_command(command),
                "https://git-scm.com/download/win",
                "Installs Git using Scoop.",
            )
    return None
