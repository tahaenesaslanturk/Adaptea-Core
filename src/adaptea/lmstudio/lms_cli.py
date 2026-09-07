from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptea.models import TelemetrySample


def resolve_executable(executable: str) -> str:
    """Resolve an executable name (e.g. 'lms', 'opencode', 'ollama', 'git', 'llama-server', 'vllm') to an absolute path if not on PATH."""
    if not executable:
        return executable
    path = Path(executable).expanduser()
    system = platform.system()
    if path.is_file() and (system == "Windows" or os.access(path, os.X_OK)):
        return str(path)

    found = shutil.which(executable)
    if found:
        return found

    name = executable.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    home = Path(os.environ.get("HOME", str(Path.home())))

    if name == "lms":
        candidates = [
            home / ".cache" / "lm-studio" / "bin" / "lms",
            home / ".lmstudio" / "bin" / "lms",
            Path("/opt/homebrew/bin/lms"),
            Path("/usr/local/bin/lms"),
            Path("/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"),
            home
            / "Applications"
            / "LM Studio.app"
            / "Contents"
            / "Resources"
            / "app"
            / ".webpack"
            / "lms",
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            candidates = [
                user_profile / ".cache" / "lm-studio" / "bin" / "lms.exe",
                user_profile / ".lmstudio" / "bin" / "lms.exe",
                local_appdata
                / "Programs"
                / "LM Studio"
                / "resources"
                / "app"
                / ".webpack"
                / "lms.exe",
                local_appdata / "Programs" / "LM Studio" / "LM Studio.exe",
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name in {"opencode", "opencode2"}:
        candidates = [
            home / ".opencode" / "bin" / "opencode",
            home / ".local" / "bin" / "opencode",
            home / "bin" / "opencode",
            home / ".cargo" / "bin" / "opencode",
            home / ".bun" / "bin" / "opencode",
            home / ".npm-global" / "bin" / "opencode",
            Path("/opt/homebrew/bin/opencode"),
            Path("/usr/local/bin/opencode"),
            Path("/usr/bin/opencode"),
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            appdata = Path(os.environ.get("APPDATA", ""))
            candidates = [
                appdata / "npm" / "opencode.cmd",
                appdata / "npm" / "opencode.ps1",
                appdata / "npm" / "opencode",
                user_profile / "AppData" / "Roaming" / "npm" / "opencode.cmd",
                local_appdata / "Programs" / "OpenCode" / "bin" / "opencode.exe",
                local_appdata / "Microsoft" / "WinGet" / "Links" / "opencode.exe",
                user_profile
                / "AppData"
                / "Local"
                / "Microsoft"
                / "WinGet"
                / "Links"
                / "opencode.exe",
                user_profile / ".opencode" / "bin" / "opencode.exe",
                user_profile / ".opencode" / "bin" / "opencode.cmd",
                user_profile / "scoop" / "shims" / "opencode.exe",
                user_profile / "scoop" / "shims" / "opencode.cmd",
                Path("C:/ProgramData/chocolatey/bin/opencode.exe"),
                user_profile / ".local" / "bin" / "opencode.exe",
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name == "ollama":
        candidates = [
            home / ".ollama" / "bin" / "ollama",
            home / ".local" / "bin" / "ollama",
            home / "bin" / "ollama",
            Path("/opt/homebrew/bin/ollama"),
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
            Path("/opt/local/bin/ollama"),
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            home / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
            Path("/Applications/Ollama.app/Contents/MacOS/Ollama"),
            home / "Applications" / "Ollama.app" / "Contents" / "MacOS" / "Ollama",
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
            program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
            candidates = [
                local_appdata / "Programs" / "Ollama" / "ollama.exe",
                user_profile / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
                program_files / "Ollama" / "ollama.exe",
                program_files_x86 / "Ollama" / "ollama.exe",
                local_appdata / "Microsoft" / "WinGet" / "Links" / "ollama.exe",
                user_profile / "scoop" / "shims" / "ollama.exe",
                Path("C:/ProgramData/chocolatey/bin/ollama.exe"),
                user_profile / ".ollama" / "bin" / "ollama.exe",
                user_profile / ".local" / "bin" / "ollama.exe",
            ]
        elif system == "Linux":
            candidates.extend(
                [
                    Path("/bin/ollama"),
                    Path("/opt/ollama/bin/ollama"),
                    Path("/var/lib/ollama/bin/ollama"),
                ]
            )
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name in {"llama-server", "llamacpp"}:
        candidates = [
            home / ".local" / "bin" / "llama-server",
            home / "bin" / "llama-server",
            Path("/opt/homebrew/bin/llama-server"),
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
            candidates = [
                local_appdata / "Programs" / "llama.cpp" / "llama-server.exe",
                program_files / "llama.cpp" / "llama-server.exe",
                local_appdata / "Microsoft" / "WinGet" / "Links" / "llama-server.exe",
                user_profile / "scoop" / "shims" / "llama-server.exe",
                Path("C:/ProgramData/chocolatey/bin/llama-server.exe"),
                user_profile / ".local" / "bin" / "llama-server.exe",
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name == "vllm":
        candidates = [
            home / ".local" / "bin" / "vllm",
            home / "bin" / "vllm",
            Path("/opt/homebrew/bin/vllm"),
            Path("/usr/local/bin/vllm"),
            Path("/usr/bin/vllm"),
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            candidates = [
                local_appdata / "Programs" / "Python" / "Scripts" / "vllm.exe",
                user_profile / "AppData" / "Roaming" / "Python" / "Scripts" / "vllm.exe",
                user_profile / ".local" / "bin" / "vllm.exe",
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name in {"hf", "huggingface-cli"}:
        candidates = [
            home / ".local" / "bin" / name,
            home / "bin" / name,
            Path("/opt/homebrew/bin") / name,
            Path("/usr/local/bin") / name,
            Path("/usr/bin") / name,
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            candidates = [
                local_appdata / "Programs" / "Python" / "Scripts" / f"{name}.exe",
                user_profile / "AppData" / "Roaming" / "Python" / "Scripts" / f"{name}.exe",
                user_profile / ".local" / "bin" / f"{name}.exe",
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name == "git":
        candidates = [
            Path("/usr/bin/git"),
            Path("/usr/local/bin/git"),
            Path("/opt/homebrew/bin/git"),
        ]
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
            program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
            program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
            candidates = [
                program_files / "Git" / "cmd" / "git.exe",
                program_files / "Git" / "bin" / "git.exe",
                program_files_x86 / "Git" / "cmd" / "git.exe",
                program_files_x86 / "Git" / "bin" / "git.exe",
                local_appdata / "Programs" / "Git" / "cmd" / "git.exe",
                user_profile / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe",
                local_appdata / "Microsoft" / "WinGet" / "Links" / "git.exe",
                user_profile / "scoop" / "shims" / "git.exe",
                user_profile / "scoop" / "apps" / "git" / "current" / "bin" / "git.exe",
                Path("C:/ProgramData/chocolatey/bin/git.exe"),
            ]
        for candidate in candidates:
            if candidate.is_file() and (system == "Windows" or os.access(candidate, os.X_OK)):
                return str(candidate)

    if name in {"npm", "npx"}:
        if system == "Windows":
            user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
            program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
            program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
            appdata = Path(os.environ.get("APPDATA", ""))
            candidates = [
                program_files / "nodejs" / f"{name}.cmd",
                program_files_x86 / "nodejs" / f"{name}.cmd",
                appdata / "npm" / f"{name}.cmd",
                user_profile / "AppData" / "Roaming" / "npm" / f"{name}.cmd",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

    return executable


def normalize_subprocess_command(args: list[str] | tuple[str, ...]) -> list[str]:
    """Ensure Windows batch scripts (.cmd, .bat) are invoked via cmd.exe /c to prevent WinError 193."""
    if not args:
        return []
    executable = resolve_executable(args[0])
    command = [executable, *args[1:]]
    if platform.system() == "Windows" and executable.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", *command]
    return command


@dataclass(slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


async def run_command(*args: str, timeout: float = 10.0) -> CommandResult:
    if not args:
        return CommandResult(0, "", "")
    command = normalize_subprocess_command(args)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandResult(-1, "", "command timed out")
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.communicate(), 5)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.communicate()
        raise
    return CommandResult(
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key.lower(), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _first(data: Any, names: set[str], expected: type[object]) -> Any:
    for key, value in _walk(data):
        normalized = key.replace("-", "_").replace(" ", "_")
        if normalized in names and isinstance(value, expected):
            return value
    return None


def parse_ps_json(text: str) -> TelemetrySample:
    data = json.loads(text)
    queue_names = {
        "queued",
        "queue_length",
        "queued_requests",
        "queuedrequests",
        "queued_predictions",
        "queuedpredictions",
        "pending_predictions",
        "pendingpredictions",
    }
    generation_names = {
        "generating",
        "is_generating",
        "isgenerating",
        "generation_active",
        "generationactive",
    }
    queued = _first(data, queue_names, int)
    generating = _first(data, generation_names, bool)
    if generating is None:
        status = _first(data, {"status", "state"}, str)
        generating = status.lower() in {"generating", "processing", "busy"} if status else None
    return TelemetrySample(
        source="lms_ps", queued_predictions=queued, generating=generating, raw=data
    )


def parse_instance_pressure(value: Any) -> dict[str, dict[str, int | bool | None]]:
    """Tolerantly extract per-instance pressure without depending on one CLI JSON shape."""
    result: dict[str, dict[str, int | bool | None]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        identifier = next(
            (
                item[key]
                for key in ("identifier", "instance_id", "instanceId", "id")
                if isinstance(item.get(key), str)
            ),
            None,
        )
        if identifier:
            queued = _first(
                item,
                {
                    "queued",
                    "queued_requests",
                    "queuedrequests",
                    "queued_predictions",
                    "queuedpredictions",
                },
                int,
            )
            generating = _first(item, {"generating", "is_generating", "isgenerating"}, bool)
            status = _first(item, {"status", "state"}, str)
            if generating is None and status:
                generating = status.lower() in {"generating", "processing", "busy"}
            result[identifier] = {
                "queued_requests": queued,
                "generation_status": generating,
            }
        for child in item.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return result


async def sample_ps(executable: str = "lms", *, timeout: float = 2.0) -> TelemetrySample | None:
    try:
        result = await run_command(executable, "ps", "--json", timeout=timeout)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        return parse_ps_json(result.stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
