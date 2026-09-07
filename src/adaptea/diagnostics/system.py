from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptea.config import Config, executable_stem
from adaptea.inference import (
    InferenceBackendError,
    configured_model,
    create_inference_backend,
    inference_connection,
)
from adaptea.lmstudio.lms_cli import CommandResult, run_command, sample_ps
from adaptea.lmstudio.models import LMModel
from adaptea.lmstudio.telemetry import sample_from_models


@dataclass(slots=True)
class LocalModel:
    key: str
    display_name: str | None = None
    architecture: str | None = None
    size_bytes: int | None = None
    max_context_length: int | None = None
    loaded: bool = False
    instance_id: str | None = None
    context_length: int | None = None
    parallel: int | None = None
    format: str | None = None
    inference_ready: bool = False
    size_vram_bytes: int | None = None

    @property
    def ready(self) -> bool:
        return self.inference_ready or self.loaded


@dataclass(slots=True)
class DiagnosticSnapshot:
    operating_system: str
    architecture: str
    python_version: str
    python_ready: bool
    git_executable: str | None
    git_version: str | None
    opencode_executable: str | None
    opencode_version: str | None
    lms_executable: str | None
    lms_version: str | None
    lmstudio_desktop: Path | None
    server_reachable: bool
    server_error: str | None
    native_api_usable: bool
    openai_api_usable: bool
    backend_kind: str = "lmstudio"
    backend_display_name: str = "LM Studio"
    backend_executable: str | None = None
    backend_version: str | None = None
    models: list[LocalModel] = field(default_factory=list)
    selected_model: LocalModel | None = None
    telemetry_usable: bool = False
    telemetry_source: str | None = None
    git_repository: bool = False
    capacity_profile: bool = False
    adaptea_config: bool = False
    opencode_configured: bool = False
    lms_bootstrap_executable: str | None = None

    @property
    def lmstudio_installed(self) -> bool:
        return self.lmstudio_desktop is not None or self.lms_executable is not None

    @property
    def required_ready(self) -> bool:
        return all(
            (
                self.python_ready,
                self.git_executable,
                self.opencode_executable,
                self.lms_executable if self.backend_kind == "lmstudio" else True,
                self.server_reachable,
                self.native_api_usable,
                self.openai_api_usable,
                self.selected_model is not None and self.selected_model.ready,
                self.opencode_configured,
                self.adaptea_config,
            )
        )


def desktop_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    if system == "Darwin":
        return [
            Path("/Applications/LM Studio.app"),
            Path.home() / "Applications" / "LM Studio.app",
        ]
    if system == "Windows":
        roots = [
            env.get("LOCALAPPDATA"),
            env.get("ProgramFiles"),
            env.get("ProgramFiles(x86)"),
        ]
        suffixes = [
            Path("Programs") / "LM Studio" / "LM Studio.exe",
            Path("LM Studio") / "LM Studio.exe",
        ]
        return [Path(root) / suffix for root in roots if root for suffix in suffixes]
    return []


def bundled_lms_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    if system == "Windows":
        home = env.get("USERPROFILE") or env.get("HOME")
        return (
            [
                Path(home) / ".lmstudio" / "bin" / "lms.exe",
                Path(home) / ".cache" / "lm-studio" / "bin" / "lms.exe",
            ]
            if home
            else []
        )
    home = env.get("HOME")
    base = Path(home) if home else Path.home()
    return [
        base / ".lmstudio" / "bin" / "lms",
        base / ".cache" / "lm-studio" / "bin" / "lms",
        Path("/opt/homebrew/bin/lms"),
        Path("/usr/local/bin/lms"),
        Path("/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"),
        base
        / "Applications"
        / "LM Studio.app"
        / "Contents"
        / "Resources"
        / "app"
        / ".webpack"
        / "lms",
    ]


def desktop_lms_candidates(desktop: Path, system: str) -> list[Path]:
    if system == "Darwin":
        return [desktop / "Contents" / "Resources" / "app" / ".webpack" / "lms"]
    if system == "Windows":
        return [desktop.parent / "resources" / "app" / ".webpack" / "lms.exe"]
    return []


def opencode_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    home = Path(env.get("HOME", str(Path.home())))
    candidates = [
        home / ".opencode" / "bin" / "opencode",
        home / ".local" / "bin" / "opencode",
        home / "bin" / "opencode",
        home / ".cargo" / "bin" / "opencode",
        home / ".bun" / "bin" / "opencode",
        home / ".npm-global" / "bin" / "opencode",
    ]
    if system == "Darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/opencode"),
                Path("/usr/local/bin/opencode"),
                Path("/usr/bin/opencode"),
            ]
        )
    elif system == "Windows":
        user_profile = Path(env.get("USERPROFILE") or env.get("HOME", str(Path.home())))
        local_appdata = Path(env.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
        appdata = Path(env.get("APPDATA", str(user_profile / "AppData" / "Roaming")))
        candidates = [
            appdata / "npm" / "opencode.cmd",
            appdata / "npm" / "opencode.ps1",
            appdata / "npm" / "opencode",
            user_profile / "AppData" / "Roaming" / "npm" / "opencode.cmd",
            local_appdata / "Programs" / "OpenCode" / "bin" / "opencode.exe",
            local_appdata / "Microsoft" / "WinGet" / "Links" / "opencode.exe",
            user_profile / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "opencode.exe",
            user_profile / ".opencode" / "bin" / "opencode.exe",
            user_profile / ".opencode" / "bin" / "opencode.cmd",
            user_profile / "scoop" / "shims" / "opencode.exe",
            user_profile / "scoop" / "shims" / "opencode.cmd",
            Path("C:/ProgramData/chocolatey/bin/opencode.exe"),
            user_profile / ".local" / "bin" / "opencode.exe",
        ]
    return candidates


def git_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    candidates = [
        Path("/usr/bin/git"),
        Path("/usr/local/bin/git"),
        Path("/opt/homebrew/bin/git"),
    ]
    if system == "Windows":
        user_profile = Path(env.get("USERPROFILE") or env.get("HOME", str(Path.home())))
        local_appdata = Path(env.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
        program_files = Path(env.get("ProgramFiles", "C:\\Program Files"))
        program_files_x86 = Path(env.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
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
    return candidates


def find_git_executable(
    configured: str = "git",
    system: str = "Darwin",
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    discovered = which(configured)
    if discovered:
        return discovered
    configured_path = Path(configured).expanduser()
    if configured_path.is_file() and (system == "Windows" or os.access(configured_path, os.X_OK)):
        return str(configured_path)
    return next(
        (
            str(path)
            for path in git_candidates(system, environment)
            if path.is_file() and (system == "Windows" or os.access(path, os.X_OK))
        ),
        None,
    )


def ollama_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    """Locations a sandboxed desktop app may not inherit through ``PATH``.

    Finder-launched macOS apps commonly receive a much smaller PATH than the user's
    terminal. Treating that as "Ollama is not installed" made every backend switch run
    the network installer again even though its CLI already existed on disk.
    """
    env = environment or os.environ
    home = Path(env.get("HOME", str(Path.home())))
    candidates = [
        home / ".ollama" / "bin" / "ollama",
        home / ".local" / "bin" / "ollama",
        home / "bin" / "ollama",
    ]
    if system == "Darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/ollama"),
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path("/opt/local/bin/ollama"),
                Path("/Applications/Ollama.app/Contents/Resources/ollama"),
                home / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
                Path("/Applications/Ollama.app/Contents/MacOS/Ollama"),
                home / "Applications" / "Ollama.app" / "Contents" / "MacOS" / "Ollama",
            ]
        )
    elif system == "Linux":
        candidates.extend(
            [
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path("/bin/ollama"),
                Path("/opt/ollama/bin/ollama"),
                Path("/var/lib/ollama/bin/ollama"),
            ]
        )
    elif system == "Windows":
        user_profile = Path(env.get("USERPROFILE") or env.get("HOME", str(Path.home())))
        local_appdata = Path(env.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
        program_files = Path(env.get("ProgramFiles", "C:\\Program Files"))
        program_files_x86 = Path(env.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
        candidates.extend(
            [
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
        )
    return candidates


def find_ollama_executable(
    configured: str,
    system: str,
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    discovered = which(configured)
    if discovered:
        return discovered
    configured_path = Path(configured).expanduser()
    if configured_path.is_file() and (system == "Windows" or os.access(configured_path, os.X_OK)):
        return str(configured_path)
    return next(
        (
            str(path)
            for path in ollama_candidates(system, environment)
            if path.is_file() and (system == "Windows" or os.access(path, os.X_OK))
        ),
        None,
    )


def llamacpp_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    home = Path(env.get("HOME", str(Path.home())))
    candidates = [
        home / ".local" / "bin" / "llama-server",
        home / "bin" / "llama-server",
    ]
    if system == "Darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/llama-server"),
                Path("/usr/local/bin/llama-server"),
                Path("/opt/local/bin/llama-server"),
            ]
        )
    elif system == "Linux":
        candidates.extend(
            [
                Path("/usr/local/bin/llama-server"),
                Path("/usr/bin/llama-server"),
                Path("/bin/llama-server"),
            ]
        )
    elif system == "Windows":
        user_profile = Path(env.get("USERPROFILE") or env.get("HOME", str(Path.home())))
        local_appdata = Path(env.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
        program_files = Path(env.get("ProgramFiles", "C:\\Program Files"))
        candidates.extend(
            [
                local_appdata / "Programs" / "llama.cpp" / "llama-server.exe",
                program_files / "llama.cpp" / "llama-server.exe",
                user_profile / ".local" / "bin" / "llama-server.exe",
            ]
        )
    return candidates


def find_llamacpp_executable(
    configured: str,
    system: str,
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    discovered = which(configured)
    if discovered:
        return discovered
    configured_path = Path(configured).expanduser()
    if configured_path.is_file() and (system == "Windows" or os.access(configured_path, os.X_OK)):
        return str(configured_path)
    return next(
        (
            str(path)
            for path in llamacpp_candidates(system, environment)
            if path.is_file() and (system == "Windows" or os.access(path, os.X_OK))
        ),
        None,
    )


def vllm_candidates(system: str, environment: dict[str, str] | None = None) -> list[Path]:
    env = environment or os.environ
    home = Path(env.get("HOME", str(Path.home())))
    candidates = [
        home / ".local" / "bin" / "vllm",
        home / "bin" / "vllm",
    ]
    if system == "Darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/vllm"),
                Path("/usr/local/bin/vllm"),
            ]
        )
    elif system == "Linux":
        candidates.extend(
            [
                Path("/usr/local/bin/vllm"),
                Path("/usr/bin/vllm"),
            ]
        )
    elif system == "Windows":
        user_profile = Path(env.get("USERPROFILE") or env.get("HOME", str(Path.home())))
        local_appdata = Path(env.get("LOCALAPPDATA", str(user_profile / "AppData" / "Local")))
        appdata = Path(env.get("APPDATA", str(user_profile / "AppData" / "Roaming")))
        candidates.extend(
            [
                local_appdata / "Programs" / "Python" / "Scripts" / "vllm.exe",
                appdata / "Python" / "Scripts" / "vllm.exe",
                user_profile / "AppData" / "Local" / "Programs" / "Python" / "Scripts" / "vllm.exe",
                user_profile / "AppData" / "Roaming" / "Python" / "Scripts" / "vllm.exe",
                user_profile / ".local" / "bin" / "vllm.exe",
            ]
        )
    return candidates


def find_vllm_executable(
    configured: str,
    system: str,
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    discovered = which(configured)
    if discovered:
        return discovered
    configured_path = Path(configured).expanduser()
    if configured_path.is_file() and (system == "Windows" or os.access(configured_path, os.X_OK)):
        return str(configured_path)
    return next(
        (
            str(path)
            for path in vllm_candidates(system, environment)
            if path.is_file() and (system == "Windows" or os.access(path, os.X_OK))
        ),
        None,
    )


def find_opencode_executable(
    configured: str,
    system: str,
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    for name in dict.fromkeys((configured, "opencode", "opencode2")):
        discovered = which(name)
        if discovered:
            return discovered
        path = Path(name).expanduser()
        if path.is_file() and (system == "Windows" or os.access(path, os.X_OK)):
            return str(path)
    return next(
        (
            str(path)
            for path in opencode_candidates(system, environment)
            if path.is_file() and (system == "Windows" or os.access(path, os.X_OK))
        ),
        None,
    )


def select_local_model(models: list[LocalModel], configured_model: str | None) -> LocalModel | None:
    if configured_model is not None:
        return next(
            (
                model
                for model in models
                if configured_model == model.key or configured_model == model.instance_id
            ),
            None,
        )
    return next((model for model in models if model.ready), None)


def find_lmstudio_desktop(system: str | None = None) -> Path | None:
    return next(
        (path for path in desktop_candidates(system or platform.system()) if path.exists()), None
    )


def find_lms_executable(
    configured: str,
    system: str,
    which: Callable[[str], str | None] = shutil.which,
    environment: dict[str, str] | None = None,
) -> str | None:
    """Resolve the ``lms`` CLI whether or not it sits on ``PATH``.

    A Finder-launched app rarely inherits ``~/.lmstudio/bin``, so a bare ``which`` call
    reported the CLI as missing and every server control that resolved it that way -- stop
    and restart among them -- returned "did nothing" instead of acting.
    """
    discovered = which(configured)
    if discovered:
        return discovered
    bundled = next(
        (str(path) for path in bundled_lms_candidates(system, environment) if path.is_file()),
        None,
    )
    if bundled:
        return bundled
    desktop = find_lmstudio_desktop(system)
    if not desktop:
        return None
    return next(
        (str(path) for path in desktop_lms_candidates(desktop, system) if path.is_file()),
        None,
    )


def parse_local_models(text: str) -> list[LocalModel]:
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = next(
            (
                item
                for key in ("models", "data", "items", "llms")
                if isinstance((item := value.get(key)), list)
            ),
            [],
        )
    else:
        rows = []
    models: list[LocalModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = next(
            (
                row.get(name)
                for name in ("key", "modelKey", "model_key", "id", "path", "identifier")
                if isinstance(row.get(name), str)
            ),
            None,
        )
        if not key:
            continue
        model_type = row.get("type")
        if isinstance(model_type, str) and model_type.lower() in {"embedding", "embeddings"}:
            continue
        models.append(
            LocalModel(
                key=key,
                display_name=_string(row, "display_name", "displayName", "name"),
                architecture=_string(row, "architecture", "arch"),
                size_bytes=_integer(row, "size_bytes", "sizeBytes", "size"),
                max_context_length=_integer(
                    row, "max_context_length", "maxContextLength", "contextLength"
                ),
                format=_string(row, "format", "compatibility_type", "compatibilityType"),
            )
        )
    return models


def from_api_models(models: list[LMModel]) -> list[LocalModel]:
    result: list[LocalModel] = []
    for model in models:
        if model.type != "llm":
            continue
        instance = model.loaded_instances[0] if model.loaded_instances else None
        raw_size = model.model_extra.get("size_bytes") if model.model_extra else None
        size_bytes = model.size_bytes if model.size_bytes is not None else raw_size
        result.append(
            LocalModel(
                key=model.key,
                display_name=model.display_name,
                architecture=model.architecture,
                size_bytes=size_bytes if isinstance(size_bytes, int) else None,
                max_context_length=model.max_context_length,
                loaded=instance is not None,
                instance_id=instance.id if instance else None,
                context_length=model.effective_context_length,
                parallel=model.effective_parallel_limit,
                format=model.format,
                inference_ready=model.ready,
                size_vram_bytes=model.size_vram_bytes,
            )
        )
    return result


def _string(row: dict[str, Any], *keys: str) -> str | None:
    return next((row[key] for key in keys if isinstance(row.get(key), str)), None)


def _integer(row: dict[str, Any], *keys: str) -> int | None:
    value = next((row[key] for key in keys if isinstance(row.get(key), int)), None)
    return int(value) if value is not None else None


def project_opencode_path(root: Path) -> Path:
    json_path = root / "opencode.json"
    jsonc_path = root / "opencode.jsonc"
    if json_path.exists():
        return json_path
    if jsonc_path.exists():
        return jsonc_path
    return json_path


def opencode_provider_configured(
    root: Path,
    executable: str | None,
    model: str | None,
    base_url: str | None = None,
    provider_id: str = "lmstudio",
) -> bool:
    from adaptea.setup.configuration import read_json_config

    paths = [root / "opencode.json", root / "opencode.jsonc"]
    for path in paths:
        if not path.exists():
            continue
        try:
            data = read_json_config(path)
        except (OSError, ValueError):
            continue
        provider_key = (
            "providers" if executable and executable_stem(executable) == "opencode2" else "provider"
        )
        providers = data.get(provider_key)
        if not isinstance(providers, dict):
            continue
        provider = providers.get(provider_id)
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        settings_key = "settings" if provider_key == "providers" else "options"
        settings = provider.get(settings_key)
        endpoint_ready = base_url is None or (
            isinstance(settings, dict) and settings.get("baseURL") == base_url.rstrip("/") + "/v1"
        )
        if endpoint_ready and (model is None or isinstance(models, dict) and model in models):
            return True
    return False


class SystemDiagnostics:
    def __init__(
        self,
        root: Path,
        config: Config,
        *,
        system: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.root = root
        self.config = config
        self.system = system or platform.system()
        self.which = which

    async def command(self, *args: str, timeout: float = 10) -> CommandResult:
        return await run_command(*args, timeout=timeout)

    async def version(self, executable: str, *alternatives: tuple[str, ...]) -> str | None:
        if not self.which(executable) and not Path(executable).is_file():
            return None
        for args in alternatives:
            try:
                result = await self.command(executable, *args)
            except OSError:
                return None
            text = (result.stdout or result.stderr).strip().splitlines()
            if result.returncode == 0:
                return text[0] if text else "available (version unavailable)"
        return "available (version unavailable)"

    async def collect(self) -> DiagnosticSnapshot:
        is_lmstudio = self.config.inference.backend == "lmstudio"
        connection = inference_connection(self.config)
        desktop = find_lmstudio_desktop(self.system) if is_lmstudio else None
        git_path = find_git_executable("git", self.system, self.which)
        configured_opencode = self.config.worker.executable
        opencode_path = find_opencode_executable(configured_opencode, self.system, self.which)
        configured_lms = self.config.lmstudio.lms_executable
        lms_path = (
            find_lms_executable(configured_lms, self.system, self.which) if is_lmstudio else None
        )
        # Desktop's bundled executable is already a functional lms CLI. Using it directly
        # avoids requiring bootstrap to edit every supported shell profile, so it is only
        # reported as the bootstrap source when nothing on PATH answered first.
        bootstrap_executable = (
            lms_path
            if lms_path
            and desktop
            and lms_path in {str(path) for path in desktop_lms_candidates(desktop, self.system)}
            else None
        )
        if is_lmstudio:
            backend_path = lms_path
        elif self.config.inference.backend == "ollama":
            backend_path = find_ollama_executable(
                self.config.ollama.executable, self.system, self.which
            )
        elif self.config.inference.backend == "llamacpp":
            backend_path = find_llamacpp_executable(
                self.config.llamacpp.executable, self.system, self.which
            )
        elif self.config.inference.backend == "vllm":
            backend_path = find_vllm_executable(
                self.config.vllm.executable, self.system, self.which
            )
        else:
            backend_path = None
        git_version, opencode_version, backend_version = await asyncio.gather(
            self.version(git_path or "git", ("--version",)) if git_path else _none(),
            self.version(opencode_path, ("--version",)) if opencode_path else _none(),
            self.version(backend_path, ("--version",), ("version",), ("--help",))
            if backend_path
            else _none(),
        )
        lms_version = backend_version if is_lmstudio else None
        telemetry_task = (
            asyncio.create_task(sample_ps(lms_path)) if is_lmstudio and lms_path else None
        )
        server_reachable = False
        server_error: str | None = None
        native_usable = False
        openai_usable = False
        api_models: list[LMModel] = []
        try:
            async with create_inference_backend(self.config, timeout=5) as client:
                api_models = await client.models()
                native_usable = True
                server_reachable = True
                await client.openai_models()
                openai_usable = True
        except InferenceBackendError as exc:
            server_error = str(exc)
        local_models: list[LocalModel] = []
        if is_lmstudio and not api_models and lms_path:
            try:
                listing = await self.command(lms_path, "ls", "--llm", "--json", timeout=10)
                if listing.returncode == 0:
                    local_models = parse_local_models(listing.stdout)
            except OSError:
                pass
        models = from_api_models(api_models) if api_models else local_models
        planner = self.config.fleet.planner() if self.config.fleet.enabled else None
        configured = planner.model if planner else configured_model(self.config)
        selected = select_local_model(models, configured)
        telemetry = await telemetry_task if telemetry_task else None
        if is_lmstudio and telemetry is None and api_models:
            telemetry = sample_from_models(api_models)
        backend_base_url = (
            self.config.lmstudio.base_url
            if is_lmstudio
            else self.config.ollama.base_url
            if self.config.inference.backend == "ollama"
            else self.config.llamacpp.base_url
            if self.config.inference.backend == "llamacpp"
            else self.config.vllm.base_url
        )
        return DiagnosticSnapshot(
            operating_system=f"{self.system} {platform.release()}",
            architecture=platform.machine() or "unknown",
            python_version=platform.python_version(),
            python_ready=sys.version_info >= (3, 12),
            git_executable=git_path,
            git_version=git_version,
            opencode_executable=opencode_path,
            opencode_version=opencode_version,
            lms_executable=lms_path,
            lms_version=lms_version,
            lmstudio_desktop=desktop,
            server_reachable=server_reachable,
            server_error=server_error,
            native_api_usable=native_usable,
            openai_api_usable=openai_usable,
            backend_kind=self.config.inference.backend,
            backend_display_name=connection.display_name.removesuffix(" (Adaptea)"),
            backend_executable=backend_path,
            backend_version=backend_version,
            models=models,
            selected_model=selected,
            telemetry_usable=telemetry is not None,
            telemetry_source=telemetry.source if telemetry else None,
            git_repository=(self.root / ".git").exists(),
            capacity_profile=(self.root / ".adaptea" / "capacity.json").exists(),
            adaptea_config=(self.root / "adaptea.toml").exists(),
            opencode_configured=opencode_provider_configured(
                self.root,
                opencode_path,
                configured or (selected.key if selected else None),
                backend_base_url,
                provider_id=self.config.inference.backend,
            ),
            lms_bootstrap_executable=bootstrap_executable,
        )


async def _none() -> None:
    return None
