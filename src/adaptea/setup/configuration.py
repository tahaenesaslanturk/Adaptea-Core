from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from adaptea.config import FleetConfig, executable_stem
from adaptea.diagnostics.system import project_opencode_path

#: One recoverable copy per configuration file, replaced on each real change.
#: Timestamped names left a new file behind on every setup run and were never removed.
BACKUP_SUFFIX = ".adaptea-backup"

#: Backups live beside the rest of Adaptea's run state, not beside the file they copy.
#: A sibling `adaptea.toml.adaptea-backup` reads as a second configuration file in the
#: project root, which is exactly the clutter the single-copy rule was meant to avoid.
BACKUP_DIRECTORY = Path(".adaptea") / "backups"


def backup_root(path: Path) -> Path:
    """The project directory a configuration file belongs to.

    Root-level files (`adaptea.toml`, `opencode.json`) back up into the project's own
    `.adaptea/backups`; a file already inside `.adaptea` backs up into the same store
    rather than creating a nested one.
    """
    parent = path.parent
    return parent.parent if parent.name == ".adaptea" else parent


def backup(path: Path) -> Path:
    destination = backup_root(path) / BACKUP_DIRECTORY / f"{path.name}{BACKUP_SUFFIX}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


def write_if_changed(path: Path, content: str) -> Path | None:
    """Write only a real change, and back up only what is about to be overwritten.

    Setup is a diagnose-repair-recheck loop, so it merges the same configuration
    repeatedly. Writing an identical file each time produced another backup, another
    mtime, and a directory of near-identical copies for no recoverable benefit.
    """
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    if existing == content:
        return None
    previous = backup(path) if existing is not None else None
    # Earlier versions wrote the copy beside the original. Remove that one now it has
    # been superseded, so upgrading cleans the project root instead of leaving a file
    # nothing writes to any more.
    legacy = path.with_name(f"{path.name}{BACKUP_SUFFIX}")
    if legacy != previous and legacy.is_file():
        legacy.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return previous


def strip_jsonc(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        character = text[index]
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == "/" and next_character == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if character == "/" and next_character == "*":
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                index += 1
            index += 2
            continue
        output.append(character)
        index += 1
    return _remove_trailing_commas("".join(output))


def _remove_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            output.append(character)
            continue
        if character == ",":
            next_nonspace = next(
                (future for future in text[index + 1 :] if not future.isspace()), ""
            )
            if next_nonspace in "}]":
                continue
        output.append(character)
    return "".join(output)


def read_json_config(path: Path) -> dict[str, Any]:
    value: Any = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def inference_provider(
    model: str,
    base_url: str,
    *,
    v2: bool,
    backend: str = "lmstudio",
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1"
    names = {
        "lmstudio": "LM Studio (local)",
        "ollama": "Ollama (local)",
        "llamacpp": "llama.cpp (local)",
        "vllm": "vLLM (local)",
    }
    display_name = names.get(backend, f"{backend} (local)")
    if v2:
        settings: dict[str, Any] = {"baseURL": endpoint}
        if backend != "lmstudio":
            settings["apiKey"] = backend
        provider = {
            "name": display_name,
            "package": "@opencode-ai/ai/providers/openai-compatible",
            "settings": settings,
            "models": {model: {"name": model}},
        }
        return provider
    options: dict[str, Any] = {"baseURL": endpoint}
    if backend != "lmstudio":
        options["apiKey"] = backend
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": display_name,
        "options": options,
        "models": {model: {"name": model}},
    }


def lmstudio_provider(model: str, base_url: str, *, v2: bool) -> dict[str, Any]:
    """Compatibility wrapper for existing setup integrations and tests."""

    return inference_provider(model, base_url, v2=v2, backend="lmstudio")


def merge_opencode_config(
    root: Path,
    executable: str,
    model: str,
    base_url: str,
    *,
    backend: str = "lmstudio",
) -> tuple[Path, Path | None]:
    return merge_opencode_models_config(root, executable, [model], base_url, backend=backend)


def merge_opencode_models_config(
    root: Path,
    executable: str,
    models: list[str],
    base_url: str,
    *,
    backend: str = "lmstudio",
) -> tuple[Path, Path | None]:
    if not models:
        raise ValueError("at least one OpenCode model is required")
    path = project_opencode_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = (
        read_json_config(path) if path.exists() else {"$schema": "https://opencode.ai/config.json"}
    )
    v2 = executable_stem(executable) == "opencode2"
    key = "providers" if v2 else "provider"
    providers = data.setdefault(key, {})
    if not isinstance(providers, dict):
        raise ValueError(f"{path}: {key} must be an object")
    existing = providers.get(backend)
    managed = inference_provider(models[0], base_url, v2=v2, backend=backend)
    managed["models"] = {model: {"name": model} for model in models}
    if isinstance(existing, dict):
        existing_models = existing.get("models")
        managed_models = managed["models"]
        if isinstance(existing_models, dict) and isinstance(managed_models, dict):
            managed["models"] = {**existing_models, **managed_models}
        existing.update(managed)
        providers[backend] = existing
    else:
        providers[backend] = managed
    return path, write_if_changed(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _toml_value(value: str | int | float | bool | list[str]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(item) for item in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def merge_approved_commands(path: Path, commands: list[str]) -> Path | None:
    """Add explicitly approved worker commands without touching anything else.

    An approval the user gave in the desktop has to survive the run that asked for it,
    and the project's ``adaptea.toml`` is where the policy already reads them from.
    Existing approvals are preserved and duplicates are dropped, so answering "always"
    twice for the same command writes the file once.
    """
    import tomllib

    existing: list[str] = []
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            data = {}
        section = data.get("worker")
        security = section.get("command_security") if isinstance(section, dict) else None
        approved = security.get("approved_commands") if isinstance(security, dict) else None
        if isinstance(approved, list):
            existing = [item for item in approved if isinstance(item, str)]
    merged = list(
        dict.fromkeys(
            [*existing, *(" ".join(command.split()) for command in commands if command.strip())]
        )
    )
    if merged == existing:
        return None
    lines = _merge_toml_section(
        text.splitlines(), "worker.command_security", {"approved_commands": merged}
    )
    return write_if_changed(path, "\n".join(lines).rstrip() + "\n")


def merge_adaptea_config(
    path: Path,
    *,
    base_url: str,
    model: str,
    lms_executable: str,
    opencode_executable: str,
    max_agents: int,
    scheduler: str = "adaptive",
    backend: str = "lmstudio",
    ollama_executable: str = "ollama",
) -> Path | None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    backend_values: dict[str, str | int | float | bool] = {
        "base_url": base_url,
        "model": model,
    }
    if backend == "lmstudio":
        backend_values["lms_executable"] = lms_executable
    else:
        backend_values["executable"] = ollama_executable
    sections: dict[str, dict[str, str | int | float | bool]] = {
        "inference": {"backend": backend},
        backend: backend_values,
        "worker": {
            "provider": "opencode",
            "executable": opencode_executable,
            "max_agents": max_agents,
            "default_scheduler": scheduler,
        },
        "controller": {"enabled": True},
    }
    lines = text.splitlines()
    for section, values in sections.items():
        lines = _merge_toml_section(lines, section, values)
    return write_if_changed(path, "\n".join(lines).rstrip() + "\n")


def _merge_toml_section(
    lines: list[str], section: str, values: Mapping[str, str | int | float | bool | list[str]]
) -> list[str]:
    header = f"[{section}]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(header)
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        return lines
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].lstrip().startswith("[")),
        len(lines),
    )
    for key, value in values.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        match = next(
            (index for index in range(start + 1, end) if pattern.match(lines[index])), None
        )
        replacement = f"{key} = {_toml_value(value)}"
        if match is None:
            lines.insert(end, replacement)
            end += 1
        else:
            lines[match] = replacement
    return lines


def merge_inference_selection(path: Path, backend: str) -> Path | None:
    if backend not in {"lmstudio", "ollama", "llamacpp", "vllm"}:
        raise ValueError(f"unsupported inference backend: {backend}")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = _merge_toml_section(lines, "inference", {"backend": backend})
    return write_if_changed(path, "\n".join(lines).rstrip() + "\n")


def merge_inference_model(path: Path, backend: str, model: str | None) -> Path | None:
    """Set or clear the selected model without touching unrelated project settings."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    header = f"[{backend}]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        if model is None:
            return None
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header, f"model = {_toml_value(model)}"])
        return write_if_changed(path, "\n".join(lines).rstrip() + "\n")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].lstrip().startswith("[")),
        len(lines),
    )
    model_line = next(
        (
            index
            for index in range(start + 1, end)
            if lines[index].split("=", 1)[0].strip() == "model"
        ),
        None,
    )
    if model_line is not None:
        if model is None:
            del lines[model_line]
        else:
            lines[model_line] = f"model = {_toml_value(model)}"
    elif model is not None:
        lines.insert(end, f"model = {_toml_value(model)}")
    return write_if_changed(path, "\n".join(lines).rstrip() + "\n")


def merge_fleet_config(path: Path, fleet: FleetConfig) -> Path | None:
    """Replace only Adaptea-managed fleet sections and preserve unrelated TOML."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            normalized = stripped.strip("[]").strip()
            skipping = normalized == "fleet" or normalized.startswith("fleet.")
        if not skipping:
            kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    if kept:
        kept.append("")
    kept.extend(
        [
            "[fleet]",
            f"enabled = {_toml_value(fleet.enabled)}",
            f"topology = {_toml_value(fleet.topology)}",
            f"max_loaded_instances = {fleet.max_loaded_instances}",
            f"memory_headroom_fraction = {fleet.memory_headroom_fraction}",
            f"instance_ttl_seconds = {fleet.instance_ttl_seconds}",
            f"minimum_residency_seconds = {fleet.minimum_residency_seconds}",
            "",
            "[fleet.routing]",
            f"high_complexity = {_toml_value(fleet.routing.high_complexity)}",
            f"low_complexity = {_toml_value(fleet.routing.low_complexity)}",
            f"medium_complexity = {_toml_value(fleet.routing.medium_complexity)}",
            f"escalate_failed_fast_task = {_toml_value(fleet.routing.escalate_failed_fast_task)}",
        ]
    )
    for model in fleet.models:
        kept.extend(
            [
                "",
                "[[fleet.models]]",
                f"name = {_toml_value(model.name)}",
                f"model = {_toml_value(model.model)}",
                f"tier = {_toml_value(model.tier)}",
                f"roles = {json.dumps(model.roles)}",
                f"instances = {_toml_value(model.instances)}",
            ]
        )
        if model.context_length is not None:
            kept.append(f"context_length = {model.context_length}")
        if model.parallel_limit is not None:
            kept.append(f"parallel_limit = {model.parallel_limit}")
    return write_if_changed(path, "\n".join(kept).rstrip() + "\n")
