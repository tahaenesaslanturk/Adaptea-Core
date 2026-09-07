from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from adaptea.config import Config, FleetModelConfig
from adaptea.fleet.models import DownloadedModel, FleetInventory, ModelInstance
from adaptea.inference import create_inference_backend
from adaptea.lmstudio.lms_cli import run_command
from adaptea.lmstudio.models import LMModel


def parse_downloaded_models(text: str) -> list[DownloadedModel]:
    try:
        value: Any = json.loads(text)
    except ValueError:
        return []
    rows = (
        value
        if isinstance(value, list)
        else value.get("models", [])
        if isinstance(value, dict)
        else []
    )
    result: list[DownloadedModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_type = row.get("type")
        if isinstance(model_type, str) and model_type.lower() not in {"llm", "language_model"}:
            continue
        key = _string(row, "modelKey", "model_key", "key", "path", "identifier")
        if key:
            result.append(
                DownloadedModel(
                    model_key=key,
                    display_name=_string(row, "displayName", "display_name", "name"),
                    architecture=_string(row, "architecture", "arch"),
                    size_bytes=_integer(row, "sizeBytes", "size_bytes", "size"),
                    max_context_length=_integer(
                        row, "maxContextLength", "max_context_length", "contextLength"
                    ),
                    format=_string(row, "format"),
                    quantization=(
                        row.get("quantization")
                        if isinstance(row.get("quantization"), dict | str)
                        else None
                    ),
                    # `lms ls --json` calls this `path`, but it is relative to the model
                    # directory. Resolution and containment checks happen below.
                    local_path=_string(row, "path"),
                )
            )
    return result


def inventory_from_models(
    models: list[LMModel], configured: list[FleetModelConfig]
) -> FleetInventory:
    downloaded: list[DownloadedModel] = []
    instances: list[ModelInstance] = []
    # Exactly one instance plans, even before anything is configured. Handing the role to
    # every loaded instance described a fleet that cannot exist and, because the desktop
    # reads its rows straight off this inventory, put three checked planners in front of
    # the user to undo. Review stays a pool; only planning is exclusive.
    planner_taken = False
    for model in models:
        if model.type != "llm":
            continue
        extra_size = model.model_extra.get("size_bytes") if model.model_extra else None
        size = model.size_bytes if model.size_bytes is not None else extra_size
        downloaded.append(
            DownloadedModel(
                model_key=model.key,
                display_name=model.display_name,
                architecture=model.architecture,
                size_bytes=size if isinstance(size, int) else None,
                max_context_length=model.max_context_length,
                format=model.format,
                quantization=model.quantization,
            )
        )
        assignment = _assignment(model, configured)
        for loaded in model.loaded_instances:
            tier = assignment.tier if assignment else "strong"
            roles: list[Literal["planner", "worker", "reviewer"]]
            if assignment:
                roles = assignment.roles
            elif configured:
                roles = []
            elif planner_taken:
                roles = ["worker", "reviewer"]
            else:
                roles = ["planner", "worker", "reviewer"]
                planner_taken = True
            instances.append(
                ModelInstance(
                    instance_id=loaded.id,
                    model_key=model.key,
                    capability_tier=tier,
                    roles=roles,
                    context_length=loaded.config.context_length,
                    # The live backend is authoritative. A saved fleet value is a legacy
                    # optional cap, not evidence that LM Studio itself can only serve one.
                    parallel_limit=loaded.config.parallel,
                )
            )
    return FleetInventory(downloaded=downloaded, instances=instances, source_notes=["native REST"])


async def discover_fleet(root: Path, config: Config) -> FleetInventory:
    del root
    try:
        async with create_inference_backend(config, timeout=10) as client:
            models = await client.models()
        inventory = inventory_from_models(models, config.fleet.models)
    except Exception:
        # Hugging Face downloads do not require a running inference server. Keeping their
        # disk inventory available while llama.cpp/vLLM is stopped is what lets the model
        # library finish a download and safely remove an unused cache entry.
        if config.inference.backend not in {"llamacpp", "vllm"}:
            raise
        inventory = FleetInventory(source_notes=[f"{config.inference.backend} server unavailable"])
    if config.inference.backend in {"llamacpp", "vllm"}:
        hf_backend = cast(Literal["llamacpp", "vllm"], config.inference.backend)
        cached = discover_huggingface_models(hf_backend)
        known = {model.model_key: model for model in inventory.downloaded}
        for model in cached:
            current = known.get(model.model_key)
            if current is None:
                inventory.downloaded.append(model)
                continue
            current.local_path = model.local_path
            current.size_bytes = current.size_bytes or model.size_bytes
            current.format = current.format or model.format
        inventory.source_notes.append("Hugging Face cache")
        return inventory
    if config.inference.backend != "lmstudio":
        inventory.source_notes = [f"{config.inference.backend} native REST"]
        return inventory
    try:
        result = await run_command(config.lmstudio.lms_executable, "ls", "--json", timeout=30)
    except OSError:
        result = None
    if result and result.returncode == 0:
        cli_models = parse_downloaded_models(result.stdout)
        for model in cli_models:
            model.local_path = resolve_lmstudio_model_path(model.local_path)
        known = {model.model_key: model for model in inventory.downloaded}
        for model in cli_models:
            current = known.get(model.model_key)
            if current is None:
                inventory.downloaded.append(model)
                continue
            # The REST and CLI surfaces expose different optional metadata across LM Studio
            # versions. Enrich known models without replacing facts already returned by REST.
            for field_name in (
                "display_name",
                "architecture",
                "size_bytes",
                "max_context_length",
                "format",
                "quantization",
                "local_path",
            ):
                if getattr(current, field_name) is None:
                    setattr(current, field_name, getattr(model, field_name))
        inventory.source_notes.append("lms ls --json")
    try:
        version = await run_command(config.lmstudio.lms_executable, "--version", timeout=10)
    except OSError:
        version = None
    if version and version.returncode == 0:
        first_line = version.stdout.strip().splitlines()
        if first_line:
            inventory.source_notes.append(f"lms runtime: {first_line[0]}")
    return inventory


def huggingface_cache_root(
    backend: Literal["llamacpp", "vllm"],
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the shared Hugging Face cache location used by llama.cpp and vLLM."""
    env = environment or os.environ
    if backend == "llamacpp" and env.get("LLAMA_CACHE"):
        return Path(env["LLAMA_CACHE"]).expanduser()
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(name):
            return Path(env[name]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return (home or Path.home()) / ".cache" / "huggingface" / "hub"


def huggingface_repo_path(
    repo_id: str,
    backend: Literal["llamacpp", "vllm"],
    cache_root: Path | None = None,
) -> Path:
    return (cache_root or huggingface_cache_root(backend)) / (
        "models--" + repo_id.replace("/", "--")
    )


def discover_huggingface_models(
    backend: Literal["llamacpp", "vllm"], cache_root: Path | None = None
) -> list[DownloadedModel]:
    """Inventory model repositories without depending on a particular `hf` CLI version."""
    root = cache_root or huggingface_cache_root(backend)
    if not root.is_dir():
        return []
    result: list[DownloadedModel] = []
    for directory in sorted(root.glob("models--*")):
        if not directory.is_dir():
            continue
        repo_id = directory.name.removeprefix("models--").replace("--", "/")
        snapshots = directory / "snapshots"
        gguf_files = list(snapshots.rglob("*.gguf")) if snapshots.is_dir() else []
        if backend == "llamacpp" and not gguf_files:
            continue
        blobs = directory / "blobs"
        size = 0
        if blobs.is_dir():
            for blob in blobs.iterdir():
                try:
                    if blob.is_file():
                        size += blob.stat().st_size
                except OSError:
                    continue
        try:
            local_path = str(directory.resolve(strict=True))
        except OSError:
            continue
        result.append(
            DownloadedModel(
                model_key=repo_id,
                display_name=repo_id.rsplit("/", 1)[-1],
                size_bytes=size or None,
                format="gguf" if backend == "llamacpp" else "huggingface",
                local_path=local_path,
            )
        )
    return result


def resolve_huggingface_model_path(
    reported: str | None,
    backend: Literal["llamacpp", "vllm"],
    cache_root: Path | None = None,
) -> str | None:
    """Accept only a whole model repository directly inside the selected HF cache."""
    if not reported:
        return None
    root = cache_root or huggingface_cache_root(backend)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = Path(reported).expanduser().resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != resolved_root or not resolved.name.startswith("models--"):
        return None
    return str(resolved)


def resolve_lmstudio_model_path(reported: str | None) -> str | None:
    """Resolve only model paths proven to live under an LM Studio download directory."""
    if not reported:
        return None
    roots: list[Path] = []
    for settings in (
        Path.home() / ".cache" / "lm-studio" / "settings.json",
        Path.home() / ".lmstudio" / "settings.json",
    ):
        try:
            value = json.loads(settings.read_text(encoding="utf-8"))
            configured = value.get("downloadsFolder") if isinstance(value, dict) else None
            if isinstance(configured, str) and configured.strip():
                roots.append(Path(configured).expanduser())
        except (OSError, ValueError):
            pass
    roots.extend(
        (Path.home() / ".cache" / "lm-studio" / "models", Path.home() / ".lmstudio" / "models")
    )
    raw = Path(reported).expanduser()
    candidates = [raw] if raw.is_absolute() else [directory / raw for directory in roots]
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        for directory in roots:
            try:
                resolved.relative_to(directory.expanduser().resolve())
            except (OSError, ValueError):
                continue
            return str(resolved)
    return None


def _assignment(model: LMModel, configured: list[FleetModelConfig]) -> FleetModelConfig | None:
    instance_ids = {instance.id for instance in model.loaded_instances}
    return next(
        (item for item in configured if item.model == model.key or item.model in instance_ids),
        None,
    )


def _string(row: dict[str, Any], *keys: str) -> str | None:
    return next((row[key] for key in keys if isinstance(row.get(key), str)), None)


def _integer(row: dict[str, Any], *keys: str) -> int | None:
    value = next((row[key] for key in keys if isinstance(row.get(key), int)), None)
    return int(value) if value is not None else None
