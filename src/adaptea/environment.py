"""The one inference environment every project runs against.

The backend, the model fleet and the roles those models play describe *this machine*,
not any one repository. Keeping a separate copy in every project's ``adaptea.toml`` meant
configuring the same thing again for each folder and watching two projects disagree about
which model plans.

There is now a single stored answer, and it is deliberately shaped like a project: it is
an ``adaptea.toml`` in Adaptea's own state directory, loaded by the same `load_config`
every other caller uses. That is what lets the diagnostics, fleet and model services run
against it with no project open at all — they only ever needed a directory holding a
config. Opening a project copies the Adaptea-managed sections into that project's file,
so the orchestrator keeps reading configuration exactly where it always has.
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptea.config import Config, FleetConfig, load_config
from adaptea.projects import user_state_directory
from adaptea.setup.configuration import (
    merge_fleet_config,
    merge_inference_model,
    merge_inference_selection,
    merge_opencode_models_config,
)


def environment_root(directory: Path | None = None) -> Path:
    """The directory that stands in for a project when configuring the environment.

    Its own folder rather than the state directory itself, so anything the fleet or
    calibration services write beside a config stays out of the way of Adaptea's other
    state files.
    """
    root = (directory or user_state_directory()) / "environment"
    root.mkdir(parents=True, exist_ok=True)
    return root


def environment_config_path(directory: Path | None = None) -> Path:
    return environment_root(directory) / "adaptea.toml"


def load_environment(directory: Path | None = None) -> Config:
    path = environment_config_path(directory)
    return load_config(path.parent, explicit=path)


def is_configured(directory: Path | None = None) -> bool:
    """Whether anything has been chosen yet, as opposed to bare defaults."""
    if not environment_config_path(directory).is_file():
        return False
    config = load_environment(directory)
    return bool(config.fleet.models or config.lmstudio.model or config.ollama.model)


def capacity_path(root: Path) -> Path:
    return root / ".adaptea" / "capacity.json"


def apply_to_project(
    root: Path,
    environment: Config | None = None,
    directory: Path | None = None,
    *,
    fleet: FleetConfig | None = None,
    capacity: dict[str, object] | None = None,
) -> list[Path]:
    """Give one project the environment it should run on.

    Returns only the paths actually rewritten: the merge writers are no-ops when the text
    would not change, so opening the same project repeatedly does not keep touching it.
    Sections Adaptea does not manage — validators, command approvals, worker settings —
    are left exactly as they are.

    Three things travel together and must never be separated. The fleet says which models
    run; the capacity profile says how many of them this machine can run at once; and the
    OpenCode catalog is what lets a worker address them at all. A project given the fleet
    without the measurement would re-measure what another project already knows, and one
    given a profile measured for a different fleet would size its runs from a number that
    never described them.

    ``fleet``/``capacity`` override the environment's own selection, which is how a
    project that names its own combination gets that one instead of the global default.
    """
    config = environment or load_environment(directory)
    selected = fleet if fleet is not None else config.fleet
    path = root / "adaptea.toml"
    written = [
        merge_inference_selection(path, config.inference.backend),
        merge_inference_model(path, "lmstudio", config.lmstudio.model),
        merge_inference_model(path, "ollama", config.ollama.model),
        merge_inference_model(path, "llamacpp", config.llamacpp.model),
        merge_inference_model(path, "vllm", config.vllm.model),
        merge_fleet_config(path, selected),
    ]
    if selected.models:
        base_url = (
            config.lmstudio.base_url
            if config.inference.backend == "lmstudio"
            else config.ollama.base_url
            if config.inference.backend == "ollama"
            else config.llamacpp.base_url
            if config.inference.backend == "llamacpp"
            else config.vllm.base_url
        )
        _opencode_path, opencode_replaced = merge_opencode_models_config(
            root,
            config.worker.executable,
            [item.model for item in selected.models],
            base_url,
            backend=config.inference.backend,
        )
        # These writers report the backup they took, not the file they wrote, so an
        # unchanged file and a newly created one both come back as None. Reporting the
        # path unconditionally made every re-open look like a rewrite.
        written.append(opencode_replaced)
    profile = capacity if capacity is not None else _environment_capacity(directory)
    written.append(_write_capacity(root, profile))
    return [item for item in written if item is not None]


def _environment_capacity(directory: Path | None = None) -> dict[str, object] | None:
    try:
        value = json.loads(capacity_path(environment_root(directory)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_capacity(root: Path, profile: dict[str, object] | None) -> Path | None:
    """Mirror a measured profile into a project, or remove a stale one.

    Returned only when something changed, so an unchanged project reports no writes and
    re-opening it stays silent.
    """
    path = capacity_path(root)
    if profile is None:
        if not path.exists():
            return None
        path.unlink()
        return path
    rendered = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    try:
        if path.read_text(encoding="utf-8") == rendered:
            return None
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def adopt_from_project(root: Path, directory: Path | None = None) -> Config | None:
    """Seed the shared environment from a project configured before it existed.

    Without this the first project opened after upgrading would have had its fleet
    replaced by empty defaults. A project that has chosen nothing donates nothing, so the
    environment stays unconfigured until someone actually configures it.
    """
    if environment_config_path(directory).is_file():
        return None
    project = load_config(root, explicit=root / "adaptea.toml")
    if not (project.fleet.models or project.lmstudio.model or project.ollama.model):
        return None
    apply_to_project(environment_root(directory), project, directory)
    return load_environment(directory)
