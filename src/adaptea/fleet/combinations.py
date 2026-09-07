"""Persist named model combinations together with the capacity they measured.

The active ``adaptea.toml`` and ``.adaptea/capacity.json`` remain the runtime inputs.
This registry is the library: selecting a combination restores both inputs atomically,
so a capacity profile can never be presented as if it described a different fleet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from adaptea.calibration.state import configuration_signature
from adaptea.config import FleetConfig
from adaptea.models import utc_now
from adaptea.reporting.report import latest_calibration

REGISTRY_VERSION = 1


def combinations_path(root: Path) -> Path:
    return root / ".adaptea" / "model-combinations.json"


def _empty() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "active_id": None, "combinations": []}


def _read(root: Path) -> dict[str, Any]:
    path = combinations_path(root)
    if not path.is_file():
        return _empty()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(value, dict) or not isinstance(value.get("combinations"), list):
        return _empty()
    return value


def _write(root: Path, registry: dict[str, Any]) -> Path:
    path = combinations_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _default_name(fleet: FleetConfig) -> str:
    names = [item.model.replace("\\", "/").rsplit("/", 1)[-1] for item in fleet.models]
    return " + ".join(names) if names else "Empty combination"


def _existing_measurement(
    root: Path, fleet: FleetConfig
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """Adopt a pre-library profile only when it describes this exact fleet shape."""
    capacity_path = root / ".adaptea" / "capacity.json"
    if not capacity_path.is_file():
        return None
    try:
        profile = json.loads(capacity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(profile, dict):
        return None
    measured = profile.get("measured_models")
    if not isinstance(measured, list):
        legacy_model = profile.get("model")
        legacy_key = legacy_model.get("key") if isinstance(legacy_model, dict) else None
        measured = [legacy_key] if legacy_key else []
    expected = sorted({item.model for item in fleet.models})
    if sorted(str(item) for item in measured) != expected:
        return None
    recorded_signature = profile.get("configuration_signature")
    if isinstance(recorded_signature, str):
        if len(expected) != len(fleet.models) or any(
            item.instances == "auto" for item in fleet.models
        ):
            return None
        desired_signature = configuration_signature(
            [
                (
                    item.model,
                    int(item.instances),
                    item.context_length,
                    item.parallel_limit,
                )
                for item in fleet.models
            ],
            1,
        )
        if recorded_signature != desired_signature:
            return None
    report: dict[str, Any] | None = None
    directory = latest_calibration(root)
    if directory is not None:
        try:
            candidate = json.loads((directory / "aggregate.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            candidate = None
        if isinstance(candidate, dict) and candidate.get("profile") == profile:
            report = candidate
    return profile, report


def _row(registry: dict[str, Any], combination_id: str) -> dict[str, Any]:
    for value in registry["combinations"]:
        if isinstance(value, dict) and value.get("id") == combination_id:
            return value
    raise KeyError(f"Unknown model combination: {combination_id}")


def save_combination(
    root: Path,
    fleet: FleetConfig,
    *,
    name: str | None = None,
    combination_id: str | None = None,
) -> dict[str, Any]:
    registry = _read(root)
    existing: dict[str, Any] | None = None
    if combination_id:
        existing = _row(registry, combination_id)
    clean_name = (name or (str(existing.get("name")) if existing else _default_name(fleet))).strip()
    if not clean_name:
        raise ValueError("Combination name cannot be empty.")
    if len(clean_name) > 80:
        raise ValueError("Combination name must be 80 characters or fewer.")
    identifier = combination_id or uuid4().hex[:12]
    fleet_value = fleet.model_dump(mode="json")
    now = utc_now()
    source = existing
    if source is None and isinstance(registry.get("active_id"), str):
        active = _row(registry, registry["active_id"])
        if active.get("fleet") == fleet_value:
            source = active
    capacity_profile = source.get("capacity_profile") if source else None
    calibration_report = source.get("calibration_report") if source else None
    measured_at = source.get("measured_at") if source else None
    # An edit creates a different runtime shape. Its old capacity stays out of the active
    # profile until calibration records a replacement for this exact saved combination.
    if existing and existing.get("fleet") != fleet_value:
        capacity_profile = None
        calibration_report = None
        measured_at = None
    if capacity_profile is None:
        migrated = _existing_measurement(root, fleet)
        if migrated is not None:
            capacity_profile, calibration_report = migrated
            measured_at = capacity_profile.get("created_at", now)
    value = {
        "id": identifier,
        "name": clean_name,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
        "fleet": fleet_value,
        "capacity_profile": capacity_profile,
        "calibration_report": calibration_report,
        "measured_at": measured_at,
    }
    if existing:
        registry["combinations"] = [
            value if item is existing else item for item in registry["combinations"]
        ]
    else:
        registry["combinations"].append(value)
    registry["active_id"] = identifier
    _write(root, registry)
    return summary(value, active=True)


def select_combination(root: Path, combination_id: str) -> tuple[FleetConfig, dict[str, Any]]:
    registry = _read(root)
    value = _row(registry, combination_id)
    fleet = FleetConfig.model_validate(value.get("fleet"))
    registry["active_id"] = combination_id
    _write(root, registry)
    capacity_path = root / ".adaptea" / "capacity.json"
    profile = value.get("capacity_profile")
    if isinstance(profile, dict):
        capacity_path.parent.mkdir(parents=True, exist_ok=True)
        capacity_path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif capacity_path.exists():
        capacity_path.unlink()
    return fleet, summary(value, active=True)


def combination_inputs(
    root: Path, combination_id: str
) -> tuple[FleetConfig, dict[str, Any] | None]:
    """The fleet and measured profile of one combination, without making it active.

    ``select_combination`` is how a person switches the machine over. This is how a
    project is handed the combination it was told to use: several projects can name
    different combinations at once, and reading one must not move the global selection
    out from under the others.
    """
    value = _row(_read(root), combination_id)
    fleet = FleetConfig.model_validate(value.get("fleet"))
    profile = value.get("capacity_profile")
    return fleet, profile if isinstance(profile, dict) else None


def delete_combination(root: Path, combination_id: str) -> None:
    registry = _read(root)
    _row(registry, combination_id)
    registry["combinations"] = [
        item for item in registry["combinations"] if item.get("id") != combination_id
    ]
    if registry.get("active_id") == combination_id:
        registry["active_id"] = None
    _write(root, registry)


def clear_active_combination(root: Path) -> None:
    registry = _read(root)
    if registry.get("active_id") is None:
        return
    registry["active_id"] = None
    _write(root, registry)


def capture_active_capacity(root: Path, report: dict[str, Any]) -> None:
    registry = _read(root)
    active = registry.get("active_id")
    if not isinstance(active, str):
        return
    value = _row(registry, active)
    capacity_path = root / ".adaptea" / "capacity.json"
    if not capacity_path.is_file():
        return
    try:
        profile = json.loads(capacity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(profile, dict):
        return
    value["capacity_profile"] = profile
    value["calibration_report"] = report
    value["measured_at"] = utc_now()
    value["updated_at"] = utc_now()
    _write(root, registry)


def active_report(root: Path) -> tuple[str, dict[str, Any] | None] | None:
    registry = _read(root)
    active = registry.get("active_id")
    if not isinstance(active, str):
        return None
    value = _row(registry, active)
    report = value.get("calibration_report")
    return active, report if isinstance(report, dict) else None


def summary(value: dict[str, Any], *, active: bool = False) -> dict[str, Any]:
    fleet = FleetConfig.model_validate(value.get("fleet"))
    report = value.get("calibration_report")
    report_profile = report.get("profile") if isinstance(report, dict) else None
    profile = report_profile if isinstance(report_profile, dict) else value.get("capacity_profile")
    return {
        "id": value["id"],
        "name": value["name"],
        "updated_at": value["updated_at"],
        "active": active,
        "models": [item.model_dump(mode="json") for item in fleet.models],
        "capacity": "current" if isinstance(value.get("capacity_profile"), dict) else "missing",
        "measured_at": value.get("measured_at"),
        "recommended_starting_concurrency": (
            profile.get("recommended_starting_concurrency") if isinstance(profile, dict) else None
        ),
        "safe_max_concurrency": (
            profile.get("safe_max_concurrency") if isinstance(profile, dict) else None
        ),
    }


def list_combinations(root: Path) -> dict[str, Any]:
    registry = _read(root)
    active = registry.get("active_id")
    return {
        "active_id": active,
        "combinations": [
            summary(value, active=value.get("id") == active)
            for value in registry["combinations"]
            if isinstance(value, dict)
        ],
    }
