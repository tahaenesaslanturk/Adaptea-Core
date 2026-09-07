"""Decide whether the measured profile still describes the configured models.

Calibration is a measurement of one machine, one model set, and one load configuration.
The user cannot be expected to know when that measurement stopped applying, so the
question is answered here from the artifacts rather than left to them.

One profile now answers both questions it used to take two to answer. ``capacity.json``
records a concurrency sweep per configured model *and* a combined sweep with every model
generating at once, so there is no separate fleet profile to keep in step. A profile that
measured one model out of three is stale, not current: it describes something the run
never does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from adaptea.config import Config
from adaptea.fleet.models import FleetInventory
from adaptea.inference import configured_model

Freshness = Literal["current", "stale", "missing", "not_required"]

#: Worst-first, so the overall state is the least fresh part that is actually required.
_SEVERITY: dict[Freshness, int] = {"missing": 3, "stale": 2, "current": 1, "not_required": 0}


@dataclass(frozen=True, slots=True)
class CalibrationState:
    """What is measured, what is not, and which calibration would fix it."""

    state: Freshness
    reason: str
    #: "quick" or "full"; None when nothing needs running.
    suggested: Literal["quick", "full"] | None
    capacity: Freshness
    #: Whether the models have been measured running together.
    topology: Freshness
    measured_model: str | None
    measured_models: list[str]
    expected_models: list[str]
    #: Configured models LM Studio currently has loaded.
    loaded_models: list[str]
    #: Every configured model is loaded, so a measurement can actually cover the set.
    ready: bool
    #: Configured models that still have to load before anything can be measured.
    pending_models: list[str]
    #: A signature that changes exactly when a new measurement is warranted, so a caller
    #: can start one automatically without re-running it for an unchanged configuration.
    signature: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "suggested": self.suggested,
            "capacity": self.capacity,
            "topology": self.topology,
            "measured_model": self.measured_model,
            "measured_models": self.measured_models,
            "expected_models": self.expected_models,
            "loaded_models": self.loaded_models,
            "ready": self.ready,
            "pending_models": self.pending_models,
            "signature": self.signature,
        }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def expected_models(config: Config, inventory: FleetInventory) -> list[str]:
    """Every model a run on this configuration would actually use."""
    if config.fleet.enabled and config.fleet.models:
        return sorted({item.model for item in config.fleet.models})
    configured = configured_model(config)
    if configured:
        # The pin may name an instance id; the inventory maps it back to its model.
        for instance in inventory.instances:
            if instance.instance_id == configured:
                return [instance.model_key]
        return [configured]
    return sorted({instance.model_key for instance in inventory.instances})


def _measured_models(profile: dict[str, Any]) -> list[str]:
    measured = profile.get("measured_models")
    if isinstance(measured, list):
        return sorted(str(item) for item in measured)
    # A version 1 profile recorded exactly one model.
    model = profile.get("model")
    key = model.get("key") if isinstance(model, dict) else None
    return [str(key)] if key else []


def _combined_models(profile: dict[str, Any]) -> list[str] | None:
    combined = profile.get("combined")
    if not isinstance(combined, dict) or not combined.get("measured"):
        return None
    models = combined.get("models")
    return sorted(str(item) for item in models) if isinstance(models, list) else []


def configuration_signature(
    models: list[tuple[str, int, int | None, int | None]], max_agents: int
) -> str:
    """Stable identity for the model load shape a capacity profile measured."""
    del max_agents  # Agent demand may change; model/runtime shape decides profile reuse.
    rows = [
        f"{key}:{instances}:{context or '-'}:{parallel or '-'}"
        for key, instances, context, parallel in sorted(models)
    ]
    return "|".join(rows)


def calibration_state(root: Path, config: Config, inventory: FleetInventory) -> CalibrationState:
    expected = expected_models(config, inventory)
    profile = _read(root / ".adaptea" / "capacity.json")
    measured = _measured_models(profile)
    measured_model = measured[0] if measured else None
    grouped: list[tuple[str, int, int | None, int | None]] = []
    for key in expected:
        instances = [item for item in inventory.instances if item.model_key == key]
        if not instances:
            continue
        grouped.append(
            (
                key,
                len(instances),
                instances[0].context_length,
                instances[0].parallel_limit,
            )
        )
    signature = configuration_signature(grouped, config.worker.max_agents)

    if not profile:
        capacity: Freshness = "missing"
        capacity_reason = "This machine has no measured concurrency profile yet."
    elif expected and sorted(measured) != expected:
        capacity = "stale"
        missing = [model for model in expected if model not in measured]
        capacity_reason = (
            f"The profile measured {', '.join(measured) or 'nothing'}, "
            f"but this project uses {', '.join(expected)}."
            + (f" Not yet measured: {', '.join(missing)}." if missing else "")
        )
    elif (
        isinstance(profile.get("configuration_signature"), str)
        and profile["configuration_signature"] != signature
    ):
        capacity = "stale"
        capacity_reason = "The selected models were loaded with different runtime settings."
    else:
        capacity = "current"
        capacity_reason = f"Concurrency measured on {', '.join(measured) or 'the selected model'}."

    combined = _combined_models(profile)
    if len(expected) <= 1:
        topology: Freshness = "not_required"
        topology_reason = "One model is configured, so there is nothing to run alongside it."
    elif not profile or combined is None:
        topology = "missing"
        topology_reason = (
            "More than one model is configured, but they were never measured generating "
            "at the same time."
        )
    elif combined != expected:
        topology = "stale"
        topology_reason = (
            f"The combined measurement covered {', '.join(combined) or 'nothing'}, "
            f"not {', '.join(expected)}."
        )
    else:
        topology = "current"
        topology_reason = "The configured models were measured running together."

    if _SEVERITY[capacity] > 1:
        state, reason = capacity, capacity_reason
    elif _SEVERITY[topology] > 1:
        state, reason = topology, topology_reason
    else:
        state, reason = "current", capacity_reason

    # Save owns loading the entire selected set. A partial fleet is not ready to measure:
    # calibrating it would save a profile that is stale the moment the remaining load
    # finishes, and would tell the user they can start work when their planner may be absent.
    loaded = sorted({instance.model_key for instance in inventory.instances})
    pending = [model for model in expected if model not in loaded]
    ready = bool(expected) and not pending
    if pending:
        reason = (
            f"{', '.join(pending)} {'is' if len(pending) == 1 else 'are'} configured but not "
            "loaded in LM Studio. Save the model selection again; Adaptea loads the complete "
            "set before it starts measurement."
        )

    # There is one calibration now; quick is what re-measures a changed configuration and
    # full is a deeper sweep the user asks for deliberately.
    suggested: Literal["quick", "full"] | None = "quick" if ready and state != "current" else None

    return CalibrationState(
        state=state,
        reason=reason,
        suggested=suggested,
        capacity=capacity,
        topology=topology,
        measured_model=measured_model,
        measured_models=measured,
        expected_models=expected,
        loaded_models=loaded,
        ready=ready,
        pending_models=pending,
        signature=signature,
    )
