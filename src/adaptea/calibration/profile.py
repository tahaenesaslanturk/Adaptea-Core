from __future__ import annotations

import platform
from typing import Any

from adaptea.calibration.aggregate import recommend
from adaptea.models import utc_now


def build_profile(
    *,
    targets: list[dict[str, Any]],
    per_model: dict[str, dict[str, Any]],
    combined: dict[str, Any],
    recommended: int,
    safe_max: int,
    fleet_capacity: int,
    max_agents: int,
) -> dict[str, Any]:
    """Describe every measured model, and the machine when they all run at once.

    ``recommended_starting_concurrency`` and ``safe_max_concurrency`` stay at the top
    level because the schedulers read them, but they now come from the combined phase
    whenever more than one model is loaded — that is the situation a run is actually in.
    """
    models: list[dict[str, Any]] = []
    for target in targets:
        aggregate = per_model.get(target["key"], {})
        model_recommended, model_safe = recommend(aggregate)
        models.append(
            {
                **target,
                "tested_concurrency": aggregate,
                "recommended_starting_concurrency": model_recommended,
                "safe_max_concurrency": min(model_safe, target["capacity"], max_agents),
            }
        )
    primary = models[0] if models else {}
    from adaptea.calibration.state import configuration_signature

    signature = configuration_signature(
        [
            (
                str(model["key"]),
                int(model.get("instances", 1)),
                (
                    int(model["load_config"]["context_length"])
                    if isinstance(model.get("load_config"), dict)
                    and isinstance(model["load_config"].get("context_length"), int)
                    else None
                ),
                int(model["parallel_limit"])
                if isinstance(model.get("parallel_limit"), int)
                else None,
            )
            for model in models
        ],
        max_agents,
    )
    return {
        "profile_version": 2,
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "lmstudio": {"backend": "LM Studio", "parallel_limit": fleet_capacity},
        "max_agents": max_agents,
        "configuration_signature": signature,
        "fleet_capacity": fleet_capacity,
        # Back-compatible single-model view: the first configured model leads.
        "model": {
            "key": primary.get("key"),
            "instance_id": primary.get("instance_id"),
            "format": primary.get("format"),
            "max_context_length": primary.get("max_context_length"),
        },
        "load_config": primary.get("load_config", {}),
        "tested_concurrency": primary.get("tested_concurrency", {}),
        "measured_models": [model["key"] for model in models],
        "models": models,
        "combined": {
            "models": [model["key"] for model in models],
            "tested_concurrency": combined,
            "measured": bool(combined),
        },
        "recommended_starting_concurrency": recommended,
        "safe_max_concurrency": min(safe_max, max(fleet_capacity, max_agents)),
        "created_at": utc_now(),
        "interpretation": "starting profile; runtime admission remains adaptive",
    }
