from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from adaptea.config import AGENT_CONTEXT_LENGTH
from adaptea.fleet.models import (
    DownloadedModel,
    FleetRecommendation,
    ModelRecommendationAssessment,
    RecommendedFleetModel,
)


def recommend_fleet(
    models: list[DownloadedModel],
    *,
    available_memory_bytes: int | None,
    estimated_memory_bytes: Mapping[str, int | None],
    max_loaded_instances: int,
    memory_headroom_fraction: float,
    requested_context_length: int = AGENT_CONTEXT_LENGTH,
) -> FleetRecommendation:
    """Build a conservative recommendation from facts LM Studio and the OS expose.

    File size is used only to rank models. Fit decisions use LM Studio's estimate at the
    requested context and an OS-reported available-memory budget. Unknown values are never
    substituted with guessed defaults.
    """
    budget = (
        int(available_memory_bytes * (1 - memory_headroom_fraction))
        if available_memory_bytes is not None
        else None
    )
    assessments: list[ModelRecommendationAssessment] = []
    eligible: list[tuple[DownloadedModel, int]] = []
    for model in models:
        estimate = estimated_memory_bytes.get(model.model_key)
        missing: list[str] = []
        if model.size_bytes is None:
            missing.append("model size")
        if model.max_context_length is None:
            missing.append("LM Studio context limit")
        elif model.max_context_length < requested_context_length:
            missing.append(
                f"maximum context {model.max_context_length:,} is below the "
                f"{requested_context_length:,}-token requirement"
            )
        if estimate is None:
            missing.append("LM Studio memory estimate")
        if budget is None:
            missing.append("available system memory")
        fits = estimate is not None and budget is not None and estimate <= budget
        if not missing and not fits:
            missing.append("estimated memory exceeds the headroom-adjusted available budget")
        assessment = ModelRecommendationAssessment(
            model_key=model.model_key,
            eligible=not missing,
            size_bytes=model.size_bytes,
            max_context_length=model.max_context_length,
            estimated_memory_bytes=estimate,
            reason=(
                "All required facts are known and the LM Studio estimate fits the memory budget."
                if not missing
                else "Not auto-selected: " + "; ".join(missing) + "."
            ),
        )
        assessments.append(assessment)
        if not missing:
            assert estimate is not None
            eligible.append((model, estimate))

    if not eligible:
        has_complete_candidate = any(
            item.size_bytes is not None
            and item.max_context_length is not None
            and estimated_memory_bytes.get(item.model_key) is not None
            for item in models
        )
        status: Literal["insufficient_data", "no_safe_fit"] = (
            "no_safe_fit" if budget is not None and has_complete_candidate else "insufficient_data"
        )
        return FleetRecommendation(
            status=status,
            requested_context_length=requested_context_length,
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=budget,
            max_loaded_instances=max_loaded_instances,
            assessments=assessments,
            explanation=[
                "Adaptea did not invent missing capacity or model metadata.",
                (
                    "No fully described model fits the conservative available-memory budget."
                    if status == "no_safe_fit"
                    else "A recommendation needs available memory, model size, context limit, "
                    "and an LM Studio memory estimate."
                ),
            ],
        )

    # A filename is not a capability benchmark. Among models proven safe at the required
    # context, file size is the only stable local signal available for the strong/fast split.
    # The UI explains this provisional choice and lets the user override it explicitly.
    eligible.sort(key=lambda item: (item[0].size_bytes or 0, item[0].model_key))
    strong_model, strong_estimate = eligible[-1]
    selected: list[RecommendedFleetModel] = [
        RecommendedFleetModel(
            name="auto-strong",
            model=strong_model.model_key,
            tier="strong",
            roles=["planner", "worker", "reviewer"],
            context_length=requested_context_length,
            explanation=(
                f"Largest fully described candidate that LM Studio estimates at "
                f"{strong_estimate:,} bytes for {requested_context_length:,} tokens."
            ),
        )
    ]
    next(
        item for item in assessments if item.model_key == strong_model.model_key
    ).selected_as = "strong"

    if max_loaded_instances > 1 and budget is not None:
        remaining = budget - strong_estimate
        fast_candidates = [
            item
            for item in eligible
            if item[0].model_key != strong_model.model_key and item[1] <= remaining
        ]
        if fast_candidates:
            fast_model, fast_estimate = fast_candidates[0]
            selected.append(
                RecommendedFleetModel(
                    name="auto-fast",
                    model=fast_model.model_key,
                    tier="fast",
                    roles=["worker"],
                    context_length=requested_context_length,
                    explanation=(
                        f"Smallest additional proven-safe candidate; combined LM Studio "
                        f"estimates remain within the {budget:,}-byte budget."
                    ),
                )
            )
            next(
                item for item in assessments if item.model_key == fast_model.model_key
            ).selected_as = "fast"

    return FleetRecommendation(
        status="recommended",
        requested_context_length=requested_context_length,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=budget,
        max_loaded_instances=max_loaded_instances,
        models=selected,
        assessments=assessments,
        explanation=[
            f"Context is fixed at the measured {requested_context_length:,}-token "
            "agent requirement.",
            f"The OS reported {available_memory_bytes:,} bytes available; "
            f"{memory_headroom_fraction:.0%} remains reserved.",
            "Adaptea reads the backend ceiling after load and calibration decides how much to use.",
        ],
    )
