from __future__ import annotations

from collections.abc import Mapping

from adaptea.config import FleetRoutingConfig
from adaptea.fleet.models import FleetInventory, ModelInstance, RoutingDecision
from adaptea.models import TaskRuntime, TaskSpec


class FleetRouter:
    """Explainable tier and least-pressure instance routing; it does not claim optimality."""

    def __init__(self, inventory: FleetInventory, policy: FleetRoutingConfig) -> None:
        self.inventory = inventory
        self.policy = policy

    def route(
        self,
        task: TaskSpec,
        running: Mapping[str, int],
        *,
        force_tier: str | None = None,
    ) -> RoutingDecision | None:
        requested, basis = self._tier(task, force_tier, running)
        candidates = self._candidates(requested, running)
        fallback = False
        if not candidates:
            candidates = self._candidates("strong" if requested == "fast" else "fast", running)
            fallback = bool(candidates)
        if not candidates:
            return None
        chosen = min(candidates, key=lambda item: (self.pressure(item, running), item.instance_id))
        pressure = self.pressure(chosen, running)
        reason = f"{basis}; {requested} pool selected"
        if fallback:
            reason += f"; requested pool unavailable, fell back to {chosen.capability_tier}"
        reason += "; instance had lowest normalized pressure"
        return RoutingDecision(
            task=task.id,
            tier=chosen.capability_tier,
            model=chosen.model_key,
            instance=chosen.instance_id,
            reason=reason,
            fallback=fallback,
            pressure_score=pressure,
            signals={
                "running": running.get(chosen.instance_id, 0),
                "parallel_limit": chosen.effective_limit,
                "queued_requests": chosen.queued_requests,
                "generation_status": chosen.generation_status,
            },
        )

    def _tier(
        self, task: TaskSpec, force_tier: str | None, running: Mapping[str, int]
    ) -> tuple[str, str]:
        if force_tier in {"fast", "strong"}:
            return force_tier, f"retry explicitly requested {force_tier}"
        if task.preferred_tier != "auto":
            return task.preferred_tier, f"preferred_tier={task.preferred_tier}"
        if task.complexity == "high" or task.risk == "high":
            configured = self.policy.high_complexity
            return (
                (configured, "high complexity or risk")
                if configured != "auto"
                else self._auto_tier(running, "high complexity policy=auto")
            )
        if task.complexity == "low" and task.risk == "low":
            configured = self.policy.low_complexity
            return (
                (configured, "low complexity and risk")
                if configured != "auto"
                else self._auto_tier(running, "low complexity policy=auto")
            )
        configured = self.policy.medium_complexity
        if configured != "auto":
            return configured, f"medium policy={configured}"
        return self._auto_tier(running, "medium policy=auto")

    def _auto_tier(self, running: Mapping[str, int], basis: str) -> tuple[str, str]:
        fast = self._candidates("fast", running)
        strong = self._candidates("strong", running)
        if fast and strong:
            fast_pressure = min(self.pressure(item, running) for item in fast)
            strong_pressure = min(self.pressure(item, running) for item in strong)
            tier = "fast" if fast_pressure <= strong_pressure else "strong"
            return tier, f"{basis}; {tier} pool had lower measured pressure"
        if fast:
            return "fast", f"{basis}; only fast capacity available"
        if strong:
            return "strong", f"{basis}; only strong capacity available"
        return "strong", f"{basis}; compatible fallback required"

    def _candidates(self, tier: str, running: Mapping[str, int]) -> list[ModelInstance]:
        return [
            instance
            for instance in self.inventory.instances
            if instance.capability_tier == tier
            and "worker" in instance.roles
            and instance.available
            and (instance.queued_requests is None or instance.queued_requests <= 0)
            and running.get(instance.instance_id, 0) < instance.effective_limit
        ]

    @staticmethod
    def pressure(instance: ModelInstance, running: Mapping[str, int]) -> float:
        limit = instance.effective_limit
        assigned = running.get(instance.instance_id, 0)
        queued = instance.queued_requests or 0
        generating = 0.15 if instance.generation_status else 0.0
        ttft = instance.measured_profile.get("median_ttft_seconds")
        ttft_penalty = min(float(ttft) / 10, 0.5) if isinstance(ttft, int | float) else 0.0
        return assigned / limit + queued / limit + generating + ttft_penalty


def assignment_counts(tasks: Mapping[str, TaskRuntime]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks.values():
        if task.assigned_instance and task.status.value in {
            "running",
            "validating",
            "reviewing",
            "retrying",
        }:
            counts[task.assigned_instance] = counts.get(task.assigned_instance, 0) + 1
    return counts
