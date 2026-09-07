"""Choose which reviewer tier judges a task, and say why.

Worker routing asks "who can implement this efficiently". Review asks a different
question: "how much does it cost if this is approved wrongly". A fast model is a
reasonable reviewer for a typo and a poor one for an authentication change, so the tier
is decided per task instead of once per run.

Every decision is explainable, configurable, and written to the run's artifacts. When no
fast reviewer is loaded the policy is inert and the run keeps its single reviewer, which
is exactly the behaviour that existed before task-based routing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from adaptea.config import ReviewerRoutingConfig
from adaptea.fleet.models import FleetInventory, ModelInstance
from adaptea.models import TaskSpec, utc_now


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    task: str
    tier: str
    instance: str
    model: str
    reason: str
    #: Which security keywords fired, so the decision can be audited rather than trusted.
    security_matches: tuple[str, ...] = ()
    fallback: bool = False
    timestamp: str = ""

    def record(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp or utc_now(),
            "task": self.task,
            "tier": self.tier,
            "instance": self.instance,
            "model": self.model,
            "reason": self.reason,
            "security_matches": list(self.security_matches),
            "fallback": self.fallback,
        }


def security_matches(task: TaskSpec, keywords: list[str]) -> tuple[str, ...]:
    """Keywords present in the task's own words, in the order the policy lists them."""
    haystack = " ".join(
        [task.title, task.description, *task.acceptance_criteria, *task.files_hint]
    ).casefold()
    return tuple(word for word in keywords if word.casefold() in haystack)


def requested_tier(
    task: TaskSpec, policy: ReviewerRoutingConfig, *, attempts: int = 1
) -> tuple[str, str, tuple[str, ...]]:
    """Return (tier, reason, security matches) without considering what is loaded."""
    matches = security_matches(task, policy.security_keywords)
    if matches:
        shown = ", ".join(matches[:3]) + ("…" if len(matches) > 3 else "")
        return (
            policy.security_sensitive,
            f"security-sensitive wording ({shown})",
            matches,
        )
    if policy.escalate_after_retry and attempts > 1:
        return "strong", f"attempt {attempts}; a retried task reviews as strong", matches
    if task.risk == "high" or task.complexity == "high":
        return policy.high_complexity, "high risk or complexity", matches
    if task.risk == "low" and task.complexity == "low":
        return policy.low_complexity, "low risk and complexity", matches
    return policy.medium_complexity, "medium risk or complexity", matches


class ReviewerRouter:
    """Pick a loaded reviewer instance for a task, preferring the requested tier."""

    def __init__(
        self,
        inventory: FleetInventory,
        policy: ReviewerRoutingConfig,
        *,
        default_instance: str,
        default_model: str,
    ) -> None:
        self.inventory = inventory
        self.policy = policy
        self.default_instance = default_instance
        self.default_model = default_model

    def _reviewers(self, tier: str) -> list[ModelInstance]:
        return [
            instance
            for instance in self.inventory.instances
            if "reviewer" in instance.roles and instance.capability_tier == tier
        ]

    @property
    def active(self) -> bool:
        """Task-based routing only means anything with reviewers in more than one tier."""
        if not self.policy.enabled:
            return False
        tiers = {
            instance.capability_tier
            for instance in self.inventory.instances
            if "reviewer" in instance.roles
        }
        return len(tiers) > 1

    def route(
        self, task: TaskSpec, running: Mapping[str, int] | None = None, *, attempts: int = 1
    ) -> ReviewerDecision:
        load = running or {}
        tier, reason, matches = requested_tier(task, self.policy, attempts=attempts)
        if not self.active:
            return ReviewerDecision(
                task=task.id,
                tier=tier,
                instance=self.default_instance,
                model=self.default_model,
                reason=(
                    f"{reason}; single reviewer configured, so the run's reviewer is used"
                    if self.policy.enabled
                    else "task-based reviewer routing disabled; run reviewer used"
                ),
                security_matches=matches,
            )
        candidates = self._reviewers(tier)
        fallback = False
        if not candidates:
            candidates = self._reviewers("strong" if tier == "fast" else "fast")
            fallback = bool(candidates)
        if not candidates:
            return ReviewerDecision(
                task=task.id,
                tier=tier,
                instance=self.default_instance,
                model=self.default_model,
                reason=f"{reason}; no reviewer instance loaded, so the run's reviewer is used",
                security_matches=matches,
                fallback=True,
            )
        chosen = min(candidates, key=lambda item: (load.get(item.instance_id, 0), item.instance_id))
        detail = f"{reason}; {tier} reviewer selected"
        if fallback:
            detail += f"; {tier} pool unavailable, fell back to {chosen.capability_tier}"
        return ReviewerDecision(
            task=task.id,
            tier=chosen.capability_tier,
            instance=chosen.instance_id,
            model=chosen.model_key,
            reason=detail,
            security_matches=matches,
            fallback=fallback,
        )
