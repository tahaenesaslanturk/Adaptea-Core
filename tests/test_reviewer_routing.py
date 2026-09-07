"""Task-based reviewer routing: low-risk work to Fast, consequential work to Strong."""

from __future__ import annotations

import pytest

from adaptea.config import Config, ReviewerRoutingConfig
from adaptea.fleet.models import FleetInventory, ModelInstance
from adaptea.fleet.reviewer_routing import ReviewerRouter, requested_tier, security_matches
from adaptea.models import TaskSpec


def instance(identifier: str, tier: str, *roles: str) -> ModelInstance:
    return ModelInstance(
        instance_id=identifier,
        model_key=f"model-{tier}",
        capability_tier=tier,  # type: ignore[arg-type]
        roles=list(roles) or ["worker"],  # type: ignore[arg-type]
        parallel_limit=2,
    )


def inventory(*instances: ModelInstance) -> FleetInventory:
    return FleetInventory(instances=list(instances))


def task(**overrides: object) -> TaskSpec:
    base: dict[str, object] = {
        "id": "t1",
        "title": "Rename a helper",
        "description": "Rename the helper for clarity.",
        "risk": "medium",
        "complexity": "medium",
    }
    base.update(overrides)
    return TaskSpec.model_validate(base)


def both_tiers() -> FleetInventory:
    return inventory(
        instance("strong-1", "strong", "planner", "worker", "reviewer"),
        instance("fast-1", "fast", "worker", "reviewer"),
    )


def router(inv: FleetInventory | None = None, policy: ReviewerRoutingConfig | None = None):
    return ReviewerRouter(
        inv or both_tiers(),
        policy or ReviewerRoutingConfig(),
        default_instance="strong-1",
        default_model="model-strong",
    )


# --- tier choice -------------------------------------------------------------------


def test_low_risk_simple_work_reviews_on_the_fast_tier() -> None:
    decision = router().route(task(risk="low", complexity="low"))
    assert decision.tier == "fast"
    assert decision.instance == "fast-1"
    assert "low risk and complexity" in decision.reason


def test_high_risk_or_complex_work_reviews_on_the_strong_tier() -> None:
    for overrides in ({"risk": "high"}, {"complexity": "high"}):
        decision = router().route(task(**overrides))
        assert decision.tier == "strong", overrides
        assert "high risk or complexity" in decision.reason


def test_medium_work_defaults_to_strong() -> None:
    """Review is the last gate before merge; ambiguity resolves upward, not downward."""
    assert router().route(task()).tier == "strong"


def test_security_sensitive_work_overrides_a_low_complexity_label() -> None:
    """A planner calls "add token refresh" simple. True of the edit, not the blast radius."""
    decision = router().route(
        task(risk="low", complexity="low", description="Add refresh for the auth token.")
    )
    assert decision.tier == "strong"
    assert "security-sensitive" in decision.reason
    assert "auth" in decision.security_matches


def test_security_keywords_are_matched_across_every_task_field() -> None:
    spec = task(
        title="Tidy up",
        description="Small cleanup.",
        acceptance_criteria=["No SQL injection is possible"],
        files_hint=["src/session/cookie.py"],
    )
    matches = security_matches(spec, ReviewerRoutingConfig().security_keywords)
    assert {"sql", "injection", "session", "cookie"} <= set(matches)


def test_security_keywords_are_configurable() -> None:
    policy = ReviewerRoutingConfig(security_keywords=["billing"])
    quiet = task(risk="low", complexity="low", description="Rework the auth token flow.")
    # "auth" is no longer a keyword for this project, so the normal policy applies.
    assert requested_tier(quiet, policy)[0] == "fast"
    loud = task(risk="low", complexity="low", description="Change the billing calculation.")
    assert requested_tier(loud, policy)[0] == "strong"


def test_a_retried_task_reviews_as_strong() -> None:
    decision = router().route(task(risk="low", complexity="low"), attempts=2)
    assert decision.tier == "strong"
    assert "attempt 2" in decision.reason


def test_retry_escalation_can_be_switched_off() -> None:
    policy = ReviewerRoutingConfig(escalate_after_retry=False)
    routed = router(policy=policy).route(task(risk="low", complexity="low"), attempts=3)
    assert routed.tier == "fast"


@pytest.mark.parametrize("tier", ["strong", "fast"])
def test_every_tier_mapping_is_configurable(tier: str) -> None:
    policy = ReviewerRoutingConfig(
        high_complexity=tier,  # type: ignore[arg-type]
        medium_complexity=tier,  # type: ignore[arg-type]
        low_complexity=tier,  # type: ignore[arg-type]
        security_sensitive=tier,  # type: ignore[arg-type]
    )
    assert requested_tier(task(risk="high"), policy)[0] == tier
    assert requested_tier(task(), policy)[0] == tier
    assert requested_tier(task(risk="low", complexity="low"), policy)[0] == tier


# --- instance selection ------------------------------------------------------------


def test_the_least_loaded_instance_of_the_chosen_tier_wins() -> None:
    inv = inventory(
        instance("strong-1", "strong", "reviewer"),
        instance("fast-1", "fast", "worker", "reviewer"),
        instance("fast-2", "fast", "worker", "reviewer"),
    )
    decision = ReviewerRouter(
        inv, ReviewerRoutingConfig(), default_instance="strong-1", default_model="model-strong"
    ).route(task(risk="low", complexity="low"), {"fast-1": 2, "fast-2": 0})
    assert decision.instance == "fast-2"


def test_routing_falls_back_when_the_requested_tier_is_not_loaded() -> None:
    inv = inventory(
        instance("strong-1", "strong", "reviewer"),
        instance("fast-1", "fast", "worker"),  # a worker, but not a reviewer
    )
    decision = ReviewerRouter(
        inv, ReviewerRoutingConfig(), default_instance="strong-1", default_model="model-strong"
    ).route(task(risk="low", complexity="low"))
    assert decision.instance == "strong-1"


# --- compatibility with what existed before ----------------------------------------


def test_a_single_reviewer_run_keeps_its_one_reviewer() -> None:
    """Only one reviewer tier loaded means nothing to route; behaviour is unchanged."""
    inv = inventory(instance("strong-1", "strong", "planner", "worker", "reviewer"))
    routed = ReviewerRouter(
        inv, ReviewerRoutingConfig(), default_instance="strong-1", default_model="model-strong"
    )
    assert not routed.active
    decision = routed.route(task(risk="low", complexity="low"))
    assert decision.instance == "strong-1"
    assert "single reviewer configured" in decision.reason


def test_routing_can_be_disabled_entirely() -> None:
    routed = router(policy=ReviewerRoutingConfig(enabled=False))
    assert not routed.active
    decision = routed.route(task(risk="low", complexity="low"))
    assert decision.instance == "strong-1"
    assert "disabled" in decision.reason


def test_configurations_written_before_reviewer_routing_still_load() -> None:
    legacy = {
        "fleet": {
            "enabled": True,
            "routing": {"high_complexity": "strong", "low_complexity": "fast"},
        }
    }
    config = Config.model_validate(legacy)
    assert config.fleet.routing.reviewer.enabled is True
    assert config.fleet.routing.reviewer.low_complexity == "fast"
    assert config.fleet.routing.high_complexity == "strong"


def test_the_planner_role_is_untouched_by_reviewer_routing() -> None:
    """Planner stays on the strong model; this policy only decides who reviews."""
    inv = both_tiers()
    planners = [i for i in inv.instances if "planner" in i.roles]
    assert [i.capability_tier for i in planners] == ["strong"]
    router(inv).route(task(risk="low", complexity="low"))
    assert [i.capability_tier for i in inv.instances if "planner" in i.roles] == ["strong"]


# --- explainability ----------------------------------------------------------------


def test_every_decision_records_an_auditable_reason() -> None:
    decision = router().route(task(risk="low", complexity="low", description="Rotate the secret."))
    record = decision.record()
    assert record["task"] == "t1"
    assert record["tier"] == "strong"
    assert record["instance"]
    assert "secret" in record["security_matches"]
    assert record["reason"]
    assert record["timestamp"]
