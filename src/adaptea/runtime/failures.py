from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from adaptea.models import FailureKind


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    max_retries: int
    escalate_to_strong: bool
    explanation: str


DEFAULT_FAILURE_POLICIES: dict[FailureKind, FailurePolicy] = {
    FailureKind.MODEL_ERROR: FailurePolicy(
        max_retries=1,
        escalate_to_strong=True,
        explanation="retry once; prefer a strong worker model when available",
    ),
    FailureKind.REVIEW_REJECTION: FailurePolicy(
        max_retries=1,
        escalate_to_strong=True,
        explanation="retry once with the review evidence; prefer a strong model when available",
    ),
    FailureKind.TIMEOUT: FailurePolicy(
        max_retries=1,
        escalate_to_strong=True,
        explanation="retry once; prefer a strong worker model when available",
    ),
    FailureKind.VALIDATION_FAILURE: FailurePolicy(
        max_retries=1,
        escalate_to_strong=True,
        explanation="retry once with validator output; prefer a strong model when available",
    ),
    FailureKind.DEPENDENCY_PROBLEM: FailurePolicy(
        max_retries=0,
        escalate_to_strong=False,
        explanation="do not retry until the failed prerequisite is resolved",
    ),
    FailureKind.INFRASTRUCTURE_ERROR: FailurePolicy(
        max_retries=1,
        escalate_to_strong=False,
        explanation="retry once on the same tier after recreating the worktree",
    ),
}
DEFAULT_MAX_TOTAL_RETRIES = 3


@dataclass(frozen=True, slots=True)
class RetryDecision:
    failure_type: FailureKind
    retry: bool
    retries_used_for_type: int
    max_retries_for_type: int
    total_retries_used: int
    max_total_retries: int
    escalate_to_strong: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failure_type"] = self.failure_type.value
        return value


def retry_policy_document(
    policies: Mapping[FailureKind, FailurePolicy] = DEFAULT_FAILURE_POLICIES,
    max_total_retries: int = DEFAULT_MAX_TOTAL_RETRIES,
) -> dict[str, Any]:
    return {
        "policy_version": 1,
        "max_total_retries_per_task": max_total_retries,
        "categories": {
            kind.value: {
                "max_retries": policy.max_retries,
                "escalate_to_strong": policy.escalate_to_strong,
                "explanation": policy.explanation,
            }
            for kind, policy in policies.items()
        },
    }


def load_retry_policy(value: object) -> tuple[dict[FailureKind, FailurePolicy], int]:
    if not isinstance(value, dict):
        return dict(DEFAULT_FAILURE_POLICIES), DEFAULT_MAX_TOTAL_RETRIES
    total = value.get("max_total_retries_per_task")
    categories = value.get("categories")
    if (
        not isinstance(total, int)
        or total < 0
        or total > DEFAULT_MAX_TOTAL_RETRIES
        or not isinstance(categories, dict)
    ):
        return dict(DEFAULT_FAILURE_POLICIES), DEFAULT_MAX_TOTAL_RETRIES
    loaded: dict[FailureKind, FailurePolicy] = {}
    for kind in FailureKind:
        row = categories.get(kind.value)
        if not isinstance(row, dict):
            return dict(DEFAULT_FAILURE_POLICIES), DEFAULT_MAX_TOTAL_RETRIES
        maximum = row.get("max_retries")
        escalate = row.get("escalate_to_strong")
        explanation = row.get("explanation")
        if (
            not isinstance(maximum, int)
            or maximum < 0
            or maximum > DEFAULT_FAILURE_POLICIES[kind].max_retries
            or not isinstance(escalate, bool)
            or not isinstance(explanation, str)
        ):
            return dict(DEFAULT_FAILURE_POLICIES), DEFAULT_MAX_TOTAL_RETRIES
        loaded[kind] = FailurePolicy(maximum, escalate, explanation)
    return loaded, total


def decide_retry(
    failure_type: FailureKind,
    retries_used_for_type: int,
    total_retries_used: int,
    policies: Mapping[FailureKind, FailurePolicy] = DEFAULT_FAILURE_POLICIES,
    max_total_retries: int = DEFAULT_MAX_TOTAL_RETRIES,
) -> RetryDecision:
    policy = policies[failure_type]
    type_budget = retries_used_for_type < policy.max_retries
    total_budget = total_retries_used < max_total_retries
    retry = type_budget and total_budget
    if retry:
        reason = (
            f"retry {retries_used_for_type + 1}/{policy.max_retries} allowed for "
            f"{failure_type.value}; total retry {total_retries_used + 1}/{max_total_retries}"
        )
    elif not type_budget:
        reason = (
            f"{failure_type.value} retry budget exhausted "
            f"({retries_used_for_type}/{policy.max_retries})"
        )
    else:
        reason = f"task retry budget exhausted ({total_retries_used}/{max_total_retries})"
    return RetryDecision(
        failure_type=failure_type,
        retry=retry,
        retries_used_for_type=retries_used_for_type,
        max_retries_for_type=policy.max_retries,
        total_retries_used=total_retries_used,
        max_total_retries=max_total_retries,
        escalate_to_strong=retry and policy.escalate_to_strong,
        reason=reason,
    )


def classify_worker_failure(exit_code: int, stderr: str, stdout: str = "") -> FailureKind:
    if exit_code == 124:
        return FailureKind.TIMEOUT
    normalized = (stderr + "\n" + stdout).lower()
    infrastructure_signals = (
        "connection refused",
        "connection reset",
        "could not resolve host",
        "network is unreachable",
        "no space left on device",
        "permission denied",
        "broken pipe",
        "econnrefused",
        "enospc",
        "no models loaded",
        "apierror",
    )
    if exit_code in {126, 127} or any(signal in normalized for signal in infrastructure_signals):
        return FailureKind.INFRASTRUCTURE_ERROR
    return FailureKind.MODEL_ERROR


def failure_title(kind: FailureKind) -> str:
    return {
        FailureKind.MODEL_ERROR: "Model error",
        FailureKind.REVIEW_REJECTION: "Review rejection",
        FailureKind.TIMEOUT: "Timeout",
        FailureKind.VALIDATION_FAILURE: "Validation failure",
        FailureKind.DEPENDENCY_PROBLEM: "Dependency problem",
        FailureKind.INFRASTRUCTURE_ERROR: "Infrastructure error",
    }[kind]
