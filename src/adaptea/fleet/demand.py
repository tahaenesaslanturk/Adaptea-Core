"""How many instances of each configured model a specific plan can actually keep busy.

The configured instance count says what the user allows, not what the work needs. Loading
every allowed instance for a plan whose tasks are all one tier spends memory on a model
that will sit idle, and the memory is exactly what a second model needed. So the counts
are derived here from the plan's own shape and the measured profile, and are only ever a
reduction: the user's configuration remains the ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from adaptea.config import Config
from adaptea.models import Plan, TaskSpec

#: A task counted toward both tiers, because routing decides at admission time.
EITHER = "either"


def planned_tier(task: TaskSpec, config: Config) -> str:
    """The tier this task will ask for, following the same rules the router follows.

    Where the policy itself says "auto", the answer genuinely is not known until
    admission reads live pressure, so the task counts toward both tiers rather than
    being guessed into one.
    """
    if task.preferred_tier != "auto":
        return task.preferred_tier
    policy = config.fleet.routing
    if task.complexity == "high" or task.risk == "high":
        configured = policy.high_complexity
    elif task.complexity == "low" and task.risk == "low":
        configured = policy.low_complexity
    else:
        configured = policy.medium_complexity
    return EITHER if configured == "auto" else configured


def peak_parallel_tasks(plan: Plan) -> int:
    """The widest set of tasks the dependency graph ever allows to run at once."""
    remaining = {task.id: set(task.depends_on) for task in plan.tasks}
    widest = 0
    done: set[str] = set()
    while remaining:
        ready = [task for task, blockers in remaining.items() if blockers <= done]
        if not ready:
            # A cycle cannot reach here (Plan validates its DAG), but never loop forever.
            break
        widest = max(widest, len(ready))
        done.update(ready)
        for task in ready:
            del remaining[task]
    return widest


def instance_demand(
    plan: Plan, config: Config, capacity: Mapping[str, Any] | None = None
) -> dict[str, int]:
    """Instances to load per model key, never above what the user configured."""
    configured = config.fleet.models
    if not configured:
        return {}
    ceiling = min(config.worker.max_agents, max(1, peak_parallel_tasks(plan)))
    measured = (capacity or {}).get("safe_max_concurrency")
    if isinstance(measured, int) and measured > 0:
        ceiling = min(ceiling, measured)
    ceiling = max(1, ceiling)

    tiers = [planned_tier(task, config) for task in plan.tasks]
    wanted = {
        "strong": sum(1 for tier in tiers if tier in {"strong", EITHER}),
        "fast": sum(1 for tier in tiers if tier in {"fast", EITHER}),
    }
    total = sum(wanted.values()) or 1

    demand: dict[str, int] = {}
    for item in configured:
        allowed = item.instances if isinstance(item.instances, int) else 1
        essential = bool({"planner", "reviewer"} & set(item.roles))
        if "worker" not in item.roles:
            # Planning and review are one call at a time; a second instance buys nothing.
            demand[item.model] = min(allowed, 1)
            continue
        share = wanted.get(item.tier, 0)
        if not share:
            demand[item.model] = 1 if essential else 0
            continue
        workers = max(1, math.ceil(ceiling * share / total))
        per_instance = max(1, item.parallel_limit or 1)
        needed = max(1, math.ceil(workers / per_instance))
        demand[item.model] = min(allowed, needed)

    budget = config.fleet.max_loaded_instances
    while sum(demand.values()) > budget:
        # Give back from whichever model has the most, never below what a role requires.
        reducible = {
            model: count for model, count in demand.items() if count > _minimum(model, configured)
        }
        if not reducible:
            break
        largest = max(reducible, key=lambda model: reducible[model])
        demand[largest] -= 1
    return demand


def _minimum(model: str, configured: list[Any]) -> int:
    item = next((entry for entry in configured if entry.model == model), None)
    if item is None:
        return 0
    return 1 if {"planner", "reviewer"} & set(item.roles) else 0
