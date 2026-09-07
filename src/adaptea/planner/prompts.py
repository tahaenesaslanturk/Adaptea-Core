from __future__ import annotations

import json

from adaptea.models import Plan


def planner_prompt(
    goal: str, validation_error: str | None = None, previous: Plan | None = None
) -> str:
    schema = json.dumps(Plan.model_json_schema(), indent=2)
    correction = (
        f"\nYour previous output was invalid:\n{validation_error}\nCorrect it."
        if validation_error
        else ""
    )
    # A follow-up message is a revision, not a fresh brief. Handing the planner the plan
    # already on screen keeps the exchange a conversation: the user says what to change
    # and the tasks they approved survive it.
    revision = ""
    if previous is not None:
        revision = (
            "\nThis is a revision. The user already has this plan and is asking you to "
            "change it, not to start over. Keep every task the request does not affect, "
            "keep their IDs stable, and change, add, or remove only what the request "
            "implies.\nCurrent plan:\n" + previous.model_dump_json(indent=2)
        )
    return f"""You are planning work for Adaptea. Inspect this repository read-only.
Do not edit files,
run destructive commands, or implement the goal. Decompose the goal into a small dependency DAG of
independently executable coding tasks. Dependencies must refer to task IDs.
Avoid invented dependencies.
For every task, classify engineering complexity as low, medium, or high and choose
preferred_tier as auto, fast, or strong. Use complexity, risk, and capability needs—not model
parameter counts or title keywords. Never emit a literal model name or LM Studio instance ID;
the runtime router chooses the current destination at admission time.
The repository may be brand new and contain no application source. When it is empty or
unbootstrapped, explicitly include the initial scaffold, dependency/configuration, and
test-bootstrap tasks needed to build the requested application from scratch. Do not require
pre-existing source code.
Return ONLY one JSON object matching this JSON Schema, with no markdown fences:
{schema}

Overall goal: {goal}
{revision}
{correction}
"""
