from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptea.config import Config
from adaptea.git.repository import git
from adaptea.inference import create_inference_backend
from adaptea.models import TaskSpec, utc_now


@dataclass(slots=True)
class ReviewResult:
    approved: bool
    reason: str
    findings: list[str]
    exit_code: int
    started_at: str
    ended_at: str
    wall_seconds: float


def reviewer_prompt(
    goal: str,
    task: TaskSpec,
    validation_output: str,
    change_set: str = "",
    proposed_rejection: tuple[str, list[str]] | None = None,
) -> str:
    verification = (
        "\nRejection verification:\n"
        "A first review proposed the rejection below. Treat it as an untrusted claim, not "
        "as evidence. Independently verify every claim against the staged diff and the "
        "rules above. Approve the implementation if the proposed rejection is hypothetical, "
        "misreads a wrapper as a direct import, ignores a client boundary, or otherwise is "
        "not proven by changed code.\n"
        f"Proposed reason: {proposed_rejection[0]}\n"
        f"Proposed findings: {json.dumps(proposed_rejection[1])}\n"
        if proposed_rejection
        else ""
    )
    return f"""You are Adaptea's read-only code reviewer.
Inspect the completed task diff in the current Git worktree. Do not edit files.
Deterministic validation already passed; it remains authoritative but does not replace review.
Check the task against its acceptance criteria and identify obvious correctness, security,
regression, and missing-test risks.

Evidence rules for a rejection:
- Reject only for a defect demonstrated by the staged diff. Every finding must name the
  changed file and the exact added expression or behavior that proves the defect.
- Do not reject hypothetical future reuse, general hardening advice, style preferences,
  or claims phrased only as something that "may", "might", "could", or would fail "if"
  used elsewhere. Those are non-blocking observations and must not appear as findings.
- Do not claim a direct library import when the diff imports a project wrapper around that
  library. Do not assume implementation details for an unchanged imported module.
- Respect framework execution boundaries shown by the code. In React/Next.js, browser APIs
  inside effects, event handlers, or explicit client components are not by themselves an
  SSR defect. Reject only when the diff proves browser-only code executes in a server path
  or at unguarded module scope.
- If the available diff cannot prove the concern, approve. Findings are blocking defects,
  not suggestions.

Overall goal: {goal}
Task: {task.title}
Description: {task.description}
Acceptance criteria: {json.dumps(task.acceptance_criteria)}
Risk: {task.risk}
Complexity: {task.complexity}
Validation output (may be truncated):
{validation_output[-12000:] or "validator passed without output"}

Staged Git change set (may be truncated):
{change_set[-24000:] or "no textual diff; verify whether the task was already satisfied"}
{verification}

Return exactly one JSON object and no prose:
{{"approved": true, "reason": "short decision reason", "findings": []}}
or
{{"approved": false, "reason": "short rejection reason", "findings": ["actionable issue"]}}
Approve when the implemented diff satisfies the task and has no evidenced blocking issue.
Reject for a concrete, diff-proven defect only. Do not reject because the project has no
automated test suite, because the change is small, or on stylistic preference alone.
"""


def extract_review(text: str) -> tuple[bool, str, list[str]]:
    decoder = json.JSONDecoder()
    candidates = [text]
    fragments: list[str] = []
    for line in text.splitlines():
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for key in ("text", "content", "message", "output"):
            value = event.get(key)
            if isinstance(value, str):
                candidates.append(value)
                fragments.append(value)
        part = event.get("part")
        if isinstance(part, dict):
            for key in ("text", "content"):
                value = part.get(key)
                if isinstance(value, str):
                    candidates.append(value)
                    fragments.append(value)
    if fragments:
        candidates.append("".join(fragments))
    for candidate in reversed(candidates):
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict) or not isinstance(value.get("approved"), bool):
                continue
            reason = value.get("reason")
            findings = value.get("findings", [])
            if not isinstance(reason, str) or not reason.strip():
                continue
            if not isinstance(findings, list) or not all(
                isinstance(item, str) for item in findings
            ):
                continue
            return value["approved"], reason.strip(), findings
    raise ValueError("reviewer output did not contain a valid review JSON object")


class OpenCodeReviewer:
    """Review a staged worker change through the pinned LM Studio model instance.

    The compatibility name is retained because reviewer routing used to launch a second
    OpenCode process. Sending the already-staged diff directly avoids tool permissions,
    session-root ambiguity, and a second coding-agent loop for this read-only decision.
    """

    def __init__(
        self, config: Config, model: str, selectable_models: list[str] | None = None
    ) -> None:
        self.config = config
        self.model = model
        self.selectable_models = selectable_models

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        validation_output: str,
    ) -> ReviewResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        change_set = (await git(worktree, "diff", "--cached", "--no-ext-diff", "HEAD")).stdout
        started_at = utc_now()
        clock = time.perf_counter()
        async with create_inference_backend(
            self.config, timeout=self.config.worker.timeout_seconds
        ) as client:
            response = await client.chat(
                self.model,
                reviewer_prompt(goal, task, validation_output, change_set),
                max_output_tokens=768,
                temperature=0,
                system_prompt=(
                    "You are a concise read-only code reviewer. Return only the requested JSON."
                ),
            )
            stdout = response.text
            approved, reason, findings = extract_review(stdout)
            if not approved:
                # A rejection can discard minutes of valid parallel work. Require a
                # separate skeptical pass before making it authoritative; this catches
                # plausible-sounding but diff-unsupported findings without weakening a
                # concrete rejection that survives independent verification.
                (artifact_dir / "reviewer-initial-response.json").write_text(
                    response.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                response = await client.chat(
                    self.model,
                    reviewer_prompt(
                        goal,
                        task,
                        validation_output,
                        change_set,
                        proposed_rejection=(reason, findings),
                    ),
                    max_output_tokens=768,
                    temperature=0,
                    system_prompt=(
                        "You are a skeptical read-only code review adjudicator. Return only "
                        "the requested JSON."
                    ),
                )
                stdout = response.text
                approved, reason, findings = extract_review(stdout)
        (artifact_dir / "reviewer-response.json").write_text(
            response.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (artifact_dir / "reviewer-output.log").write_text(stdout + "\n", encoding="utf-8")
        code = 0
        result = ReviewResult(
            approved=approved,
            reason=reason,
            findings=findings,
            exit_code=code,
            started_at=started_at,
            ended_at=utc_now(),
            wall_seconds=time.perf_counter() - clock,
        )
        (artifact_dir / "review.json").write_text(
            json.dumps(
                {
                    "approved": result.approved,
                    "reason": result.reason,
                    "findings": result.findings,
                    "model": self.model,
                    "started_at": result.started_at,
                    "ended_at": result.ended_at,
                    "wall_seconds": result.wall_seconds,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result
