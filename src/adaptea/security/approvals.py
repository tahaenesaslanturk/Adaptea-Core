"""Turn a denied command into a question for the user instead of a dead end.

The worker's command policy is deliberately narrow: anything outside the safe-default
list is denied unless the project has approved it in advance. That is the right default
for an unattended run, but it made an ordinary attended run fail on ordinary work — a
``mkdir``, an install, a one-off script — with nothing the user could do except stop,
edit ``adaptea.toml`` by hand, and start again.

This module keeps the same policy and adds the missing third answer. A denial raises an
approval request; the desktop asks the user; an approval is applied to the live policy
(and, when the user wants it remembered, written back to ``adaptea.toml``) so the task's
next attempt runs the command instead of being blocked by it again. Commands in the
blocked category are never asked about: no answer from a user can make ``sudo rm -rf /``
safe, which is what "blocked" means.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from adaptea.config import CommandSecurityConfig
from adaptea.models import utc_now
from adaptea.security.commands import decide_command

#: What the user can answer. "once" approves the command for the rest of this run;
#: "always" also writes it into the project's ``adaptea.toml``.
ApprovalDecision = Literal["once", "always", "deny"]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One command a worker tried to run and was not allowed to."""

    request_id: str
    command: str
    task_id: str
    run_id: str
    requested_at: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ApprovalOutcome:
    request: ApprovalRequest
    decision: ApprovalDecision
    approved: bool
    remembered: bool
    #: Where the approval was persisted, when it was.
    config_path: str | None = None


@dataclass(slots=True)
class CommandApprovalBroker:
    """Collects denied commands, answers them, and applies an approval to the run.

    The broker mutates the very ``CommandSecurityConfig`` the orchestrator hands to each
    worker, so an approval takes effect on the next attempt of the task without
    restarting the run. It deliberately does not interrupt the attempt that was denied:
    OpenCode has already been told no, and killing a worker mid-edit to re-ask is worse
    than letting it finish and retrying with the approval in place.
    """

    root: Path
    security: CommandSecurityConfig
    run_id: str = ""
    #: Notified as soon as a request exists, so the desktop can ask while work continues.
    notify: Callable[[ApprovalRequest], None] | None = None
    pending: dict[str, ApprovalRequest] = field(default_factory=dict)
    #: Approvals granted per task, phrased for the next attempt's retry context.
    notes: dict[str, list[str]] = field(default_factory=dict)
    history: list[ApprovalOutcome] = field(default_factory=list)

    def request(self, command: str, *, task_id: str) -> ApprovalRequest | None:
        """Raise a question about ``command``, or return None when there is none to ask.

        A command that the policy would now allow (because an identical one was already
        approved), one that is blocked outright, and one already awaiting an answer all
        produce no new request.
        """
        normalized = " ".join(command.strip().split())
        if not normalized:
            return None
        decision = decide_command(normalized, self.security)
        if decision.action == "allow" or decision.category == "blocked":
            return None
        if any(item.command == normalized for item in self.pending.values()):
            return None
        request = ApprovalRequest(
            request_id=uuid.uuid4().hex,
            command=normalized,
            task_id=task_id,
            run_id=self.run_id,
            requested_at=utc_now(),
            reason=decision.reason,
        )
        self.pending[request.request_id] = request
        if self.notify is not None:
            # A worker must never fail because nothing is listening for the question.
            try:
                self.notify(request)
            except Exception:  # noqa: BLE001 - notification is best-effort
                pass
        return request

    def resolve(self, request_id: str, decision: ApprovalDecision) -> ApprovalOutcome:
        request = self.pending.pop(request_id, None)
        if request is None:
            raise KeyError(f"No command approval is waiting for an answer: {request_id}")
        approved = decision in ("once", "always")
        written: Path | None = None
        if approved:
            if request.command not in self.security.approved_commands:
                # Mutating the list in place is what makes the approval reach the next
                # attempt: the orchestrator's workers read this same object each launch.
                self.security.approved_commands.append(request.command)
            if decision == "always":
                written = remember_approval(self.root, request.command)
            self.notes.setdefault(request.task_id, []).append(
                f"The user has approved the command `{request.command}`. "
                "It is allowed now; run it if the task still needs it."
            )
        outcome = ApprovalOutcome(
            request=request,
            decision=decision,
            approved=approved,
            remembered=decision == "always",
            config_path=str(written) if written else None,
        )
        self.history.append(outcome)
        return outcome

    def snapshot(self) -> list[ApprovalRequest]:
        """Every question still waiting, so a desktop that reconnects sees them all."""
        return list(self.pending.values())

    def context_for(self, task_id: str) -> list[str]:
        return list(self.notes.get(task_id, ()))


def remember_approval(root: Path, command: str) -> Path | None:
    """Persist one approved command into the project's ``adaptea.toml``.

    Returns the configuration file the approval now lives in. A command that was
    already approved writes nothing and still reports the same file, because the
    question the caller is answering is where the approval is, not whether this call
    is what put it there.
    """
    from adaptea.setup.configuration import merge_approved_commands

    path = root / "adaptea.toml"
    merge_approved_commands(path, [command])
    return path if path.is_file() else None
