from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest

from adaptea.config import Config
from adaptea.lmstudio.lms_cli import CommandResult
from adaptea.lmstudio.models import ChatResponse
from adaptea.models import TaskSpec
from adaptea.reviewer.opencode import OpenCodeReviewer, extract_review, reviewer_prompt


def test_reviewer_extracts_structured_decision_from_opencode_events() -> None:
    payload = {"approved": False, "reason": "missing boundary test", "findings": ["test zero"]}
    event = json.dumps({"part": {"text": json.dumps(payload)}})

    approved, reason, findings = extract_review(event)

    assert approved is False
    assert reason == "missing boundary test"
    assert findings == ["test zero"]


def test_reviewer_prompt_is_read_only_and_contains_validation_and_criteria() -> None:
    task = TaskSpec(
        id="auth",
        title="Auth",
        description="Add auth",
        acceptance_criteria=["logout revokes session"],
        risk="high",
    )

    prompt = reviewer_prompt("secure app", task, "12 tests passed", "+ revoke_session()")

    assert "Do not edit files" in prompt
    assert "logout revokes session" in prompt
    assert "12 tests passed" in prompt
    assert "+ revoke_session()" in prompt
    assert '"approved": true' in prompt
    assert "Every finding must name the" in prompt
    assert "Do not reject hypothetical future reuse" in prompt
    assert "inside effects, event handlers" in prompt
    assert "Do not claim a direct library import" in prompt


def test_reviewer_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="valid review JSON"):
        extract_review("Looks fine to me")


def test_rejection_verification_treats_the_first_review_as_an_untrusted_claim() -> None:
    task = TaskSpec(id="ui", title="UI", description="Make animation SSR-safe")

    prompt = reviewer_prompt(
        "safe app",
        task,
        "build passed",
        "+ import ScrollAnimationWrapper from './ScrollAnimationWrapper'",
        proposed_rejection=(
            "direct GSAP import may break SSR",
            ["Projects.jsx may fail if reused elsewhere"],
        ),
    )

    assert "Treat it as an untrusted claim" in prompt
    assert "misreads a wrapper as a direct import" in prompt
    assert "Projects.jsx may fail if reused elsewhere" in prompt


@pytest.mark.asyncio
async def test_a_rejection_requires_a_skeptical_confirmation_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = [
        {"approved": False, "reason": "wrapper may break SSR", "findings": ["UI may fail"]},
        {"approved": True, "reason": "the diff uses a client wrapper", "findings": []},
    ]

    class Backend:
        prompts: list[str] = []

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        async def chat(self, model: str, prompt: str, **kwargs: Any) -> ChatResponse:
            del model, kwargs
            self.prompts.append(prompt)
            decision = decisions[len(self.prompts) - 1]
            return ChatResponse(output=[{"type": "message", "content": json.dumps(decision)}])

    async def staged_diff(*args: Any, **kwargs: Any) -> CommandResult:
        del args, kwargs
        return CommandResult(0, "+ import ClientWrapper from './ClientWrapper'", "")

    backend = Backend()
    monkeypatch.setattr(
        "adaptea.reviewer.opencode.create_inference_backend", lambda *a, **k: backend
    )
    monkeypatch.setattr("adaptea.reviewer.opencode.git", staged_diff)
    reviewer = OpenCodeReviewer(Config(), "reviewer")
    task = TaskSpec(id="ui", title="UI", description="Make it SSR-safe")

    result = await reviewer.run(tmp_path, tmp_path / "artifacts", "safe app", task, "passed")

    assert result.approved is True
    assert len(backend.prompts) == 2
    assert "Treat it as an untrusted claim" in backend.prompts[1]
    assert (tmp_path / "artifacts" / "reviewer-initial-response.json").is_file()
