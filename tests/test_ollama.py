from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from adaptea.config import Config
from adaptea.inference import BackendCapabilities
from adaptea.lmstudio.models import LMModel
from adaptea.models import Plan, TaskSpec
from adaptea.ollama.client import OllamaClient
from adaptea.reviewer.opencode import ReviewResult
from adaptea.runtime.controller import Orchestrator, create_run
from adaptea.workers.opencode import WorkerResult


@pytest.mark.asyncio
async def test_ollama_discovers_models_and_only_reports_documented_capacity() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3-coder:latest",
                            "model": "qwen3-coder:latest",
                            "size": 4_200_000_000,
                            "details": {
                                "format": "gguf",
                                "family": "qwen3",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3-coder:latest",
                            "model": "qwen3-coder:latest",
                            "size_vram": 3_800_000_000,
                            "context_length": 32768,
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            assert json.loads(request.content) == {"model": "qwen3-coder:latest"}
            return httpx.Response(
                200,
                json={
                    "details": {
                        "format": "gguf",
                        "family": "qwen3",
                        "quantization_level": "Q4_K_M",
                    },
                    "capabilities": ["completion", "tools"],
                    "model_info": {"qwen3.context_length": 131072},
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(404)

    async with OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler)) as client:
        models = await client.models()
        catalog = await client.openai_models()

    assert catalog["object"] == "list"
    assert requests == [
        ("GET", "/api/tags"),
        ("GET", "/api/ps"),
        ("POST", "/api/show"),
        ("GET", "/v1/models"),
    ]
    assert len(models) == 1
    model = models[0]
    assert model.key == "qwen3-coder:latest"
    assert model.ready is True
    assert model.loaded is False
    assert model.destination == model.key
    assert model.size_bytes == 4_200_000_000
    assert model.size_vram_bytes == 3_800_000_000
    assert model.context_length == 32768
    assert model.max_context_length == 131072
    assert model.effective_parallel_limit is None
    assert model.capabilities == {"completion": True, "tools": True}


@pytest.mark.asyncio
async def test_ollama_chat_normalizes_usage_without_inventing_ttft() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen3-coder",
            "messages": [
                {"role": "system", "content": "Return only text."},
                {"role": "user", "content": "hello"},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 64},
        }
        return httpx.Response(
            200,
            json={
                "model": "qwen3-coder",
                "message": {"role": "assistant", "content": "hi"},
                "prompt_eval_count": 12,
                "eval_count": 20,
                "eval_duration": 2_000_000_000,
                "load_duration": 500_000_000,
            },
        )

    async with OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler)) as client:
        response = await client.chat(
            "qwen3-coder",
            "hello",
            max_output_tokens=64,
            system_prompt="Return only text.",
        )

    assert response.text == "hi"
    assert response.stats.input_tokens == 12
    assert response.stats.total_output_tokens == 20
    assert response.stats.tokens_per_second == 10
    assert response.stats.model_load_time_seconds == 0.5
    assert response.stats.time_to_first_token_seconds is None


class FakeOllamaBackend:
    capabilities = BackendCapabilities(
        model_lifecycle=False,
        live_pressure_metrics=False,
        safe_default_parallel_limit=1,
    )

    async def __aenter__(self) -> FakeOllamaBackend:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def models(self) -> list[LMModel]:
        return [LMModel(type="llm", key="qwen3-coder", inference_ready=True)]


class FakeLMStudioBackend(FakeOllamaBackend):
    """Same surface, but advertising the pressure stream LM Studio really does expose."""

    capabilities = BackendCapabilities(
        model_lifecycle=True,
        live_pressure_metrics=True,
        safe_default_parallel_limit=4,
    )

    async def models(self) -> list[LMModel]:
        return [LMModel(type="llm", key="coder", inference_ready=True, max_context_length=65536)]


class _NoopWorker:
    """Produces no changes, so the run reaches its end without touching a model."""

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        dependency_summaries: list[str],
        retry_context: str | None = None,
        progress: Callable[[dict[str, str]], None] | None = None,
    ) -> WorkerResult:
        del worktree, goal, dependency_summaries, retry_context, progress
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return WorkerResult(0, "start", "end", 0.01, "session", f"noop {task.id}")


class _ApprovingReviewer:
    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        validation_output: str,
    ) -> ReviewResult:
        del worktree, artifact_dir, goal, task, validation_output
        return ReviewResult(True, "fixture approval", [], 0, "start", "end", 0.01)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_ollama_unknown_parallel_capacity_forces_safe_adaptive_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    monkeypatch.setattr(
        "adaptea.runtime.controller.create_inference_backend",
        lambda *_args, **_kwargs: FakeOllamaBackend(),
    )
    config = Config(
        inference={"backend": "ollama"},
        ollama={"model": "qwen3-coder"},
        worker={"max_agents": 8},
    )
    plan = Plan(
        goal="Safe fallback",
        tasks=[TaskSpec(id="one", title="One", description="Do one thing")],
    )

    state = await create_run(tmp_path, config, plan, "adaptive", 8, None)

    assert state.parallel_limit == 1
    assert state.target_concurrency == 1
    assert state.planner_model == "qwen3-coder"


@pytest.mark.asyncio
async def test_ollama_run_never_starts_the_lm_studio_log_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama exposes no pressure stream, so the lms process must never be spawned.

    Starting it anyway would either fail noisily on a machine without LM Studio or, worse,
    attach to an unrelated LM Studio instance and feed the controller telemetry that has
    nothing to do with the backend actually serving the run.
    """
    _init_repo(tmp_path)
    started: list[str] = []

    class TattlingObserver:
        def __init__(self, executable: str) -> None:
            started.append(executable)

        async def start(self, _callback: object) -> bool:
            started.append("start")
            return True

        async def stop(self) -> None:
            return None

    monkeypatch.setattr("adaptea.runtime.controller.LogObserver", TattlingObserver)
    monkeypatch.setattr(
        "adaptea.runtime.controller.create_inference_backend",
        lambda *_args, **_kwargs: FakeOllamaBackend(),
    )
    config = Config(
        inference={"backend": "ollama"},
        ollama={"model": "qwen3-coder"},
        worker={"max_agents": 4},
    )
    plan = Plan(goal="No telemetry", tasks=[TaskSpec(id="one", title="One", description="Do it")])
    state = await create_run(tmp_path, config, plan, "adaptive", 4, None)

    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker = _NoopWorker()  # type: ignore[assignment]
    orchestrator.reviewer = _ApprovingReviewer()  # type: ignore[assignment]
    await orchestrator.run()

    assert started == [], f"the lms log observer was constructed for an Ollama run: {started}"

    events = [
        json.loads(line)
        for line in (orchestrator.run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    degraded = next(event for event in events if event["event"] == "telemetry_degraded")
    # The reason must name the real cause rather than blaming a missing lms binary.
    assert degraded["source"] == "backend does not expose pressure"
    assert degraded["safe_target"] == 1


@pytest.mark.asyncio
async def test_lmstudio_run_still_starts_its_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Ollama gate must not have switched telemetry off for LM Studio as well."""
    _init_repo(tmp_path)
    started: list[str] = []

    class RecordingObserver:
        def __init__(self, executable: str) -> None:
            started.append(executable)

        async def start(self, _callback: object) -> bool:
            return True

        async def stop(self) -> None:
            return None

    monkeypatch.setattr("adaptea.runtime.controller.LogObserver", RecordingObserver)
    monkeypatch.setattr(
        "adaptea.runtime.controller.create_inference_backend",
        lambda *_args, **_kwargs: FakeLMStudioBackend(),
    )
    config = Config(
        lmstudio={"model": "coder", "lms_executable": "lms-under-test"},
        worker={"max_agents": 4},
    )
    plan = Plan(goal="Telemetry", tasks=[TaskSpec(id="one", title="One", description="Do it")])
    state = await create_run(tmp_path, config, plan, "adaptive", 4, None)

    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker = _NoopWorker()  # type: ignore[assignment]
    orchestrator.reviewer = _ApprovingReviewer()  # type: ignore[assignment]
    await orchestrator.run()

    assert started == ["lms-under-test"]
