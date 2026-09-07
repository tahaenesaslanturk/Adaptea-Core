from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from adaptea.config import Config
from adaptea.lmstudio.client import LMStudioClient, select_loaded_model
from adaptea.lmstudio.lms_cli import parse_ps_json, sample_ps
from adaptea.lmstudio.log_stream import parse_log_event
from adaptea.lmstudio.models import LMModel
from adaptea.lmstudio.telemetry import RuntimeTelemetrySampler, sample_from_models
from adaptea.services import ApplicationServices


@pytest.mark.asyncio
async def test_native_models_auth_and_chat_stats() -> None:
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "future": "ignored",
                    "models": [
                        {
                            "type": "llm",
                            "key": "coder",
                            "display_name": "Coder",
                            "format": "gguf",
                            "max_context_length": 65536,
                            "loaded_instances": [
                                {
                                    "id": "coder@1",
                                    "config": {
                                        "context_length": 8192,
                                        "parallel": 4,
                                        "eval_batch_size": 512,
                                        "flash_attention": True,
                                        "future_field": 42,
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
        if request.url.path == "/api/v1/chat":
            body = json.loads(request.content)
            assert body["store"] is False
            assert body["max_output_tokens"] == 32
            return httpx.Response(
                200,
                json={
                    "model_instance_id": "coder@1",
                    "output": [{"type": "message", "content": "hello"}],
                    "stats": {
                        "input_tokens": 12,
                        "total_output_tokens": 31,
                        "reasoning_output_tokens": 2,
                        "tokens_per_second": 22.5,
                        "time_to_first_token_seconds": 0.4,
                    },
                },
            )
        return httpx.Response(404)

    async with LMStudioClient(
        "http://test", "secret", transport=httpx.MockTransport(handler)
    ) as client:
        models = await client.models()
        selected = select_loaded_model(models, None)
        assert selected is not None
        assert selected.loaded_instances[0].config.parallel == 4
        assert selected.loaded_instances[0].config.context_length == 8192
        response = await client.chat("coder", "hi", max_output_tokens=32)
        assert response.text == "hello"
        assert response.stats.tokens_per_second == 22.5
    assert seen_auth == ["Bearer secret", "Bearer secret"]


def test_tolerant_ps_and_log_parsing() -> None:
    sample = parse_ps_json(
        json.dumps(
            {
                "futureEnvelope": {
                    "models": [{"runtime": {"state": "generating", "queuedRequests": 3}}]
                }
            }
        )
    )
    assert sample.generating is True
    assert sample.queued_predictions == 3
    unknown = parse_ps_json('{"models":[{"newShape":true}]}')
    assert unknown.generating is None
    assert unknown.queued_predictions is None
    logged = parse_log_event(
        '{"type":"llm.prediction.output","stats":{"tokensPerSecond":31.2,"timeToFirstTokenSeconds":0.8}}'
    )
    assert logged is not None and logged.tokens_per_second == 31.2


@pytest.mark.asyncio
async def test_absent_lms_is_degraded_not_fatal() -> None:
    assert await sample_ps("definitely-no-such-lms-executable") is None


def test_native_models_are_a_live_telemetry_fallback() -> None:
    models = [
        LMModel.model_validate(
            {
                "type": "llm",
                "key": "qwen/coder",
                "loaded_instances": [{"id": "qwen/coder@1", "config": {}, "state": "generating"}],
            }
        )
    ]

    sample = sample_from_models(models)

    assert sample.source == "native"
    assert sample.generating is True
    assert sample.raw["models"][0]["key"] == "qwen/coder"


@pytest.mark.asyncio
async def test_native_telemetry_fallback_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LMModel.model_validate(
        {
            "type": "llm",
            "key": "qwen/coder",
            "loaded_instances": [{"id": "qwen/coder@1", "config": {}}],
        }
    )
    calls = 0
    now = 100.0

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def models(self) -> list[LMModel]:
            nonlocal calls
            calls += 1
            return [model]

    async def no_cli_sample(_executable: str) -> None:
        return None

    monkeypatch.setattr("adaptea.lmstudio.telemetry.LMStudioClient", FakeClient)
    monkeypatch.setattr("adaptea.lmstudio.telemetry.sample_ps", no_cli_sample)
    monkeypatch.setattr("adaptea.lmstudio.telemetry.time.monotonic", lambda: now)
    sampler = RuntimeTelemetrySampler("lms", "http://test", native_poll_seconds=10)

    assert await sampler.sample() is not None
    now = 105.0
    assert await sampler.sample() is None
    now = 110.0
    assert await sampler.sample() is not None
    assert calls == 2


@pytest.mark.asyncio
async def test_fast_lmstudio_health_reports_the_loaded_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = LMModel.model_validate(
        {
            "type": "llm",
            "key": "qwen/coder",
            "loaded_instances": [{"id": "qwen/coder@1", "config": {}}],
        }
    )

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def models(self) -> list[LMModel]:
            return [model]

    monkeypatch.setattr(
        "adaptea.services.create_inference_backend",
        lambda *_args, **_kwargs: FakeClient(),
    )
    service = ApplicationServices()
    service.load_config = lambda _root: Config(lmstudio={"model": "qwen/coder"})  # type: ignore[method-assign]

    health = await service.lmstudio_health(tmp_path)

    assert health["reachable"] is True
    assert health["loaded_models"] == ["qwen/coder"]
    assert health["selected_model"] == "qwen/coder"
