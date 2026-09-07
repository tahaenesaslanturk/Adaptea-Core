from __future__ import annotations

import asyncio
from typing import Any

import httpx

from adaptea.inference.backend import BackendCapabilities, InferenceBackendError
from adaptea.lmstudio.models import ChatResponse, ChatStats, LMModel


class OllamaError(InferenceBackendError):
    pass


class OllamaClient:
    """Ollama adapter normalized to Adaptea's inference contract."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        token: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        # Ollama does not publish queue pressure or a server concurrency ceiling. One
        # in-flight request is the only defensible automatic default.
        return BackendCapabilities(
            model_lifecycle=False,
            live_pressure_metrics=False,
            safe_default_parallel_limit=1,
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(f"Ollama {method} {path} failed: {exc}") from exc

    async def models(self) -> list[LMModel]:
        tags, running = await asyncio.gather(
            self._request("GET", "/api/tags"),
            self._request("GET", "/api/ps"),
        )
        tag_rows = _object_rows(tags, "models", "/api/tags")
        running_rows = _object_rows(running, "models", "/api/ps")
        active = {
            name: row for row in running_rows if (name := _string(row, "model", "name")) is not None
        }
        named_rows = [(name, row) for row in tag_rows if (name := _model_name(row))]
        details = await asyncio.gather(*(self._show_optional(name) for name, _row in named_rows))
        shown = {name: detail for (name, _row), detail in zip(named_rows, details, strict=True)}
        return [
            _normalize_model(row, active.get(name), shown.get(name, {}))
            for row in tag_rows
            if (name := _model_name(row))
        ]

    async def _show_optional(self, model: str) -> dict[str, Any]:
        try:
            value = await self._request("POST", "/api/show", json={"model": model})
        except OllamaError:
            return {}
        return value if isinstance(value, dict) else {}

    async def openai_models(self) -> dict[str, Any]:
        value = await self._request("GET", "/v1/models")
        if not isinstance(value, dict):
            raise OllamaError("OpenAI-compatible /v1/models returned a non-object")
        return value

    async def unload(self, model: str) -> dict[str, Any]:
        """Unload a model from GPU/RAM by setting keep_alive to 0."""
        try:
            value = await self._request(
                "POST",
                "/api/generate",
                json={"model": model, "keep_alive": 0},
            )
            return value if isinstance(value, dict) else {}
        except OllamaError:
            return {}

    async def chat(
        self,
        model: str,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        value = await self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                },
            },
        )
        if not isinstance(value, dict):
            raise OllamaError("native /api/chat returned a non-object")
        message = value.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        eval_count = _integer(value, "eval_count")
        eval_duration = _integer(value, "eval_duration")
        tokens_per_second = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_count is not None and eval_duration and eval_duration > 0
            else None
        )
        load_duration = _integer(value, "load_duration")
        return ChatResponse(
            model_instance_id=_string(value, "model"),
            output=([{"type": "message", "content": content}] if isinstance(content, str) else []),
            stats=ChatStats(
                input_tokens=_integer(value, "prompt_eval_count"),
                total_output_tokens=eval_count,
                tokens_per_second=tokens_per_second,
                model_load_time_seconds=(
                    load_duration / 1_000_000_000 if load_duration is not None else None
                ),
                # The non-streaming response has no documented time-to-first-token metric.
                time_to_first_token_seconds=None,
            ),
        )


def _normalize_model(
    tag: dict[str, Any], running: dict[str, Any] | None, shown: dict[str, Any]
) -> LMModel:
    name = _model_name(tag)
    assert name is not None
    details = shown.get("details") if isinstance(shown.get("details"), dict) else tag.get("details")
    details = details if isinstance(details, dict) else {}
    model_info = shown.get("model_info")
    model_info = model_info if isinstance(model_info, dict) else {}
    contexts = [
        value
        for key, value in model_info.items()
        if key.endswith(".context_length") and isinstance(value, int) and value > 0
    ]
    active_context = _integer(running or {}, "context_length")
    quantization = _string(details, "quantization_level")
    capabilities = shown.get("capabilities")
    capability_map = (
        {item: True for item in capabilities if isinstance(item, str)}
        if isinstance(capabilities, list)
        else None
    )
    return LMModel(
        type="llm",
        key=name,
        display_name=name,
        architecture=_string(details, "family"),
        quantization={"level": quantization} if quantization else None,
        max_context_length=max(contexts) if contexts else None,
        format=_string(details, "format"),
        capabilities=capability_map,
        inference_ready=True,
        size_bytes=_integer(tag, "size"),
        size_vram_bytes=_integer(running or {}, "size_vram"),
        context_length=active_context,
        parallel_limit=None,
    )


def _object_rows(value: Any, key: str, endpoint: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        raise OllamaError(f"Ollama {endpoint} returned an invalid model list")
    return [row for row in value[key] if isinstance(row, dict)]


def _model_name(row: dict[str, Any]) -> str | None:
    return _string(row, "model", "name")


def _string(row: dict[str, Any], *keys: str) -> str | None:
    return next((row[key] for key in keys if isinstance(row.get(key), str)), None)


def _integer(row: dict[str, Any], *keys: str) -> int | None:
    value = next((row[key] for key in keys if isinstance(row.get(key), int)), None)
    return int(value) if value is not None else None
