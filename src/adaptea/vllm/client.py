from __future__ import annotations

from typing import Any

import httpx

from adaptea.inference.backend import BackendCapabilities, InferenceBackendError
from adaptea.lmstudio.models import ChatResponse, ChatStats, LMModel


class VLLMError(InferenceBackendError):
    pass


class VLLMClient:
    """vLLM adapter normalized to Adaptea's inference contract."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        token: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        # vLLM provides live Prometheus metrics (/metrics) for KV-cache pressure and queue depth.
        return BackendCapabilities(
            model_lifecycle=False,
            live_pressure_metrics=True,
            safe_default_parallel_limit=8,
        )

    async def __aenter__(self) -> VLLMClient:
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
            raise VLLMError(f"vLLM {method} {path} failed: {exc}") from exc

    async def models(self) -> list[LMModel]:
        data = await self.openai_models()
        model_rows = data.get("data", []) if isinstance(data, dict) else []
        models: list[LMModel] = []
        for row in model_rows:
            if not isinstance(row, dict):
                continue
            model_id = row.get("id")
            if not isinstance(model_id, str):
                continue
            max_context = row.get("max_model_len")
            models.append(
                LMModel(
                    type="llm",
                    key=model_id,
                    display_name=model_id,
                    architecture=row.get("root") if isinstance(row.get("root"), str) else None,
                    quantization=None,
                    max_context_length=int(max_context) if isinstance(max_context, int) else None,
                    format="vllm",
                    capabilities={"completion": True, "tools": True},
                    inference_ready=True,
                    size_bytes=None,
                    size_vram_bytes=None,
                    context_length=int(max_context) if isinstance(max_context, int) else None,
                    parallel_limit=None,
                )
            )
        if not models:
            models.append(
                LMModel(
                    type="llm",
                    key="default",
                    display_name="default",
                    architecture=None,
                    quantization=None,
                    max_context_length=None,
                    format="vllm",
                    capabilities={"completion": True, "tools": True},
                    inference_ready=True,
                    size_bytes=None,
                    size_vram_bytes=None,
                    context_length=None,
                    parallel_limit=None,
                )
            )
        return models

    async def openai_models(self) -> dict[str, Any]:
        value = await self._request("GET", "/v1/models")
        if not isinstance(value, dict):
            raise VLLMError("OpenAI-compatible /v1/models returned a non-object")
        return value

    async def metrics(self) -> dict[str, float]:
        """Scrape Prometheus /metrics endpoint from vLLM."""
        try:
            response = await self._client.get("/metrics")
            response.raise_for_status()
            text = response.text
        except Exception:
            return {}

        metrics: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]
                try:
                    metrics[name] = float(parts[1])
                except ValueError:
                    pass
        return metrics

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

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False,
        }

        value = await self._request("POST", "/v1/chat/completions", json=payload)
        if not isinstance(value, dict):
            raise VLLMError("OpenAI-compatible /v1/chat/completions returned a non-object")

        choices = value.get("choices")
        content = None
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                msg = first_choice.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")

        usage = value.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None

        return ChatResponse(
            model_instance_id=str(value.get("model") or model),
            output=([{"type": "message", "content": content}] if isinstance(content, str) else []),
            stats=ChatStats(
                input_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
                total_output_tokens=int(completion_tokens)
                if isinstance(completion_tokens, int)
                else None,
                tokens_per_second=None,
                model_load_time_seconds=None,
                time_to_first_token_seconds=None,
            ),
        )
