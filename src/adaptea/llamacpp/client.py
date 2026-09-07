from __future__ import annotations

import asyncio
from typing import Any

import httpx

from adaptea.inference.backend import BackendCapabilities, InferenceBackendError
from adaptea.lmstudio.models import ChatResponse, ChatStats, LMModel


class LlamaCppError(InferenceBackendError):
    pass


class LlamaCppClient:
    """llama.cpp (llama-server) adapter normalized to Adaptea's inference contract."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
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
        # llama-server exposes slot-level concurrency (/slots, /props) and supports continuous batching.
        return BackendCapabilities(
            model_lifecycle=False,
            live_pressure_metrics=True,
            safe_default_parallel_limit=4,
        )

    async def __aenter__(self) -> LlamaCppClient:
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
            raise LlamaCppError(f"llama.cpp {method} {path} failed: {exc}") from exc

    async def models(self) -> list[LMModel]:
        # llama-server exposes /props for server/model properties and /v1/models for OpenAI listing
        props_task = asyncio.create_task(self._safe_props())
        openai_task = asyncio.create_task(self._safe_openai_models())
        slots_task = asyncio.create_task(self._safe_slots())

        props, openai_data, slots = await asyncio.gather(props_task, openai_task, slots_task)
        total_slots = len(slots) if isinstance(slots, list) else None

        default_settings = (
            props.get("default_generation_settings", {}) if isinstance(props, dict) else {}
        )
        n_ctx = default_settings.get("n_ctx") if isinstance(default_settings, dict) else None
        if not n_ctx and isinstance(props, dict):
            n_ctx = props.get("n_ctx") or props.get("total_slots")

        # Extract model key from OpenAI models, props, or fallback
        model_keys: list[str] = []
        if isinstance(openai_data, dict) and isinstance(openai_data.get("data"), list):
            for item in openai_data["data"]:
                if isinstance(item, dict) and (key := item.get("id")):
                    model_keys.append(str(key))

        if not model_keys and isinstance(props, dict):
            model_path = props.get("model_path") or props.get("model_alias") or props.get("model")
            if isinstance(model_path, str) and model_path.strip():
                model_keys.append(model_path.strip().rsplit("/", 1)[-1].rsplit("\\", 1)[-1])

        if not model_keys:
            model_keys.append("default")

        return [
            LMModel(
                type="llm",
                key=key,
                display_name=key,
                architecture="gguf",
                quantization=None,
                max_context_length=int(n_ctx) if isinstance(n_ctx, int) else None,
                format="gguf",
                capabilities={"completion": True, "tools": True},
                inference_ready=True,
                size_bytes=None,
                size_vram_bytes=None,
                context_length=int(n_ctx) if isinstance(n_ctx, int) else None,
                parallel_limit=total_slots,
            )
            for key in model_keys
        ]

    async def _safe_props(self) -> dict[str, Any]:
        try:
            res = await self._request("GET", "/props")
            return res if isinstance(res, dict) else {}
        except Exception:
            return {}

    async def _safe_slots(self) -> list[dict[str, Any]]:
        try:
            res = await self._request("GET", "/slots")
            return res if isinstance(res, list) else []
        except Exception:
            return []

    async def _safe_openai_models(self) -> dict[str, Any]:
        try:
            res = await self._request("GET", "/v1/models")
            return res if isinstance(res, dict) else {}
        except Exception:
            return {}

    async def openai_models(self) -> dict[str, Any]:
        value = await self._request("GET", "/v1/models")
        if not isinstance(value, dict):
            raise LlamaCppError("OpenAI-compatible /v1/models returned a non-object")
        return value

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
            raise LlamaCppError("OpenAI-compatible /v1/chat/completions returned a non-object")

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

        # llama.cpp server can include timings in the response
        timings = value.get("timings", {})
        predicted_per_second = (
            timings.get("predicted_per_second") if isinstance(timings, dict) else None
        )
        prompt_ms = timings.get("prompt_ms") if isinstance(timings, dict) else None
        time_to_first_token_seconds = (
            (prompt_ms / 1000.0) if isinstance(prompt_ms, (int, float)) and prompt_ms > 0 else None
        )

        return ChatResponse(
            model_instance_id=str(value.get("model") or model),
            output=([{"type": "message", "content": content}] if isinstance(content, str) else []),
            stats=ChatStats(
                input_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
                total_output_tokens=int(completion_tokens)
                if isinstance(completion_tokens, int)
                else None,
                tokens_per_second=float(predicted_per_second)
                if isinstance(predicted_per_second, (int, float))
                else None,
                model_load_time_seconds=None,
                time_to_first_token_seconds=time_to_first_token_seconds,
            ),
        )
