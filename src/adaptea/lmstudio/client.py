from __future__ import annotations

from typing import Any

import httpx

from adaptea.inference.backend import BackendCapabilities, InferenceBackendError
from adaptea.lmstudio.models import ChatResponse, LMModel, ModelsResponse


class LMStudioError(InferenceBackendError):
    pass


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
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
        return BackendCapabilities(
            model_lifecycle=True,
            live_pressure_metrics=True,
            safe_default_parallel_limit=4,
        )

    async def __aenter__(self) -> LMStudioClient:
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
            raise LMStudioError(f"LM Studio {method} {path} failed: {exc}") from exc

    async def models(self) -> list[LMModel]:
        data = await self._request("GET", "/api/v1/models")
        return ModelsResponse.model_validate(data).models

    async def openai_models(self) -> dict[str, Any]:
        value = await self._request("GET", "/v1/models")
        if not isinstance(value, dict):
            raise LMStudioError("OpenAI-compatible /v1/models returned a non-object")
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
        payload: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        data = await self._request("POST", "/api/v1/chat", json=payload)
        return ChatResponse.model_validate(data)

    async def load(self, model: str, **load_config: Any) -> dict[str, Any]:
        payload = {"model": model, "echo_load_config": True, **load_config}
        value = await self._request("POST", "/api/v1/models/load", json=payload)
        if not isinstance(value, dict):
            raise LMStudioError("model load returned a non-object")
        return value

    async def unload(self, instance_id: str) -> dict[str, Any]:
        value = await self._request(
            "POST", "/api/v1/models/unload", json={"instance_id": instance_id}
        )
        if not isinstance(value, dict):
            raise LMStudioError("model unload returned a non-object")
        return value


def select_loaded_model(models: list[LMModel], configured: str | None) -> LMModel | None:
    loaded = [model for model in models if model.ready and model.type == "llm"]
    if configured:
        for model in loaded:
            if model.key == configured or any(
                item.id == configured for item in model.loaded_instances
            ):
                return model
        return None
    return loaded[0] if loaded else None
