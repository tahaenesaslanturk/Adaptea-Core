from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable

from adaptea.config import Config
from adaptea.lmstudio.models import ChatResponse, LMModel

BackendKind = Literal["lmstudio", "ollama", "llamacpp", "vllm"]
type InferenceModel = LMModel
type InferenceResponse = ChatResponse


class InferenceBackendError(RuntimeError):
    """Base error raised by an inference backend."""


@dataclass(frozen=True, slots=True)
class BackendConnection:
    """Provider details needed by OpenAI-compatible external clients."""

    kind: BackendKind
    display_name: str
    openai_base_url: str
    api_token: str | None = None
    api_token_environment: str | None = None
    fallback_api_key: str | None = None


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Only capability facts that influence safe orchestration decisions."""

    model_lifecycle: bool
    live_pressure_metrics: bool
    safe_default_parallel_limit: int | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    """Small normalized surface used by Adaptea's inference consumers.

    Model lifecycle and telemetry are deliberately excluded: those remain provider-specific.
    """

    @property
    def capabilities(self) -> BackendCapabilities: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def close(self) -> None: ...

    async def models(self) -> list[InferenceModel]: ...

    async def openai_models(self) -> dict[str, Any]: ...

    async def chat(
        self,
        model: str,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0,
        system_prompt: str | None = None,
    ) -> InferenceResponse: ...


def inference_connection(config: Config) -> BackendConnection:
    """Return connection metadata for the selected backend.

    The explicit branch makes adding a backend an exhaustively typed change. Existing configs
    omit ``[inference]`` and therefore continue to select LM Studio.
    """

    if config.inference.backend == "lmstudio":
        return BackendConnection(
            kind="lmstudio",
            display_name="LM Studio (Adaptea)",
            openai_base_url=config.lmstudio.base_url.rstrip("/") + "/v1",
            api_token=config.lmstudio.api_token,
            api_token_environment="LM_API_TOKEN" if config.lmstudio.api_token else None,
        )
    if config.inference.backend == "ollama":
        return BackendConnection(
            kind="ollama",
            display_name="Ollama (Adaptea)",
            openai_base_url=config.ollama.base_url.rstrip("/") + "/v1",
            api_token=config.ollama.api_token,
            api_token_environment="OLLAMA_API_KEY" if config.ollama.api_token else None,
            fallback_api_key="ollama" if config.ollama.api_token is None else None,
        )
    if config.inference.backend == "llamacpp":
        return BackendConnection(
            kind="llamacpp",
            display_name="llama.cpp (Adaptea)",
            openai_base_url=config.llamacpp.base_url.rstrip("/") + "/v1",
            api_token=config.llamacpp.api_token,
            api_token_environment="LLAMACPP_API_KEY" if config.llamacpp.api_token else None,
            fallback_api_key="llamacpp" if config.llamacpp.api_token is None else None,
        )
    if config.inference.backend == "vllm":
        return BackendConnection(
            kind="vllm",
            display_name="vLLM (Adaptea)",
            openai_base_url=config.vllm.base_url.rstrip("/") + "/v1",
            api_token=config.vllm.api_token,
            api_token_environment="VLLM_API_KEY" if config.vllm.api_token else None,
            fallback_api_key="vllm" if config.vllm.api_token is None else None,
        )


def create_inference_backend(config: Config, *, timeout: float = 120.0) -> InferenceBackend:
    """Construct the selected inference backend from application configuration."""

    if config.inference.backend == "lmstudio":
        from adaptea.lmstudio.client import LMStudioClient

        return LMStudioClient(
            config.lmstudio.base_url,
            config.lmstudio.api_token,
            timeout=timeout,
        )
    if config.inference.backend == "ollama":
        from adaptea.ollama.client import OllamaClient

        return OllamaClient(
            config.ollama.base_url,
            config.ollama.api_token,
            timeout=timeout,
        )
    if config.inference.backend == "llamacpp":
        from adaptea.llamacpp.client import LlamaCppClient

        return LlamaCppClient(
            config.llamacpp.base_url,
            config.llamacpp.api_token,
            timeout=timeout,
        )
    if config.inference.backend == "vllm":
        from adaptea.vllm.client import VLLMClient

        return VLLMClient(
            config.vllm.base_url,
            config.vllm.api_token,
            timeout=timeout,
        )


def configured_model(config: Config) -> str | None:
    if config.inference.backend == "lmstudio":
        return config.lmstudio.model
    if config.inference.backend == "ollama":
        return config.ollama.model
    if config.inference.backend == "llamacpp":
        return config.llamacpp.model
    if config.inference.backend == "vllm":
        return config.vllm.model


def set_configured_model(config: Config, model: str) -> None:
    if config.inference.backend == "lmstudio":
        config.lmstudio.model = model
    elif config.inference.backend == "ollama":
        config.ollama.model = model
    elif config.inference.backend == "llamacpp":
        config.llamacpp.model = model
    elif config.inference.backend == "vllm":
        config.vllm.model = model
