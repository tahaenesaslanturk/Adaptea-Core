from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoadedConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    context_length: int | None = None
    eval_batch_size: int | None = None
    parallel: int | None = None
    flash_attention: bool | None = None
    num_experts: int | None = None
    offload_kv_cache_to_gpu: bool | None = None


class LoadedInstance(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    config: LoadedConfig = Field(default_factory=LoadedConfig)


class LMModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    key: str
    display_name: str | None = None
    publisher: str | None = None
    architecture: str | None = None
    quantization: dict[str, Any] | None = None
    loaded_instances: list[LoadedInstance] = Field(default_factory=list)
    max_context_length: int | None = None
    format: str | None = None
    capabilities: dict[str, Any] | None = None
    # Provider-neutral facts used by the inference abstraction. LM Studio continues to
    # derive readiness and capacity from loaded_instances when these are absent.
    inference_ready: bool | None = None
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    context_length: int | None = None
    parallel_limit: int | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.loaded_instances)

    @property
    def ready(self) -> bool:
        return self.inference_ready if self.inference_ready is not None else self.loaded

    @property
    def destination(self) -> str:
        return self.loaded_instances[0].id if self.loaded_instances else self.key

    @property
    def effective_context_length(self) -> int | None:
        if self.context_length is not None:
            return self.context_length
        return self.loaded_instances[0].config.context_length if self.loaded_instances else None

    @property
    def effective_parallel_limit(self) -> int | None:
        if self.parallel_limit is not None:
            return self.parallel_limit
        return self.loaded_instances[0].config.parallel if self.loaded_instances else None


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    models: list[LMModel]


class ChatStats(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int | None = None
    total_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_seconds: float | None = None
    model_load_time_seconds: float | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_instance_id: str | None = None
    output: list[dict[str, Any]] = Field(default_factory=list)
    stats: ChatStats = Field(default_factory=ChatStats)
    response_id: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(
            str(item.get("content", "")) for item in self.output if item.get("type") == "message"
        )
