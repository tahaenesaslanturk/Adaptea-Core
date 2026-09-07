from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from adaptea.models import utc_now


class InstanceStatus(StrEnum):
    REQUESTED = "requested"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    UNLOADING = "unloading"
    FAILED = "failed"


class DownloadedModel(BaseModel):
    model_key: str
    display_name: str | None = None
    architecture: str | None = None
    size_bytes: int | None = None
    max_context_length: int | None = None
    format: str | None = None
    quantization: dict[str, Any] | str | None = None
    # Absolute only after discovery has verified the path inside a backend-owned model
    # directory. LM Studio files and Hugging Face repository caches use it for recoverable
    # deletion without accepting an arbitrary path from a CLI or API response.
    local_path: str | None = None


class ModelRecommendationAssessment(BaseModel):
    model_key: str
    eligible: bool = False
    selected_as: Literal["fast", "strong"] | None = None
    size_bytes: int | None = None
    max_context_length: int | None = None
    estimated_memory_bytes: int | None = None
    reason: str


class RecommendedFleetModel(BaseModel):
    name: str
    model: str
    tier: Literal["fast", "strong"]
    roles: list[Literal["planner", "worker", "reviewer"]]
    instances: int = 1
    context_length: int
    # The backend reports its actual ceiling after load. A recommendation must not pin a
    # model to one request before the machine has even been measured.
    parallel_limit: int | None = None
    explanation: str


class FleetRecommendation(BaseModel):
    status: Literal["recommended", "insufficient_data", "no_safe_fit"]
    requested_context_length: int
    available_memory_bytes: int | None
    memory_budget_bytes: int | None
    max_loaded_instances: int
    models: list[RecommendedFleetModel] = Field(default_factory=list)
    assessments: list[ModelRecommendationAssessment] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)


class ModelInstance(BaseModel):
    instance_id: str
    model_key: str
    capability_tier: Literal["fast", "strong"]
    roles: list[Literal["planner", "worker", "reviewer"]] = Field(default_factory=list)
    context_length: int | None = None
    parallel_limit: int | None = None
    status: InstanceStatus = InstanceStatus.READY
    queued_requests: int | None = None
    generation_status: bool | None = None
    measured_profile: dict[str, Any] = Field(default_factory=dict)
    running_workers: int = 0
    admission_target: int | None = None

    @property
    def effective_limit(self) -> int:
        return max(1, self.admission_target or self.parallel_limit or 4)

    @property
    def available(self) -> bool:
        return self.status in {InstanceStatus.READY, InstanceStatus.IDLE, InstanceStatus.BUSY}


class FleetInventory(BaseModel):
    downloaded: list[DownloadedModel] = Field(default_factory=list)
    instances: list[ModelInstance] = Field(default_factory=list)
    discovered_at: str = Field(default_factory=utc_now)
    source_notes: list[str] = Field(default_factory=list)

    def instance(self, instance_id: str) -> ModelInstance | None:
        return next((item for item in self.instances if item.instance_id == instance_id), None)


class RoutingDecision(BaseModel):
    timestamp: str = Field(default_factory=utc_now)
    task: str
    tier: Literal["fast", "strong"]
    model: str
    instance: str
    reason: str
    fallback: bool = False
    pressure_score: float
    signals: dict[str, Any] = Field(default_factory=dict)


class TopologyInstance(BaseModel):
    model: str
    tier: Literal["fast", "strong"]
    count: int = Field(ge=1)
    workers_per_instance: int = Field(default=1, ge=1)


class TopologyCandidate(BaseModel):
    id: str
    instances: list[TopologyInstance]
    total_instances: int = Field(ge=1)
    total_workers: int = Field(ge=1)
    feasible: bool = True
    feasibility_reason: str = "within configured bounds"


class TopologyResult(BaseModel):
    candidate: TopologyCandidate
    pass_rate: float | None = None
    median_completion_seconds: float | None = None
    direct_wall_seconds: float | None = None
    median_ttft_seconds: float | None = None
    generation_tokens_per_second: float | None = None
    resource_pressure: float | None = None
    repetitions: int = 0
    successful_repetitions: int = 0
    measurement_success_rate: float | None = Field(default=None, ge=0, le=1)
