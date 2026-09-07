from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

# OpenCode sends a large system prompt plus every tool schema before a single byte of
# repository content, and the plan agent adds the planning contract on top. A model loaded
# with 32k context overflows on the very first planning request — LM Studio answers "The
# number of tokens to keep from the initial prompt is greater than the context length" —
# which surfaces to the user as a plan that never finishes.
AGENT_CONTEXT_LENGTH = 65536

# Below this a coding agent cannot reliably complete one request, so doctor stops calling it
# ready.
MINIMUM_AGENT_CONTEXT_LENGTH = 32768


class LMStudioConfig(BaseModel):
    base_url: str = "http://127.0.0.1:1234"
    model: str | None = None
    api_token: str | None = None
    lms_executable: str = "lms"
    telemetry_poll_seconds: float = Field(default=1.5, ge=0.25)


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str | None = None
    api_token: str | None = None
    executable: str = "ollama"


class LlamaCppConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8080"
    model: str | None = None
    api_token: str | None = None
    executable: str = "llama-server"
    context_length: int | None = None


class VLLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    model: str | None = None
    api_token: str | None = None
    executable: str = "vllm"


class InferenceConfig(BaseModel):
    """Select the inference implementation without changing provider-specific settings."""

    backend: Literal["lmstudio", "ollama", "llamacpp", "vllm"] = "lmstudio"


class CommandSecurityConfig(BaseModel):
    approved_commands: list[str] = Field(default_factory=list)

    @field_validator("approved_commands")
    @classmethod
    def validate_approvals(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if any(value in {"*", "**"} for value in cleaned):
            raise ValueError("approved_commands cannot approve every command")
        return cleaned


class WorkerConfig(BaseModel):
    provider: str = "opencode"
    executable: str = "opencode"
    max_agents: int = Field(default=8, ge=1)
    timeout_seconds: int = Field(default=900, ge=30, le=7200)
    planner_timeout_seconds: int | None = Field(default=None, ge=60, le=14400)
    default_scheduler: Literal["adaptive", "fixed", "naive"] = "adaptive"
    command_security: CommandSecurityConfig = Field(default_factory=CommandSecurityConfig)

    @property
    def planner_timeout(self) -> int:
        """Planning gets a longer leash than one coding worker.

        A worker is one of many and is expected to finish quickly; planning is a single
        call that gates the entire run, and a large local model can legitimately spend
        far longer on it. Sharing the worker deadline made planning fail on exactly the
        models it matters most for.
        """
        return self.planner_timeout_seconds or max(self.timeout_seconds, 1800)


class ProjectConfig(BaseModel):
    # Empty means "detect the validator this project actually uses"; see adaptea.validation.
    # A hard-coded pytest default rejects every correct change in a repository without tests.
    test_command: list[str] = Field(default_factory=list)
    no_tests_is_failure: bool = False


class ControllerConfig(BaseModel):
    enabled: bool = True
    decision_window_seconds: float = 10.0
    cooldown_seconds: float = 10.0
    pressure_samples: int = 3
    healthy_samples: int = 3
    ttft_degradation_ratio: float = 1.5
    throughput_degradation_ratio: float = 0.7


class CalibrationConfig(BaseModel):
    repetitions: int = Field(default=3, ge=1)
    concurrency: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    seed: int = 2026


def _default_security_keywords() -> list[str]:
    """Substrings that make a task security-sensitive regardless of its declared risk.

    A planner routinely labels "add token refresh" as low complexity, which is true of the
    edit and false of the consequences. These are matched case-insensitively against the
    task title, description, acceptance criteria, and file hints.
    """
    return [
        "auth",
        "credential",
        "password",
        "secret",
        "token",
        "session",
        "cookie",
        "crypt",
        "hash",
        "signature",
        "certificate",
        "permission",
        "privilege",
        "sandbox",
        "subprocess",
        "shell",
        "eval",
        "deserial",
        "sql",
        "injection",
        "xss",
        "csrf",
        "cors",
        "sanitize",
        "escape",
    ]


class ReviewerRoutingConfig(BaseModel):
    """Which reviewer tier judges a task.

    Review is not the same decision as implementation. A fast model is a reasonable
    reviewer for a typo fix and a poor one for an authentication change, so the tier is
    chosen per task rather than once per run. When no fast reviewer is loaded this whole
    policy is inert and the run keeps its single reviewer, which is the pre-existing
    behaviour.
    """

    enabled: bool = True
    high_complexity: Literal["strong", "fast"] = "strong"
    medium_complexity: Literal["strong", "fast"] = "strong"
    low_complexity: Literal["strong", "fast"] = "fast"
    #: Security-sensitive work overrides the complexity answer. Set to "fast" only if you
    #: genuinely want a small model approving security changes.
    security_sensitive: Literal["strong", "fast"] = "strong"
    #: A task the worker had to retry has already proved it is harder than it looked.
    escalate_after_retry: bool = True
    security_keywords: list[str] = Field(default_factory=_default_security_keywords)


class FleetRoutingConfig(BaseModel):
    high_complexity: Literal["strong", "fast", "auto"] = "strong"
    low_complexity: Literal["strong", "fast", "auto"] = "fast"
    medium_complexity: Literal["strong", "fast", "auto"] = "auto"
    escalate_failed_fast_task: bool = True
    reviewer: ReviewerRoutingConfig = Field(default_factory=ReviewerRoutingConfig)


def _default_worker_roles() -> list[Literal["planner", "worker", "reviewer"]]:
    return ["worker"]


class FleetModelConfig(BaseModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tier: Literal["fast", "strong"]
    roles: list[Literal["planner", "worker", "reviewer"]] = Field(
        default_factory=_default_worker_roles
    )
    instances: Annotated[int, Field(ge=1)] | Literal["auto"] = "auto"
    context_length: int | None = Field(default=None, ge=1)
    parallel_limit: int | None = Field(default=None, ge=1)


class FleetConfig(BaseModel):
    enabled: bool = False
    topology: Literal["auto", "explicit"] = "auto"
    max_loaded_instances: int = Field(default=3, ge=1)
    memory_headroom_fraction: float = Field(default=0.20, ge=0.05, le=0.75)
    instance_ttl_seconds: int = Field(default=3600, ge=60)
    minimum_residency_seconds: int = Field(default=300, ge=0)
    models: list[FleetModelConfig] = Field(default_factory=list)
    routing: FleetRoutingConfig = Field(default_factory=FleetRoutingConfig)

    def planner(self) -> FleetModelConfig | None:
        return next((model for model in self.models if "planner" in model.roles), None)

    def reviewer(self) -> FleetModelConfig | None:
        return next(
            (model for model in self.models if "reviewer" in model.roles),
            self.planner(),
        )


class Config(BaseModel):
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    fleet: FleetConfig = Field(default_factory=FleetConfig)


def executable_stem(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.removesuffix(".exe")


def find_config(start: Path) -> Path | None:
    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / "adaptea.toml"
        if candidate.is_file():
            return candidate
    return None


def load_config(root: Path, explicit: Path | None = None) -> Config:
    # ``explicit`` may deliberately name a project-local file that does not exist yet.
    # In that case the project starts from defaults; falling through to a parent config
    # would leak another project's model fleet into a newly opened folder.
    path = explicit if explicit is not None else find_config(root)
    data: dict[str, object] = {}
    if path and path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    config = Config.model_validate(data)
    config.lmstudio.api_token = os.getenv("LM_API_TOKEN", config.lmstudio.api_token)
    config.ollama.api_token = os.getenv("OLLAMA_API_KEY", config.ollama.api_token)
    config.llamacpp.api_token = os.getenv("LLAMACPP_API_KEY", config.llamacpp.api_token)
    config.vllm.api_token = os.getenv("VLLM_API_KEY", config.vllm.api_token)
    if executable := os.getenv("ADAPTEA_OPENCODE_EXECUTABLE"):
        config.worker.executable = executable
    elif (
        config.worker.executable == "opencode"
        and not shutil.which("opencode")
        and shutil.which("opencode2")
    ):
        config.worker.executable = "opencode2"
    if executable := os.getenv("ADAPTEA_LMS_EXECUTABLE"):
        config.lmstudio.lms_executable = executable
    if base_url := os.getenv("ADAPTEA_LMSTUDIO_BASE_URL"):
        config.lmstudio.base_url = base_url
    if base_url := os.getenv("ADAPTEA_OLLAMA_BASE_URL"):
        config.ollama.base_url = base_url
    if executable := os.getenv("ADAPTEA_OLLAMA_EXECUTABLE"):
        config.ollama.executable = executable
    if base_url := os.getenv("ADAPTEA_LLAMACPP_BASE_URL"):
        config.llamacpp.base_url = base_url
    if executable := os.getenv("ADAPTEA_LLAMACPP_EXECUTABLE"):
        config.llamacpp.executable = executable
    if base_url := os.getenv("ADAPTEA_VLLM_BASE_URL"):
        config.vllm.base_url = base_url
    if executable := os.getenv("ADAPTEA_VLLM_EXECUTABLE"):
        config.vllm.executable = executable
    return config
