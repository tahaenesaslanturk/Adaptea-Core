from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    VALIDATING = "validating"
    REVIEWING = "reviewing"
    MERGED = "merged"
    FAILED = "failed"
    RETRYING = "retrying"
    NEEDS_MANUAL_RESOLUTION = "needs_manual_resolution"
    BLOCKED = "blocked"


class FailureKind(StrEnum):
    MODEL_ERROR = "model_error"
    REVIEW_REJECTION = "review_rejection"
    TIMEOUT = "timeout"
    VALIDATION_FAILURE = "validation_failure"
    DEPENDENCY_PROBLEM = "dependency_problem"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    files_hint: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "medium"
    complexity: Literal["low", "medium", "high"] = "medium"
    preferred_tier: Literal["auto", "fast", "strong"] = "auto"


class MergeConflictState(BaseModel):
    status: Literal["detected", "preparing", "ready", "preparation_failed", "resolved"]
    detected_at: str = Field(default_factory=utc_now)
    source_branch: str
    integration_branch: str
    original_worktree: str
    resolution_worktree: str
    resolution_branch: str
    conflicting_files: list[str] = Field(default_factory=list)
    related_tasks: list[str] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    preparation_error: str | None = None
    resolved_at: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_dag(self) -> Plan:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"task {task.id} has unknown dependencies: {sorted(missing)}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id} depends on itself")
        visiting: set[str] = set()
        visited: set[str] = set()
        edges = {task.id: task.depends_on for task in self.tasks}

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("task dependency graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for dependency in edges[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for task_id in ids:
            visit(task_id)
        return self


class TaskRuntime(BaseModel):
    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    branch: str | None = None
    worktree: str | None = None
    attempts: int = 0
    started_at: str | None = None
    #: Start of the worker process currently contributing to ``wall_seconds``. This is
    #: separate from the task's first start so live duration does not double-count a
    #: completed attempt while its retry is running.
    attempt_started_at: str | None = None
    ended_at: str | None = None
    validation_exit_code: int | None = None
    validation_passed: bool | None = None
    validation_detail: str | None = None
    validation_command: list[str] = Field(default_factory=list)
    worker_exit_code: int | None = None
    #: The most recent action the coding agent took, so a long task shows what it is doing
    #: rather than only that it is running. Cleared when the attempt ends.
    activity: str | None = None
    summary: str | None = None
    failure: str | None = None
    failure_type: FailureKind | None = None
    failure_reason: str | None = None
    failure_counts: dict[str, int] = Field(default_factory=dict)
    failure_history: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    retry_counts_by_type: dict[str, int] = Field(default_factory=dict)
    retry_exhausted: bool = False
    retry_pending: bool = False
    pending_retry_context: str | None = None
    merge_conflict: MergeConflictState | None = None
    assigned_tier: Literal["fast", "strong"] | None = None
    assigned_model: str | None = None
    assigned_instance: str | None = None
    routing_reason: str | None = None
    routing_history: list[dict[str, Any]] = Field(default_factory=list)
    attempt_history: list[dict[str, Any]] = Field(default_factory=list)
    fast_escalations: int = 0
    fast_escalation_wasted_seconds: float = 0.0
    reviewer_model: str | None = None
    reviewer_instance: str | None = None
    reviewer_tier: str | None = None
    reviewer_routing_reason: str | None = None
    review_approved: bool | None = None
    reviewer_reason: str | None = None
    reviewer_findings: list[str] = Field(default_factory=list)
    review_attempts: int = 0
    review_rejections: int = 0
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    wall_seconds: float | None = None


class RunState(BaseModel):
    run_id: str
    goal: str
    repository: str
    integration_branch: str
    source_branch: str | None = None
    source_commit: str | None = None
    scheduler: Literal["adaptive", "fixed", "naive"]
    target_concurrency: int
    #: An adaptive run's explicitly requested starting point, when the user set one.
    #: None means "start from the measured profile", which is 1 until anything is measured.
    starting_concurrency: int | None = None
    user_ceiling: int
    parallel_limit: int
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    tasks: dict[str, TaskRuntime]
    fleet_enabled: bool = False
    planner_model: str | None = None
    reviewer_model: str | None = None
    fleet_topology: dict[str, Any] = Field(default_factory=dict)
    applied_branch: str | None = None
    applied_commit: str | None = None
    applied_at: str | None = None
    pushed_remote: str | None = None
    pushed_branch: str | None = None
    pushed_at: str | None = None

    def refresh_readiness(self) -> None:
        merged = {key for key, task in self.tasks.items() if task.status == TaskStatus.MERGED}
        terminal_failures = {
            key
            for key, task in self.tasks.items()
            if task.status in {TaskStatus.FAILED, TaskStatus.NEEDS_MANUAL_RESOLUTION}
        }
        for task in self.tasks.values():
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.BLOCKED}:
                continue
            blocked_dependencies = [
                dependency
                for dependency in task.spec.depends_on
                if dependency in terminal_failures
                or self.tasks[dependency].status == TaskStatus.BLOCKED
            ]
            if blocked_dependencies:
                task.status = TaskStatus.BLOCKED
                detail = ", ".join(
                    f"{dependency} "
                    f"({self.tasks[dependency].failure_type or self.tasks[dependency].status})"
                    for dependency in blocked_dependencies
                )
                reason = f"blocked by failed dependencies: {detail}"
                if (
                    task.failure_type != FailureKind.DEPENDENCY_PROBLEM
                    or task.failure_reason != reason
                ):
                    task.failure_counts[FailureKind.DEPENDENCY_PROBLEM.value] = (
                        task.failure_counts.get(FailureKind.DEPENDENCY_PROBLEM.value, 0) + 1
                    )
                    task.failure_history.append(
                        {
                            "timestamp": utc_now(),
                            "attempt": task.attempts,
                            "type": FailureKind.DEPENDENCY_PROBLEM.value,
                            "reason": reason,
                            "decision": "blocked",
                            "retry": False,
                            "max_retries": 0,
                        }
                    )
                task.failure_type = FailureKind.DEPENDENCY_PROBLEM
                task.failure_reason = reason
                task.failure = (
                    f"Dependency problem: {reason}. Automatic retry is disabled because the "
                    "prerequisite must succeed first."
                )
                task.retry_exhausted = True
            elif all(dep in merged for dep in task.spec.depends_on):
                task.status = TaskStatus.READY
            else:
                task.status = TaskStatus.PENDING
        self.updated_at = utc_now()

    @property
    def complete(self) -> bool:
        terminal = {
            TaskStatus.MERGED,
            TaskStatus.FAILED,
            TaskStatus.NEEDS_MANUAL_RESOLUTION,
            TaskStatus.BLOCKED,
        }
        return all(task.status in terminal for task in self.tasks.values())


class TelemetrySample(BaseModel):
    timestamp: str = Field(default_factory=utc_now)
    generating: bool | None = None
    queued_predictions: int | None = None
    tokens_per_second: float | None = None
    ttft_seconds: float | None = None
    source: Literal["native", "lms_ps", "lms_log", "runtime"]
    raw: dict[str, Any] = Field(default_factory=dict)


class ControllerDecision(BaseModel):
    timestamp: str = Field(default_factory=utc_now)
    old_target: int
    new_target: int
    reason: str
    signals: dict[str, Any]


def safe_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
