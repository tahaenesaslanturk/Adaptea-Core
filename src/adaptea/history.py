from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from adaptea.models import RunState, TaskStatus
from adaptea.runtime.controller import prepare_resume
from adaptea.runtime.state import StateStore


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    repository: Path
    goal: str
    scheduler: str
    status: str
    created_at: str
    duration_seconds: float | None
    merged: int
    tasks: int
    resumable: bool
    run_dir: Path


def run_status(state: RunState) -> str:
    statuses = {task.status for task in state.tasks.values()}
    if statuses == {TaskStatus.MERGED}:
        return "COMPLETED"
    if any(
        status
        in {
            TaskStatus.RUNNING,
            TaskStatus.VALIDATING,
            TaskStatus.REVIEWING,
            TaskStatus.RETRYING,
        }
        for status in statuses
    ):
        return "RUNNING"
    if any(
        status in {TaskStatus.FAILED, TaskStatus.NEEDS_MANUAL_RESOLUTION} for status in statuses
    ):
        return "FAILED"
    return "RESUMABLE" if not state.complete else "PARTIAL"


def is_resumable(state: RunState) -> bool:
    return not all(task.status == TaskStatus.MERGED for task in state.tasks.values())


def list_runs(projects: list[Path]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for project in projects:
        runs = project / ".adaptea" / "runs"
        if not runs.is_dir():
            continue
        for directory in runs.iterdir():
            state_path = directory / "state.json"
            if not state_path.is_file():
                continue
            try:
                state = StateStore(directory).load()
                duration = _duration(directory, state)
            except (OSError, ValueError):
                continue
            records.append(
                RunRecord(
                    run_id=state.run_id,
                    repository=Path(state.repository),
                    goal=state.goal,
                    scheduler=state.scheduler,
                    status=run_status(state),
                    created_at=state.created_at,
                    duration_seconds=duration,
                    merged=sum(task.status == TaskStatus.MERGED for task in state.tasks.values()),
                    tasks=len(state.tasks),
                    resumable=is_resumable(state),
                    run_dir=directory,
                )
            )
    return sorted(records, key=lambda row: row.created_at, reverse=True)


def _duration(directory: Path, state: RunState) -> float | None:
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = None
        if isinstance(value, dict):
            if isinstance(value.get("wall_seconds"), (int, float)):
                return float(value["wall_seconds"])
            finished = value.get("finished_at")
            started = value.get("started_at")
            if isinstance(finished, str) and isinstance(started, str):
                try:
                    return max(
                        0.0,
                        (
                            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
                        ).total_seconds(),
                    )
                except (ValueError, TypeError):
                    pass

    task_seconds = [
        task.wall_seconds
        for task in state.tasks.values()
        if isinstance(task.wall_seconds, (int, float)) and task.wall_seconds > 0
    ]
    total_task_seconds = float(sum(task_seconds)) if task_seconds else 0.0

    # If tasks have started_at and ended_at timestamps, calculate execution span
    task_starts = [
        datetime.fromisoformat(task.started_at)
        for task in state.tasks.values()
        if isinstance(task.started_at, str)
    ]
    task_ends = [
        datetime.fromisoformat(task.ended_at)
        for task in state.tasks.values()
        if isinstance(task.ended_at, str)
    ]
    if task_starts and task_ends:
        try:
            earliest = min(task_starts)
            latest = max(task_ends)
            if latest >= earliest:
                span = (latest - earliest).total_seconds()
                # If there was an overnight sleep or long pause between tasks, active task sum is much more accurate
                if total_task_seconds > 0 and span > total_task_seconds * 3:
                    return max(0.0, total_task_seconds)
                return max(0.0, span)
        except Exception:
            pass

    if total_task_seconds > 0:
        return total_task_seconds

    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            finished = value.get("finished_at") if isinstance(value, dict) else None
            if isinstance(finished, str):
                return max(
                    0.0,
                    (
                        datetime.fromisoformat(finished) - datetime.fromisoformat(state.created_at)
                    ).total_seconds(),
                )
        except Exception:
            pass
    return None


def prepare_run_resume(record: RunRecord) -> RunState:
    store = StateStore(record.run_dir)
    state = prepare_resume(store.load(), record.run_dir)
    store.save(state)
    return state


def delete_run_metadata(record: RunRecord) -> None:
    """Delete only metadata files; worktrees and branches are deliberately untouched."""
    removable = {
        "state.json",
        "plan.json",
        "manifest.json",
        "summary.json",
        "events.jsonl",
        "telemetry.jsonl",
        "controller-decisions.jsonl",
        "fleet-controller-decisions.jsonl",
        "routing-decisions.jsonl",
        "runtime-status.jsonl",
        "command-security-decisions.jsonl",
        "failure-decisions.jsonl",
    }
    for path in record.run_dir.iterdir():
        if path.is_file() and path.name in removable:
            path.unlink()
