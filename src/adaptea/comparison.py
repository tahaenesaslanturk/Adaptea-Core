from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from adaptea.config import load_config
from adaptea.git.repository import git
from adaptea.models import Plan, RunState, TaskStatus, utc_now
from adaptea.runtime.controller import Orchestrator, create_run


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    label: str
    root: Path
    state: RunState


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    serial: RunState
    adaptive: RunState
    directory: Path


async def prepare_comparison(
    root: Path, plan: Plan, max_agents: int
) -> tuple[ComparisonRun, ComparisonRun, Path]:
    """Create two isolated worktrees at the exact same clean Git commit."""
    status = await git(root, "status", "--porcelain")
    if status.stdout.strip():
        raise RuntimeError("Comparison requires a clean repository so both runs start identically.")
    comparison_id = f"compare-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
    directory = root / ".adaptea" / "comparisons" / comparison_id
    serial_root = directory / "serial-source"
    adaptive_root = directory / "adaptive-source"
    directory.mkdir(parents=True)
    for target in (serial_root, adaptive_root):
        result = await git(root, "worktree", "add", "--detach", str(target), "HEAD", check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not prepare clean comparison worktree: {result.stderr.strip()}"
            )
        copy_local_configuration(root, target)
    config_serial = load_config(serial_root)
    config_adaptive = load_config(adaptive_root)
    serial_state = await create_run(
        serial_root, config_serial, plan, "fixed", max_agents=1, concurrency=1
    )
    adaptive_state = await create_run(
        adaptive_root, config_adaptive, plan, "adaptive", max_agents=max_agents, concurrency=None
    )
    manifest = {
        "comparison_id": comparison_id,
        "created_at": utc_now(),
        "source_repository": str(root),
        "source_commit": (await git(root, "rev-parse", "HEAD")).stdout.strip(),
        "serial": {"root": str(serial_root), "run_id": serial_state.run_id},
        "adaptive": {"root": str(adaptive_root), "run_id": adaptive_state.run_id},
    }
    (directory / "comparison.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return (
        ComparisonRun("A — Serial / C=1", serial_root, serial_state),
        ComparisonRun("B — Adaptive", adaptive_root, adaptive_state),
        directory,
    )


async def execute_comparison(root: Path, plan: Plan, max_agents: int) -> ComparisonResult:
    serial, adaptive, directory = await prepare_comparison(root, plan, max_agents)
    serial_final = await Orchestrator(serial.root, load_config(serial.root), serial.state).run()
    adaptive_final = await Orchestrator(
        adaptive.root, load_config(adaptive.root), adaptive.state
    ).run()
    result = ComparisonResult(serial_final, adaptive_final, directory)
    (directory / "result.json").write_text(
        json.dumps(comparison_summary(result), indent=2) + "\n", encoding="utf-8"
    )
    return result


def comparison_summary(result: ComparisonResult) -> dict[str, object]:
    def summarize(state: RunState) -> dict[str, object]:
        merged = sum(task.status == TaskStatus.MERGED for task in state.tasks.values())
        total = len(state.tasks)
        samples = _runtime_samples(state)
        running = _integer_values(samples, "running")
        targets = _integer_values(samples, "target")
        return {
            "run_id": state.run_id,
            "tasks": total,
            "merged": merged,
            "pass_rate": merged / total if total else 0.0,
            "retries": sum(max(task.attempts - 1, 0) for task in state.tasks.values()),
            "final_target": state.target_concurrency,
            "duration_seconds": max(
                0.0,
                (
                    datetime.fromisoformat(state.updated_at)
                    - datetime.fromisoformat(state.created_at)
                ).total_seconds(),
            ),
            "peak_workers": max(running, default=0),
            "average_workers": sum(running) / len(running) if running else 0.0,
            "target_min": min(targets, default=state.target_concurrency),
            "target_max": max(targets, default=state.target_concurrency),
        }

    return {"serial": summarize(result.serial), "adaptive": summarize(result.adaptive)}


def _runtime_samples(state: RunState) -> list[dict[str, object]]:
    path = Path(state.repository) / ".adaptea" / "runs" / state.run_id / "runtime-status.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _integer_values(rows: list[dict[str, object]], key: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int):
            values.append(value)
    return values


def copy_local_configuration(source: Path, target: Path) -> None:
    for name in ("adaptea.toml", "opencode.json", "opencode.jsonc"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)
    capacity = source / ".adaptea" / "capacity.json"
    if capacity.is_file():
        destination = target / ".adaptea" / "capacity.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(capacity, destination)
