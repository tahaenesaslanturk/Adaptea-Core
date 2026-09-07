from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from adaptea.models import RunState, utc_now


class StateStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "state.json"

    def save(self, state: RunState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state.updated_at = utc_now()
        temporary = self.path.with_suffix(f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
        temporary.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> RunState:
        return RunState.model_validate_json(self.path.read_text(encoding="utf-8"))


def latest_run(root: Path) -> Path | None:
    runs = root / ".adaptea" / "runs"
    if not runs.exists():
        return None
    candidates = [path for path in runs.iterdir() if (path / "state.json").exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
