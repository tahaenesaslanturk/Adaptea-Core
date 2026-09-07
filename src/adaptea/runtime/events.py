from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptea.models import utc_now


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def event(path: Path, kind: str, **data: Any) -> None:
    append_jsonl(path, {"timestamp": utc_now(), "event": kind, **data})
