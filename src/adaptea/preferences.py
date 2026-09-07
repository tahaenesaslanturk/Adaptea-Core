from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ProjectPreferences(BaseModel):
    """Desktop behavior that belongs to one project but should not dirty its Git tree."""

    # Workers always commit on private worktree branches because that is how runs stay
    # resumable. This setting controls only what happens to the user's branch after the
    # reviewed integration branch passes. Push is deliberately a non-force push to the
    # conventional ``origin`` remote; a rejection leaves the local commit intact.
    completion_action: Literal["none", "commit", "commit_and_push"] = "commit"
    # Some users want to inspect/edit every generated plan; others want a submitted chat
    # message to flow straight into execution. Keep review-first as the conservative
    # default and persist the choice outside the project working tree.
    auto_start_plans: bool = False
    # Stopping a run terminates its workers, which is what ends generation. Whether the
    # models themselves are unloaded is a separate question, and the answer that matches
    # how a run is actually stopped — to change something and start again — is to keep
    # them loaded: reloading a local model costs minutes of the next run.
    stop_models_on_stop: bool = False
    # Unload models from memory/VRAM whenever a run finishes (completes or fails)
    stop_models_on_finish: bool = False
    # Which saved model combination this project runs on. The combinations themselves are
    # global — one machine, one set of downloaded models, one measurement per set — so a
    # project stores only the choice. None means "whatever the environment has selected",
    # which is what almost every project wants and what a new project starts with.
    model_combination_id: str | None = None

    @property
    def auto_commit_completed_runs(self) -> bool:
        """Compatibility for older callers while the boolean preference is retired."""
        return self.completion_action != "none"


def preferences_path(root: Path) -> Path:
    git_entry = root / ".git"
    git_directory = git_entry
    if git_entry.is_file():
        try:
            marker, raw_path = git_entry.read_text(encoding="utf-8").strip().split(":", 1)
        except (OSError, ValueError):
            marker = ""
            raw_path = ""
        if marker.lower() == "gitdir" and raw_path.strip():
            candidate = Path(raw_path.strip())
            git_directory = candidate if candidate.is_absolute() else root / candidate
    if git_directory.is_dir():
        # Git metadata is local to this checkout and can never make the working tree dirty.
        return git_directory.resolve() / "adaptea" / "preferences.json"
    return root / ".adaptea" / "preferences.json"


def load_project_preferences(root: Path) -> ProjectPreferences:
    try:
        value = json.loads(preferences_path(root).read_text(encoding="utf-8"))
        # 0.5.x stored a boolean. Preserve the user's choice instead of letting a saved
        # ``false`` fall back to the new local-commit default after an upgrade.
        if isinstance(value, dict) and "completion_action" not in value:
            legacy = value.get("auto_commit_completed_runs")
            if isinstance(legacy, bool):
                value["completion_action"] = "commit" if legacy else "none"
        return ProjectPreferences.model_validate(value)
    except (OSError, ValueError):
        return ProjectPreferences()


def save_project_preferences(root: Path, preferences: ProjectPreferences) -> Path:
    path = preferences_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(preferences.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
