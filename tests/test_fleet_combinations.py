from __future__ import annotations

import json
from pathlib import Path

from adaptea.config import FleetConfig, FleetModelConfig
from adaptea.fleet.combinations import (
    active_report,
    capture_active_capacity,
    delete_combination,
    list_combinations,
    save_combination,
    select_combination,
)


def _fleet(model: str, name: str = "strong") -> FleetConfig:
    return FleetConfig(
        enabled=True,
        topology="explicit",
        models=[
            FleetModelConfig(
                name=name,
                model=model,
                tier="strong",
                roles=["planner", "worker", "reviewer"],
                instances=1,
                context_length=65_536,
            )
        ],
    )


def _measure(root: Path, model: str) -> tuple[dict[str, object], dict[str, object]]:
    profile: dict[str, object] = {
        "model": {"key": model},
        "measured_models": [model],
        "tested_concurrency": {"1": {}},
    }
    report: dict[str, object] = {
        "profile": {
            "recommended_starting_concurrency": 1,
            "safe_max_concurrency": 2,
        },
        "models": [model],
    }
    directory = root / ".adaptea"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capacity.json").write_text(json.dumps(profile), encoding="utf-8")
    capture_active_capacity(root, report)
    return profile, report


def test_selection_restores_only_that_combinations_capacity(tmp_path: Path) -> None:
    first = save_combination(tmp_path, _fleet("coder-30b"), name="Strong")
    first_profile, first_report = _measure(tmp_path, "coder-30b")
    second = save_combination(tmp_path, _fleet("coder-8b"), name="Fast")

    library = list_combinations(tmp_path)
    assert library["active_id"] == second["id"]
    assert [item["capacity"] for item in library["combinations"]] == ["current", "missing"]

    selected_fleet, selected = select_combination(tmp_path, str(second["id"]))
    assert selected_fleet.models[0].model == "coder-8b"
    assert selected["capacity"] == "missing"
    assert not (tmp_path / ".adaptea" / "capacity.json").exists()
    assert active_report(tmp_path) == (second["id"], None)

    restored_fleet, restored = select_combination(tmp_path, str(first["id"]))
    assert restored_fleet.models[0].model == "coder-30b"
    assert restored["capacity"] == "current"
    assert json.loads((tmp_path / ".adaptea" / "capacity.json").read_text()) == first_profile
    assert active_report(tmp_path) == (first["id"], first_report)


def test_save_as_new_reuses_capacity_only_for_the_same_fleet(tmp_path: Path) -> None:
    original = save_combination(tmp_path, _fleet("coder-30b"), name="Original")
    _measure(tmp_path, "coder-30b")

    copy = save_combination(tmp_path, _fleet("coder-30b"), name="Copy")
    different = save_combination(tmp_path, _fleet("coder-8b"), name="Different")

    assert original["capacity"] == "missing"
    assert copy["capacity"] == "current"
    assert different["capacity"] == "missing"


def test_first_save_adopts_a_matching_existing_capacity_profile(tmp_path: Path) -> None:
    profile = {
        "measured_models": ["coder-30b"],
        "configuration_signature": "coder-30b:1:65536:-",
        "recommended_starting_concurrency": 1,
        "safe_max_concurrency": 2,
    }
    directory = tmp_path / ".adaptea"
    directory.mkdir(parents=True)
    (directory / "capacity.json").write_text(json.dumps(profile), encoding="utf-8")

    saved = save_combination(tmp_path, _fleet("coder-30b"), name="Existing setup")

    assert saved["capacity"] == "current"
    assert saved["recommended_starting_concurrency"] == 1
    assert saved["safe_max_concurrency"] == 2


def test_deleting_the_active_combination_keeps_other_saved_entries(tmp_path: Path) -> None:
    first = save_combination(tmp_path, _fleet("coder-30b"), name="Strong")
    second = save_combination(tmp_path, _fleet("coder-8b"), name="Fast")

    delete_combination(tmp_path, str(second["id"]))

    library = list_combinations(tmp_path)
    assert library["active_id"] is None
    assert [item["id"] for item in library["combinations"]] == [first["id"]]
