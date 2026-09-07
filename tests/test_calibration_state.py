from __future__ import annotations

import json
from pathlib import Path

from adaptea.calibration.state import calibration_state, configuration_signature
from adaptea.config import Config, FleetModelConfig
from adaptea.fleet.models import FleetInventory, ModelInstance


def _instance(model: str, context: int = 65536, parallel: int = 2) -> ModelInstance:
    return ModelInstance(
        instance_id=f"{model}-1",
        model_key=model,
        capability_tier="strong",
        roles=["planner", "worker", "reviewer"],
        context_length=context,
        parallel_limit=parallel,
    )


def _capacity(root: Path, *models: str, together: bool = False) -> None:
    directory = root / ".adaptea"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capacity.json").write_text(
        json.dumps(
            {
                "model": {"key": models[0]},
                "measured_models": list(models),
                "tested_concurrency": {"1": {}},
                "combined": {"models": list(models), "measured": together},
            }
        ),
        encoding="utf-8",
    )


def test_a_machine_with_no_profile_asks_for_a_quick_measurement(tmp_path: Path) -> None:
    config = Config()
    config.lmstudio.model = "coder-30b"
    state = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-30b")]))
    assert state.state == "missing"
    assert state.suggested == "quick"
    assert state.capacity == "missing"
    # One instance cannot be compared against another topology.
    assert state.topology == "not_required"


def test_changing_the_model_invalidates_the_measured_profile(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    config = Config()
    config.lmstudio.model = "coder-8b"
    state = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-8b")]))
    assert state.state == "stale"
    assert state.suggested == "quick"
    assert "coder-30b" in state.reason and "coder-8b" in state.reason


def test_an_unchanged_model_needs_no_further_measurement(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    config = Config()
    config.lmstudio.model = "coder-30b"
    state = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-30b")]))
    assert state.state == "current"
    assert state.suggested is None
    assert state.measured_model == "coder-30b"


def _two_model_config() -> Config:
    config = Config()
    config.lmstudio.model = "coder-30b"
    config.fleet.enabled = True
    config.fleet.models = [
        FleetModelConfig(
            name="strong", model="coder-30b", tier="strong", roles=["planner", "worker"]
        ),
        FleetModelConfig(name="fast", model="coder-8b", tier="fast", roles=["worker"]),
    ]
    return config


def test_measuring_one_of_two_configured_models_is_not_a_current_profile(
    tmp_path: Path,
) -> None:
    _capacity(tmp_path, "coder-30b")
    inventory = FleetInventory(instances=[_instance("coder-30b"), _instance("coder-8b")])
    state = calibration_state(tmp_path, _two_model_config(), inventory)
    # The old profile called this current because it only ever asked about one model.
    assert state.capacity == "stale"
    assert state.suggested == "quick"
    assert "coder-8b" in state.reason


def test_both_models_measured_separately_still_owe_a_combined_run(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b", "coder-8b")
    inventory = FleetInventory(instances=[_instance("coder-30b"), _instance("coder-8b")])
    state = calibration_state(tmp_path, _two_model_config(), inventory)
    assert state.capacity == "current"
    assert state.topology == "missing"
    assert state.suggested == "quick"


def test_a_fleet_measured_alone_and_together_is_current(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b", "coder-8b", together=True)
    inventory = FleetInventory(instances=[_instance("coder-30b"), _instance("coder-8b")])
    state = calibration_state(tmp_path, _two_model_config(), inventory)
    assert state.state == "current"
    assert state.topology == "current"
    assert state.suggested is None


def test_reloading_the_same_saved_combination_reuses_its_calibration(tmp_path: Path) -> None:
    config = _two_model_config()
    inventory = FleetInventory(instances=[_instance("coder-30b"), _instance("coder-8b")])
    signature = configuration_signature(
        [("coder-30b", 1, 65536, 2), ("coder-8b", 1, 65536, 2)],
        config.worker.max_agents,
    )
    _capacity(tmp_path, "coder-30b", "coder-8b", together=True)
    path = tmp_path / ".adaptea" / "capacity.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["configuration_signature"] = signature
    path.write_text(json.dumps(profile), encoding="utf-8")

    first = calibration_state(tmp_path, config, inventory)
    reloaded = calibration_state(tmp_path, config, inventory)

    assert first.signature == reloaded.signature == signature
    assert reloaded.state == "current"
    assert reloaded.suggested is None


def test_the_signature_changes_only_when_a_new_measurement_is_warranted(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    config = Config()
    config.lmstudio.model = "coder-30b"
    first = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-30b")]))
    same = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-30b")]))
    assert first.signature == same.signature
    changed = calibration_state(
        tmp_path, config, FleetInventory(instances=[_instance("coder-30b", context=32768)])
    )
    assert changed.signature != first.signature


def test_a_profile_is_reused_until_the_model_runtime_shape_changes(tmp_path: Path) -> None:
    signature = configuration_signature([("coder-30b", 1, 65536, 2)], 8)
    _capacity(tmp_path, "coder-30b")
    path = tmp_path / ".adaptea" / "capacity.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["configuration_signature"] = signature
    path.write_text(json.dumps(profile), encoding="utf-8")
    config = Config()
    config.lmstudio.model = "coder-30b"

    same = calibration_state(tmp_path, config, FleetInventory(instances=[_instance("coder-30b")]))
    changed = calibration_state(
        tmp_path,
        config,
        FleetInventory(instances=[_instance("coder-30b", context=32768)]),
    )

    assert same.state == "current" and same.suggested is None
    assert changed.state == "stale" and changed.suggested == "quick"


def test_a_partly_loaded_fleet_waits_for_the_complete_saved_selection(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    # Save owns loading the whole selection. Measuring the first model while the second is
    # still loading would produce a stale-on-arrival profile.
    inventory = FleetInventory(instances=[_instance("coder-30b")])
    state = calibration_state(tmp_path, _two_model_config(), inventory)
    assert state.ready is False
    assert state.pending_models == ["coder-8b"]
    assert state.suggested is None
    assert "complete set" in state.reason


def test_nothing_loaded_is_nothing_to_measure(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    state = calibration_state(tmp_path, _two_model_config(), FleetInventory(instances=[]))
    assert state.ready is False
    assert state.suggested is None


def test_measurement_is_offered_once_every_configured_model_is_loaded(tmp_path: Path) -> None:
    _capacity(tmp_path, "coder-30b")
    inventory = FleetInventory(instances=[_instance("coder-30b"), _instance("coder-8b")])
    state = calibration_state(tmp_path, _two_model_config(), inventory)
    assert state.ready is True
    assert state.pending_models == []
    assert state.suggested == "quick"
