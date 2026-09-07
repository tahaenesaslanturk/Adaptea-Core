from pathlib import Path
from types import SimpleNamespace

import pytest

from adaptea.config import Config, FleetModelConfig
from adaptea.inference import configured_model
from adaptea.lmstudio.models import LMModel, LoadedInstance
from adaptea.smoke import LAYER_REMEDIES, VERIFICATION_LAYERS, MVPSmokeTest


def test_default_smoke_fixture_is_a_bounded_hello_world_pipeline() -> None:
    plan = MVPSmokeTest._fixture_plan()

    assert [task.id for task in plan.tasks] == ["page"]
    assert "Hello World" in plan.tasks[0].description


def test_parallel_smoke_fixture_preserves_dependency_coverage() -> None:
    plan = MVPSmokeTest._fixture_plan(parallel=True)

    assert [task.id for task in plan.tasks] == ["page", "styles", "connect-page"]
    assert plan.tasks[-1].depends_on == ["page", "styles"]


def test_verification_chain_order_and_failed_layer_remedy(tmp_path: Path) -> None:
    smoke = MVPSmokeTest(tmp_path, Config())

    smoke.record("Native API", False, "Connection refused.", layer="LM Studio")

    assert VERIFICATION_LAYERS == ("Git", "LM Studio", "Model", "OpenCode", "Adaptea")
    assert smoke.steps[0].layer == "LM Studio"
    assert smoke.steps[0].remedy == LAYER_REMEDIES["LM Studio"]


@pytest.mark.asyncio
async def test_smoke_stops_at_first_failed_verification_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def diagnose(self) -> object:
            return SimpleNamespace(git_executable=None)

    monkeypatch.setattr("adaptea.smoke.SetupManager", Manager)
    smoke = MVPSmokeTest(tmp_path, Config())

    await smoke._run(tmp_path / "fixture")

    assert [(step.layer, step.success) for step in smoke.steps] == [("Git", False)]
    assert smoke.steps[0].remedy == LAYER_REMEDIES["Git"]


class _CapturedRun(Exception):
    """Stops the smoke fixture once create_run has received its configuration."""


def _fleet_config() -> Config:
    """A saved fleet whose planner is loaded while `lmstudio.model` is stale."""
    config = Config()
    config.lmstudio.model = "publisher/unloaded-8b"
    config.fleet.enabled = True
    config.fleet.models = [
        FleetModelConfig(
            name="planner",
            model="publisher/loaded-30b",
            tier="strong",
            roles=["planner", "worker", "reviewer"],
        )
    ]
    return config


def _install_verified_model_stubs(monkeypatch: pytest.MonkeyPatch, model: LMModel) -> None:
    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def diagnose(self) -> object:
            return SimpleNamespace(
                git_executable="/usr/bin/git",
                server_reachable=True,
                native_api_usable=True,
                server_error=None,
                selected_model=SimpleNamespace(key=model.key, ready=True),
                opencode_executable="/usr/local/bin/opencode",
            )

        async def smoke_test(self, _snapshot: object) -> object:
            return SimpleNamespace(success=True, detail="OpenCode recognizes the model.")

    class Client:
        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def models(self) -> list[LMModel]:
            return [model]

        async def chat(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                model_dump=lambda mode="json": {"text": "ADAPTEA_DIRECT_OK"},
            )

    monkeypatch.setattr("adaptea.smoke.SetupManager", Manager)
    monkeypatch.setattr("adaptea.smoke.create_setup_logger", lambda _root: (None, Path("log")))
    monkeypatch.setattr("adaptea.smoke.create_inference_backend", lambda *_a, **_k: Client())


async def test_smoke_fixture_runs_the_model_it_just_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling fleet for the fixture must not fall back to a stale `lmstudio.model`.

    `diagnose` resolves the selected model from the fleet planner, so a fleet whose
    activation unloaded the model named in `lmstudio.model` would verify one model and
    then run the pipeline with another that is no longer inference-ready.
    """
    loaded = LMModel(
        type="llm",
        key="publisher/loaded-30b",
        loaded_instances=[LoadedInstance(id="publisher/loaded-30b:1")],
    )
    _install_verified_model_stubs(monkeypatch, loaded)
    captured: dict[str, Config] = {}

    async def create_run(_root: Path, config: Config, *_args: object, **_kwargs: object) -> object:
        captured["config"] = config
        raise _CapturedRun

    monkeypatch.setattr("adaptea.smoke.create_run", create_run)
    smoke = MVPSmokeTest(tmp_path, _fleet_config())
    fixture = tmp_path / "fixture"
    fixture.mkdir()

    with pytest.raises(_CapturedRun):
        await smoke._run(fixture)

    assert configured_model(captured["config"]) == "publisher/loaded-30b"
