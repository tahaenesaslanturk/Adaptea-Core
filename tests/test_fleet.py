from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import (
    Config,
    ControllerConfig,
    FleetConfig,
    FleetModelConfig,
    FleetRoutingConfig,
    load_config,
)
from adaptea.fleet.calibration import (
    FleetCalibrationRunner,
    generate_topology_candidates,
    machine_signature,
    profile_is_stale,
    select_topology,
)
from adaptea.fleet.controller import InstanceAdmissionController
from adaptea.fleet.demand import instance_demand
from adaptea.fleet.discovery import (
    discover_huggingface_models,
    huggingface_cache_root,
    inventory_from_models,
    parse_downloaded_models,
    resolve_huggingface_model_path,
)
from adaptea.fleet.lifecycle import (
    InstanceLifecycleManager,
    ensure_configured_instances,
    parse_resource_estimate,
    parse_vm_stat_available,
)
from adaptea.fleet.models import (
    DownloadedModel,
    FleetInventory,
    ModelInstance,
    TopologyCandidate,
    TopologyResult,
)
from adaptea.fleet.recommendation import recommend_fleet
from adaptea.fleet.routing import FleetRouter
from adaptea.lmstudio.lms_cli import CommandResult, parse_instance_pressure
from adaptea.lmstudio.models import LMModel
from adaptea.models import Plan, RunState, TaskRuntime, TaskSpec, TaskStatus
from adaptea.reviewer.opencode import ReviewResult
from adaptea.runtime.controller import Orchestrator
from adaptea.runtime.state import StateStore
from adaptea.services import ApplicationServices
from adaptea.setup.configuration import merge_fleet_config
from adaptea.validation import ValidationOutcome
from adaptea.workers.opencode import WorkerResult


def model(key: str, *instances: tuple[str, int | None]) -> LMModel:
    return LMModel.model_validate(
        {
            "type": "llm",
            "key": key,
            "format": "gguf",
            "loaded_instances": [
                {"id": identifier, "config": {"context_length": 4096, "parallel": parallel}}
                for identifier, parallel in instances
            ],
        }
    )


def fleet_config(*models: FleetModelConfig) -> Config:
    config = Config()
    config.fleet = FleetConfig(enabled=True, models=list(models))
    config.worker.max_agents = 4
    config.lmstudio.telemetry_poll_seconds = 0.01
    config.lmstudio.lms_executable = "missing-lms"
    config.project.test_command = ["python", "-c", "raise SystemExit(0)"]
    return config


def test_discovery_one_many_duplicate_and_missing_optional_fields() -> None:
    configured = [
        FleetModelConfig(
            name="strong",
            model="strong-model",
            tier="strong",
            roles=["planner", "worker"],
        ),
        FleetModelConfig(name="fast", model="fast-model", tier="fast", roles=["worker"]),
    ]
    inventory = inventory_from_models(
        [
            model("strong-model", ("strong-1", 1)),
            model("fast-model", ("fast-1", 2), ("fast-2", None)),
        ],
        configured,
    )
    assert len(inventory.downloaded) == 2
    assert [item.instance_id for item in inventory.instances] == [
        "strong-1",
        "fast-1",
        "fast-2",
    ]
    assert inventory.instance("fast-2").parallel_limit is None  # type: ignore[union-attr]
    assert inventory.instance("fast-1").capability_tier == "fast"  # type: ignore[union-attr]
    with_unconfigured = inventory_from_models(
        [
            model("strong-model", ("strong-1", 1)),
            model("not-in-fleet", ("external-1", 4)),
        ],
        configured,
    )
    assert with_unconfigured.instance("external-1").roles == []  # type: ignore[union-attr]
    # With nothing configured every loaded instance is fair game for work, but exactly one
    # of them plans: three checked planners is a fleet that cannot run, and the desktop
    # reads its rows straight off this inventory.
    unconfigured = inventory_from_models(
        [
            model("first", ("first-1", 1)),
            model("second", ("second-1", 1)),
            model("third", ("third-1", 1)),
        ],
        [],
    )
    assert [item.roles for item in unconfigured.instances] == [
        ["planner", "worker", "reviewer"],
        ["worker", "reviewer"],
        ["worker", "reviewer"],
    ]
    downloaded = parse_downloaded_models(
        json.dumps(
            {
                "models": [
                    {
                        "type": "llm",
                        "modelKey": "one",
                        "sizeBytes": 12,
                        "maxContextLength": 131072,
                    },
                    {"type": "embedding", "modelKey": "skip"},
                ]
            }
        )
    )
    assert [item.model_key for item in downloaded] == ["one"]
    assert downloaded[0].max_context_length == 131072


def test_huggingface_cache_inventory_filters_llamacpp_to_gguf(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    gguf_repo = cache / "models--publisher--coder-GGUF"
    gguf_blob = gguf_repo / "blobs" / "weights"
    gguf_blob.parent.mkdir(parents=True)
    gguf_blob.write_bytes(b"123456")
    snapshot = gguf_repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "coder-Q4_K_M.gguf").symlink_to(gguf_blob)
    transformer_repo = cache / "models--publisher--transformer"
    (transformer_repo / "snapshots" / "revision").mkdir(parents=True)
    (transformer_repo / "snapshots" / "revision" / "model.safetensors").write_bytes(b"x")

    llama_models = discover_huggingface_models("llamacpp", cache)
    vllm_models = discover_huggingface_models("vllm", cache)

    assert [item.model_key for item in llama_models] == ["publisher/coder-GGUF"]
    assert llama_models[0].size_bytes == 6
    assert {item.model_key for item in vllm_models} == {
        "publisher/coder-GGUF",
        "publisher/transformer",
    }
    assert resolve_huggingface_model_path(str(gguf_repo), "llamacpp", cache) == str(
        gguf_repo.resolve()
    )
    assert resolve_huggingface_model_path(str(snapshot), "llamacpp", cache) is None


def test_huggingface_cache_root_matches_backend_environment(tmp_path: Path) -> None:
    assert (
        huggingface_cache_root("llamacpp", {"LLAMA_CACHE": str(tmp_path / "llama")}, tmp_path)
        == tmp_path / "llama"
    )
    assert (
        huggingface_cache_root("vllm", {"HF_HOME": str(tmp_path / "hf")}, tmp_path)
        == tmp_path / "hf" / "hub"
    )


def test_fleet_recommendation_uses_context_estimates_memory_and_lmstudio_limit() -> None:
    models = [
        DownloadedModel(model_key="small", size_bytes=4_000, max_context_length=131072),
        DownloadedModel(model_key="medium", size_bytes=8_000, max_context_length=131072),
        DownloadedModel(model_key="short", size_bytes=12_000, max_context_length=32768),
    ]
    recommendation = recommend_fleet(
        models,
        available_memory_bytes=20_000,
        estimated_memory_bytes={"small": 4_000, "medium": 10_000},
        max_loaded_instances=2,
        memory_headroom_fraction=0.20,
    )
    assert recommendation.status == "recommended"
    assert recommendation.memory_budget_bytes == 16_000
    assert [(item.model, item.tier) for item in recommendation.models] == [
        ("medium", "strong"),
        ("small", "fast"),
    ]
    assert all(item.context_length == 65536 for item in recommendation.models)
    assert all(item.parallel_limit is None for item in recommendation.models)
    short = next(item for item in recommendation.assessments if item.model_key == "short")
    assert short.eligible is False
    assert "below the 65,536-token requirement" in short.reason


def test_fleet_recommendation_refuses_to_guess_missing_facts() -> None:
    recommendation = recommend_fleet(
        [DownloadedModel(model_key="unknown", max_context_length=131072)],
        available_memory_bytes=None,
        estimated_memory_bytes={},
        max_loaded_instances=3,
        memory_headroom_fraction=0.20,
    )
    assert recommendation.status == "insufficient_data"
    assert recommendation.models == []
    assert "model size" in recommendation.assessments[0].reason
    assert "available system memory" in recommendation.assessments[0].reason


def test_fleet_recommendation_does_not_exceed_instance_or_combined_memory_limit() -> None:
    models = [
        DownloadedModel(model_key="small", size_bytes=4_000, max_context_length=65536),
        DownloadedModel(model_key="large", size_bytes=9_000, max_context_length=65536),
    ]
    recommendation = recommend_fleet(
        models,
        available_memory_bytes=12_500,
        estimated_memory_bytes={"small": 4_000, "large": 9_000},
        max_loaded_instances=1,
        memory_headroom_fraction=0.20,
    )
    assert [item.model for item in recommendation.models] == ["large"]


def test_vm_stat_available_memory_parser() -> None:
    assert (
        parse_vm_stat_available(
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               100.\n"
            "Pages inactive:                           200.\n"
            "Pages speculative:                         10.\n"
            "Pages purgeable:                           20.\n"
        )
        == 330 * 16384
    )


@pytest.mark.asyncio
async def test_fleet_status_skips_slow_estimates_on_the_startup_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.lmstudio.model = "small"
    config.fleet.max_loaded_instances = 2
    inventory = FleetInventory(
        downloaded=[
            DownloadedModel(model_key="small", size_bytes=4_000, max_context_length=131072),
            DownloadedModel(model_key="large", size_bytes=8_000, max_context_length=131072),
        ]
    )

    async def discover(_root: Path, _config: Config) -> FleetInventory:
        return inventory

    async def command(*args: str, timeout: float = 10) -> CommandResult:
        del args, timeout
        raise AssertionError("fleet status must not estimate every downloaded model")

    monkeypatch.setattr("adaptea.fleet.discovery.discover_fleet", discover)
    monkeypatch.setattr("adaptea.fleet.lifecycle.available_memory_bytes", lambda: 20_000)
    monkeypatch.setattr("adaptea.lmstudio.lms_cli.run_command", command)
    service = ApplicationServices()
    service.load_config = lambda _root: config  # type: ignore[method-assign]

    status = await service.fleet_status(tmp_path)
    recommendation = status["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["status"] == "insufficient_data"
    assert recommendation["models"] == []
    assert recommendation["requested_context_length"] == 65536
    assert status["configured_models"] == [
        {
            "name": "primary",
            "model": "small",
            "tier": "strong",
            "roles": ["planner", "worker", "reviewer"],
            "instances": 1,
            "context_length": None,
            "parallel_limit": None,
        }
    ]


def test_legacy_and_fleet_configuration_round_trip(tmp_path: Path) -> None:
    legacy = load_config(tmp_path)
    assert legacy.fleet.enabled is False
    path = tmp_path / "adaptea.toml"
    path.write_text('[project]\ntest_command = ["python", "-m", "pytest"]\n', encoding="utf-8")
    fleet = FleetConfig(
        enabled=True,
        topology="explicit",
        models=[
            FleetModelConfig(
                name="planner",
                model="strong",
                tier="strong",
                roles=["planner", "worker"],
                instances=1,
            ),
            FleetModelConfig(
                name="fast", model="fast", tier="fast", roles=["worker"], instances="auto"
            ),
        ],
    )
    merge_fleet_config(path, fleet)
    loaded = load_config(tmp_path)
    assert loaded.fleet.enabled
    assert loaded.fleet.planner().model == "strong"  # type: ignore[union-attr]
    assert loaded.fleet.models[0].instances == 1
    assert loaded.fleet.models[1].instances == "auto"
    assert loaded.project.test_command[-1] == "pytest"


@pytest.mark.asyncio
async def test_saving_fleet_unloads_models_removed_by_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = FleetInventory(
        instances=[
            ModelInstance(
                instance_id="keep-1",
                model_key="keep-model",
                capability_tier="strong",
            ),
            ModelInstance(
                instance_id="remove-1",
                model_key="remove-model",
                capability_tier="fast",
            ),
        ]
    )
    unloaded: list[str] = []

    async def discover(_root: Path, _config: Config) -> FleetInventory:
        return inventory

    class Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def unload(self, instance_id: str) -> dict[str, object]:
            unloaded.append(instance_id)
            return {}

    monkeypatch.setattr("adaptea.fleet.discovery.discover_fleet", discover)
    monkeypatch.setattr("adaptea.services.LMStudioClient", Client)
    service = ApplicationServices()
    configured = Config()
    configured.fleet.enabled = True
    configured.fleet.models = [
        FleetModelConfig(name="keep", model="keep-model", tier="strong"),
        FleetModelConfig(name="remove", model="remove-model", tier="fast"),
    ]
    service.load_config = lambda _root: configured  # type: ignore[method-assign]
    desired = FleetConfig(
        enabled=True,
        models=[FleetModelConfig(name="keep", model="keep-model", tier="strong")],
    )

    assert await service.unload_unconfigured_fleet_models(tmp_path, desired) == ["remove-1"]
    assert unloaded == ["remove-1"]


@pytest.mark.asyncio
async def test_unload_combination_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from adaptea.fleet.combinations import save_combination

    saved = save_combination(
        tmp_path,
        FleetConfig(
            enabled=True,
            models=[FleetModelConfig(name="target", model="target-model", tier="strong")],
        ),
        name="Target Combo",
    )

    inventory = FleetInventory(
        instances=[
            ModelInstance(
                instance_id="target-1",
                model_key="target-model",
                capability_tier="strong",
            ),
            ModelInstance(
                instance_id="other-1",
                model_key="other-model",
                capability_tier="fast",
            ),
        ]
    )
    unloaded: list[str] = []

    async def discover(_root: Path, _config: Config) -> FleetInventory:
        return inventory

    class Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def unload(self, instance_id: str) -> dict[str, object]:
            unloaded.append(instance_id)
            return {}

    monkeypatch.setattr("adaptea.fleet.discovery.discover_fleet", discover)
    monkeypatch.setattr("adaptea.services.LMStudioClient", Client)
    service = ApplicationServices()
    configured = Config()
    service.load_config = lambda _root: configured  # type: ignore[method-assign]

    result = await service.unload_combination_models(tmp_path, str(saved["id"]))
    assert result == ["target-1"]
    assert unloaded == ["target-1"]


@pytest.mark.asyncio
async def test_stop_models_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = FleetInventory(
        instances=[
            ModelInstance(
                instance_id="model-inst-1",
                model_key="active-model",
                capability_tier="strong",
            ),
        ]
    )
    unloaded: list[str] = []

    async def discover(_root: Path, _config: Config) -> FleetInventory:
        return inventory

    class Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def unload(self, instance_id: str) -> dict[str, object]:
            unloaded.append(instance_id)
            return {}

    monkeypatch.setattr("adaptea.fleet.discovery.discover_fleet", discover)
    monkeypatch.setattr("adaptea.services.LMStudioClient", Client)
    service = ApplicationServices()
    configured = Config()
    configured.inference.backend = "lmstudio"
    configured.lmstudio.model = "active-model"
    service.load_config = lambda _root: configured  # type: ignore[method-assign]

    result = await service.stop_models(tmp_path, force=True)
    assert result == ["model-inst-1"]
    assert unloaded == ["model-inst-1"]


@pytest.mark.asyncio
async def test_stop_models_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unloaded: list[str] = []

    class OllamaClientMock:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> OllamaClientMock:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def models(self) -> list[object]:
            return []

        async def unload(self, model: str) -> dict[str, object]:
            unloaded.append(model)
            return {}

    monkeypatch.setattr("adaptea.ollama.client.OllamaClient", OllamaClientMock)
    service = ApplicationServices()
    configured = Config()
    configured.inference.backend = "ollama"
    configured.ollama.model = "qwen3-coder:latest"
    service.load_config = lambda _root: configured  # type: ignore[method-assign]

    result = await service.stop_models(tmp_path, force=True)
    assert result == ["qwen3-coder:latest"]
    assert unloaded == ["qwen3-coder:latest"]


@pytest.mark.asyncio
async def test_ensure_ready_models_loads_unloaded_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def load(self, model: str, identifier: str, **_kwargs: object) -> ModelInstance:
            loaded.append(f"{model}:{identifier}")
            return ModelInstance(
                instance_id=identifier,
                model_key=model,
                capability_tier="strong",
            )

    async def discover(_root: Path, _config: Config) -> FleetInventory:
        return FleetInventory(instances=[])

    monkeypatch.setattr("adaptea.fleet.discovery.discover_fleet", discover)
    monkeypatch.setattr("adaptea.fleet.lifecycle.InstanceLifecycleManager", Manager)

    service = ApplicationServices()
    configured = Config()
    configured.inference.backend = "lmstudio"
    configured.fleet.enabled = False
    configured.lmstudio.model = "qwen/single-model"
    service.load_config = lambda _root: configured  # type: ignore[method-assign]

    assert await service.ensure_ready_models(tmp_path) is True
    assert len(loaded) == 1
    assert "qwen/single-model" in loaded[0]


def routing_inventory() -> FleetInventory:
    return FleetInventory(
        instances=[
            ModelInstance(
                instance_id="fast-1",
                model_key="fast-model",
                capability_tier="fast",
                roles=["worker"],
                parallel_limit=2,
            ),
            ModelInstance(
                instance_id="fast-2",
                model_key="fast-model",
                capability_tier="fast",
                roles=["worker"],
                parallel_limit=1,
            ),
            ModelInstance(
                instance_id="strong-1",
                model_key="strong-model",
                capability_tier="strong",
                roles=["planner", "worker"],
                parallel_limit=1,
            ),
        ]
    )


def test_routing_tiers_overrides_fallback_pressure_and_ceilings() -> None:
    router = FleetRouter(routing_inventory(), FleetRoutingConfig())
    low = TaskSpec(id="low", title="Low", description="Low", risk="low", complexity="low")
    high = TaskSpec(id="high", title="High", description="High", complexity="high")
    assert router.route(low, {}).tier == "fast"  # type: ignore[union-attr]
    assert router.route(high, {}).tier == "strong"  # type: ignore[union-attr]
    override = low.model_copy(update={"preferred_tier": "strong"})
    assert router.route(override, {}).tier == "strong"  # type: ignore[union-attr]
    least = router.route(low, {"fast-1": 1})
    assert least is not None and least.instance == "fast-2"
    assert router.route(low, {"fast-1": 2, "fast-2": 1}).tier == "strong"  # type: ignore[union-attr]
    congested = routing_inventory()
    congested.instance("fast-1").queued_requests = 2  # type: ignore[union-attr]
    assert FleetRouter(congested, FleetRoutingConfig()).route(low, {}).instance == "fast-2"  # type: ignore[union-attr]
    only_strong = FleetInventory(instances=[routing_inventory().instances[-1]])
    fallback = FleetRouter(only_strong, FleetRoutingConfig()).route(low, {})
    assert fallback is not None and fallback.fallback and fallback.tier == "strong"


def test_per_instance_pressure_parser_handles_multiple_shapes() -> None:
    pressure = parse_instance_pressure(
        {
            "models": [
                {"identifier": "a", "runtime": {"queuedRequests": 2, "state": "busy"}},
                {"instance_id": "b", "queued_predictions": 0, "generating": False},
            ]
        }
    )
    assert pressure["a"] == {"queued_requests": 2, "generation_status": True}
    assert pressure["b"] == {"queued_requests": 0, "generation_status": False}


def test_instance_controller_hysteresis_reduces_only_pressured_pool_without_preemption() -> None:
    inventory = routing_inventory()
    fast = inventory.instance("fast-1")
    assert fast is not None
    clock = [0.0]
    controller = InstanceAdmissionController(
        fast,
        ControllerConfig(pressure_samples=2, healthy_samples=2, cooldown_seconds=0),
        clock=lambda: clock[0],
    )
    fast.admission_target = 2
    assert controller.observe(queued_requests=1, generating=True, running=2) is None
    decision = controller.observe(queued_requests=1, generating=True, running=2)
    assert decision is not None and decision.new_target == 1
    assert fast.effective_limit == 1
    assert inventory.instance("fast-2").effective_limit == 1  # type: ignore[union-attr]
    routed = FleetRouter(inventory, FleetRoutingConfig()).route(
        TaskSpec(id="next", title="Next", description="Next", risk="low", complexity="low"),
        {"fast-1": 2},
    )
    assert routed is not None and routed.instance == "fast-2"
    # The two already-running workers remain counted; admission control never preempts them.
    assert decision.signals["running"] == 2


@pytest.mark.asyncio
async def test_lifecycle_estimate_unique_load_failure_and_safe_unload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate = parse_resource_estimate(
        "small", "Estimated Total Memory: 4 GB\nEstimated GPU Memory: 3 GB", 16_000_000_000, 0.2
    )
    assert estimate.safely_fits is True
    current_lms = parse_resource_estimate(
        "small",
        "Estimated GPU Memory: 11.11 GiB\nEstimated Total Memory: 11.11 GiB",
        128 * 1024**3,
        0.2,
    )
    assert current_lms.estimated_total_bytes == int(11.11 * 1024**3)
    assert current_lms.safely_fits is True
    config = fleet_config(
        FleetModelConfig(name="fast", model="small", tier="fast", roles=["worker"])
    )
    inventory = FleetInventory(
        downloaded=[DownloadedModel(model_key="small", size_bytes=100_000_000)]
    )
    calls: list[tuple[str, ...]] = []

    async def command(*args: str, timeout: float = 10) -> CommandResult:
        del timeout
        calls.append(args)
        if "--estimate-only" in args:
            return CommandResult(0, "Estimated Total Memory: 1 GB", "")
        return CommandResult(0, "loaded", "")

    monkeypatch.setattr("adaptea.fleet.lifecycle.run_command", command)
    monkeypatch.setattr("adaptea.fleet.lifecycle.physical_memory_bytes", lambda: 16_000_000_000)
    manager = InstanceLifecycleManager(config, inventory)
    loaded = await manager.load("small", "adaptea-fast-1")
    second = await manager.load("small", "adaptea-fast-2")
    assert loaded.instance_id == "adaptea-fast-1"
    assert second.instance_id == "adaptea-fast-2"
    assert any("--identifier" in call for call in calls)
    assert any(call[1:4] == ("load", "small", "--estimate-only") for call in calls)
    with pytest.raises(ValueError, match="already exists"):
        await manager.load("small", "adaptea-fast-1")
    loaded.running_workers = 1
    with pytest.raises(RuntimeError, match="active coding worker"):
        await manager.unload(loaded)

    async def failure(*args: str, timeout: float = 10) -> CommandResult:
        del timeout
        return (
            CommandResult(0, "Estimated Total Memory: 1 GB", "")
            if "--estimate-only" in args
            else CommandResult(1, "", "guardrail refused")
        )

    loaded.running_workers = 0
    inventory.instances.clear()
    monkeypatch.setattr("adaptea.fleet.lifecycle.run_command", failure)
    with pytest.raises(RuntimeError, match="guardrail refused"):
        await manager.load("small", "failed-instance")


@pytest.mark.asyncio
async def test_user_requested_load_is_not_blocked_by_the_configured_instance_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberate Load must behave like assigning the same model as Strong or Fast.

    The desktop writes ``max_loaded_instances`` as the number of instances the saved fleet
    asks for, so a one-model fleet leaves the ceiling at one. Applying that to a second
    model the user is explicitly turning on made Load report a full fleet, while choosing a
    tier for the same model succeeded — the assignment raised the ceiling on the way past.
    """
    config = fleet_config(
        FleetModelConfig(name="fast", model="small", tier="fast", roles=["worker"])
    )
    config.fleet.max_loaded_instances = 1
    inventory = FleetInventory(
        downloaded=[
            DownloadedModel(model_key="small", size_bytes=100_000_000),
            DownloadedModel(model_key="other", size_bytes=100_000_000),
        ]
    )

    async def command(*args: str, timeout: float = 10) -> CommandResult:
        del timeout
        if "--estimate-only" in args:
            return CommandResult(0, "Estimated Total Memory: 1 GB", "")
        return CommandResult(0, "loaded", "")

    monkeypatch.setattr("adaptea.fleet.lifecycle.run_command", command)
    monkeypatch.setattr("adaptea.fleet.lifecycle.physical_memory_bytes", lambda: 16_000_000_000)
    manager = InstanceLifecycleManager(config, inventory)
    await manager.load("small", "adaptea-small-1")

    # Unattended loading still stops at the ceiling the configuration asked for.
    with pytest.raises(RuntimeError, match="maximum loaded fleet instances"):
        await manager.load("other", "adaptea-other-1")

    second = await manager.load("other", "adaptea-other-1", requested_by_user=True)
    assert second.instance_id == "adaptea-other-1"
    assert len(inventory.instances) == 2

    # Memory is the guard that does not move: an estimate that does not fit still refuses.
    async def too_large(*args: str, timeout: float = 10) -> CommandResult:
        del timeout
        if "--estimate-only" in args:
            return CommandResult(0, "Estimated Total Memory: 64 GB", "")
        return CommandResult(0, "loaded", "")

    monkeypatch.setattr("adaptea.fleet.lifecycle.run_command", too_large)
    with pytest.raises(RuntimeError, match="exceeds"):
        await manager.load("other", "adaptea-other-2", requested_by_user=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "executable",
    [
        "/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms",
        r"C:\Program Files\LM Studio\resources\app\.webpack\lms.exe",
    ],
)
async def test_lifecycle_process_paths_remain_single_arguments(
    executable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fleet_config(
        FleetModelConfig(name="fast", model="small", tier="fast", roles=["worker"])
    )
    config.lmstudio.lms_executable = executable
    calls: list[tuple[str, ...]] = []

    async def command(*args: str, timeout: float = 10) -> CommandResult:
        del timeout
        calls.append(args)
        return CommandResult(0, "Estimated Total Memory: 1 GB", "")

    monkeypatch.setattr("adaptea.fleet.lifecycle.run_command", command)
    monkeypatch.setattr("adaptea.fleet.lifecycle.physical_memory_bytes", lambda: 16_000_000_000)
    await InstanceLifecycleManager(config, FleetInventory()).estimate("small", 4096)
    assert calls == [(executable, "load", "small", "--estimate-only", "--context-length", "4096")]


@pytest.mark.asyncio
async def test_configured_instance_identifier_uses_model_name_not_internal_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fleet_config(
        FleetModelConfig(
            name="fast-1",
            model="qwen/qwen3.8-27b",
            tier="fast",
            roles=["worker"],
            instances=1,
        )
    )
    inventory = FleetInventory()
    identifiers: list[str] = []

    async def load(
        _self: InstanceLifecycleManager,
        model: str,
        identifier: str,
        *,
        context_length: int | None = None,
        explicit_override: bool = False,
    ) -> ModelInstance:
        del context_length, explicit_override
        identifiers.append(identifier)
        instance = ModelInstance(
            instance_id=identifier,
            model_key=model,
            capability_tier="fast",
        )
        inventory.instances.append(instance)
        return instance

    monkeypatch.setattr(InstanceLifecycleManager, "load", load)
    assert await ensure_configured_instances(tmp_path, config, inventory)
    assert identifiers == ["adaptea-qwen3.8-27b-1"]


def test_topology_candidates_feasibility_selection_and_staleness() -> None:
    config = fleet_config(
        FleetModelConfig(name="strong", model="strong", tier="strong", roles=["planner", "worker"]),
        FleetModelConfig(name="fast", model="fast", tier="fast", roles=["worker"]),
    )
    inventory = FleetInventory(
        downloaded=[DownloadedModel(model_key="strong"), DownloadedModel(model_key="fast")],
        instances=[
            ModelInstance(
                instance_id="strong-1",
                model_key="strong",
                capability_tier="strong",
                roles=["planner", "worker"],
                parallel_limit=4,
            ),
            ModelInstance(
                instance_id="fast-1",
                model_key="fast",
                capability_tier="fast",
                roles=["worker"],
                parallel_limit=2,
            ),
        ],
    )
    candidates = generate_topology_candidates(config, inventory, {"strong": 1, "fast": 1})
    assert {row.id for row in candidates} >= {"strong-1", "fast-1", "strong-1-fast-1"}
    assert next(row for row in candidates if row.id == "strong-1-fast-2").feasible is False
    expanded = generate_topology_candidates(config, inventory)
    expanded_ids = {row.id for row in expanded}
    assert {"strong-1-w4", "strong-2x-w1", "fast-1-w2", "fast-2x-w1"} <= expanded_ids
    assert "strong-1-fast-1-sw1-fw2" in expanded_ids
    assert "strong-1-fast-1-sw2-fw2" in expanded_ids
    good = TopologyResult(
        candidate=candidates[0], pass_rate=1.0, median_completion_seconds=20, repetitions=3
    )
    fast_bad = TopologyResult(
        candidate=candidates[1], pass_rate=0.8, median_completion_seconds=10, repetitions=3
    )
    selected, reason = select_topology([good, fast_bad])
    assert selected is good
    assert "Validation" in reason
    profile = {"machine_signature": machine_signature(inventory)}
    assert not profile_is_stale(profile, inventory)
    inventory.instances[0].context_length = 8192
    assert profile_is_stale(profile, inventory)


def test_topology_selection_rejects_a_fast_but_unrepeatable_measurement() -> None:
    stable = TopologyResult(
        candidate=TopologyCandidate(
            id="stable",
            instances=[],
            total_instances=1,
            total_workers=1,
        ),
        generation_tokens_per_second=20,
        repetitions=3,
        successful_repetitions=3,
        measurement_success_rate=1.0,
    )
    flaky = TopologyResult(
        candidate=TopologyCandidate(
            id="flaky",
            instances=[],
            total_instances=1,
            total_workers=2,
        ),
        generation_tokens_per_second=100,
        repetitions=3,
        successful_repetitions=1,
        measurement_success_rate=1 / 3,
    )
    selected, reason = select_topology([flaky, stable])
    assert selected is stable
    assert "repeatable" in reason


@pytest.mark.asyncio
async def test_mocked_interleaved_fleet_calibration_writes_profile(tmp_path: Path) -> None:
    config = fleet_config(
        FleetModelConfig(name="single", model="coder", tier="strong", roles=["planner", "worker"])
    )
    inventory = FleetInventory(
        downloaded=[DownloadedModel(model_key="coder", format="gguf")],
        instances=[
            ModelInstance(
                instance_id="coder-1",
                model_key="coder",
                capability_tier="strong",
                roles=["planner", "worker"],
                parallel_limit=4,
            )
        ],
    )
    seen: list[str] = []

    async def benchmark(candidate: TopologyCandidate) -> TopologyResult:
        seen.append(candidate.id)
        return TopologyResult(
            candidate=candidate,
            median_ttft_seconds=1.0,
            generation_tokens_per_second=float(candidate.total_workers),
            repetitions=1,
        )

    async def validate(candidate: TopologyCandidate) -> tuple[float, float, int]:
        return 1.0, 30.0 / candidate.total_workers, 0

    runner = FleetCalibrationRunner(
        tmp_path,
        config,
        inventory,
        client=object(),  # type: ignore[arg-type]
        benchmark=benchmark,
        agent_validator=validate,
    )
    directory = await runner.run(repetitions=2, safe_instance_limits={"coder": 2})
    assert directory.is_dir()
    assert len(seen) >= 4
    profile = json.loads((tmp_path / ".adaptea" / "fleet.json").read_text())
    assert profile["profile_version"] == 2
    assert profile["recommended_topology"]


@pytest.mark.asyncio
async def test_agent_validation_shortlist_covers_each_topology_family(tmp_path: Path) -> None:
    config = fleet_config(
        FleetModelConfig(
            name="strong",
            model="strong",
            tier="strong",
            roles=["planner", "worker"],
        ),
        FleetModelConfig(name="fast", model="fast", tier="fast", roles=["worker"]),
    )
    config.fleet.max_loaded_instances = 3
    inventory = FleetInventory(
        instances=[
            ModelInstance(
                instance_id="strong-1",
                model_key="strong",
                capability_tier="strong",
                roles=["planner", "worker"],
                parallel_limit=2,
            ),
            ModelInstance(
                instance_id="fast-1",
                model_key="fast",
                capability_tier="fast",
                roles=["worker"],
                parallel_limit=2,
            ),
        ]
    )
    validated: list[TopologyCandidate] = []

    async def benchmark(candidate: TopologyCandidate) -> TopologyResult:
        return TopologyResult(
            candidate=candidate,
            generation_tokens_per_second=float(candidate.total_workers),
            repetitions=1,
        )

    async def validate(candidate: TopologyCandidate) -> tuple[float, float, int]:
        validated.append(candidate)
        return 1.0, 10.0, 0

    runner = FleetCalibrationRunner(
        tmp_path,
        config,
        inventory,
        client=object(),  # type: ignore[arg-type]
        benchmark=benchmark,
        agent_validator=validate,
    )
    await runner.run(repetitions=1, safe_instance_limits={"strong": 2, "fast": 2})
    families = {
        "replicated"
        if any(row.count > 1 for row in candidate.instances)
        else "continuous_batching"
        if any(row.workers_per_instance > 1 for row in candidate.instances)
        else "heterogeneous"
        if len({row.model for row in candidate.instances}) > 1
        else "continuous_batching"
        for candidate in validated
    }
    assert families == {"continuous_batching", "replicated", "heterogeneous"}


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    (path / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


class FleetFakeWorker:
    def __init__(self, destination: str) -> None:
        self.destination = destination

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        dependency_summaries: list[str],
        retry_context: str | None = None,
        progress: Callable[[dict[str, str]], None] | None = None,
    ) -> WorkerResult:
        del goal, dependency_summaries, retry_context, progress
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (worktree / f"{task.id}.txt").write_text(self.destination, encoding="utf-8")
        return WorkerResult(0, "start", "end", 0.01, None, f"done on {self.destination}")


class FleetFakeReviewer:
    def __init__(self, approvals: list[bool] | None = None) -> None:
        self.approvals = approvals or [True]
        self.calls = 0

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        validation_output: str,
    ) -> ReviewResult:
        del worktree, artifact_dir, goal, task, validation_output
        approved = self.approvals[min(self.calls, len(self.approvals) - 1)]
        self.calls += 1
        return ReviewResult(
            approved,
            "approved" if approved else "missing regression coverage",
            [] if approved else ["add a regression test"],
            0,
            "start",
            "end",
            0.01,
        )


def make_fleet_state(root: Path, plan: Plan, inventory: FleetInventory, target: int) -> RunState:
    state = RunState(
        run_id="run-fleet",
        goal=plan.goal,
        repository=str(root),
        integration_branch="adaptea/run-fleet/integration",
        scheduler="fixed",
        target_concurrency=target,
        user_ceiling=target,
        parallel_limit=target,
        tasks={task.id: TaskRuntime(spec=task) for task in plan.tasks},
        fleet_enabled=True,
        planner_model="strong-1",
        reviewer_model="strong-1",
        fleet_topology=inventory.model_dump(mode="json"),
    )
    state.refresh_readiness()
    run_dir = root / ".adaptea" / "runs" / state.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"model": "strong-model"}), encoding="utf-8")
    (run_dir / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    for name in (
        "events.jsonl",
        "telemetry.jsonl",
        "controller-decisions.jsonl",
        "routing-decisions.jsonl",
    ):
        (run_dir / name).touch()
    StateStore(run_dir).save(state)
    return state


@pytest.mark.asyncio
async def test_same_model_multi_instance_routing_and_heterogeneous_pinning(tmp_path: Path) -> None:
    init_repo(tmp_path)
    inventory = routing_inventory()
    plan = Plan(
        goal="fleet",
        tasks=[
            TaskSpec(id="docs", title="Docs", description="Docs", risk="low", complexity="low"),
            TaskSpec(id="tests", title="Tests", description="Tests", risk="low", complexity="low"),
            TaskSpec(id="core", title="Core", description="Core", complexity="high"),
        ],
    )
    state = make_fleet_state(tmp_path, plan, inventory, 3)
    config = fleet_config(
        FleetModelConfig(name="fast", model="fast-model", tier="fast", roles=["worker"]),
        FleetModelConfig(
            name="strong",
            model="strong-model",
            tier="strong",
            roles=["planner", "worker"],
        ),
    )
    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker_factory = lambda destination, _models: FleetFakeWorker(destination)  # type: ignore[assignment]
    reviewer = FleetFakeReviewer()
    orchestrator.reviewer_factory = lambda _destination, _models: reviewer  # type: ignore[assignment]
    final = await orchestrator.run()
    assert final.tasks["docs"].assigned_instance in {"fast-1", "fast-2"}
    assert final.tasks["tests"].assigned_instance in {"fast-1", "fast-2"}
    assert final.tasks["core"].assigned_instance == "strong-1"
    assert all(task.status == TaskStatus.MERGED for task in final.tasks.values())
    decisions = [
        json.loads(line)
        for line in (tmp_path / ".adaptea" / "runs" / state.run_id / "routing-decisions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["instance"] for row in decisions} >= {"fast-1", "strong-1"}


@pytest.mark.asyncio
async def test_fast_validation_failure_escalates_once_to_strong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_repo(tmp_path)
    plan = Plan(
        goal="escalate",
        tasks=[
            TaskSpec(
                id="simple", title="Simple", description="Simple", risk="low", complexity="low"
            )
        ],
    )
    inventory = routing_inventory()
    state = make_fleet_state(tmp_path, plan, inventory, 1)
    config = fleet_config(
        FleetModelConfig(name="fast", model="fast-model", tier="fast", roles=["worker"]),
        FleetModelConfig(
            name="strong",
            model="strong-model",
            tier="strong",
            roles=["planner", "worker"],
        ),
    )
    validations = 0

    async def run_validation(*_args: Any, **_kwargs: Any) -> ValidationOutcome:
        nonlocal validations
        validations += 1
        if validations == 1:
            return ValidationOutcome(False, 1, "validator exited 1", ["pytest"])
        return ValidationOutcome(True, 0, "validation passed", ["pytest"])

    monkeypatch.setattr("adaptea.runtime.controller.run_validation", run_validation)
    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker_factory = lambda destination, _models: FleetFakeWorker(destination)  # type: ignore[assignment]
    reviewer = FleetFakeReviewer()
    orchestrator.reviewer_factory = lambda _destination, _models: reviewer  # type: ignore[assignment]
    final = await orchestrator.run()
    task = final.tasks["simple"]
    assert task.status == TaskStatus.MERGED
    assert task.fast_escalations == 1
    assert task.fast_escalation_wasted_seconds == pytest.approx(0.01)
    assert [row["tier"] for row in task.routing_history] == ["fast", "strong"]
    assert task.attempts == 2
    assert task.retry_count == 1
    assert task.failure_counts == {"validation_failure": 1}
    assert task.failure_history[0]["escalated_to_strong"] is True
    summary = json.loads(
        (tmp_path / ".adaptea" / "runs" / state.run_id / "summary.json").read_text()
    )
    assert summary["tiers"]["fast"]["failed"] == 1
    assert summary["tiers"]["strong"]["passed"] == 1
    assert summary["models"]["strong-model"]["retries"] == 1


@pytest.mark.asyncio
async def test_reviewer_rejection_repairs_once_and_escalates_fast_task(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    plan = Plan(
        goal="review repair",
        tasks=[
            TaskSpec(
                id="simple",
                title="Simple",
                description="Simple",
                risk="low",
                complexity="low",
            )
        ],
    )
    inventory = routing_inventory()
    state = make_fleet_state(tmp_path, plan, inventory, 1)
    config = fleet_config(
        FleetModelConfig(name="fast", model="fast-model", tier="fast", roles=["worker"]),
        FleetModelConfig(
            name="strong",
            model="strong-model",
            tier="strong",
            roles=["planner", "worker", "reviewer"],
        ),
    )
    reviewer = FleetFakeReviewer([False, True])
    orchestrator = Orchestrator(tmp_path, config, state)
    orchestrator.worker_factory = lambda destination, _models: FleetFakeWorker(destination)  # type: ignore[assignment]
    orchestrator.reviewer_factory = lambda _destination, _models: reviewer  # type: ignore[assignment]

    final = await orchestrator.run()

    task = final.tasks["simple"]
    assert task.status == TaskStatus.MERGED
    assert task.review_attempts == 2
    assert task.review_rejections == 1
    assert task.fast_escalations == 1
    assert task.assigned_tier == "strong"
    assert task.review_approved is True
    assert len(task.review_history) == 2
    assert task.retry_count == 1
    assert task.failure_counts == {"review_rejection": 1}
    assert task.failure_history[0]["escalated_to_strong"] is True
    summary = json.loads(
        (tmp_path / ".adaptea" / "runs" / state.run_id / "summary.json").read_text()
    )
    assert summary["reviews_total"] == 2
    assert summary["reviewer_rejections"] == 1
    assert summary["review_approval_rate"] == pytest.approx(0.5)


def _demand_config(strong_instances: int = 2, fast_instances: int = 2) -> Config:
    config = Config()
    config.worker.max_agents = 6
    config.fleet.enabled = True
    config.fleet.max_loaded_instances = 6
    config.fleet.routing.high_complexity = "strong"
    config.fleet.routing.low_complexity = "fast"
    config.fleet.routing.medium_complexity = "strong"
    config.fleet.models = [
        FleetModelConfig(
            name="strong",
            model="coder-30b",
            tier="strong",
            roles=["planner", "worker", "reviewer"],
            instances=strong_instances,
            parallel_limit=1,
        ),
        FleetModelConfig(
            name="fast",
            model="coder-8b",
            tier="fast",
            roles=["worker"],
            instances=fast_instances,
            parallel_limit=1,
        ),
    ]
    return config


def _task(identifier: str, complexity: str, depends: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        id=identifier,
        title=identifier,
        description=identifier,
        depends_on=depends or [],
        risk="low" if complexity == "low" else complexity,  # type: ignore[arg-type]
        complexity=complexity,  # type: ignore[arg-type]
    )


def test_a_plan_of_one_tier_does_not_load_the_other_tier() -> None:
    plan = Plan(goal="all hard", tasks=[_task(f"t{index}", "high") for index in range(4)])
    demand = instance_demand(plan, _demand_config())
    # Nothing in this plan can route to the fast pool, so holding it in memory buys
    # nothing and costs the strong model room.
    assert demand["coder-8b"] == 0
    assert demand["coder-30b"] == 2


def test_demand_never_exceeds_what_the_user_configured() -> None:
    plan = Plan(goal="lots", tasks=[_task(f"t{index}", "high") for index in range(12)])
    demand = instance_demand(plan, _demand_config(strong_instances=1))
    assert demand["coder-30b"] == 1


def test_a_serial_chain_asks_for_one_worker_not_the_whole_ceiling() -> None:
    chain = [_task("t0", "high")]
    for index in range(1, 5):
        chain.append(_task(f"t{index}", "high", depends=[f"t{index - 1}"]))
    demand = instance_demand(Plan(goal="serial", tasks=chain), _demand_config())
    # Every task waits for the one before it, so a second instance would never be used.
    assert demand["coder-30b"] == 1


def test_a_measured_ceiling_lowers_the_instances_that_get_loaded() -> None:
    plan = Plan(goal="wide", tasks=[_task(f"t{index}", "high") for index in range(6)])
    unmeasured = instance_demand(plan, _demand_config())
    measured = instance_demand(plan, _demand_config(), {"safe_max_concurrency": 1})
    assert unmeasured["coder-30b"] == 2
    assert measured["coder-30b"] == 1


def test_a_mixed_plan_keeps_both_tiers_loaded() -> None:
    plan = Plan(
        goal="mixed",
        tasks=[_task("hard", "high"), _task("easy", "low"), _task("also-easy", "low")],
    )
    demand = instance_demand(plan, _demand_config())
    assert demand["coder-30b"] >= 1
    assert demand["coder-8b"] >= 1
