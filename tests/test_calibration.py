from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptea.calibration.aggregate import aggregate_samples, recommend
from adaptea.calibration.runner import CalibrationRunner, cap_candidates, interleaved_order
from adaptea.config import Config, FleetModelConfig
from adaptea.lmstudio.models import ChatResponse, LMModel
from adaptea.reporting.report import benchmark_summary


def test_caps_and_seeded_interleaving() -> None:
    assert cap_candidates([1, 2, 4, 8], 4) == [1, 2, 4]
    first = interleaved_order([1, 2, 4], 3, 7)
    assert first == interleaved_order([1, 2, 4], 3, 7)
    assert first != [1, 1, 1, 2, 2, 2, 4, 4, 4]
    assert all(sorted(first[index : index + 3]) == [1, 2, 4] for index in range(0, 9, 3))


def test_aggregate_uses_medians_and_recommendation() -> None:
    samples = [
        {
            "concurrency": 1,
            "valid": True,
            "latency_seconds": value,
            "batch_wall_seconds": value,
            "input_tokens": 10,
            "output_tokens": 20,
            "ttft_seconds": value / 10,
            "tokens_per_second": 20 / value,
        }
        for value in [1.0, 3.0, 2.0]
    ] + [
        {
            "concurrency": 2,
            "valid": True,
            "latency_seconds": 1.5,
            "batch_wall_seconds": 1.5,
            "input_tokens": 10,
            "output_tokens": 20,
            "ttft_seconds": 0.2,
            "tokens_per_second": 15,
        }
        for _ in range(4)
    ]
    aggregate = aggregate_samples(samples)
    assert aggregate["1"]["median_latency_seconds"] == 2.0
    starting, safe = recommend(aggregate)
    assert 1 <= starting <= safe <= 2


def test_the_ceiling_a_run_may_ask_for_is_always_measured() -> None:
    # LM Studio serves two at a time, but the user allows eight agents. Refusing to
    # measure eight is why max_agents was never represented in a profile.
    assert cap_candidates([1, 2, 4, 8], 2, 8) == [1, 2, 4, 8]
    assert cap_candidates([1, 2], 2, 6) == [1, 2, 6]


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.destinations: list[str] = []

    async def models(self) -> list[LMModel]:
        return [
            LMModel.model_validate(
                {
                    "type": "llm",
                    "key": "coder",
                    "format": "gguf",
                    "max_context_length": 8192,
                    "loaded_instances": [
                        {"id": "coder@1", "config": {"context_length": 4096, "parallel": 2}}
                    ],
                }
            )
        ]

    async def chat(self, *args: object, **kwargs: object) -> ChatResponse:
        self.calls += 1
        if args:
            self.destinations.append(str(args[0]))
        maximum = int(kwargs["max_output_tokens"])
        return ChatResponse.model_validate(
            {
                "output": [{"type": "message", "content": "x"}],
                "stats": {
                    "input_tokens": 20,
                    "total_output_tokens": maximum,
                    "tokens_per_second": 20,
                    "time_to_first_token_seconds": 0.1,
                },
            }
        )


class TwoModelClient(FakeClient):
    async def models(self) -> list[LMModel]:
        return [
            LMModel.model_validate(
                {
                    "type": "llm",
                    "key": key,
                    "loaded_instances": [{"id": f"{key}@1", "config": {"parallel": 2}}],
                }
            )
            for key in ("coder", "helper")
        ]


class LoadedOtherModelClient(FakeClient):
    async def models(self) -> list[LMModel]:
        return [
            LMModel.model_validate(
                {
                    "type": "llm",
                    "key": "qwen/other",
                    "loaded_instances": [{"id": "qwen-other@1", "config": {"parallel": 1}}],
                }
            )
        ]


@pytest.mark.asyncio
async def test_calibration_artifacts_and_warmup_exclusion(tmp_path: Path) -> None:
    config = Config()
    config.calibration.seed = 4
    config.worker.max_agents = 2
    fake = FakeClient()
    directory = await CalibrationRunner(tmp_path, config, fake).run(
        [1, 2, 8], repetitions=1, quick=True
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    samples = (directory / "samples.jsonl").read_text().splitlines()
    order = manifest["models"][0]["order"]
    assert order == [2, 1] or order == [1, 2]
    assert len(samples) == 3
    assert fake.calls == 4  # one unmeasured warm-up plus 1+2 measured requests
    assert (directory / "report.csv").exists()
    assert (directory / "report.html").exists()
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text())
    assert profile["measured_models"] == ["coder"]
    # A single model has nothing to be measured alongside.
    assert profile["combined"]["measured"] is False


@pytest.mark.asyncio
async def test_the_sweep_reaches_the_agent_ceiling(tmp_path: Path) -> None:
    config = Config()
    config.worker.max_agents = 8  # the model itself only serves two at a time
    fake = FakeClient()
    directory = await CalibrationRunner(tmp_path, config, fake).run(repetitions=1, quick=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert sorted(manifest["models"][0]["order"]) == [1, 2, 4, 8]
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text())
    assert "8" in profile["models"][0]["tested_concurrency"]


@pytest.mark.asyncio
async def test_every_configured_model_is_measured_alone_and_together(tmp_path: Path) -> None:
    config = Config()
    config.worker.max_agents = 2
    config.fleet.enabled = True
    config.fleet.models = [
        FleetModelConfig(name="strong", model="coder", tier="strong", roles=["planner", "worker"]),
        FleetModelConfig(name="fast", model="helper", tier="fast", roles=["worker"]),
    ]
    fake = TwoModelClient()
    await CalibrationRunner(tmp_path, config, fake).run([1, 2], repetitions=1, quick=True)
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text())
    assert profile["measured_models"] == ["coder", "helper"]
    assert profile["combined"]["measured"] is True
    assert sorted(profile["combined"]["models"]) == ["coder", "helper"]
    # The combined phase alternates models rather than filling one and then the other.
    assert {"coder@1", "helper@1"} <= set(fake.destinations)


@pytest.mark.asyncio
async def test_a_pin_that_is_not_loaded_does_not_block_measuring_what_is(tmp_path: Path) -> None:
    # The pin names a model LM Studio does not have up, while three others are loaded and
    # ready. Refusing measured nothing and asked the user to select a model the error
    # itself could not load; the setup checks are where an absent pin is reported.
    config = Config(lmstudio={"model": "qwen/qwen3.8-27b"})
    await CalibrationRunner(tmp_path, config, LoadedOtherModelClient()).run(quick=True)
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text(encoding="utf-8"))
    assert profile["measured_models"] == ["qwen/other"]


def test_benchmark_comparison_defines_observed_fixed_oracle() -> None:
    rows = []
    for configuration, wall in (
        ("fixed-c1", 10.0),
        ("fixed-c2", 6.0),
        ("adaptive", 7.0),
        ("naive", 9.0),
    ):
        rows.extend(
            {
                "configuration": configuration,
                "wall_seconds": wall + offset,
                "pass_rate": 1.0,
                "tasks_per_second": 1 / wall,
            }
            for offset in (-0.1, 0.0, 0.1)
        )
    aggregate = benchmark_summary(rows)
    assert aggregate["oracle_fixed"]["configuration"] == "fixed-c2"
    assert aggregate["formal_comparison"] is True


@pytest.mark.asyncio
async def test_calibration_measures_the_part_of_the_fleet_that_is_loaded(
    tmp_path: Path,
) -> None:
    config = Config()
    config.fleet.enabled = True
    config.fleet.models = [
        FleetModelConfig(name="strong", model="coder", tier="strong", roles=["planner", "worker"]),
        FleetModelConfig(name="fast", model="helper", tier="fast", roles=["worker"]),
    ]
    # Only "coder" is loaded, and nothing is going to load "helper" on its own. Refusing
    # left the machine with no profile at all rather than one that says what it covers.
    await CalibrationRunner(tmp_path, config, FakeClient()).run(quick=True)
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text(encoding="utf-8"))
    assert profile["measured_models"] == ["coder"]


@pytest.mark.asyncio
async def test_calibration_refuses_when_none_of_the_fleet_is_loaded(tmp_path: Path) -> None:
    config = Config()
    config.fleet.enabled = True
    config.fleet.models = [
        FleetModelConfig(name="fast", model="helper", tier="fast", roles=["planner", "worker"]),
    ]
    with pytest.raises(RuntimeError, match="None of the configured models"):
        await CalibrationRunner(tmp_path, config, FakeClient()).run(quick=True)


@pytest.mark.asyncio
async def test_a_pin_naming_an_unloaded_model_measures_what_is_running(tmp_path: Path) -> None:
    config = Config()
    # A leftover pin blocked every measurement on a machine with models loaded and ready,
    # and told the user to go and select a model that the error itself could not load.
    config.lmstudio.model = "prism-ml/bonsai-27b"
    await CalibrationRunner(tmp_path, config, FakeClient()).run(quick=True)
    profile = json.loads((tmp_path / ".adaptea" / "capacity.json").read_text(encoding="utf-8"))
    assert profile["measured_models"] == ["coder"]
