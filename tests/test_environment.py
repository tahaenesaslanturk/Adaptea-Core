from pathlib import Path

from adaptea.config import load_config
from adaptea.environment import (
    adopt_from_project,
    apply_to_project,
    environment_config_path,
    environment_root,
    is_configured,
    load_environment,
)

FLEET = """
[inference]
backend = "ollama"

[ollama]
model = "qwen3-coder:30b"

[fleet]
enabled = true
topology = "auto"
max_loaded_instances = 2
memory_headroom_fraction = 0.2
instance_ttl_seconds = 3600
minimum_residency_seconds = 300

[[fleet.models]]
name = "strong-1"
model = "qwen3-coder:30b"
tier = "strong"
roles = ["planner", "worker"]
"""


def test_unconfigured_until_something_is_chosen(tmp_path: Path) -> None:
    assert is_configured(tmp_path) is False
    apply_to_project(environment_root(tmp_path), load_config(tmp_path), tmp_path)
    # Writing bare defaults is not a configuration; nothing has been chosen.
    assert is_configured(tmp_path) is False


def test_one_environment_reaches_every_project(tmp_path: Path) -> None:
    environment_config_path(tmp_path).write_text(FLEET, encoding="utf-8")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    # A project with its own unrelated settings keeps them; only the Adaptea-managed
    # inference and fleet sections are replaced.
    (second / "adaptea.toml").write_text(
        '[project]\ntest_command = ["pytest", "-q"]\n\n[fleet]\nenabled = false\n',
        encoding="utf-8",
    )

    for project in (first, second):
        apply_to_project(project, directory=tmp_path)

    for project in (first, second):
        config = load_config(project, explicit=project / "adaptea.toml")
        assert config.inference.backend == "ollama"
        assert config.ollama.model == "qwen3-coder:30b"
        assert config.fleet.enabled is True
        assert [model.model for model in config.fleet.models] == ["qwen3-coder:30b"]
    assert load_config(second, explicit=second / "adaptea.toml").project.test_command == [
        "pytest",
        "-q",
    ]


def test_applying_twice_rewrites_nothing(tmp_path: Path) -> None:
    environment_config_path(tmp_path).write_text(FLEET, encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    assert apply_to_project(project, directory=tmp_path)
    # Opening the same project again must not keep touching its files.
    assert apply_to_project(project, directory=tmp_path) == []


def test_first_configured_project_seeds_the_environment(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "adaptea.toml").write_text(FLEET, encoding="utf-8")

    # Upgrading must not replace a fleet someone already configured with empty defaults.
    adopted = adopt_from_project(project, tmp_path)

    assert adopted is not None
    assert [model.model for model in load_environment(tmp_path).fleet.models] == ["qwen3-coder:30b"]
    # A project that has chosen nothing donates nothing, and a seeded environment is
    # never overwritten by the next project to be opened.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert adopt_from_project(empty, tmp_path) is None
    assert load_environment(tmp_path).fleet.models
