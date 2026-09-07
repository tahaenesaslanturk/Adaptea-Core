from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from adaptea.config import Config, FleetConfig, FleetModelConfig, load_config
from adaptea.diagnostics.system import (
    DiagnosticSnapshot,
    LocalModel,
    opencode_provider_configured,
)
from adaptea.lmstudio.lms_cli import CommandResult
from adaptea.setup.manager import SetupManager, command_failure_reason


class SimulationDiagnostics:
    def __init__(
        self, root: Path, config: Config, system: str, *, model_present: bool = False
    ) -> None:
        self.root = root
        self.config = config
        self.system = system
        self.git = False
        self.lms = False
        self.server = False
        self.opencode = False
        self.desktop = False
        self.model_present = model_present
        self.model_loaded = False

    def which(self, name: str) -> str | None:
        always = {
            "Darwin": {"brew": "/mock/brew", "curl": "/usr/bin/curl"},
            "Windows": {
                "winget": r"C:\Windows\winget.exe",
                "choco": r"C:\Tools\choco.exe",
            },
        }[self.system]
        if name in always:
            return always[name]
        if name == "git" and self.git:
            return (
                r"C:\Program Files\Git\bin\git.exe" if self.system == "Windows" else "/usr/bin/git"
            )
        if name in {"lms", self.config.lmstudio.lms_executable} and self.lms:
            return r"C:\LM Studio\bin\lms.exe" if self.system == "Windows" else "/mock/lms"
        if name in {"opencode", self.config.worker.executable} and self.opencode:
            return r"C:\Tools\opencode.exe" if self.system == "Windows" else "/mock/opencode"
        return None

    async def collect(self) -> DiagnosticSnapshot:
        model = (
            LocalModel(
                key="qwen/coder",
                display_name="Qwen Coder",
                architecture="qwen",
                size_bytes=4_000_000_000,
                max_context_length=32768,
                loaded=self.model_loaded,
                instance_id="qwen/coder" if self.model_loaded else None,
                context_length=8192 if self.model_loaded else None,
                parallel=4 if self.model_loaded else None,
                format="gguf",
            )
            if self.model_present
            else None
        )
        opencode = self.which("opencode")
        lms = self.which("lms")
        configured = opencode_provider_configured(
            self.root,
            opencode,
            self.config.lmstudio.model,
            self.config.lmstudio.base_url,
        )
        selected = (
            model
            if model and self.config.lmstudio.model in {None, model.key, model.instance_id}
            else None
        )
        return DiagnosticSnapshot(
            operating_system=self.system,
            architecture="arm64" if self.system == "Darwin" else "AMD64",
            python_version="3.12",
            python_ready=True,
            git_executable=self.which("git"),
            git_version="git 2" if self.git else None,
            opencode_executable=opencode,
            opencode_version="opencode 1" if self.opencode else None,
            lms_executable=lms,
            lms_version="lms 0.1" if self.lms else None,
            lmstudio_desktop=Path("/Applications/LM Studio.app") if self.desktop else None,
            server_reachable=self.server,
            server_error=None if self.server else "stopped",
            native_api_usable=self.server,
            openai_api_usable=self.server,
            models=[model] if model else [],
            selected_model=selected,
            telemetry_usable=self.server,
            git_repository=False,
            capacity_profile=False,
            adaptea_config=(self.root / "adaptea.toml").exists(),
            opencode_configured=configured,
            lms_bootstrap_executable=(
                "/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"
                if self.desktop and not self.lms
                else None
            ),
        )


class SimulationRunner:
    def __init__(self, state: SimulationDiagnostics, *, fail_component: str | None = None) -> None:
        self.state = state
        self.fail_component = fail_component
        self.commands: list[tuple[str, ...]] = []

    async def run(self, *args: str, **_kwargs: Any) -> CommandResult:
        self.commands.append(args)
        joined = " ".join(args)
        if self.fail_component and self.fail_component in joined:
            return CommandResult(1, "", "simulated installer failure")
        if "Git.Git" in args or args[-2:] == ("install", "git"):
            self.state.git = True
        if "install.sh" in joined or "install.ps1" in joined:
            self.state.lms = True
        if "bootstrap -y" in joined:
            self.state.lms = True
        if "server start" in joined:
            self.state.server = True
        if any("opencode" in item.lower() for item in args) and (
            any(Path(item).name == "brew" for item in args)
            or "choco" in args
            or "scoop" in args
            or "winget" in args
            or "npm" in args
        ):
            self.state.opencode = True
        if "get" in args:
            self.state.model_present = True
        return CommandResult(
            0,
            "adaptea/qwen/coder" if args[-2:] == ("models", "adaptea") else "ADAPTEA_SETUP_OK",
            "",
        )


def manager_for(
    tmp_path: Path,
    system: str,
    *,
    model_present: bool = False,
    confirm_install: bool = True,
    runner_failure: str | None = None,
) -> tuple[SetupManager, SimulationDiagnostics, SimulationRunner]:
    config = Config()
    state = SimulationDiagnostics(tmp_path, config, system, model_present=model_present)
    runner = SimulationRunner(state, fail_component=runner_failure)
    manager = SetupManager(
        tmp_path,
        config,
        diagnostics=state,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
        logger=logging.getLogger(f"test.{system}.{id(tmp_path)}"),
        confirm_install=lambda _action: confirm_install,
        confirm=lambda _message: True,
        select_model=lambda models: models[0],
        request_model_identifier=lambda: "qwen/coder",
        request_context_length=lambda _model: 8192,
        request_max_agents=lambda _current: 6,
    )

    async def load_model(_model: str, _context: int | None = None) -> None:
        state.model_loaded = True

    manager.load_model = load_model  # type: ignore[method-assign]
    return manager, state, runner


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["Darwin", "Windows"])
async def test_mocked_setup_end_to_end_cross_platform(tmp_path: Path, system: str) -> None:
    manager, state, runner = manager_for(tmp_path, system)
    outcome = await manager.fix_all(automatic=True)
    assert outcome.failures == []
    assert outcome.snapshot.required_ready
    assert state.git and state.lms and state.server and state.model_loaded and state.opencode
    assert (tmp_path / "adaptea.toml").exists()
    assert (tmp_path / "opencode.json").exists()
    commands = [" ".join(command) for command in runner.commands]
    assert any("server start" in command for command in commands)
    assert any("get qwen/coder --gguf" in command for command in commands)


@pytest.mark.asyncio
async def test_already_configured_machine_is_idempotent(tmp_path: Path) -> None:
    manager, state, runner = manager_for(tmp_path, "Darwin", model_present=True)
    notices: list[str] = []
    manager.notice = notices.append
    state.git = state.lms = state.server = state.opencode = state.model_loaded = True
    manager.config.lmstudio.model = "qwen/coder"
    from adaptea.setup.configuration import merge_adaptea_config, merge_opencode_config

    merge_opencode_config(tmp_path, "/mock/opencode", "qwen/coder", "http://127.0.0.1:1234")
    merge_adaptea_config(
        tmp_path / "adaptea.toml",
        base_url="http://127.0.0.1:1234",
        model="qwen/coder",
        lms_executable="/mock/lms",
        opencode_executable="/mock/opencode",
        max_agents=8,
    )
    outcome = await manager.fix_all(automatic=True)
    assert outcome.snapshot.required_ready
    assert not any("install" in " ".join(command) for command in runner.commands)
    assert notices[0].startswith("Inspecting the local toolchain")
    assert notices[-1] == "Safe repair finished; all required checks are ready."


@pytest.mark.asyncio
async def test_automatic_repair_does_not_choose_or_load_an_unselected_model(
    tmp_path: Path,
) -> None:
    manager, state, _runner = manager_for(tmp_path, "Darwin", model_present=True)
    state.git = state.lms = state.server = state.opencode = True
    outcome = await manager.fix_all(automatic=True)
    assert not state.model_loaded
    assert not any("Loaded qwen/coder" in message for message in outcome.messages)


@pytest.mark.asyncio
async def test_single_loaded_model_wins_over_multiple_downloaded_models(tmp_path: Path) -> None:
    manager, state, _runner = manager_for(tmp_path, "Darwin", model_present=True)
    state.model_loaded = True
    snapshot = await state.collect()
    other = LocalModel(key="qwen/other", display_name="Other downloaded model")
    snapshot = replace(snapshot, models=[snapshot.models[0], other])

    assert manager._configured_or_single_model(snapshot) == snapshot.models[0]


@pytest.mark.asyncio
async def test_single_loaded_model_wins_over_unloaded_configured_model(tmp_path: Path) -> None:
    manager, state, _runner = manager_for(tmp_path, "Darwin", model_present=True)
    state.model_loaded = True
    manager.config.lmstudio.model = "qwen/old"
    snapshot = await state.collect()
    old = LocalModel(key="qwen/old", display_name="Old configured model")
    snapshot = replace(snapshot, models=[snapshot.models[0], old])

    assert manager._configured_or_single_model(snapshot) == snapshot.models[0]


@pytest.mark.asyncio
async def test_selected_model_replaces_the_fleet_planner(tmp_path: Path) -> None:
    config = Config(
        fleet=FleetConfig(
            enabled=True,
            models=[
                FleetModelConfig(
                    name="old-strong",
                    model="qwen/old",
                    tier="strong",
                    roles=["planner", "worker"],
                ),
                FleetModelConfig(
                    name="new-fast",
                    model="qwen/new",
                    tier="fast",
                    roles=["worker"],
                ),
            ],
        )
    )
    manager = SetupManager(tmp_path, config)
    new = LocalModel(key="qwen/new", display_name="New", loaded=True)

    assert manager._align_fleet_planner(new)
    assert config.fleet.planner() is not None
    assert config.fleet.planner().model == "qwen/new"  # type: ignore[union-attr]
    assert config.fleet.planner().tier == "strong"  # type: ignore[union-attr]
    assert config.fleet.models[0].roles == ["worker"]


@pytest.mark.asyncio
async def test_safe_repair_preserves_stale_selection_instead_of_guessing_from_loaded_model(
    tmp_path: Path,
) -> None:
    config = Config(
        lmstudio={"model": "qwen/old"},
        fleet=FleetConfig(
            enabled=True,
            models=[
                FleetModelConfig(
                    name="old-strong",
                    model="qwen/old",
                    tier="strong",
                    roles=["planner", "worker"],
                )
            ],
        ),
    )
    state = SimulationDiagnostics(tmp_path, config, "Darwin", model_present=True)
    state.git = state.lms = state.server = state.opencode = state.model_loaded = True
    runner = SimulationRunner(state)
    manager = SetupManager(
        tmp_path,
        config,
        diagnostics=state,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
    )
    from adaptea.setup.configuration import (
        merge_adaptea_config,
        merge_fleet_config,
        merge_opencode_config,
    )

    merge_adaptea_config(
        tmp_path / "adaptea.toml",
        base_url=config.lmstudio.base_url,
        model="qwen/old",
        lms_executable="/mock/lms",
        opencode_executable="/mock/opencode",
        max_agents=8,
    )
    merge_fleet_config(tmp_path / "adaptea.toml", config.fleet)
    merge_opencode_config(tmp_path, "/mock/opencode", "qwen/old", config.lmstudio.base_url)

    outcome = await manager.fix_all(automatic=True)

    persisted = load_config(tmp_path, tmp_path / "adaptea.toml")
    assert not outcome.snapshot.required_ready
    assert persisted.lmstudio.model == "qwen/old"
    assert persisted.fleet.planner() is not None
    assert persisted.fleet.planner().model == "qwen/old"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_desktop_bundled_lms_is_bootstrapped(tmp_path: Path) -> None:
    manager, state, runner = manager_for(tmp_path, "Darwin")
    state.git = state.opencode = state.desktop = True
    outcome = await manager.fix_all(automatic=True)
    assert state.lms
    assert any("bootstrap -y" in " ".join(command) for command in runner.commands)
    assert any("Bootstrapped" in message for message in outcome.messages)


@pytest.mark.asyncio
async def test_external_opencode_install_is_seen_on_recheck(tmp_path: Path) -> None:
    manager, state, _runner = manager_for(tmp_path, "Darwin")
    assert (await manager.diagnose()).opencode_executable is None
    state.opencode = True
    assert (await manager.diagnose()).opencode_executable == "/mock/opencode"


@pytest.mark.asyncio
async def test_cancelled_installation_is_safe(tmp_path: Path) -> None:
    manager, state, runner = manager_for(tmp_path, "Darwin", confirm_install=False)
    outcome = await manager.fix_all(automatic=True)
    assert not state.git and not state.lms and not state.opencode
    assert runner.commands == []
    assert any("Cancelled" in message for message in outcome.messages)
    assert not outcome.snapshot.required_ready


@pytest.mark.asyncio
async def test_failed_installer_is_reported_without_exception(tmp_path: Path) -> None:
    manager, state, _runner = manager_for(tmp_path, "Darwin", runner_failure="brew install git")
    outcome = await manager.fix_all(automatic=True)
    assert not state.git
    assert any("Git installation failed" in failure for failure in outcome.failures)


def test_failure_reason_prefers_permission_error_over_stack_tail() -> None:
    result = CommandResult(
        1,
        "",
        "/bin/sh: ~/.bash_profile: Permission denied\nError: failed\n  at stack:1",
    )
    assert command_failure_reason(result).endswith("Permission denied")


@pytest.mark.asyncio
async def test_server_start_retries_via_daemon(tmp_path: Path) -> None:
    config = Config()
    state = SimulationDiagnostics(tmp_path, config, "Darwin")
    state.lms = True

    class RetryRunner(SimulationRunner):
        async def run(self, *args: str, **kwargs: Any) -> CommandResult:
            self.commands.append(args)
            starts = [command for command in self.commands if "server" in command]
            if "server" in args and len(starts) == 1:
                return CommandResult(1, "", "daemon stopped")
            if "server" in args:
                state.server = True
            return CommandResult(0, "", "")

    runner = RetryRunner(state)
    manager = SetupManager(
        tmp_path,
        config,
        diagnostics=state,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
    )
    assert await manager.start_server("/mock/lms")
    joined = [" ".join(command) for command in runner.commands]
    assert joined == [
        "/mock/lms server start --port 1234 --bind 127.0.0.1",
        "/mock/lms daemon up",
        "/mock/lms server start --port 1234 --bind 127.0.0.1",
    ]


@pytest.mark.asyncio
async def test_smoke_test_uses_harmless_temporary_directory(tmp_path: Path) -> None:
    manager, state, runner = manager_for(tmp_path, "Darwin", model_present=True)
    state.git = state.lms = state.server = state.opencode = state.model_loaded = True
    manager.config.lmstudio.model = "qwen/coder"
    from adaptea.setup.configuration import merge_opencode_config

    merge_opencode_config(tmp_path, "/mock/opencode", "qwen/coder", "http://127.0.0.1:1234")
    result = await manager.smoke_test(await state.collect())
    assert result.success
    command = runner.commands[-1]
    assert command[-2:] == ("models", "adaptea")


def test_repeated_setup_leaves_one_backup_and_rewrites_nothing(tmp_path: Path) -> None:
    """Setup re-runs constantly; it must not leave a file behind each time."""
    from adaptea.setup.configuration import merge_adaptea_config

    backups = tmp_path / ".adaptea" / "backups"
    path = tmp_path / "adaptea.toml"
    arguments = {
        "base_url": "http://127.0.0.1:1234",
        "model": "coder-30b",
        "lms_executable": "lms",
        "opencode_executable": "opencode",
        "max_agents": 4,
    }
    assert merge_adaptea_config(path, **arguments) is None  # type: ignore[arg-type]
    first = path.read_text(encoding="utf-8")

    # An identical merge is a no-op: no rewrite, no backup.
    assert merge_adaptea_config(path, **arguments) is None  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == first
    assert not backups.exists()

    # A real change backs up once, and a second real change reuses that one name.
    assert merge_adaptea_config(path, **{**arguments, "model": "coder-8b"}) is not None  # type: ignore[arg-type]
    assert merge_adaptea_config(path, **{**arguments, "model": "coder-14b"}) is not None  # type: ignore[arg-type]
    assert [item.name for item in backups.iterdir()] == ["adaptea.toml.adaptea-backup"]
    assert "coder-14b" in path.read_text(encoding="utf-8")

    # The project root keeps exactly one configuration file and the state directory.
    assert sorted(item.name for item in tmp_path.iterdir()) == [".adaptea", "adaptea.toml"]


def test_legacy_sibling_backup_is_removed_on_the_next_write(tmp_path: Path) -> None:
    """Upgrading cleans the project root instead of leaving an orphaned copy behind."""
    from adaptea.setup.configuration import write_if_changed

    path = tmp_path / "opencode.json"
    path.write_text("{}\n", encoding="utf-8")
    legacy = tmp_path / "opencode.json.adaptea-backup"
    legacy.write_text("{}\n", encoding="utf-8")

    write_if_changed(path, '{"model": "coder"}\n')

    assert not legacy.exists()
    assert (tmp_path / ".adaptea" / "backups" / "opencode.json.adaptea-backup").is_file()


def test_setup_logs_are_pruned_to_a_bounded_window(tmp_path: Path) -> None:
    """`.adaptea/logs` must not grow by one file for every diagnose-repair cycle."""
    from adaptea.setup.manager import prune_setup_logs

    directory = tmp_path / "logs"
    directory.mkdir()
    for index in range(15):
        (directory / f"setup-2026010{index // 10}-0000{index % 10}.log").write_text("x")

    removed = prune_setup_logs(directory, keep=10)

    assert len(removed) == 5
    assert len(list(directory.glob("setup-*.log"))) == 10
