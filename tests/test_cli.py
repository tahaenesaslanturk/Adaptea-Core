from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptea import __version__
from adaptea.cli import app

runner = CliRunner()


def test_help_lists_definition_of_done_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "setup",
        "doctor",
        "calibrate",
        "benchmark",
        "command-policy",
        "report",
        "plan",
        "run",
        "status",
        "resume",
        "tui",
        "smoke-test",
    ):
        assert command in result.stdout


def test_no_command_launches_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[Path] = []
    monkeypatch.setattr("adaptea.repl.start_repl", lambda root: launched.append(root))
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert launched == [Path.cwd().resolve()]


def test_dashboard_flag_launches_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[Path] = []
    monkeypatch.setattr("adaptea.tui.run_tui", lambda root: launched.append(root))
    result = runner.invoke(app, ["--dashboard"])
    assert result.exit_code == 0
    assert launched == [Path.cwd().resolve()]


def test_version_without_command() -> None:
    """Assert against the package version, not a literal that goes stale on every bump."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"adaptea {__version__}"


def test_run_rejects_missing_goal_and_plan() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "goal" in (result.stdout + result.stderr).lower()


def test_command_policy_presents_three_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["command-policy"])
    assert result.exit_code == 0
    assert "Safe default" in result.stdout
    assert "Approval required" in result.stdout
    assert "Blocked" in result.stdout
    assert "no project approvals" in result.stdout


def test_setup_reports_invalid_toml_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("adaptea.toml").write_text("[lmstudio\nbroken", encoding="utf-8")
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 1
    assert "What failed" in result.stdout
    assert "Traceback" not in result.stdout
