from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from adaptea.diagnostics.doctor import checks_from_snapshot
from adaptea.diagnostics.system import (
    DiagnosticSnapshot,
    LocalModel,
    bundled_lms_candidates,
    desktop_candidates,
    desktop_lms_candidates,
    parse_local_models,
    select_local_model,
)
from adaptea.setup.ui import setup_table


def snapshot(**overrides: object) -> DiagnosticSnapshot:
    values: dict[str, object] = {
        "operating_system": "Darwin 25",
        "architecture": "arm64",
        "python_version": "3.12",
        "python_ready": True,
        "git_executable": "/usr/bin/git",
        "git_version": "git 2",
        "opencode_executable": None,
        "opencode_version": None,
        "lms_executable": None,
        "lms_version": None,
        "lmstudio_desktop": None,
        "server_reachable": False,
        "server_error": "stopped",
        "native_api_usable": False,
        "openai_api_usable": False,
        "models": [],
        "selected_model": None,
        "telemetry_usable": False,
        "git_repository": False,
        "capacity_profile": False,
        "adaptea_config": False,
        "opencode_configured": False,
    }
    values.update(overrides)
    return DiagnosticSnapshot(**values)  # type: ignore[arg-type]


def test_lms_ls_json_tolerates_current_and_future_fields() -> None:
    result = parse_local_models(
        json.dumps(
            {
                "future": True,
                "models": [
                    {
                        "modelKey": "qwen/coder",
                        "displayName": "Qwen Coder",
                        "architecture": "qwen",
                        "sizeBytes": 4_000_000_000,
                        "maxContextLength": 32768,
                        "type": "llm",
                        "unknown": {"x": 1},
                    },
                    {"key": "embed", "type": "embedding"},
                ],
            }
        )
    )
    assert len(result) == 1
    assert result[0].key == "qwen/coder"
    assert result[0].size_bytes == 4_000_000_000


def test_cross_platform_desktop_paths() -> None:
    mac = desktop_candidates("Darwin")
    assert Path("/Applications/LM Studio.app") in mac
    windows = desktop_candidates("Windows", {"LOCALAPPDATA": r"C:\Users\Ada\AppData\Local"})
    assert any(str(path).endswith("LM Studio.exe") for path in windows)
    assert any("Ada" in str(path) for path in windows)


def test_cross_platform_bundled_lms_paths() -> None:
    mac = bundled_lms_candidates("Darwin", {"HOME": "/Users/ada"})
    assert Path("/Users/ada/.cache/lm-studio/bin/lms") in mac
    assert Path("/Users/ada/.lmstudio/bin/lms") in mac
    windows = bundled_lms_candidates("Windows", {"USERPROFILE": r"C:\Users\Ada"})
    assert windows[0].parts[-3:] == (".lmstudio", "bin", "lms.exe")


def test_cross_platform_desktop_bundled_lms_paths() -> None:
    mac = desktop_lms_candidates(Path("/Applications/LM Studio.app"), "Darwin")
    assert mac[0].parts[-5:] == ("Contents", "Resources", "app", ".webpack", "lms")
    windows = desktop_lms_candidates(Path(r"C:\Apps\LM Studio\LM Studio.exe"), "Windows")
    assert windows[0].parts[-4:] == ("resources", "app", ".webpack", "lms.exe")


def test_unconfigured_model_selects_loaded_model_not_first_unloaded() -> None:
    models = [
        LocalModel(key="first", loaded=False),
        LocalModel(key="second", loaded=True, instance_id="second"),
    ]
    assert select_local_model(models, None) == models[1]
    assert select_local_model(models, "first") == models[0]


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"opencode_executable": None}, "OpenCode"),
        ({"lms_executable": None}, "LM Studio / llmster"),
        ({"lms_executable": "lms", "server_reachable": False}, "LM Studio server"),
        ({"lms_executable": "lms", "server_reachable": True, "models": []}, "Model"),
        (
            {
                "lms_executable": "lms",
                "server_reachable": True,
                "models": [LocalModel(key="coder")],
                "selected_model": LocalModel(key="coder"),
            },
            "Model",
        ),
        ({"opencode_configured": False}, "OpenCode → LM Studio"),
    ],
)
def test_setup_statuses_cover_missing_states(overrides: dict[str, object], label: str) -> None:
    table = setup_table(snapshot(**overrides))
    assert label in render(table)


def test_doctor_and_setup_share_the_same_snapshot(tmp_path: Path) -> None:
    state = snapshot()
    doctor = {check.name: check.level for check in checks_from_snapshot(state, tmp_path)}
    table = setup_table(state)
    rendered = render(table)
    assert doctor["OpenCode"] == "FAIL"
    assert "OpenCode" in rendered and "Missing" in rendered
    assert state.required_ready is False


def render(table: object) -> str:
    console = Console(record=True, width=120)
    console.print(table)
    return console.export_text()
