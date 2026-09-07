from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from adaptea.projects import (
    RecentProjects,
    create_project,
    detect_project_type,
    ensure_adaptea_ignored,
    filesystem_roots,
    inspect_project,
    user_state_directory,
    validate_project_name,
)


def test_platform_native_state_paths() -> None:
    windows = user_state_directory("Windows", {"LOCALAPPDATA": r"C:\Users\Ada\AppData\Local"})
    assert PureWindowsPath(str(windows)).parts[-1] == "Adaptea"
    darwin = user_state_directory("Darwin", {})
    assert darwin.parts[-3:] == ("Library", "Application Support", "Adaptea")
    assert filesystem_roots("Windows", {"USERPROFILE": r"C:\Users\Ada"})


def test_recent_projects_prunes_missing_directories(tmp_path: Path) -> None:
    state = RecentProjects(tmp_path / "state" / "recents.json")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    state.remember(first)
    state.remember(second)
    second.rmdir()
    assert [Path(row.path) for row in state.load()] == [first.resolve()]


def test_recent_project_can_be_removed_without_deleting_its_folder(tmp_path: Path) -> None:
    state = RecentProjects(tmp_path / "state" / "recents.json")
    project = tmp_path / "project"
    project.mkdir()
    state.remember(project)

    assert state.forget(project)
    assert state.load() == []
    assert project.is_dir()


def test_runtime_ignore_is_added_without_replacing_existing_rules(tmp_path: Path) -> None:
    ignore = tmp_path / ".gitignore"
    ignore.write_text("node_modules/\n.env", encoding="utf-8")

    assert ensure_adaptea_ignored(tmp_path)
    assert ignore.read_text(encoding="utf-8") == "node_modules/\n.env\n.adaptea/\n"
    assert not ensure_adaptea_ignored(tmp_path)


@pytest.mark.asyncio
async def test_create_empty_project_initializes_git_and_detects_type(tmp_path: Path) -> None:
    info = await create_project(tmp_path, "hospital-opl-platform")
    assert info.path == tmp_path / "hospital-opl-platform"
    assert info.is_git
    assert info.git_branch == "main"
    assert (info.path / ".gitignore").is_file()
    assert (await inspect_project(info.path)).git_status == "Clean"
    (info.path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    assert detect_project_type(info.path) == "Python"


@pytest.mark.parametrize("name", ["", "..", "folder/name", r"folder\name", "bad:name"])
def test_project_names_are_safe_on_macos_and_windows(name: str) -> None:
    with pytest.raises(ValueError):
        validate_project_name(name)
