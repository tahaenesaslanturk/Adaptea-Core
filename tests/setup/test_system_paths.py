from __future__ import annotations

from pathlib import Path

from adaptea.diagnostics.system import find_ollama_executable, find_opencode_executable


def test_finds_opencode_installed_outside_gui_path(tmp_path: Path) -> None:
    executable = tmp_path / ".opencode" / "bin" / "opencode"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    found = find_opencode_executable(
        "opencode",
        "Darwin",
        which=lambda _name: None,
        environment={"HOME": str(tmp_path)},
    )

    assert found == str(executable)


def test_finds_ollama_installed_outside_gui_path(tmp_path: Path) -> None:
    executable = tmp_path / ".ollama" / "bin" / "ollama"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    found = find_ollama_executable(
        "ollama",
        "Darwin",
        which=lambda _name: None,
        environment={"HOME": str(tmp_path)},
    )

    assert found == str(executable)


def test_finds_ollama_installed_in_local_bin(tmp_path: Path) -> None:
    executable = tmp_path / ".local" / "bin" / "ollama"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    found = find_ollama_executable(
        "ollama",
        "Linux",
        which=lambda _name: None,
        environment={"HOME": str(tmp_path)},
    )

    assert found == str(executable)


def test_finds_ollama_installed_on_windows(tmp_path: Path) -> None:
    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)

    found = find_ollama_executable(
        "ollama",
        "Windows",
        which=lambda _name: None,
        environment={"LOCALAPPDATA": str(tmp_path), "HOME": str(tmp_path)},
    )

    assert found == str(executable)


def test_finds_git_installed_on_windows(tmp_path: Path) -> None:
    from adaptea.diagnostics.system import find_git_executable

    executable = tmp_path / "Git" / "cmd" / "git.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")

    found = find_git_executable(
        "git",
        "Windows",
        which=lambda _name: None,
        environment={"ProgramFiles": str(tmp_path)},
    )
    assert found == str(executable)


def test_finds_opencode_npm_cmd_on_windows(tmp_path: Path) -> None:
    executable = tmp_path / "npm" / "opencode.cmd"
    executable.parent.mkdir(parents=True)
    executable.write_text("@echo off\n", encoding="utf-8")

    found = find_opencode_executable(
        "opencode",
        "Windows",
        which=lambda _name: None,
        environment={"APPDATA": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert found == str(executable)


def test_finds_llamacpp_on_windows(tmp_path: Path) -> None:
    from adaptea.diagnostics.system import find_llamacpp_executable

    executable = tmp_path / "Programs" / "llama.cpp" / "llama-server.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")

    found = find_llamacpp_executable(
        "llama-server",
        "Windows",
        which=lambda _name: None,
        environment={"LOCALAPPDATA": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert found == str(executable)


def test_finds_vllm_on_windows(tmp_path: Path) -> None:
    from adaptea.diagnostics.system import find_vllm_executable

    executable = tmp_path / "Programs" / "Python" / "Scripts" / "vllm.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")

    found = find_vllm_executable(
        "vllm",
        "Windows",
        which=lambda _name: None,
        environment={"LOCALAPPDATA": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert found == str(executable)


def test_normalize_subprocess_command_windows_batch(monkeypatch) -> None:
    from adaptea.lmstudio.lms_cli import normalize_subprocess_command

    monkeypatch.setattr("platform.system", lambda: "Windows")
    result = normalize_subprocess_command(["C:\\npm\\opencode.cmd", "run", "--pure"])
    assert result == ["cmd.exe", "/c", "C:\\npm\\opencode.cmd", "run", "--pure"]

    result_exe = normalize_subprocess_command(["C:\\bin\\lms.exe", "server", "start"])
    assert result_exe == ["C:\\bin\\lms.exe", "server", "start"]
