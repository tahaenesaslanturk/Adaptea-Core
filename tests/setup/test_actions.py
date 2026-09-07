from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from adaptea.setup.actions import (
    SetupCommandRunner,
    git_install_action,
    llmster_install_action,
    lms_install_action,
    lmstudio_desktop_install_action,
    opencode_install_action,
)


def finder(*available: str):
    return lambda name: f"/mock/{name}" if name in available else None


def test_official_llmster_commands_require_remote_script_confirmation() -> None:
    mac = llmster_install_action("Darwin")
    linux = llmster_install_action("Linux")
    windows = llmster_install_action("Windows")
    assert mac is not None and mac.remote_script
    assert mac.display_command == "curl -fsSL https://lmstudio.ai/install.sh | bash"
    assert linux is not None and linux.remote_script
    assert linux.display_command == "curl -fsSL https://lmstudio.ai/install.sh | bash"
    assert windows is not None and windows.remote_script
    assert windows.display_command == "irm https://lmstudio.ai/install.ps1 | iex"
    assert lms_install_action("Darwin") == mac


def test_lmstudio_desktop_install_action() -> None:
    mac = lmstudio_desktop_install_action("Darwin", finder("brew"))
    assert mac is not None
    assert mac.component == "LM Studio Desktop"
    assert mac.command == ("/mock/brew", "install", "--cask", "lm-studio")
    assert lmstudio_desktop_install_action("Darwin", finder()) is None
    assert lmstudio_desktop_install_action("Windows") is None


def test_opencode_chooses_safest_available_official_method() -> None:
    mac = opencode_install_action("Darwin", finder("brew", "npm", "curl"))
    assert mac is not None and mac.command[:2] == ("/mock/brew", "install")
    windows = opencode_install_action("Windows", finder("scoop", "npm"))
    assert windows is not None and windows.command == ("scoop", "install", "opencode")
    npm = opencode_install_action("Windows", finder("npm"))
    assert npm is not None and npm.command == ("npm", "install", "-g", "opencode-ai")


def test_git_install_only_when_reliable_manager_exists() -> None:
    assert git_install_action("Darwin", finder()) is None
    mac = git_install_action("Darwin", finder("brew"))
    assert mac is not None and mac.command == ("brew", "install", "git")
    windows = git_install_action("Windows", finder("winget"))
    assert windows is not None and windows.command[0] == "winget"


@pytest.mark.asyncio
async def test_verbose_command_output_streams_before_process_finishes() -> None:
    output: list[str] = []
    runner = SetupCommandRunner(logging.getLogger("test.streaming"), output.append, verbose=True)
    task = asyncio.create_task(
        runner.run(
            sys.executable,
            "-c",
            "import time; print('first', flush=True); time.sleep(0.2); print('second')",
        )
    )
    await asyncio.sleep(0.1)
    assert output == ["first"]
    result = await task
    assert result.returncode == 0
    assert output == ["first", "second"]


@pytest.mark.asyncio
async def test_verbose_command_output_removes_terminal_escape_codes() -> None:
    output: list[str] = []
    runner = SetupCommandRunner(logging.getLogger("test.ansi"), output.append, verbose=True)
    result = await runner.run(
        sys.executable,
        "-c",
        "print('\\033[0;2mInstalling\\033[0m OpenCode')",
    )
    assert result.returncode == 0
    assert output == ["Installing OpenCode"]


@pytest.mark.asyncio
async def test_verbose_command_output_streams_carriage_return_progress() -> None:
    output: list[str] = []
    runner = SetupCommandRunner(logging.getLogger("test.progress"), output.append, verbose=True)
    result = await runner.run(
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('10%\\r55%\\r100%\\n'); sys.stdout.flush()",
    )
    assert result.returncode == 0
    assert output == ["10%", "55%", "100%"]


@pytest.mark.asyncio
async def test_terminate_interrupts_downloader_before_forcing_it_closed() -> None:
    output: list[str] = []
    runner = SetupCommandRunner(logging.getLogger("test.interrupt"), output.append, verbose=True)
    task = asyncio.create_task(
        runner.run(
            sys.executable,
            "-c",
            "import signal,sys,time; "
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
            "print('ready', flush=True); time.sleep(60)",
        )
    )

    # Keep the signal handler expression small without relying on shell behavior.
    await asyncio.sleep(0.1)
    assert output == ["ready"]
    await runner.terminate()
    result = await task

    assert result.returncode == 0
