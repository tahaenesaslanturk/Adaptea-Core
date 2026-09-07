"""Deterministic validation of a finished worker change.

Adaptea validates every worker change with a real command before it is allowed to merge.
The command has to describe the project that is actually being worked on: a repository with
no automated test suite cannot be validated by running one, and reporting that absence as a
failure would reject correct work. This module resolves the effective command for a worktree
and interprets its exit code.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from adaptea.lmstudio.lms_cli import resolve_executable

# pytest reports "no tests were collected" with a dedicated exit code. For a project that
# has no suite at all this is an observation, not a failed validation.
PYTEST_NO_TESTS_COLLECTED = 5

# `npm init` writes this placeholder script, which always exits non-zero.
_NPM_PLACEHOLDER = "no test specified"

# `python -m pytest` without pytest installed exits 1, the same code a genuine test failure
# uses, so the interpreter's own message is what separates the two.
_PYTEST_MISSING = "No module named pytest"

NO_SUITE_DETAIL = (
    "no automated test suite detected in this project; deterministic validation had "
    "nothing to execute"
)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The result of validating one worktree."""

    passed: bool
    exit_code: int
    detail: str
    command: list[str] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        return bool(self.command)


def python_interpreter() -> str:
    """Return a Python interpreter that exists on this host.

    `python` is frequently absent on macOS and on minimal Linux images, and in a packaged
    build `sys.executable` is the Adaptea sidecar rather than an interpreter.
    """
    current = Path(sys.executable) if sys.executable else None
    if (
        not getattr(sys, "frozen", False)
        and current
        and current.is_file()
        and current.stem.lower().startswith("python")
    ):
        return str(current)
    for candidate in ("python3", "python"):
        if resolved := shutil.which(candidate):
            return resolved
    # Never fall back to sys.executable: in a packaged build that is the Adaptea sidecar,
    # and running it with `-m pytest` would launch Adaptea instead of a validator. Naming
    # an interpreter that is absent fails loudly and correctly.
    return "python3"


def _has_pytest_suite(worktree: Path) -> bool:
    if (worktree / "conftest.py").is_file():
        return True
    for name in ("tests", "test"):
        directory = worktree / name
        if directory.is_dir() and next(directory.rglob("*.py"), None) is not None:
            return True
    for pattern in ("test_*.py", "*_test.py", "*/test_*.py", "*/*_test.py"):
        if next(worktree.glob(pattern), None) is not None:
            return True
    return False


def _has_npm_test(worktree: Path) -> bool:
    manifest = worktree / "package.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return False
    script = scripts.get("test")
    return (
        isinstance(script, str) and bool(script.strip()) and _NPM_PLACEHOLDER not in script.lower()
    )


def detect_test_command(worktree: Path) -> list[str]:
    """Infer the project's validator, or return an empty command when it has none."""
    if _has_pytest_suite(worktree):
        return [python_interpreter(), "-m", "pytest"]
    if _has_npm_test(worktree) and (npm := shutil.which("npm")):
        return [npm, "test"]
    return []


def _resolve_executable(command: list[str]) -> list[str]:
    """Repoint a bare `python`/`python3` command at an interpreter that exists."""
    if not command:
        return []
    head, *rest = command
    if head.lower() in {"python", "python3", "python.exe", "python3.exe"} and not shutil.which(
        head
    ):
        return [python_interpreter(), *rest]
    return list(command)


def resolve_test_command(worktree: Path, configured: list[str]) -> list[str]:
    """Return the command that will actually validate `worktree`.

    An explicitly configured `project.test_command` always wins; an empty one means
    "detect what this project uses".
    """
    if configured:
        return _resolve_executable(configured)
    return detect_test_command(worktree)


_PYTEST_MISSING = "No module named pytest"


def interpret(
    command: list[str], exit_code: int, *, no_tests_is_failure: bool, output: str = ""
) -> ValidationOutcome:
    if exit_code == 0:
        return ValidationOutcome(True, 0, "validation passed", command)
    is_pytest = any(part.lower().removesuffix(".exe") == "pytest" for part in command)
    if is_pytest and _PYTEST_MISSING in output:
        return ValidationOutcome(
            False,
            exit_code,
            f"this project has a pytest suite, but {command[0]} cannot import pytest; "
            "install pytest for that interpreter or set project.test_command explicitly",
            command,
        )
    if is_pytest and exit_code == PYTEST_NO_TESTS_COLLECTED and not no_tests_is_failure:
        return ValidationOutcome(True, 0, NO_SUITE_DETAIL, command)
    return ValidationOutcome(False, exit_code, f"validator exited {exit_code}", command)


async def run_validation(
    worktree: Path,
    configured: list[str],
    log_path: Path,
    *,
    no_tests_is_failure: bool = False,
) -> ValidationOutcome:
    """Validate `worktree`, writing the validator's combined output to `log_path`."""
    command = resolve_test_command(worktree, configured)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not command:
        if no_tests_is_failure:
            outcome = ValidationOutcome(False, 1, NO_SUITE_DETAIL, [])
        else:
            outcome = ValidationOutcome(True, 0, NO_SUITE_DETAIL, [])
        log_path.write_text(f"[adaptea] {outcome.detail}\n", encoding="utf-8")
        return outcome
    if not worktree.is_dir():
        detail = f"worktree directory does not exist: {worktree}"
        log_path.write_text(f"[adaptea] {detail}\n", encoding="utf-8")
        return ValidationOutcome(False, 127, detail, command)
    try:
        executable = resolve_executable(command[0])
        process = await asyncio.create_subprocess_exec(
            executable,
            *command[1:],
            cwd=worktree,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (OSError, ValueError) as exc:
        detail = f"validator {command[0]!r} could not be executed: {exc}"
        log_path.write_text(f"[adaptea] {detail}\n", encoding="utf-8")
        return ValidationOutcome(False, 127, detail, command)
    output, _ = await process.communicate()
    exit_code = process.returncode or 0
    outcome = interpret(
        command,
        exit_code,
        no_tests_is_failure=no_tests_is_failure,
        output=output.decode(errors="replace"),
    )
    log_path.write_bytes(
        f"[adaptea] validator: {json.dumps(command)}\n".encode()
        + output
        + f"\n[adaptea] {outcome.detail}\n".encode()
    )
    return outcome
