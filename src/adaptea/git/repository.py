from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from adaptea.lmstudio.lms_cli import resolve_executable

ADAPTEA_GIT_AUTHOR_NAME = "Adaptea"
# GitHub associates this noreply address with the official ``adaptea[bot]`` account,
# including the App avatar. It contains no credential and does not grant repository access.
ADAPTEA_GIT_AUTHOR_EMAIL = "322406691+adaptea[bot]@users.noreply.github.com"


@dataclass(slots=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str
    conflicting_files: list[str] = field(default_factory=list)


async def git(root: Path, *args: str, check: bool = True) -> GitResult:
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")
    process = await asyncio.create_subprocess_exec(
        resolve_executable("git"),
        *args,
        cwd=root,
        env=environment,
        # git must never be able to sit waiting on a credential prompt.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    result = GitResult(
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def sanitize_branch(value: str, max_length: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-").lower()
    safe = re.sub(r"[.-]{2,}", "-", safe)
    reserved = {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if safe in reserved:
        safe = f"task-{safe}"
    return (safe or "task")[:max_length].rstrip(".-")


async def ensure_repository(root: Path) -> None:
    result = await git(root, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "adaptea run requires an existing Git repository with at least one commit"
        )
    head = await git(root, "rev-parse", "HEAD", check=False)
    if head.returncode != 0:
        raise RuntimeError("adaptea run requires a repository with at least one commit")
