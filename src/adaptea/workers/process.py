"""Run an OpenCode subprocess while reporting what it is doing, not just that it runs.

``asyncio.subprocess.Process.communicate`` returns only once the process exits, so a
planner or worker that runs for minutes produces no observable progress until it is
finished. These helpers drain both pipes as they fill, hand each completed stdout line to
a callback, and keep the same bytes ``communicate`` would have returned.

Output is read in chunks rather than with ``readline``: OpenCode embeds full tool output
in a single JSON line, which routinely exceeds asyncio's default line limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from adaptea.workers.activity import Activity, summarize_line

#: How long a terminated process is given to exit before it is killed.
_GRACE_SECONDS = 5.0
_CHUNK = 65536


@dataclass(slots=True)
class ProcessOutput:
    stdout: bytes
    stderr: bytes
    #: True when the process was terminated for exceeding its deadline.
    timed_out: bool = False


async def _drain(
    stream: asyncio.StreamReader | None,
    sink: list[bytes],
    on_line: Callable[[bytes], None] | None = None,
) -> None:
    if stream is None:
        return
    pending = b""
    while True:
        chunk = await stream.read(_CHUNK)
        if not chunk:
            break
        sink.append(chunk)
        if on_line is None:
            continue
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            on_line(line)
    if on_line and pending:
        on_line(pending)


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        return

    try:
        await asyncio.wait_for(process.wait(), _GRACE_SECONDS)
    except TimeoutError:
        with contextlib.suppress(Exception):
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
            else:
                process.kill()
        with contextlib.suppress(Exception):
            await process.wait()


async def communicate_with_activity(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    on_activity: Callable[[Activity], None] | None = None,
    on_line: Callable[[bytes], None] | None = None,
    workspace: Path | None = None,
) -> ProcessOutput:
    """Collect both pipes, reporting each observed action as it is parsed.

    ``on_line`` receives every complete stdout line, summarised or not, so a caller can
    persist the raw event stream while ``on_activity`` drives what the user sees.

    On timeout the process is stopped and whatever it produced is still returned, so a
    stalled run remains diagnosable. Cancellation stops the process and re-raises.
    """
    out: list[bytes] = []
    err: list[bytes] = []

    def handle(line: bytes) -> None:
        if on_line is not None:
            on_line(line)
        if on_activity is None:
            return
        activity = summarize_line(line.decode("utf-8", errors="replace"), workspace=workspace)
        if activity is not None:
            on_activity(activity)

    pipes = asyncio.gather(
        _drain(process.stdout, out, handle if (on_activity or on_line) else None),
        _drain(process.stderr, err),
    )
    try:
        await asyncio.wait_for(asyncio.shield(pipes), timeout)
        await process.wait()
    except TimeoutError:
        await _stop(process)
        with contextlib.suppress(Exception):
            await pipes
        return ProcessOutput(b"".join(out), b"".join(err), timed_out=True)
    except asyncio.CancelledError:
        pipes.cancel()
        with contextlib.suppress(Exception):
            await pipes
        await _stop(process)
        raise
    return ProcessOutput(b"".join(out), b"".join(err))
