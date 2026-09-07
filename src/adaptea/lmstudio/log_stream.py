from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from adaptea.lmstudio.lms_cli import resolve_executable
from adaptea.models import TelemetrySample


def parse_log_event(line: str) -> TelemetrySample | None:
    try:
        data: Any = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get("stats")
    stats: dict[str, Any] = nested if isinstance(nested, dict) else data
    speed = stats.get("tokensPerSecond", stats.get("tokens_per_second"))
    ttft = stats.get("timeToFirstTokenSeconds", stats.get("time_to_first_token_seconds"))
    return TelemetrySample(
        source="lms_log",
        tokens_per_second=float(speed) if isinstance(speed, int | float) else None,
        ttft_seconds=float(ttft) if isinstance(ttft, int | float) else None,
        raw=data,
    )


class LogObserver:
    def __init__(self, executable: str = "lms") -> None:
        self.executable = executable
        self.process: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task[None] | None = None

    async def start(self, callback: Callable[[TelemetrySample], Awaitable[None]]) -> bool:
        try:
            self.process = await asyncio.create_subprocess_exec(
                resolve_executable(self.executable),
                "log",
                "stream",
                "--source",
                "model",
                "--filter",
                "output",
                "--json",
                "--stats",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return False
        self.task = asyncio.create_task(self._read(callback))
        return True

    async def _read(self, callback: Callable[[TelemetrySample], Awaitable[None]]) -> None:
        assert self.process and self.process.stdout
        while line := await self.process.stdout.readline():
            if sample := parse_log_event(line.decode(errors="replace")):
                await callback(sample)

    async def stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), 3)
            if self.process.returncode is None:
                self.process.kill()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
