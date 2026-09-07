from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

from adaptea.lmstudio.client import LMStudioClient, LMStudioError
from adaptea.lmstudio.lms_cli import sample_ps
from adaptea.lmstudio.models import LMModel
from adaptea.models import TelemetrySample


def sample_from_models(models: list[LMModel]) -> TelemetrySample:
    """Create a live native-API sample when CLI pressure telemetry is unavailable."""
    raw = {"models": [model.model_dump(mode="json") for model in models]}
    from adaptea.lmstudio.lms_cli import parse_ps_json

    sample = parse_ps_json(json.dumps(raw))
    sample.source = "native"
    return sample


class RuntimeTelemetrySampler:
    def __init__(
        self,
        executable: str,
        base_url: str,
        token: str | None = None,
        native_poll_seconds: float = 10.0,
    ) -> None:
        self.executable = executable
        self.base_url = base_url
        self.token = token
        self.cli_available: bool | None = None
        self.native_poll_seconds = native_poll_seconds
        self._last_native_attempt: float | None = None

    async def sample(self) -> TelemetrySample | None:
        if self.cli_available is not False:
            sample = await sample_ps(self.executable)
            if sample is not None:
                self.cli_available = True
                return sample
            self.cli_available = False
        now = time.monotonic()
        if (
            self._last_native_attempt is not None
            and now - self._last_native_attempt < self.native_poll_seconds
        ):
            return None
        self._last_native_attempt = now
        try:
            async with LMStudioClient(self.base_url, self.token, timeout=2) as client:
                return sample_from_models(await client.models())
        except (LMStudioError, ValueError):
            return None


async def poll_ps(
    executable: str,
    interval: float,
    callback: Callable[[TelemetrySample], Awaitable[None]],
    stop: asyncio.Event,
    base_url: str = "http://127.0.0.1:1234",
    token: str | None = None,
) -> None:
    sampler = RuntimeTelemetrySampler(executable, base_url, token)
    while not stop.is_set():
        sample = await sampler.sample()
        if sample:
            await callback(sample)
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except TimeoutError:
            pass
