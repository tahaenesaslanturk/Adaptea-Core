from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from adaptea.config import ControllerConfig
from adaptea.fleet.models import ModelInstance
from adaptea.models import ControllerDecision


@dataclass(slots=True)
class InstanceAdmissionController:
    instance: ModelInstance
    config: ControllerConfig
    clock: Callable[[], float] = time.monotonic
    pressure: deque[bool] = field(default_factory=deque)
    healthy: deque[bool] = field(default_factory=deque)
    last_change: float = field(default=-1e12)

    def observe(
        self, *, queued_requests: int | None, generating: bool | None, running: int
    ) -> ControllerDecision | None:
        target = self.instance.effective_limit
        pressured = queued_requests is not None and queued_requests > 0
        healthy = queued_requests == 0 and (generating is not True or running <= target)
        self.pressure.append(pressured)
        self.healthy.append(healthy)
        while len(self.pressure) > self.config.pressure_samples:
            self.pressure.popleft()
        while len(self.healthy) > self.config.healthy_samples:
            self.healthy.popleft()
        now = self.clock()
        if now - self.last_change < self.config.cooldown_seconds:
            return None
        new_target = target
        reason = ""
        if len(self.pressure) == self.config.pressure_samples and all(self.pressure):
            new_target = max(1, target - 1)
            reason = "sustained per-instance queue pressure"
        elif (
            len(self.healthy) == self.config.healthy_samples
            and all(self.healthy)
            and running >= target
        ):
            ceiling = self.instance.parallel_limit or target
            new_target = min(ceiling, target + 1)
            reason = "sustained per-instance inference headroom"
        if new_target == target:
            return None
        self.instance.admission_target = new_target
        self.last_change = now
        self.pressure.clear()
        self.healthy.clear()
        return ControllerDecision(
            old_target=target,
            new_target=new_target,
            reason=reason,
            signals={
                "instance_id": self.instance.instance_id,
                "queued_requests": queued_requests,
                "generating": generating,
                "running": running,
            },
        )
