from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class SchedulerSnapshot:
    target: int
    running: int
    ready: int
    user_ceiling: int
    parallel_limit: int

    @property
    def fan_out(self) -> int:
        return min(self.target, self.ready, self.user_ceiling, self.parallel_limit)

    @property
    def admissions(self) -> int:
        # Admission-only: a lower target never implies terminating running workers.
        return max(0, min(self.target, self.user_ceiling, self.parallel_limit) - self.running)


class Scheduler(ABC):
    @property
    @abstractmethod
    def target(self) -> int: ...

    def admissions(self, running: int, ready: int, ceiling: int, parallel: int) -> int:
        snapshot = SchedulerSnapshot(self.target, running, ready, ceiling, parallel)
        return min(ready, snapshot.admissions)
