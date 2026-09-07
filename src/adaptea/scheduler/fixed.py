from dataclasses import dataclass

from adaptea.scheduler.base import Scheduler


@dataclass(slots=True)
class FixedScheduler(Scheduler):
    concurrency: int

    @property
    def target(self) -> int:
        return self.concurrency
