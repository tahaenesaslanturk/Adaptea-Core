from dataclasses import dataclass

from adaptea.scheduler.base import Scheduler


@dataclass(slots=True)
class NaiveScheduler(Scheduler):
    max_agents: int

    @property
    def target(self) -> int:
        return self.max_agents
