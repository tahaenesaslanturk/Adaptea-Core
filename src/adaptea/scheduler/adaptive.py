from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from adaptea.config import ControllerConfig
from adaptea.models import ControllerDecision, TelemetrySample
from adaptea.scheduler.base import Scheduler


@dataclass(slots=True)
class Baseline:
    ttft_seconds: float | None = None
    tokens_per_second: float | None = None


class AdaptiveScheduler(Scheduler):
    def __init__(
        self,
        starting: int,
        safe_max: int,
        config: ControllerConfig,
        baseline: Baseline | None = None,
    ) -> None:
        self._target = max(1, min(starting, safe_max))
        self.safe_max = max(1, safe_max)
        self.config = config
        self.baseline = baseline or Baseline()
        self.samples: deque[TelemetrySample] = deque(
            maxlen=max(config.pressure_samples, config.healthy_samples, 20)
        )
        self.last_change = 0.0

    @property
    def target(self) -> int:
        return self._target

    def observe(
        self,
        sample: TelemetrySample,
        *,
        running: int,
        ready: int,
        now: float | None = None,
    ) -> ControllerDecision | None:
        self.samples.append(sample)
        clock = time.monotonic() if now is None else now
        if clock - self.last_change < self.config.cooldown_seconds:
            return None
        queue_samples = [item for item in self.samples if item.queued_predictions is not None][
            -self.config.pressure_samples :
        ]
        queue_pressure = len(queue_samples) >= self.config.pressure_samples and all(
            item.queued_predictions is not None and item.queued_predictions > 0
            for item in queue_samples
        )
        recent_pressure = list(self.samples)
        ttft_pressure = self._sustained_ratio(
            recent_pressure,
            "ttft_seconds",
            self.baseline.ttft_seconds,
            self.config.ttft_degradation_ratio,
            self.config.pressure_samples,
            greater=True,
        )
        rate_pressure = self._sustained_ratio(
            recent_pressure,
            "tokens_per_second",
            self.baseline.tokens_per_second,
            self.config.throughput_degradation_ratio,
            self.config.pressure_samples,
            greater=False,
        )
        if (queue_pressure or ttft_pressure or rate_pressure) and self._target > 1:
            reasons = []
            if queue_pressure:
                reasons.append("LM Studio queue remained non-zero")
            if ttft_pressure:
                reasons.append("TTFT degraded relative to calibration")
            if rate_pressure:
                reasons.append("generation throughput degraded relative to calibration")
            return self._change(self._target - 1, "; ".join(reasons), sample, running, ready, clock)

        healthy = list(self.samples)[-self.config.healthy_samples :]
        enough_health = len(healthy) >= self.config.healthy_samples
        # Queue depth is optional in the LM Studio `lms ps --json` contract. Keep an
        # unknown queue distinct from an explicitly empty one, but do not let a missing
        # optional field pin adaptive runs to one worker forever. When every recent
        # sample omits queue depth, a fully occupied target plus ready work is enough to
        # probe one step higher. Any queue/latency/throughput pressure still wins above.
        healthy_queue = [item for item in self.samples if item.queued_predictions is not None][
            -self.config.healthy_samples :
        ]
        queue_clear = len(healthy_queue) >= self.config.healthy_samples and all(
            item.queued_predictions == 0 for item in healthy_queue
        )
        queue_unavailable = enough_health and not any(
            item.queued_predictions is not None for item in self.samples
        )
        # Some `lms ps` builds expose neither queue depth nor generation performance.
        # A sequence of those empty samples is not evidence of spare capacity.  Only
        # probe without queue depth when the model log has supplied enough real latency
        # or throughput observations to make the probe accountable.
        measured_health = [
            item
            for item in self.samples
            if item.ttft_seconds is not None or item.tokens_per_second is not None
        ][-self.config.healthy_samples :]
        saturated_without_queue = (
            queue_unavailable
            and len(measured_health) >= self.config.healthy_samples
            and running >= self._target
        )
        no_bad_ttft = all(
            item.ttft_seconds is None
            or self.baseline.ttft_seconds is None
            or item.ttft_seconds < self.baseline.ttft_seconds * self.config.ttft_degradation_ratio
            for item in healthy
        )
        no_bad_rate = all(
            item.tokens_per_second is None
            or self.baseline.tokens_per_second is None
            or item.tokens_per_second
            > self.baseline.tokens_per_second * self.config.throughput_degradation_ratio
            for item in healthy
        )
        if (
            ready > 0
            and (queue_clear or saturated_without_queue)
            and no_bad_ttft
            and no_bad_rate
            and self._target < self.safe_max
        ):
            reason = (
                "ready backlog and sustained LM Studio headroom"
                if queue_clear
                else "cautious probe: ready backlog, occupied target, and live generation "
                "telemetry; queue depth unavailable"
            )
            return self._change(
                self._target + 1,
                reason,
                sample,
                running,
                ready,
                clock,
            )
        return None

    @staticmethod
    def _sustained_ratio(
        samples: list[TelemetrySample],
        field: str,
        baseline: float | None,
        ratio: float,
        required: int,
        *,
        greater: bool,
    ) -> bool:
        if baseline is None or not samples:
            return False
        values = [
            getattr(sample, field) for sample in samples if getattr(sample, field) is not None
        ]
        values = values[-required:]
        if len(values) < required:
            return False
        threshold = baseline * ratio
        return all(
            value > threshold if greater else value < threshold
            for value in values
            if value is not None
        )

    def _change(
        self,
        target: int,
        reason: str,
        sample: TelemetrySample,
        running: int,
        ready: int,
        now: float,
    ) -> ControllerDecision:
        old = self._target
        self._target = max(1, min(target, self.safe_max))
        self.last_change = now
        return ControllerDecision(
            old_target=old,
            new_target=self._target,
            reason=reason,
            signals={
                "queued_predictions": sample.queued_predictions,
                "ttft_seconds": sample.ttft_seconds,
                "tokens_per_second": sample.tokens_per_second,
                "running_workers": running,
                "ready_tasks": ready,
            },
        )
