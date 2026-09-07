from __future__ import annotations

from adaptea.config import ControllerConfig
from adaptea.models import TelemetrySample
from adaptea.scheduler.adaptive import AdaptiveScheduler, Baseline
from adaptea.scheduler.fixed import FixedScheduler
from adaptea.scheduler.naive import NaiveScheduler


def test_fixed_naive_and_hard_ceilings() -> None:
    assert FixedScheduler(4).admissions(running=1, ready=9, ceiling=3, parallel=8) == 2
    assert NaiveScheduler(8).admissions(running=2, ready=9, ceiling=8, parallel=4) == 2
    assert FixedScheduler(4).admissions(running=4, ready=9, ceiling=8, parallel=8) == 0


def test_adaptive_pressure_hysteresis_cooldown_and_no_preemption() -> None:
    config = ControllerConfig(pressure_samples=2, healthy_samples=2, cooldown_seconds=10)
    scheduler = AdaptiveScheduler(4, 6, config, Baseline(1.0, 20.0))
    pressure = TelemetrySample(source="lms_ps", queued_predictions=2)
    assert scheduler.observe(pressure, running=4, ready=4, now=20) is None
    down = scheduler.observe(pressure, running=4, ready=4, now=21)
    assert down is not None and down.new_target == 3
    # Running workers continue; target reduction only closes admissions.
    assert scheduler.admissions(running=4, ready=4, ceiling=8, parallel=8) == 0
    assert scheduler.observe(pressure, running=3, ready=4, now=25) is None
    assert scheduler.observe(pressure, running=3, ready=4, now=32).new_target == 2  # type: ignore[union-attr]
    healthy = TelemetrySample(
        source="lms_ps", queued_predictions=0, ttft_seconds=0.8, tokens_per_second=21
    )
    assert scheduler.observe(healthy, running=2, ready=4, now=45) is None
    up = scheduler.observe(healthy, running=2, ready=4, now=46)
    assert up is not None and up.new_target == 3


def test_unknown_queue_probes_up_only_when_the_current_target_is_occupied() -> None:
    config = ControllerConfig(pressure_samples=2, healthy_samples=2, cooldown_seconds=0)
    scheduler = AdaptiveScheduler(1, 3, config)
    unknown = TelemetrySample(source="lms_log", tokens_per_second=20)
    scheduler.observe(unknown, running=0, ready=4, now=1)
    assert scheduler.observe(unknown, running=0, ready=4, now=2) is None
    assert scheduler.target == 1

    up = scheduler.observe(unknown, running=1, ready=4, now=3)

    assert up is not None and up.new_target == 2
    assert up.reason == (
        "cautious probe: ready backlog, occupied target, and live generation telemetry; "
        "queue depth unavailable"
    )


def test_unknown_queue_without_performance_telemetry_does_not_invent_headroom() -> None:
    config = ControllerConfig(pressure_samples=2, healthy_samples=2, cooldown_seconds=0)
    scheduler = AdaptiveScheduler(1, 3, config)
    empty = TelemetrySample(source="lms_ps")

    scheduler.observe(empty, running=1, ready=4, now=1)
    decision = scheduler.observe(empty, running=1, ready=4, now=2)

    assert decision is None
    assert scheduler.target == 1


def test_mixed_known_and_unknown_queue_samples_do_not_claim_headroom() -> None:
    config = ControllerConfig(pressure_samples=2, healthy_samples=2, cooldown_seconds=0)
    scheduler = AdaptiveScheduler(1, 3, config)
    scheduler.observe(
        TelemetrySample(source="lms_ps", queued_predictions=0),
        running=1,
        ready=4,
        now=1,
    )

    decision = scheduler.observe(TelemetrySample(source="lms_log"), running=1, ready=4, now=2)

    assert decision is None
    assert scheduler.target == 1

    # Queue-less log events may be interleaved with queue-aware process samples. Two
    # explicit zeroes still establish headroom without treating the unknown as zero.
    decision = scheduler.observe(
        TelemetrySample(source="lms_ps", queued_predictions=0),
        running=1,
        ready=4,
        now=3,
    )
    assert decision is not None and decision.new_target == 2


def test_an_adaptive_run_can_be_told_where_to_start() -> None:
    """Without a measured profile the safe start is 1, but that says nothing about the
    machine. An explicit start is honoured and admission keeps adapting from there."""
    from adaptea.config import Config
    from adaptea.runtime.controller import build_scheduler
    from adaptea.scheduler.adaptive import AdaptiveScheduler

    config = Config()
    measured = build_scheduler("adaptive", config, {}, parallel=8, concurrency=None, ceiling=8)
    assert isinstance(measured, AdaptiveScheduler)
    assert measured.target == 1

    requested = build_scheduler("adaptive", config, {}, parallel=8, concurrency=4, ceiling=8)
    assert isinstance(requested, AdaptiveScheduler)
    assert requested.target == 4
    # The backend ceiling and the user's own limit still bound an explicit start.
    assert build_scheduler("adaptive", config, {}, parallel=2, concurrency=6, ceiling=8).target == 2
    assert build_scheduler("adaptive", config, {}, parallel=8, concurrency=6, ceiling=3).target == 3
