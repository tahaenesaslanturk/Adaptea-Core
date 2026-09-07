from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from adaptea.config import CommandSecurityConfig, Config
from adaptea.fleet.controller import InstanceAdmissionController
from adaptea.fleet.demand import instance_demand
from adaptea.fleet.discovery import inventory_from_models
from adaptea.fleet.lifecycle import ensure_configured_instances
from adaptea.fleet.models import FleetInventory
from adaptea.fleet.reviewer_routing import ReviewerDecision, ReviewerRouter
from adaptea.fleet.routing import FleetRouter, assignment_counts
from adaptea.git.integration import commit_worker_changes, merge_branch
from adaptea.git.repository import (
    ADAPTEA_GIT_AUTHOR_EMAIL,
    ADAPTEA_GIT_AUTHOR_NAME,
    ensure_repository,
    git,
)
from adaptea.git.worktrees import WorktreeManager
from adaptea.inference import configured_model, create_inference_backend
from adaptea.lmstudio.client import select_loaded_model
from adaptea.lmstudio.lms_cli import parse_instance_pressure
from adaptea.lmstudio.log_stream import LogObserver
from adaptea.lmstudio.telemetry import RuntimeTelemetrySampler
from adaptea.models import (
    FailureKind,
    MergeConflictState,
    Plan,
    RunState,
    TaskRuntime,
    TaskStatus,
    TelemetrySample,
    utc_now,
)
from adaptea.planner.opencode import verify_opencode_models
from adaptea.reviewer.opencode import OpenCodeReviewer
from adaptea.runtime.events import append_jsonl, event
from adaptea.runtime.failures import (
    classify_worker_failure,
    decide_retry,
    failure_title,
    load_retry_policy,
    retry_policy_document,
)
from adaptea.runtime.state import StateStore
from adaptea.scheduler.adaptive import AdaptiveScheduler, Baseline
from adaptea.scheduler.base import Scheduler
from adaptea.scheduler.fixed import FixedScheduler
from adaptea.scheduler.naive import NaiveScheduler
from adaptea.security.approvals import (
    ApprovalDecision,
    ApprovalOutcome,
    ApprovalRequest,
    CommandApprovalBroker,
)
from adaptea.security.commands import policy_document
from adaptea.validation import run_validation
from adaptea.workers.activity import Activity
from adaptea.workers.opencode import OpenCodeWorker


class RuntimeSampler(Protocol):
    async def sample(self) -> TelemetrySample | None: ...


class NoRuntimeTelemetry:
    async def sample(self) -> None:
        return None


def new_run_id() -> str:
    return f"run-{utc_now()[:10]}-{uuid.uuid4().hex[:10]}"


def read_capacity(root: Path) -> dict[str, Any]:
    path = root / ".adaptea" / "capacity.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def build_scheduler(
    mode: Literal["adaptive", "fixed", "naive"],
    config: Config,
    capacity: dict[str, Any],
    parallel: int,
    concurrency: int | None,
    ceiling: int,
) -> Scheduler:
    if mode == "fixed":
        if concurrency is None:
            raise ValueError("--concurrency is required with --scheduler fixed")
        return FixedScheduler(min(concurrency, ceiling, parallel))
    if mode == "naive":
        return NaiveScheduler(min(ceiling, parallel))
    measured = int(capacity.get("recommended_starting_concurrency", 1))
    starting = concurrency if concurrency is not None else measured
    safe = int(capacity.get("safe_max_concurrency", parallel))
    if concurrency is not None:
        # An explicit start may exceed what was measured, but never the backend ceiling
        # or the user's own limit; admission stays adaptive from there.
        safe = max(safe, concurrency)
    tested = capacity.get("tested_concurrency", {})
    row = tested.get(str(starting), {}) if isinstance(tested, dict) else {}
    baseline = Baseline(
        ttft_seconds=_float_or_none(row.get("median_ttft_seconds"))
        if isinstance(row, dict)
        else None,
        tokens_per_second=_float_or_none(row.get("generation_tokens_per_second"))
        if isinstance(row, dict)
        else None,
    )
    return AdaptiveScheduler(
        min(starting, ceiling, parallel),
        min(safe, ceiling, parallel),
        config.controller,
        baseline,
    )


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _fleet_capacity(root: Path, fleet: FleetInventory, parallel: int) -> dict[str, Any]:
    path = root / ".adaptea" / "fleet.json"
    if not path.is_file():
        return {}
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    from adaptea.fleet.calibration import profile_is_stale

    if not isinstance(profile, dict) or profile_is_stale(profile, fleet):
        return {}
    topology = profile.get("recommended_topology") if isinstance(profile, dict) else None
    rows = topology.get("instances") if isinstance(topology, dict) else None
    if not isinstance(rows, list):
        return {}
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = row.get("model")
        workers = row.get("workers_per_instance")
        count = row.get("count")
        if not isinstance(model, str) or not isinstance(workers, int) or not isinstance(count, int):
            continue
        matching = [instance for instance in fleet.instances if instance.model_key == model]
        for instance in matching[:count]:
            instance.admission_target = min(workers, instance.parallel_limit or workers)
            total += instance.admission_target
    if total < 1:
        return {}
    return {
        "recommended_starting_concurrency": min(total, parallel),
        "safe_max_concurrency": min(total, parallel),
        "tested_concurrency": {},
    }


async def create_run(
    root: Path,
    config: Config,
    plan: Plan,
    scheduler_mode: Literal["adaptive", "fixed", "naive"],
    max_agents: int | None,
    concurrency: int | None,
) -> RunState:
    await ensure_repository(root)
    source_branch_result = await git(
        root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
    )
    source_commit_result = await git(root, "rev-parse", "HEAD")
    source_branch = (
        source_branch_result.stdout.strip() if source_branch_result.returncode == 0 else None
    )
    source_commit = source_commit_result.stdout.strip()
    async with create_inference_backend(config, timeout=10) as client:
        models = await client.models()
        model = select_loaded_model(models, configured_model(config))
        backend_capabilities = client.capabilities
    if config.inference.backend != "lmstudio" and config.fleet.enabled:
        raise RuntimeError("Fleet mode is currently supported only by the LM Studio backend.")
    fleet = inventory_from_models(models, config.fleet.models) if config.fleet.enabled else None
    # Load for the work in hand rather than for the configured maximum: a plan whose tasks
    # all want one tier should not spend memory holding the other tier idle.
    demand = instance_demand(plan, config, read_capacity(root)) if fleet is not None else None
    if fleet is not None and await ensure_configured_instances(root, config, fleet, demand):
        async with create_inference_backend(config, timeout=30) as client:
            models = await client.models()
        fleet = inventory_from_models(models, config.fleet.models)
    parallel: int | None = None
    if fleet is not None:
        workers = [
            instance
            for instance in fleet.instances
            if "worker" in instance.roles and instance.available
        ]
        if not workers:
            raise RuntimeError(
                "fleet mode requires at least one loaded instance configured for the worker role"
            )
        parallel = sum(instance.effective_limit for instance in workers)
        planner = next(
            (instance.instance_id for instance in fleet.instances if "planner" in instance.roles),
            None,
        )
        if planner is None:
            raise RuntimeError(
                "fleet mode requires a loaded instance configured for the planner role"
            )
        reviewer = next(
            (instance.instance_id for instance in fleet.instances if "reviewer" in instance.roles),
            planner,
        )
        await verify_opencode_models(
            config,
            list(
                dict.fromkeys([instance.instance_id for instance in workers] + [planner, reviewer])
            ),
            root,
        )
        model = next(
            (
                candidate
                for candidate in models
                if any(instance.id == planner for instance in candidate.loaded_instances)
            ),
            None,
        )
    if not model:
        raise RuntimeError("no configured and inference-ready model was found")
    if fleet is None:
        parallel = model.effective_parallel_limit
    if parallel is None:
        parallel = backend_capabilities.safe_default_parallel_limit or max(
            1, config.worker.max_agents
        )
    ceiling = max_agents or config.worker.max_agents
    if fleet:
        capacity = _fleet_capacity(root, fleet, parallel)
    else:
        capacity = read_capacity(root) if config.inference.backend == "lmstudio" else {}
        profile_model = capacity.get("model")
        profile_lmstudio = capacity.get("lmstudio")
        if (
            not isinstance(profile_model, dict)
            or profile_model.get("key") != model.key
            or not isinstance(profile_lmstudio, dict)
            or profile_lmstudio.get("parallel_limit") != parallel
        ):
            capacity = {}
    scheduler = build_scheduler(scheduler_mode, config, capacity, parallel, concurrency, ceiling)
    run_id = new_run_id()
    manager = WorktreeManager(root, run_id)
    state = RunState(
        run_id=run_id,
        goal=plan.goal,
        repository=str(root),
        integration_branch=manager.integration_branch,
        source_branch=source_branch,
        source_commit=source_commit,
        scheduler=scheduler_mode,
        target_concurrency=scheduler.target,
        starting_concurrency=concurrency if scheduler_mode == "adaptive" else None,
        user_ceiling=ceiling,
        parallel_limit=parallel,
        tasks={task.id: TaskRuntime(spec=task) for task in plan.tasks},
        fleet_enabled=fleet is not None,
        planner_model=(
            next(
                (
                    instance.instance_id
                    for instance in fleet.instances
                    if "planner" in instance.roles
                ),
                None,
            )
            if fleet
            else model.destination
        ),
        reviewer_model=(
            next(
                (
                    instance.instance_id
                    for instance in fleet.instances
                    if "reviewer" in instance.roles
                ),
                next(
                    (
                        instance.instance_id
                        for instance in fleet.instances
                        if "planner" in instance.roles
                    ),
                    None,
                ),
            )
            if fleet
            else model.destination
        ),
        fleet_topology=fleet.model_dump(mode="json") if fleet else {},
    )
    state.refresh_readiness()
    run_dir = root / ".adaptea" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at": state.created_at,
        "backend": config.inference.backend,
        "model": model.key,
        "model_instance": state.planner_model or model.destination,
        "reviewer_instance": state.reviewer_model or model.destination,
        "parallel_limit": parallel,
        "scheduler": scheduler_mode,
        "user_ceiling": ceiling,
        "integration_branch": manager.integration_branch,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "agent_request_relationship": (
            "one OpenCode worker may issue many inference backend requests"
        ),
        "fleet_enabled": state.fleet_enabled,
        "fleet_topology": state.fleet_topology,
        "command_security": policy_document(config.worker.command_security),
        "retry_policy": retry_policy_document(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for filename in (
        "telemetry.jsonl",
        "controller-decisions.jsonl",
        "fleet-controller-decisions.jsonl",
        "routing-decisions.jsonl",
        "runtime-status.jsonl",
        "events.jsonl",
        "command-security-decisions.jsonl",
        "failure-decisions.jsonl",
    ):
        (run_dir / filename).touch()
    StateStore(run_dir).save(state)
    return state


class Orchestrator:
    def __init__(
        self,
        root: Path,
        config: Config,
        state: RunState,
        status_callback: Callable[[RunState, TelemetrySample | None, int], None] | None = None,
        activity_callback: Callable[[str, dict[str, str]], None] | None = None,
        approval_callback: Callable[[ApprovalRequest], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.state = state
        self.run_dir = root / ".adaptea" / "runs" / state.run_id
        self.store = StateStore(self.run_dir)
        self.manager = WorktreeManager(root, state.run_id)
        self.capacity = read_capacity(root)
        self.scheduler = build_scheduler(
            state.scheduler,
            config,
            self.capacity,
            state.parallel_limit,
            state.target_concurrency if state.scheduler == "fixed" else state.starting_concurrency,
            state.user_ceiling,
        )
        if isinstance(self.scheduler, AdaptiveScheduler):
            self.scheduler._target = state.target_concurrency
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.failure_policies, self.max_total_retries = load_retry_policy(
            manifest.get("retry_policy")
        )
        stored_security = manifest.get("command_security")
        if isinstance(stored_security, dict):
            categories = stored_security.get("categories")
            approval = categories.get("approval_required") if isinstance(categories, dict) else None
            approved = approval.get("approved_patterns") if isinstance(approval, dict) else None
            if isinstance(approved, list) and all(isinstance(item, str) for item in approved):
                self.config = config.model_copy(deep=True)
                self.config.worker.command_security = CommandSecurityConfig(
                    approved_commands=approved
                )
        self.model = str(manifest["model"])
        self.fleet_inventory = (
            FleetInventory.model_validate(state.fleet_topology) if state.fleet_enabled else None
        )
        self.router = (
            FleetRouter(self.fleet_inventory, config.fleet.routing)
            if self.fleet_inventory
            else None
        )
        self.instance_controllers = (
            {
                instance.instance_id: InstanceAdmissionController(instance, config.controller)
                for instance in self.fleet_inventory.instances
                if "worker" in instance.roles
            }
            if self.fleet_inventory and state.scheduler == "adaptive" and config.controller.enabled
            else {}
        )
        # A denied command is a question, not only a log line. The broker mutates the
        # very security config the workers below are constructed with, so an approval
        # reaches the task's next attempt without restarting the run.
        self.approvals = CommandApprovalBroker(
            root=root,
            security=self.config.worker.command_security,
            run_id=state.run_id,
            notify=approval_callback,
        )
        self.worker = OpenCodeWorker(
            self.config,
            self.model,
            on_activity=self._record_activity,
            on_blocked_command=self._request_command_approval,
        )
        self.worker_factory: Callable[[str, list[str] | None], OpenCodeWorker] = (
            lambda destination, selectable: OpenCodeWorker(
                self.config,
                destination,
                selectable,
                on_activity=self._record_activity,
                on_blocked_command=self._request_command_approval,
            )
        )
        selectable = (
            [instance.instance_id for instance in self.fleet_inventory.instances]
            if self.fleet_inventory
            else None
        )
        self.reviewer = OpenCodeReviewer(
            self.config, state.reviewer_model or state.planner_model or self.model, selectable
        )
        self.reviewer_factory: Callable[[str, list[str] | None], OpenCodeReviewer] = (
            lambda destination, models: OpenCodeReviewer(self.config, destination, models)
        )
        self.running: dict[str, asyncio.Task[None]] = {}
        self.state_lock = asyncio.Lock()
        self.merge_lock = asyncio.Lock()
        self.last_sample: TelemetrySample | None = None
        self.telemetry_sampler: RuntimeSampler = (
            RuntimeTelemetrySampler(
                config.lmstudio.lms_executable,
                config.lmstudio.base_url,
                config.lmstudio.api_token,
            )
            if config.inference.backend == "lmstudio"
            else NoRuntimeTelemetry()
        )
        self.status_callback = status_callback
        self.activity_callback = activity_callback
        self.admission_enabled = True
        self.abort_requested = False

    def _task_progress(self, task_id: str) -> Callable[[dict[str, str]], None] | None:
        """Bind the run's activity callback to one task, so steps arrive attributed."""
        if self.activity_callback is None:
            return None
        callback = self.activity_callback
        return lambda entry: callback(task_id, entry)

    def _record_activity(self, task_id: str, activity: Activity) -> None:
        """Keep the latest observed action on the task so run status carries it.

        The desktop feed is a scrolling history; this is the single current line shown
        on the task itself. It is display state, not run state: deliberately not
        persisted through the state store, because a resumed run has no live worker.
        """
        task = self.state.tasks.get(task_id)
        if task is not None:
            task.activity = activity.text

    def _request_command_approval(self, task_id: str, command: str) -> None:
        """Ask the user about a command the policy refused for this task."""
        request = self.approvals.request(command, task_id=task_id)
        if request is None:
            return
        event(
            self.run_dir / "events.jsonl",
            "command_approval_requested",
            task_id=task_id,
            request_id=request.request_id,
            command=request.command,
        )

    def resolve_command_approval(
        self, request_id: str, decision: ApprovalDecision
    ) -> ApprovalOutcome:
        """Apply the user's answer, so the next attempt of the task runs under it."""
        outcome = self.approvals.resolve(request_id, decision)
        event(
            self.run_dir / "events.jsonl",
            "command_approval_resolved",
            task_id=outcome.request.task_id,
            request_id=request_id,
            command=outcome.request.command,
            decision=decision,
            approved=outcome.approved,
            remembered=outcome.remembered,
        )
        return outcome

    def pending_command_approvals(self) -> list[ApprovalRequest]:
        return self.approvals.snapshot()

    def stop_admitting(self) -> None:
        """Pause new admissions without preempting any healthy running worker."""
        self.admission_enabled = False
        event(self.run_dir / "events.jsonl", "admission_paused", running=len(self.running))

    def resume_admitting(self) -> None:
        self.admission_enabled = True
        event(self.run_dir / "events.jsonl", "admission_resumed", running=len(self.running))

    def abort(self) -> None:
        """Abort only after frontend confirmation; worker subprocesses handle cancellation."""
        self.abort_requested = True
        self.admission_enabled = False
        event(self.run_dir / "events.jsonl", "run_abort_requested", running=len(self.running))
        active = {
            TaskStatus.RUNNING,
            TaskStatus.VALIDATING,
            TaskStatus.REVIEWING,
            TaskStatus.RETRYING,
        }
        for task in self.state.tasks.values():
            if task.status not in active and task.status != TaskStatus.MERGED:
                task.status = TaskStatus.FAILED
                task.failure = "run aborted by user"
                task.ended_at = utc_now()
        for future in self.running.values():
            future.cancel()
        self.store.save(self.state)

    async def _ensure_models_loaded(self) -> None:
        if self.config.inference.backend != "lmstudio":
            return
        try:
            from adaptea.fleet.discovery import discover_fleet

            inventory = await discover_fleet(self.root, self.config)
            if self.config.fleet.enabled and self.config.fleet.models:
                from adaptea.fleet.lifecycle import ensure_configured_instances

                await ensure_configured_instances(self.root, self.config, inventory)
            else:
                model_key = configured_model(self.config)
                if model_key and not any(
                    item.model_key == model_key or item.instance_id == model_key
                    for item in inventory.instances
                ):
                    import re

                    from adaptea.fleet.lifecycle import InstanceLifecycleManager

                    manager = InstanceLifecycleManager(self.config, inventory)
                    model_name = model_key.rsplit("/", 1)[-1]
                    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
                    await manager.load(
                        model_key,
                        f"adaptea-{safe_name or 'model'}-1",
                        requested_by_user=True,
                    )
        except Exception:
            pass

    async def run(self) -> RunState:
        self.started_at = utc_now()
        await self._ensure_models_loaded()
        await self.manager.create_integration()
        await self._cleanup_recovered_worktrees()
        await self._recover_manual_resolutions()
        self.state.refresh_readiness()
        self.store.save(self.state)
        observer = (
            LogObserver(self.config.lmstudio.lms_executable)
            if self.config.inference.backend == "lmstudio"
            else None
        )
        observer_available = await observer.start(self.observe) if observer else False
        if not observer_available:
            event(
                self.run_dir / "events.jsonl",
                "telemetry_degraded",
                source=("lms log stream" if observer else "backend does not expose pressure"),
                safe_target=self.state.target_concurrency,
            )
        event(
            self.run_dir / "events.jsonl",
            "run_started",
            resumed=any(task.attempts for task in self.state.tasks.values()),
        )
        self._record_unpersisted_failures()
        try:
            while not self.state.complete or self.running:
                await self._reap()
                self.state.refresh_readiness()
                self._record_unpersisted_failures()
                await self._admit()
                await self._sample_runtime()
                append_jsonl(
                    self.run_dir / "runtime-status.jsonl",
                    {
                        "timestamp": utc_now(),
                        "target": self.state.target_concurrency,
                        "running": len(self.running),
                        "ready": sum(
                            task.status == TaskStatus.READY for task in self.state.tasks.values()
                        ),
                    },
                )
                if self.status_callback:
                    self.status_callback(self.state, self.last_sample, len(self.running))
                if not self.running and not any(
                    task.status == TaskStatus.READY for task in self.state.tasks.values()
                ):
                    self.state.refresh_readiness()
                    if self.state.complete:
                        break
                    raise RuntimeError("run is stalled with no ready or running tasks")
                if (
                    not self.running
                    and not self.admission_enabled
                    and any(task.status == TaskStatus.READY for task in self.state.tasks.values())
                ):
                    await asyncio.sleep(self.config.lmstudio.telemetry_poll_seconds)
                    continue
                if self.running:
                    await asyncio.wait(
                        self.running.values(),
                        timeout=self.config.lmstudio.telemetry_poll_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await asyncio.sleep(0)
            await self._reap()
        finally:
            if observer:
                await observer.stop()
            for future in self.running.values():
                if not future.done():
                    future.cancel()
            if self.running:
                with contextlib.suppress(Exception):
                    await asyncio.gather(*self.running.values(), return_exceptions=True)
            self.store.save(self.state)
        self._write_summary()
        event(self.run_dir / "events.jsonl", "run_finished")
        if self.state.tasks and all(
            task.status == TaskStatus.MERGED for task in self.state.tasks.values()
        ):
            # The integration branch is the durable result. Its checkout can be
            # recreated by Git if needed and otherwise duplicates a whole project under
            # .adaptea after every successful run.
            await self.manager.remove(self.manager.integration_path)
        return self.state

    async def _cleanup_recovered_worktrees(self) -> None:
        worktree_root = self.manager.root.resolve()
        for task in self.state.tasks.values():
            if not task.worktree or task.status not in {TaskStatus.READY, TaskStatus.PENDING}:
                continue
            candidate = Path(task.worktree).resolve()
            if (
                candidate.is_relative_to(worktree_root)
                and candidate != self.manager.integration_path
            ):
                await self.manager.remove(candidate)
                task.worktree = None
                task.branch = None
        self.store.save(self.state)

    async def _admit(self) -> None:
        if not self.admission_enabled:
            return
        ready = [task for task in self.state.tasks.values() if task.status == TaskStatus.READY]
        count = self.scheduler.admissions(
            len(self.running), len(ready), self.state.user_ceiling, self.state.parallel_limit
        )
        for task in ready[:count]:
            if self.router:
                if task.pending_retry_context and task.assigned_instance and task.assigned_tier:
                    event(
                        self.run_dir / "events.jsonl",
                        "retry_assignment_preserved",
                        task_id=task.spec.id,
                        instance=task.assigned_instance,
                        tier=task.assigned_tier,
                    )
                else:
                    decision = self.router.route(task.spec, assignment_counts(self.state.tasks))
                    if decision is None:
                        continue
                    self._assign(
                        task, decision.model, decision.instance, decision.tier, decision.reason
                    )
                    append_jsonl(
                        self.run_dir / "routing-decisions.jsonl",
                        decision.model_dump(mode="json"),
                    )
            else:
                task.assigned_model = self.model
                task.assigned_instance = self.model
                task.routing_reason = "legacy single-model configuration"
            task.status = TaskStatus.RUNNING
            task.attempts += 1
            admitted_at = utc_now()
            if task.started_at is None:
                task.started_at = admitted_at
            task.attempt_started_at = admitted_at
            self.store.save(self.state)
            event(
                self.run_dir / "events.jsonl",
                "worker_admitted",
                task_id=task.spec.id,
                attempt=task.attempts,
                target=self.scheduler.target,
            )
            self.running[task.spec.id] = asyncio.create_task(self._execute(task))

    def _route_reviewer(self, task: TaskRuntime, run_reviewer: str) -> ReviewerDecision:
        default_model = (
            next(
                (
                    instance.model_key
                    for instance in self.fleet_inventory.instances
                    if instance.instance_id == run_reviewer
                ),
                self.model,
            )
            if self.fleet_inventory
            else self.model
        )
        if not self.fleet_inventory:
            # Legacy single-model mode has exactly one reviewer; nothing to route.
            return ReviewerDecision(
                task=task.spec.id,
                tier="strong",
                instance=run_reviewer,
                model=default_model,
                reason="single-model configuration; run reviewer used",
            )
        router = ReviewerRouter(
            self.fleet_inventory,
            self.config.fleet.routing.reviewer,
            default_instance=run_reviewer,
            default_model=default_model,
        )
        return router.route(task.spec, assignment_counts(self.state.tasks), attempts=task.attempts)

    async def _execute(self, task: TaskRuntime) -> None:
        retry_context = task.pending_retry_context
        task.pending_retry_context = None
        while True:
            if task.retry_pending:
                task.retry_pending = False
                task.pending_retry_context = None
                task.status = TaskStatus.RUNNING
                task.attempt_started_at = utc_now()
                self.store.save(self.state)
            path, branch = await self.manager.create_task(task.spec.id, task.attempts)
            task.worktree = str(path)
            task.branch = branch
            self.store.save(self.state)
            dependencies = [
                self.state.tasks[dependency].summary or f"{dependency} merged"
                for dependency in task.spec.depends_on
            ]
            plan_context = [
                f"{t.spec.id}: {t.spec.title} [{t.status.value.upper() if t.spec.id != task.spec.id else 'CURRENT TASK'}]"
                for t in self.state.tasks.values()
            ]
            artifact_dir = self.run_dir / "tasks" / task.spec.id / f"attempt-{task.attempts}"
            try:
                destination = task.assigned_instance or self.model
                selectable = (
                    [instance.instance_id for instance in self.fleet_inventory.instances]
                    if self.fleet_inventory
                    else None
                )
                worker = (
                    self.worker_factory(destination, selectable) if self.router else self.worker
                )
                import inspect

                worker_kwargs: dict[str, Any] = {
                    "progress": self._task_progress(task.spec.id),
                }
                if "plan_context" in inspect.signature(worker.run).parameters:
                    worker_kwargs["plan_context"] = plan_context
                # An approval granted during an earlier attempt is only useful if the
                # worker is told the command is available to it now.
                approved_context = self.approvals.context_for(task.spec.id)
                result = await worker.run(
                    path,
                    artifact_dir,
                    self.state.goal,
                    task.spec,
                    dependencies,
                    "\n".join([*filter(None, [retry_context]), *approved_context]) or None,
                    **worker_kwargs,
                )
                self._collect_command_security(task, artifact_dir)
            except asyncio.CancelledError:
                self._collect_command_security(task, artifact_dir)
                raise
            except (OSError, RuntimeError) as exc:
                self._collect_command_security(task, artifact_dir)
                task.attempt_started_at = None
                retry_context = await self._handle_failure(
                    task,
                    FailureKind.INFRASTRUCTURE_ERROR,
                    f"worker launch failed: {exc}",
                    path,
                    artifact_dir,
                )
                if retry_context is not None:
                    continue
                return
            task.worker_exit_code = result.exit_code
            task.attempt_started_at = None
            # The agent has stopped acting; a stale "Editing app.py" would outlive the work.
            task.activity = None
            task.summary = result.summary
            task.wall_seconds = (task.wall_seconds or 0) + result.wall_seconds
            if result.exit_code != 0:
                timed_out_with_changes = result.exit_code == 124 and bool(
                    (await git(path, "status", "--porcelain")).stdout.strip()
                )
                if not timed_out_with_changes:
                    stderr_path = artifact_dir / "stderr.log"
                    stderr = (
                        stderr_path.read_text(encoding="utf-8", errors="replace")
                        if stderr_path.is_file()
                        else ""
                    )
                    stdout_path = artifact_dir / "stdout.log"
                    stdout = (
                        stdout_path.read_text(encoding="utf-8", errors="replace")
                        if stdout_path.is_file()
                        else ""
                    )
                    failure_type = classify_worker_failure(result.exit_code, stderr, stdout)
                    detail = f"OpenCode exited {result.exit_code}"
                    extracted_error: str | None = None
                    for line in stdout.splitlines():
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict) and parsed.get("type") == "error":
                                err = parsed.get("error")
                                if isinstance(err, dict):
                                    msg = err.get("data", {}).get("message") or err.get("message")
                                    if msg:
                                        extracted_error = str(msg)
                                        break
                        except Exception:
                            continue
                    if extracted_error:
                        detail += f": {extracted_error}"
                    elif stderr.strip():
                        detail += f": {stderr.strip()[-1000:]}"
                    self._record_attempt(task, result.wall_seconds, False, failure_type.value)
                    retry_context = await self._handle_failure(
                        task,
                        failure_type,
                        detail,
                        path,
                        artifact_dir,
                        wall_seconds=result.wall_seconds,
                    )
                    if retry_context is not None:
                        continue
                    return
                self._record_salvaged_timeout(task, artifact_dir)
                task.summary = (
                    task.summary
                    + "\n[Adaptea] OpenCode reached its time limit after producing changes; "
                    "deterministic validation will decide whether they are usable."
                ).strip()
                event(
                    self.run_dir / "events.jsonl",
                    "worker_timeout_salvaged",
                    task_id=task.spec.id,
                    timeout_seconds=self.config.worker.timeout_seconds,
                )
            task.status = TaskStatus.VALIDATING
            self.store.save(self.state)
            validation = await run_validation(
                path,
                self.config.project.test_command,
                artifact_dir / "validation.log",
                no_tests_is_failure=self.config.project.no_tests_is_failure,
            )
            validation_code = validation.exit_code
            task.validation_exit_code = validation_code
            task.validation_passed = validation.passed
            task.validation_detail = validation.detail
            task.validation_command = validation.command
            event(
                self.run_dir / "events.jsonl",
                "validation_completed",
                task_id=task.spec.id,
                attempt=task.attempts,
                passed=validation.passed,
                exit_code=validation_code,
                command=validation.command,
                detail=validation.detail,
            )
            self._record_attempt(
                task,
                result.wall_seconds,
                validation.passed,
                validation.detail,
            )
            if not validation.passed:
                retry_context = await self._handle_failure(
                    task,
                    FailureKind.VALIDATION_FAILURE,
                    validation.detail,
                    path,
                    artifact_dir,
                    wall_seconds=result.wall_seconds,
                )
                if retry_context is not None:
                    continue
                return

            run_reviewer = self.state.reviewer_model or self.state.planner_model or self.model
            # Review tier is a per-task decision: how costly is a wrong approval here.
            # Without a fleet, or with reviewers in only one tier, this resolves to the
            # run's single reviewer exactly as it did before task-based routing.
            reviewer_decision = self._route_reviewer(task, run_reviewer)
            reviewer_instance = reviewer_decision.instance
            task.reviewer_instance = reviewer_instance
            task.reviewer_tier = reviewer_decision.tier
            task.reviewer_routing_reason = reviewer_decision.reason
            task.reviewer_model = (
                next(
                    (
                        instance.model_key
                        for instance in self.fleet_inventory.instances
                        if instance.instance_id == reviewer_instance
                    ),
                    self.model,
                )
                if self.fleet_inventory
                else self.model
            )
            append_jsonl(self.run_dir / "reviewer-decisions.jsonl", reviewer_decision.record())
            event(
                self.run_dir / "events.jsonl",
                "reviewer_routed",
                task_id=task.spec.id,
                tier=reviewer_decision.tier,
                instance=reviewer_instance,
                reason=reviewer_decision.reason,
                security_matches=list(reviewer_decision.security_matches),
            )
            task.status = TaskStatus.REVIEWING
            self.store.save(self.state)
            # Stage only inside the isolated worktree so the reviewer receives the exact
            # patch that will be committed. A rejected worktree is discarded safely.
            await git(path, "add", "--all")
            validation_path = artifact_dir / "validation.log"
            validation_output = (
                validation_path.read_text(encoding="utf-8", errors="replace")
                if validation_path.is_file()
                else ""
            )
            selectable = (
                [instance.instance_id for instance in self.fleet_inventory.instances]
                if self.fleet_inventory
                else None
            )
            reviewer = (
                self.reviewer_factory(reviewer_instance, selectable)
                if self.fleet_inventory
                else self.reviewer
            )
            try:
                review = await reviewer.run(
                    path,
                    artifact_dir,
                    self.state.goal,
                    task.spec,
                    validation_output,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failure_type = (
                    FailureKind.MODEL_ERROR
                    if isinstance(exc, ValueError)
                    else FailureKind.INFRASTRUCTURE_ERROR
                )
                retry_context = await self._handle_failure(
                    task,
                    failure_type,
                    f"reviewer failed: {exc}",
                    path,
                    artifact_dir,
                    wall_seconds=result.wall_seconds,
                )
                if retry_context is not None:
                    continue
                return
            task.review_attempts += 1
            task.review_approved = review.approved
            task.reviewer_reason = review.reason
            task.reviewer_findings = review.findings
            task.review_history.append(
                {
                    "attempt": task.attempts,
                    "approved": review.approved,
                    "reason": review.reason,
                    "findings": review.findings,
                    "model": task.reviewer_model,
                    "instance": reviewer_instance,
                    "wall_seconds": review.wall_seconds,
                }
            )
            event(
                self.run_dir / "events.jsonl",
                "review_completed",
                task_id=task.spec.id,
                approved=review.approved,
                reviewer_instance=reviewer_instance,
                reason=review.reason,
            )
            if not review.approved:
                task.review_rejections += 1
                retry_context = await self._handle_failure(
                    task,
                    FailureKind.REVIEW_REJECTION,
                    "reviewer rejected the implementation: "
                    f"{review.reason}; findings={json.dumps(review.findings)}",
                    path,
                    artifact_dir,
                    wall_seconds=result.wall_seconds,
                )
                if retry_context is not None:
                    continue
                return
            await commit_worker_changes(path, task.spec.id)
            async with self.merge_lock:
                merge = await merge_branch(self.manager.integration_path, branch)
            if merge.returncode == 0:
                task.status = TaskStatus.MERGED
                task.failure = None
                task.failure_type = None
                task.failure_reason = None
                task.retry_exhausted = False
                task.ended_at = utc_now()
                self.store.save(self.state)
                event(
                    self.run_dir / "events.jsonl",
                    "task_merged",
                    task_id=task.spec.id,
                    branch=branch,
                )
                await self.manager.remove(path)
                task.worktree = None
                self.store.save(self.state)
                return
            conflict = (merge.stdout + "\n" + merge.stderr)[-4000:]
            resolution_path, resolution_branch = self.manager.resolution_spec(
                task.spec.id, task.attempts
            )
            task.merge_conflict = MergeConflictState(
                status="detected",
                source_branch=branch,
                integration_branch=self.state.integration_branch,
                original_worktree=str(path),
                resolution_worktree=str(resolution_path),
                resolution_branch=resolution_branch,
                conflicting_files=merge.conflicting_files,
                related_tasks=[task.spec.id],
            )
            self.store.save(self.state)
            (artifact_dir / "merge-conflict.log").write_text(conflict, encoding="utf-8")
            retry_context = await self._handle_failure(
                task,
                FailureKind.INFRASTRUCTURE_ERROR,
                f"merge conflict against the current integration branch: {conflict}",
                path,
                artifact_dir,
                wall_seconds=result.wall_seconds,
                terminal_status=TaskStatus.NEEDS_MANUAL_RESOLUTION,
            )
            if retry_context is not None:
                task.merge_conflict = None
                self.store.save(self.state)
                continue
            await self._prepare_merge_resolution(
                task,
                branch,
                path,
                artifact_dir,
                merge.conflicting_files,
            )
            return

    async def _prepare_merge_resolution(
        self,
        task: TaskRuntime,
        source_branch: str,
        original_worktree: Path,
        artifact_dir: Path,
        conflicting_files: list[str],
    ) -> None:
        resolution_path, resolution_branch = self.manager.resolution_spec(
            task.spec.id, task.attempts
        )
        existing = task.merge_conflict
        conflict = MergeConflictState(
            status="preparing",
            detected_at=existing.detected_at if existing else utc_now(),
            source_branch=source_branch,
            integration_branch=self.state.integration_branch,
            original_worktree=str(original_worktree),
            resolution_worktree=str(resolution_path),
            resolution_branch=resolution_branch,
            conflicting_files=sorted(set(conflicting_files)),
            related_tasks=existing.related_tasks if existing else [task.spec.id],
            manual_steps=existing.manual_steps if existing else [],
        )
        task.merge_conflict = conflict
        self.store.save(self.state)
        self._write_merge_conflict_artifact(artifact_dir, conflict)

        try:
            related = await self._related_conflict_tasks(task, conflict.conflicting_files)
            resolution_path, resolution_branch = await self.manager.create_resolution(
                task.spec.id, task.attempts
            )
            status = await git(resolution_path, "status", "--porcelain", check=False)
            merge_head = await git(
                resolution_path,
                "rev-parse",
                "--verify",
                "-q",
                "MERGE_HEAD",
                check=False,
            )
            source_ancestor = await git(
                resolution_path,
                "merge-base",
                "--is-ancestor",
                source_branch,
                "HEAD",
                check=False,
            )
            preparation_result = None
            if (
                source_ancestor.returncode != 0
                and merge_head.returncode != 0
                and not status.stdout.strip()
            ):
                preparation_result = await git(
                    resolution_path,
                    "-c",
                    f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
                    "-c",
                    f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
                    "merge",
                    "--no-ff",
                    "--no-commit",
                    source_branch,
                    check=False,
                )
            elif (
                source_ancestor.returncode != 0
                and merge_head.returncode != 0
                and status.stdout.strip()
            ):
                raise RuntimeError(
                    "resolution worktree contains uncommitted changes without an active merge; "
                    "Adaptea left them untouched"
                )
            unmerged = await git(
                resolution_path,
                "diff",
                "--name-only",
                "--diff-filter=U",
                check=False,
            )
            observed_files = sorted(
                {line.strip() for line in unmerged.stdout.splitlines() if line.strip()}
            )
            if (
                preparation_result is not None
                and preparation_result.returncode != 0
                and not observed_files
            ):
                raise RuntimeError(
                    "could not replay the task branch in the resolution worktree: "
                    f"{preparation_result.stderr.strip() or preparation_result.stdout.strip()}"
                )
            conflict.status = "ready"
            conflict.resolution_worktree = str(resolution_path)
            conflict.resolution_branch = resolution_branch
            conflict.conflicting_files = observed_files or conflict.conflicting_files
            conflict.related_tasks = related
            conflict.manual_steps = self._manual_conflict_steps(conflict)
            conflict.preparation_error = None
            task.failure = (
                "Infrastructure error: merge conflict persisted after the bounded retry. "
                f"Resolve {len(conflict.conflicting_files)} conflicting file(s) manually in "
                f"{resolution_path}, commit the result, then resume this run. Adaptea did not "
                "attempt semantic conflict resolution."
            )
            task.failure_reason = "merge conflict requires manual resolution"
            self.store.save(self.state)
            self._write_merge_conflict_artifact(artifact_dir, conflict)
            event(
                self.run_dir / "events.jsonl",
                "merge_resolution_worktree_ready",
                task_id=task.spec.id,
                worktree=str(resolution_path),
                branch=resolution_branch,
                conflicting_files=conflict.conflicting_files,
                related_tasks=conflict.related_tasks,
            )
        except (OSError, RuntimeError) as exc:
            conflict.status = "preparation_failed"
            conflict.preparation_error = str(exc)
            conflict.manual_steps = self._manual_conflict_steps(conflict)
            task.failure = (
                "Infrastructure error: merge conflict persisted and the safe resolution "
                f"worktree could not be prepared: {exc}. No semantic resolution was attempted."
            )
            self.store.save(self.state)
            self._write_merge_conflict_artifact(artifact_dir, conflict)
            event(
                self.run_dir / "events.jsonl",
                "merge_resolution_worktree_failed",
                task_id=task.spec.id,
                error=str(exc),
            )

    async def _related_conflict_tasks(
        self, task: TaskRuntime, conflicting_files: list[str]
    ) -> list[str]:
        related = {task.spec.id, *task.spec.depends_on}
        conflicts = set(conflicting_files)
        if conflicts:
            for candidate in self.state.tasks.values():
                if not candidate.branch or candidate.spec.id == task.spec.id:
                    continue
                changed = await git(
                    self.root,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    candidate.branch,
                    check=False,
                )
                if conflicts.intersection(changed.stdout.splitlines()):
                    related.add(candidate.spec.id)
        return sorted(related)

    @staticmethod
    def _manual_conflict_steps(conflict: MergeConflictState) -> list[str]:
        files = ", ".join(conflict.conflicting_files) or "the files reported by git status"
        return [
            f"Open the isolated resolution worktree: {conflict.resolution_worktree}",
            "Run `git status --short` and confirm the merge is confined to this worktree.",
            f"Edit the conflicting files ({files}) and remove conflict markers after choosing "
            "the intended combined behavior. Adaptea has not attempted semantic resolution "
            "or chosen either side for you.",
            "Run the relevant tests, then stage only resolved files with `git add -- <files>`.",
            f"Commit the manual resolution on `{conflict.resolution_branch}`.",
            "Return to Adaptea and Resume the run; Adaptea will verify and merge your commit.",
        ]

    @staticmethod
    def _write_merge_conflict_artifact(artifact_dir: Path, conflict: MergeConflictState) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / "merge-conflict.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(conflict.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    async def _recover_manual_resolutions(self) -> None:
        for task in self.state.tasks.values():
            if task.status != TaskStatus.NEEDS_MANUAL_RESOLUTION:
                continue
            conflict = task.merge_conflict
            artifact_dir = self.run_dir / "tasks" / task.spec.id / f"attempt-{task.attempts}"
            if conflict is None:
                if task.branch and task.worktree:
                    await self._prepare_merge_resolution(
                        task,
                        task.branch,
                        Path(task.worktree),
                        artifact_dir,
                        [],
                    )
                    conflict = task.merge_conflict
                else:
                    continue
            if conflict and conflict.status != "resolved":
                await self._prepare_merge_resolution(
                    task,
                    conflict.source_branch,
                    Path(conflict.original_worktree),
                    artifact_dir,
                    conflict.conflicting_files,
                )
                await self._finalize_manual_resolution(task, artifact_dir)

    async def _finalize_manual_resolution(self, task: TaskRuntime, artifact_dir: Path) -> None:
        conflict = task.merge_conflict
        if conflict is None or conflict.status != "ready":
            return
        resolution = Path(conflict.resolution_worktree)
        if not resolution.is_dir():
            return
        unmerged = await git(resolution, "diff", "--name-only", "--diff-filter=U", check=False)
        merge_head = await git(resolution, "rev-parse", "--verify", "-q", "MERGE_HEAD", check=False)
        status = await git(resolution, "status", "--porcelain", check=False)
        source_ancestor = await git(
            resolution,
            "merge-base",
            "--is-ancestor",
            conflict.source_branch,
            "HEAD",
            check=False,
        )
        if (
            unmerged.stdout.strip()
            or merge_head.returncode == 0
            or status.stdout.strip()
            or source_ancestor.returncode != 0
        ):
            return

        async with self.merge_lock:
            merge = await merge_branch(self.manager.integration_path, conflict.resolution_branch)
        if merge.returncode != 0:
            refresh = await git(
                resolution,
                "-c",
                f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
                "-c",
                f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
                "merge",
                "--no-ff",
                "--no-commit",
                self.state.integration_branch,
                check=False,
            )
            current = await git(
                resolution,
                "diff",
                "--name-only",
                "--diff-filter=U",
                check=False,
            )
            conflict.conflicting_files = sorted(
                {
                    *merge.conflicting_files,
                    *(line.strip() for line in current.stdout.splitlines() if line.strip()),
                }
            )
            conflict.related_tasks = await self._related_conflict_tasks(
                task, conflict.conflicting_files
            )
            conflict.manual_steps = self._manual_conflict_steps(conflict)
            task.failure = (
                "The integration branch advanced while the manual resolution was open. "
                "Resolve the newly reported files in the same resolution worktree, commit, "
                "and resume again. No semantic resolution was attempted."
            )
            if refresh.returncode == 0:
                task.failure += " Git prepared the updated merge; it still requires your commit."
            self.store.save(self.state)
            self._write_merge_conflict_artifact(artifact_dir, conflict)
            return

        conflict.status = "resolved"
        conflict.resolved_at = utc_now()
        task.status = TaskStatus.MERGED
        task.failure = None
        task.failure_type = None
        task.failure_reason = None
        task.retry_exhausted = False
        task.ended_at = conflict.resolved_at
        self.store.save(self.state)
        self._write_merge_conflict_artifact(artifact_dir, conflict)
        event(
            self.run_dir / "events.jsonl",
            "manual_merge_resolution_integrated",
            task_id=task.spec.id,
            resolution_branch=conflict.resolution_branch,
        )
        await self.manager.remove(Path(conflict.original_worktree))
        await self.manager.remove(resolution)

    async def _handle_failure(
        self,
        task: TaskRuntime,
        failure_type: FailureKind,
        detail: str,
        path: Path,
        artifact_dir: Path,
        *,
        wall_seconds: float = 0.0,
        terminal_status: TaskStatus = TaskStatus.FAILED,
        resume_via_admission: bool = False,
    ) -> str | None:
        key = failure_type.value
        task.failure_counts[key] = task.failure_counts.get(key, 0) + 1
        decision = decide_retry(
            failure_type,
            task.retry_counts_by_type.get(key, 0),
            task.retry_count,
            self.failure_policies,
            self.max_total_retries,
        )
        escalated = False
        escalation_reason: str | None = None
        if (
            decision.retry
            and decision.escalate_to_strong
            and self.router
            and task.assigned_tier == "fast"
            and self.config.fleet.routing.escalate_failed_fast_task
            and task.fast_escalations < 1
        ):
            routing = self.router.route(
                task.spec,
                assignment_counts(self.state.tasks),
                force_tier="strong",
            )
            if routing and routing.tier == "strong":
                escalated = True
                escalation_reason = routing.reason
                task.fast_escalations += 1
                task.fast_escalation_wasted_seconds += wall_seconds
                self._assign(
                    task,
                    routing.model,
                    routing.instance,
                    routing.tier,
                    routing.reason + f"; {key} retry escalated to strong",
                )
                append_jsonl(
                    self.run_dir / "routing-decisions.jsonl",
                    routing.model_dump(mode="json") | {"escalation": f"{key}_to_strong"},
                )

        title = failure_title(failure_type)
        timestamp = utc_now()
        record = {
            "timestamp": timestamp,
            "task_id": task.spec.id,
            "attempt": task.attempts,
            "type": key,
            "reason": detail,
            "decision": "retry" if decision.retry else "fail",
            "retry": decision.retry,
            "retries_used_for_type": decision.retries_used_for_type,
            "max_retries_for_type": decision.max_retries_for_type,
            "total_retries_used": decision.total_retries_used,
            "max_total_retries": decision.max_total_retries,
            "escalated_to_strong": escalated,
            "escalation_reason": escalation_reason,
            "policy_reason": decision.reason,
            "artifact_recorded": True,
        }
        task.failure_type = failure_type
        task.failure_reason = detail
        task.failure_history.append(record)
        append_jsonl(artifact_dir / "failure-decisions.jsonl", record)
        append_jsonl(self.run_dir / "failure-decisions.jsonl", record)
        event(
            self.run_dir / "events.jsonl",
            "task_failure_classified",
            task_id=task.spec.id,
            attempt=task.attempts,
            failure_type=key,
            retry=decision.retry,
            escalated_to_strong=escalated,
            reason=detail,
            policy_reason=decision.reason,
        )

        if not decision.retry:
            task.status = terminal_status
            task.retry_exhausted = True
            task.retry_pending = False
            task.pending_retry_context = None
            task.failure = f"{title}: {detail}. {decision.reason}."
            task.ended_at = timestamp
            self.store.save(self.state)
            return None

        task.retry_count += 1
        task.retry_counts_by_type[key] = task.retry_counts_by_type.get(key, 0) + 1
        target = "strong model" if escalated else "the current assigned model"
        context = (
            f"{title} on attempt {task.attempts}: {detail}. "
            f"A bounded retry was approved by policy ({decision.reason}) using {target}. "
            "Address this exact failure and do not repeat the unsuccessful approach."
        )
        task.failure = f"{title}: {detail}. {decision.reason}; retry scheduled on {target}."
        task.retry_exhausted = False
        task.pending_retry_context = context
        await self.manager.remove(path)
        task.worktree = None
        task.branch = None
        if resume_via_admission:
            task.status = TaskStatus.PENDING
            task.retry_pending = False
        else:
            task.status = TaskStatus.RETRYING
            task.attempts += 1
            task.retry_pending = True
        self.store.save(self.state)
        event(
            self.run_dir / "events.jsonl",
            "task_retry",
            task_id=task.spec.id,
            failure_type=key,
            next_attempt=task.attempts if not resume_via_admission else task.attempts + 1,
            escalated_to_strong=escalated,
        )
        return context

    def _record_salvaged_timeout(self, task: TaskRuntime, artifact_dir: Path) -> None:
        key = FailureKind.TIMEOUT.value
        task.failure_counts[key] = task.failure_counts.get(key, 0) + 1
        detail = "worker timed out after producing changes; deterministic validation will decide"
        record = {
            "timestamp": utc_now(),
            "task_id": task.spec.id,
            "attempt": task.attempts,
            "type": key,
            "reason": detail,
            "decision": "salvage",
            "retry": False,
            "escalated_to_strong": False,
            "artifact_recorded": True,
        }
        task.failure_type = FailureKind.TIMEOUT
        task.failure_reason = detail
        task.failure = f"Timeout: {detail}."
        task.failure_history.append(record)
        append_jsonl(artifact_dir / "failure-decisions.jsonl", record)
        append_jsonl(self.run_dir / "failure-decisions.jsonl", record)

    def _record_unpersisted_failures(self) -> None:
        for task in self.state.tasks.values():
            for record in task.failure_history:
                if record.get("artifact_recorded") is True:
                    continue
                append_jsonl(
                    self.run_dir / "failure-decisions.jsonl",
                    record | {"task_id": task.spec.id},
                )
                record["artifact_recorded"] = True

    def _collect_command_security(self, task: TaskRuntime, artifact_dir: Path) -> None:
        source = artifact_dir / "command-security-decisions.jsonl"
        if not source.is_file():
            return
        allowed = 0
        denied = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                decision = json.loads(line)
            except ValueError:
                continue
            if not isinstance(decision, dict):
                continue
            action = decision.get("action")
            allowed += int(action == "allow")
            denied += int(action == "deny")
            append_jsonl(
                self.run_dir / "command-security-decisions.jsonl",
                decision | {"task_id": task.spec.id, "attempt": task.attempts},
            )
        event(
            self.run_dir / "events.jsonl",
            "worker_command_security_recorded",
            task_id=task.spec.id,
            attempt=task.attempts,
            allowed=allowed,
            denied=denied,
        )

    def _assign(self, task: TaskRuntime, model: str, instance: str, tier: str, reason: str) -> None:
        task.assigned_model = model
        task.assigned_instance = instance
        task.assigned_tier = tier  # type: ignore[assignment]
        task.routing_reason = reason
        task.routing_history.append(
            {
                "timestamp": utc_now(),
                "tier": tier,
                "model": model,
                "instance": instance,
                "reason": reason,
                "attempt": task.attempts + 1,
            }
        )

    @staticmethod
    def _record_attempt(
        task: TaskRuntime, wall_seconds: float, validation_passed: bool, outcome: str
    ) -> None:
        task.attempt_history.append(
            {
                "attempt": task.attempts,
                "is_retry": task.attempts > 1,
                "tier": task.assigned_tier,
                "model": task.assigned_model,
                "instance": task.assigned_instance,
                "wall_seconds": wall_seconds,
                "validation_passed": validation_passed,
                "outcome": outcome,
                "token_use": None,
            }
        )

    async def _reap(self) -> None:
        finished = [task_id for task_id, future in self.running.items() if future.done()]
        for task_id in finished:
            future = self.running.pop(task_id)
            try:
                await future
            except asyncio.CancelledError:
                task = self.state.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.failure = "run aborted by user"
                task.ended_at = utc_now()
                event(self.run_dir / "events.jsonl", "worker_aborted", task_id=task_id)
            except Exception as exc:
                task = self.state.tasks[task_id]
                path = (
                    Path(task.worktree)
                    if task.worktree
                    else self.manager.root / f"uncreated-{task.spec.id}-{task.attempts}"
                )
                artifact_dir = self.run_dir / "tasks" / task.spec.id / f"attempt-{task.attempts}"
                await self._handle_failure(
                    task,
                    FailureKind.INFRASTRUCTURE_ERROR,
                    f"unhandled worker infrastructure error: {exc}",
                    path,
                    artifact_dir,
                    resume_via_admission=True,
                )
        self.state.refresh_readiness()
        self.store.save(self.state)

    async def _sample_runtime(self) -> None:
        sample = await self.telemetry_sampler.sample()
        if sample:
            await self.observe(sample)

    async def observe(self, sample: TelemetrySample) -> None:
        self.last_sample = sample
        append_jsonl(self.run_dir / "telemetry.jsonl", sample.model_dump(mode="json"))
        if self.fleet_inventory:
            for instance_id, pressure in parse_instance_pressure(sample.raw).items():
                instance = self.fleet_inventory.instance(instance_id)
                if instance:
                    queued = pressure.get("queued_requests")
                    generating = pressure.get("generation_status")
                    instance.queued_requests = queued if isinstance(queued, int) else None
                    instance.generation_status = (
                        generating if isinstance(generating, bool) else None
                    )
            running_by_instance = assignment_counts(self.state.tasks)
            for instance_id, controller in self.instance_controllers.items():
                instance = controller.instance
                decision = controller.observe(
                    queued_requests=instance.queued_requests,
                    generating=instance.generation_status,
                    running=running_by_instance.get(instance_id, 0),
                )
                if decision:
                    append_jsonl(
                        self.run_dir / "fleet-controller-decisions.jsonl",
                        decision.model_dump(mode="json"),
                    )
        if not isinstance(self.scheduler, AdaptiveScheduler):
            return
        ready = sum(task.status == TaskStatus.READY for task in self.state.tasks.values())
        decision = self.scheduler.observe(sample, running=len(self.running), ready=ready)
        if decision:
            self.state.target_concurrency = decision.new_target
            append_jsonl(
                self.run_dir / "controller-decisions.jsonl", decision.model_dump(mode="json")
            )
            self.store.save(self.state)

    def _write_summary(self) -> None:
        merged = sum(task.status == TaskStatus.MERGED for task in self.state.tasks.values())
        total = len(self.state.tasks)
        attempts = sum(task.attempts for task in self.state.tasks.values())
        assignments: dict[str, dict[str, Any]] = {}
        for task in self.state.tasks.values():
            key = task.assigned_instance or task.assigned_model or "legacy-unknown"
            row = assignments.setdefault(
                key,
                {
                    "model": task.assigned_model,
                    "tier": task.assigned_tier,
                    "tasks_attempted": 0,
                    "tasks_passed": 0,
                    "tasks_failed": 0,
                    "durations_seconds": [],
                    "retries": 0,
                },
            )
            row["tasks_attempted"] += 1
            row["tasks_passed"] += int(task.status == TaskStatus.MERGED)
            row["tasks_failed"] += int(task.status != TaskStatus.MERGED)
            row["retries"] += max(0, task.attempts - 1)
            duration = _task_duration(task)
            if duration is not None:
                row["durations_seconds"].append(duration)
        for row in assignments.values():
            durations = row.pop("durations_seconds")
            row["median_duration_seconds"] = (
                sorted(durations)[len(durations) // 2] if durations else None
            )
            attempted = row["tasks_attempted"]
            row["pass_rate"] = row["tasks_passed"] / attempted if attempted else None
        models, tiers = _attempt_aggregates(self.state.tasks)
        runtime_rows = []
        runtime_path = self.run_dir / "runtime-status.jsonl"
        if runtime_path.is_file():
            for line in runtime_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    runtime_rows.append(value)
        running_samples = [
            row["running"] for row in runtime_rows if isinstance(row.get("running"), int)
        ]
        security_rows: list[dict[str, Any]] = []
        security_path = self.run_dir / "command-security-decisions.jsonl"
        if security_path.is_file():
            for line in security_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    security_rows.append(value)
        finished_at = utc_now()
        started_at = getattr(self, "started_at", None) or self.state.created_at
        wall_seconds: float | None = None
        try:
            wall_seconds = max(
                0.0,
                (
                    datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
                ).total_seconds(),
            )
        except (ValueError, TypeError):
            pass
        summary = {
            "run_id": self.state.run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "tasks_total": total,
            "tasks_merged": merged,
            "pass_rate": merged / total if total else 0,
            "worker_starts": attempts,
            "worker_retries": sum(max(0, task.attempts - 1) for task in self.state.tasks.values()),
            "scheduler": self.state.scheduler,
            "final_target": self.state.target_concurrency,
            "statuses": {task_id: task.status for task_id, task in self.state.tasks.items()},
            "fleet_enabled": self.state.fleet_enabled,
            "planner_model": self.state.planner_model,
            "reviewer_model": self.state.reviewer_model,
            "topology": self.state.fleet_topology,
            "instances": assignments,
            "models": models,
            "tiers": tiers,
            "models_used": sorted(
                {task.assigned_model for task in self.state.tasks.values() if task.assigned_model}
            ),
            "instances_used": sorted(assignments),
            "fast_to_strong_escalations": sum(
                task.fast_escalations for task in self.state.tasks.values()
            ),
            "fast_escalation_wasted_seconds": sum(
                task.fast_escalation_wasted_seconds for task in self.state.tasks.values()
            ),
            "reviews_total": sum(task.review_attempts for task in self.state.tasks.values()),
            "reviewer_rejections": sum(
                task.review_rejections for task in self.state.tasks.values()
            ),
            "review_approval_rate": (
                sum(
                    int(review.get("approved") is True)
                    for task in self.state.tasks.values()
                    for review in task.review_history
                )
                / sum(task.review_attempts for task in self.state.tasks.values())
                if sum(task.review_attempts for task in self.state.tasks.values())
                else None
            ),
            "peak_simultaneous_workers": max(running_samples, default=0),
            "average_simultaneous_workers": (
                sum(running_samples) / len(running_samples) if running_samples else 0
            ),
            "command_security": {
                "decisions": len(security_rows),
                "allowed": sum(row.get("action") == "allow" for row in security_rows),
                "denied": sum(row.get("action") == "deny" for row in security_rows),
                "blocked": sum(row.get("category") == "blocked" for row in security_rows),
            },
            "failures": {
                kind.value: {
                    "occurrences": sum(
                        task.failure_counts.get(kind.value, 0) for task in self.state.tasks.values()
                    ),
                    "retries": sum(
                        task.retry_counts_by_type.get(kind.value, 0)
                        for task in self.state.tasks.values()
                    ),
                    "terminal_tasks": sum(
                        task.failure_type == kind and task.status != TaskStatus.MERGED
                        for task in self.state.tasks.values()
                    ),
                }
                for kind in FailureKind
            },
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )


def _task_duration(task: TaskRuntime) -> float | None:
    if not task.started_at or not task.ended_at:
        return None
    try:
        return max(
            0.0,
            (
                datetime.fromisoformat(task.ended_at) - datetime.fromisoformat(task.started_at)
            ).total_seconds(),
        )
    except ValueError:
        return None


def _attempt_aggregates(
    tasks: dict[str, TaskRuntime],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    models: dict[str, dict[str, Any]] = {}
    tiers: dict[str, dict[str, Any]] = {}
    for task in tasks.values():
        for attempt in task.attempt_history:
            for destination, groups in (
                (attempt.get("model"), models),
                (attempt.get("tier"), tiers),
            ):
                if not isinstance(destination, str):
                    continue
                row = groups.setdefault(
                    destination,
                    {
                        "attempts": 0,
                        "passed": 0,
                        "failed": 0,
                        "retries": 0,
                        "durations_seconds": [],
                    },
                )
                row["attempts"] += 1
                row["retries"] += int(attempt.get("is_retry") is True)
                passed = attempt.get("validation_passed") is True
                row["passed"] += int(passed)
                row["failed"] += int(not passed)
                duration = attempt.get("wall_seconds")
                if isinstance(duration, int | float):
                    row["durations_seconds"].append(float(duration))
    for groups in (models, tiers):
        for row in groups.values():
            durations = row.pop("durations_seconds")
            row["median_duration_seconds"] = statistics.median(durations) if durations else None
            row["pass_rate"] = row["passed"] / row["attempts"] if row["attempts"] else None
    return models, tiers


#: Statuses a run ends in with work left undone. Nothing automatic will move them again:
#: the retry budget is spent, or a prerequisite of theirs failed.
_EXHAUSTED = {TaskStatus.FAILED, TaskStatus.NEEDS_MANUAL_RESOLUTION, TaskStatus.BLOCKED}


def reopen_exhausted(state: RunState) -> int:
    """Give terminally failed tasks a fresh budget because the user asked for one.

    Automatic resume deliberately does not do this. A task whose retries ran out failed
    for a reason, and clearing that on every resume would loop on it forever. But a person
    who has read the failure and fixed the cause is making a judgement no policy can make,
    and until now the interface had no way to express it: a failed run offered nothing at
    all. What happened is kept in ``failure_history``; only the budget is renewed.
    """
    reopened = 0
    for task in state.tasks.values():
        if task.status not in _EXHAUSTED:
            continue
        task.failure_history.append(
            {
                "timestamp": utc_now(),
                "attempt": task.attempts,
                "type": task.failure_type.value if task.failure_type else "unclassified",
                "reason": task.failure_reason or task.failure or "task ended without merging",
                "decision": "retry",
                "retry": True,
                "policy_reason": "the user asked for another attempt",
                "source": "user_retry",
            }
        )
        task.status = TaskStatus.PENDING
        task.retry_exhausted = False
        task.retry_pending = False
        task.retry_count = 0
        task.retry_counts_by_type = {}
        task.pending_retry_context = (
            "A previous attempt ended without merging and you asked to try again. "
            f"What failed before: {task.failure_reason or task.failure or 'unknown'}. "
            "Start from the current integration branch and address that directly."
        )
        task.failure = None
        task.failure_type = None
        task.failure_reason = None
        if task.merge_conflict and task.merge_conflict.status == "detected":
            task.merge_conflict = None
        reopened += 1
    state.refresh_readiness()
    return reopened


def prepare_resume(
    state: RunState, run_dir: Path | None = None, *, retry_failed: bool = False
) -> RunState:
    snapshot: object = None
    if run_dir is not None:
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict):
            snapshot = manifest.get("retry_policy")
    policies, max_total_retries = load_retry_policy(snapshot)
    for task in state.tasks.values():
        if task.status == TaskStatus.RETRYING:
            # The old attempt number is reserved immediately before an in-process retry.
            # Put it back so admission consumes it exactly once after a process restart.
            task.attempts = max(0, task.attempts - 1)
            task.retry_pending = False
            if task.merge_conflict and task.merge_conflict.status == "detected":
                task.merge_conflict = None
            task.status = TaskStatus.PENDING
            task.failure = "Recovered before the scheduled retry started; retry budget preserved."
            continue
        if (
            task.merge_conflict
            and task.merge_conflict.status == "detected"
            and task.status in {TaskStatus.RUNNING, TaskStatus.VALIDATING, TaskStatus.REVIEWING}
        ):
            key = FailureKind.INFRASTRUCTURE_ERROR.value
            task.failure_counts[key] = task.failure_counts.get(key, 0) + 1
            decision = decide_retry(
                FailureKind.INFRASTRUCTURE_ERROR,
                task.retry_counts_by_type.get(key, 0),
                task.retry_count,
                policies,
                max_total_retries,
            )
            reason = "merge conflict was detected before the Adaptea process stopped"
            record = {
                "timestamp": utc_now(),
                "attempt": task.attempts,
                "type": key,
                "reason": reason,
                "decision": "retry" if decision.retry else "manual_resolution",
                "retry": decision.retry,
                "policy_reason": decision.reason,
                "source": "resume_merge_conflict",
            }
            task.failure_history.append(record)
            task.failure_type = FailureKind.INFRASTRUCTURE_ERROR
            task.failure_reason = reason
            if decision.retry:
                task.retry_count += 1
                task.retry_counts_by_type[key] = task.retry_counts_by_type.get(key, 0) + 1
                task.pending_retry_context = (
                    "A merge conflict was detected before interruption. Resume approved the "
                    f"single clean-worktree retry ({decision.reason})."
                )
                task.merge_conflict = None
                task.retry_exhausted = False
                task.status = TaskStatus.PENDING
                task.failure = f"Infrastructure error: {reason}. {decision.reason}."
            else:
                task.retry_exhausted = True
                task.status = TaskStatus.NEEDS_MANUAL_RESOLUTION
                task.failure = (
                    f"Infrastructure error: {reason}. {decision.reason}. A safe manual "
                    "resolution worktree will be prepared on resume."
                )
            continue
        if task.status in {
            TaskStatus.RUNNING,
            TaskStatus.VALIDATING,
            TaskStatus.REVIEWING,
        }:
            key = FailureKind.INFRASTRUCTURE_ERROR.value
            task.failure_counts[key] = task.failure_counts.get(key, 0) + 1
            decision = decide_retry(
                FailureKind.INFRASTRUCTURE_ERROR,
                task.retry_counts_by_type.get(key, 0),
                task.retry_count,
                policies,
                max_total_retries,
            )
            reason = "Adaptea process stopped while the task attempt was active"
            record = {
                "timestamp": utc_now(),
                "attempt": task.attempts,
                "type": key,
                "reason": reason,
                "decision": "retry" if decision.retry else "fail",
                "retry": decision.retry,
                "policy_reason": decision.reason,
                "source": "resume",
            }
            task.failure_history.append(record)
            task.failure_type = FailureKind.INFRASTRUCTURE_ERROR
            task.failure_reason = reason
            if decision.retry:
                task.retry_count += 1
                task.retry_counts_by_type[key] = task.retry_counts_by_type.get(key, 0) + 1
                task.pending_retry_context = (
                    f"Infrastructure error: {reason}. Resume approved one bounded retry "
                    f"({decision.reason}). Recreate the work cleanly."
                )
                task.retry_exhausted = False
                task.status = TaskStatus.PENDING
                task.failure = f"Infrastructure error: {reason}. {decision.reason}."
            else:
                task.retry_exhausted = True
                task.status = TaskStatus.FAILED
                task.failure = f"Infrastructure error: {reason}. {decision.reason}."
            continue
        if task.status == TaskStatus.READY:
            task.status = TaskStatus.PENDING
    if retry_failed:
        reopen_exhausted(state)
    state.refresh_readiness()
    return state
