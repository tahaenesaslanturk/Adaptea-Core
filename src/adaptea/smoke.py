from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from adaptea.config import Config, load_config
from adaptea.diagnostics.system import project_opencode_path
from adaptea.git.repository import git
from adaptea.git.worktrees import WorktreeManager
from adaptea.inference import (
    InferenceBackendError,
    configured_model,
    create_inference_backend,
    set_configured_model,
)
from adaptea.lmstudio.client import select_loaded_model
from adaptea.models import Plan, TaskSpec, TaskStatus
from adaptea.runtime.controller import Orchestrator, create_run
from adaptea.setup.manager import SetupManager, create_setup_logger
from adaptea.validation import run_validation

VerificationLayer = Literal["Git", "LM Studio", "Ollama", "Model", "OpenCode", "Adaptea"]
VERIFICATION_LAYERS: tuple[VerificationLayer, ...] = (
    "Git",
    "LM Studio",
    "Model",
    "OpenCode",
    "Adaptea",
)

LAYER_REMEDIES: dict[VerificationLayer, str] = {
    "Git": "Install Git, make sure it is available on PATH, then run Verify setup again.",
    "LM Studio": ("Open LM Studio, start its local server on the configured endpoint, then retry."),
    "Ollama": "Start Ollama on the configured endpoint, then retry.",
    "Model": ("Make the selected model available with at least 32,768 context tokens, then retry."),
    "OpenCode": (
        "Install OpenCode and re-run Setup so the project-local inference provider is written."
    ),
    "Adaptea": (
        "Open the setup log, fix the reported worker, validation, or merge error, then retry."
    ),
}


@dataclass(frozen=True, slots=True)
class SmokeStep:
    name: str
    success: bool
    detail: str
    layer: VerificationLayer = "Adaptea"
    remedy: str | None = None


@dataclass(slots=True)
class SmokeTestResult:
    success: bool
    steps: list[SmokeStep] = field(default_factory=list)
    duration_seconds: float = 0.0
    artifact_directory: Path | None = None
    worker_retries: int = 0


ProgressCallback = Callable[[SmokeStep], None]
ActivityCallback = Callable[[str], None]


class MVPSmokeTest:
    """Exercise the real local-model, worker, worktree, validation, and merge pipeline."""

    def __init__(
        self,
        source_root: Path,
        config: Config | None = None,
        *,
        progress: ProgressCallback | None = None,
        activity: ActivityCallback | None = None,
        keep_fixture: bool = False,
        scheduler_mode: Literal["adaptive", "fixed", "naive"] = "adaptive",
        concurrency: int | None = None,
        minimum_peak_workers: int = 1,
    ) -> None:
        self.source_root = source_root.resolve()
        self.config = config or load_config(source_root)
        self.progress = progress
        self.activity = activity
        self.keep_fixture = keep_fixture
        self.scheduler_mode = scheduler_mode
        self.concurrency = concurrency
        self.minimum_peak_workers = minimum_peak_workers
        self.worker_retries = 0
        self.steps: list[SmokeStep] = []

    def announce(self, message: str) -> None:
        if self.activity:
            self.activity(message)

    def record(
        self,
        name: str,
        success: bool,
        detail: str,
        *,
        layer: VerificationLayer = "Adaptea",
        remedy: str | None = None,
    ) -> None:
        step = SmokeStep(
            name,
            success,
            detail,
            layer,
            None if success else remedy or LAYER_REMEDIES[layer],
        )
        self.steps.append(step)
        if self.progress:
            self.progress(step)

    async def run(self) -> SmokeTestResult:
        started = time.perf_counter()
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if self.keep_fixture:
            fixture = Path(tempfile.mkdtemp(prefix="adaptea-mvp-smoke-"))
        else:
            temporary = tempfile.TemporaryDirectory(prefix="adaptea-mvp-smoke-")
            fixture = Path(temporary.name)
        try:
            await self._run(fixture)
        except Exception as exc:
            self.record("Smoke test", False, _concise_error(exc), layer="Adaptea")
        success = bool(self.steps) and all(step.success for step in self.steps)
        result = SmokeTestResult(
            success=success,
            steps=self.steps.copy(),
            duration_seconds=time.perf_counter() - started,
            artifact_directory=fixture if self.keep_fixture else None,
            worker_retries=self.worker_retries,
        )
        if temporary is not None:
            temporary.cleanup()
        return result

    async def _run(self, fixture: Path) -> None:
        backend_name = "Ollama" if self.config.inference.backend == "ollama" else "LM Studio"
        backend_layer: VerificationLayer = (
            "Ollama" if self.config.inference.backend == "ollama" else "LM Studio"
        )
        self.announce(f"Verifying Git → {backend_name} → model → OpenCode → Adaptea…")
        logger, _log_path = create_setup_logger(self.source_root)
        manager = SetupManager(
            self.source_root,
            self.config,
            logger=logger,
            notice=self.announce,
        )
        snapshot = await manager.diagnose()
        git_ready = bool(snapshot.git_executable)
        self.record(
            "Git executable",
            git_ready,
            f"Git is available at {snapshot.git_executable}."
            if git_ready
            else "Git was not found on PATH or at a supported installation path.",
            layer="Git",
        )
        if not git_ready:
            return

        backend_ready = snapshot.server_reachable and snapshot.native_api_usable
        self.record(
            f"{backend_name} server",
            backend_ready,
            f"{backend_name}'s local server and native model API are reachable."
            if backend_ready
            else snapshot.server_error
            or f"{backend_name}'s local server or native model API is not reachable.",
            layer=backend_layer,
        )
        if not backend_ready:
            return

        self.announce(f"Asking the selected {backend_name} model for a deterministic marker…")
        if snapshot.selected_model is None or not snapshot.selected_model.ready:
            # Which model, and what is up instead. "No selected model is inference-ready"
            # named neither, so the one thing the reader had to know — that the planner
            # this project is configured for is not among the models they loaded — was
            # the one thing the failure did not say.
            planner = self.config.fleet.planner() if self.config.fleet.enabled else None
            wanted = planner.model if planner else configured_model(self.config)
            role = "planner model" if planner else "selected model"
            loaded = ", ".join(sorted({model.key for model in snapshot.models if model.ready}))
            self.record(
                "Selected model",
                False,
                (
                    f"{wanted} is this project's {role}, and it is not loaded in "
                    f"{backend_name}. Loaded: {loaded or 'nothing'}."
                    if wanted
                    else f"No model is loaded in {backend_name}."
                ),
                layer="Model",
                remedy=(
                    f"Load {wanted} in Environment -> Models, or give one of the loaded "
                    "models the planner role there and save."
                    if wanted
                    else "Choose a model in Environment -> Models and save; that loads it."
                ),
            )
            return
        try:
            async with create_inference_backend(self.config, timeout=120) as client:
                selected = select_loaded_model(await client.models(), snapshot.selected_model.key)
                if selected is None:
                    self.record(
                        "Selected model",
                        False,
                        f"The selected model disappeared from {backend_name} before inference.",
                        layer="Model",
                    )
                    return
                response = await client.chat(
                    selected.key,
                    "Reply with exactly ADAPTEA_DIRECT_OK",
                    max_output_tokens=24,
                )
        except InferenceBackendError as exc:
            self.record(
                backend_name,
                False,
                f"The {backend_name} connection was interrupted: {_concise_error(exc)}",
                layer=backend_layer,
            )
            return
        direct_text = json.dumps(response.model_dump(mode="json"))
        direct_ok = "ADAPTEA_DIRECT_OK" in direct_text
        self.record(
            "Direct model inference",
            direct_ok,
            "Direct local-model inference succeeded."
            if direct_ok
            else "The direct model response did not contain the expected marker.",
            layer="Model",
        )
        if not direct_ok:
            return

        self.announce("Checking that OpenCode recognizes the selected local model…")
        if not snapshot.opencode_executable:
            self.record(
                "OpenCode executable",
                False,
                "OpenCode was not found on PATH or at a supported installation path.",
                layer="OpenCode",
            )
            return
        opencode_result = await manager.smoke_test(snapshot)
        self.record(
            "OpenCode model catalog",
            opencode_result.success,
            opencode_result.detail,
            layer="OpenCode",
        )
        if not opencode_result.success:
            return

        self.announce("Creating an isolated Git fixture and integration branch…")
        try:
            await self._prepare_fixture(fixture)
        except Exception as exc:
            self.record(
                "Git fixture",
                False,
                f"The isolated Git fixture could not be created: {_concise_error(exc)}",
                layer="Git",
                remedy=(
                    "Check Git permissions and identity configuration, then run Verify setup again."
                ),
            )
            return
        self.record(
            "Worktree creation",
            True,
            f"Deterministic fixture created at {fixture}.",
            layer="Adaptea",
        )
        parallel_fixture = self.minimum_peak_workers > 1
        plan = self._fixture_plan(parallel=parallel_fixture)
        fixture_config = self.config.model_copy(deep=True)
        # The general system test proves the minimum usable pipeline with the selected loaded
        # model. Optional multi-model topology is measured separately by Fleet Calibration.
        fixture_config.fleet.enabled = False
        # Turning the fleet off also drops the planner that `diagnose` resolved the selected
        # model from, so the fixture would fall back to `lmstudio.model`. Saving a fleet
        # unloads the instances it leaves out, which can strand that key on a model that is
        # no longer inference-ready — the run then failed with "no configured and
        # inference-ready model was found" after the Model layer had just passed. Verify and
        # run the same model.
        set_configured_model(fixture_config, selected.key)
        fixture_config.worker.timeout_seconds = min(fixture_config.worker.timeout_seconds, 120)
        fixture_config.project.test_command = [
            "python",
            "-c",
            "import pathlib; assert pathlib.Path('.git').exists()",
        ]
        state = await create_run(
            fixture,
            fixture_config,
            plan,
            self.scheduler_mode,
            max_agents=min(max(1, self.concurrency or 3), self.config.worker.max_agents),
            concurrency=self.concurrency,
        )
        peak_workers = 0
        saw_ready_after_work = False
        last_activity = 0.0

        def status_callback(current: object, _sample: object, _running: int) -> None:
            nonlocal peak_workers, saw_ready_after_work, last_activity
            runtime = current
            tasks = getattr(runtime, "tasks", {})
            coding = sum(
                task.status in {TaskStatus.RUNNING, TaskStatus.VALIDATING, TaskStatus.RETRYING}
                for task in tasks.values()
            )
            peak_workers = max(peak_workers, coding)
            if parallel_fixture and tasks["connect-page"].status == TaskStatus.READY:
                saw_ready_after_work = True
            now = time.monotonic()
            if now - last_activity >= 8:
                complete = sum(
                    task.status in {TaskStatus.MERGED, TaskStatus.FAILED} for task in tasks.values()
                )
                reviewing = sum(task.status == TaskStatus.REVIEWING for task in tasks.values())
                self.announce(
                    f"Fixture running · {coding} coding · {reviewing} reviewing · "
                    f"{complete}/{len(tasks)} finished"
                )
                last_activity = now

        self.announce("Starting isolated coding workers, validation, review, and merge…")
        orchestrator = Orchestrator(fixture, fixture_config, state, status_callback=status_callback)
        final = await orchestrator.run()
        self.worker_retries = sum(max(0, task.attempts - 1) for task in final.tasks.values())
        all_merged = all(task.status == TaskStatus.MERGED for task in final.tasks.values())
        self.record(
            "Parallel workers" if parallel_fixture else "Coding worker",
            peak_workers >= self.minimum_peak_workers,
            f"Observed peak of {peak_workers} workers; "
            f"at least {self.minimum_peak_workers} required.",
            layer="Adaptea",
        )
        self.record(
            "Admission controller",
            final.scheduler == self.scheduler_mode,
            f"{self.scheduler_mode.title()} admission target finished at "
            f"{final.target_concurrency}.",
            layer="Adaptea",
        )
        # A step's detail is what the user is shown when the system test fails, so it has to
        # describe what actually happened. Reporting the success sentence for a failed step
        # produced "System test failed: Every task passed the configured validation command."
        unvalidated = [task for task in final.tasks.values() if task.validation_passed is not True]
        self.record(
            "Validation",
            not unvalidated,
            "Every task passed the configured validation command."
            if not unvalidated
            else "; ".join(
                f"{task.spec.id}: "
                + (
                    task.validation_detail
                    or task.failure
                    or f"stopped at {task.status.value} before validation ran"
                )
                for task in unvalidated
            ),
            layer="Adaptea",
        )
        unmerged = [task for task in final.tasks.values() if task.status != TaskStatus.MERGED]
        self.record(
            "Merge",
            all_merged,
            "Every passing branch merged sequentially."
            if all_merged
            else "; ".join(
                f"{task.spec.id}: {task.failure or task.status.value}" for task in unmerged
            ),
            layer="Adaptea",
        )
        if parallel_fixture:
            dependency_ok = (
                saw_ready_after_work
                or final.tasks["connect-page"].status == TaskStatus.MERGED
                and all(
                    final.tasks[dependency].status == TaskStatus.MERGED
                    for dependency in ("page", "styles")
                )
            )
            self.record(
                "Dependency unlock",
                dependency_ok,
                "The dependent task ran only after both prerequisites merged."
                if dependency_ok
                else "The dependent task did not wait for both prerequisites to merge.",
                layer="Adaptea",
            )
        fixture_repo = Path(final.repository)
        worktree_mgr = WorktreeManager(fixture_repo, final.run_id)
        integration = await worktree_mgr.create_integration()
        final_test = await run_validation(
            integration,
            [
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "page = Path('index.html').read_text().lower(); "
                    "assert '<!doctype html>' in page; "
                    "assert 'hello world' in page; "
                    + (
                        "styles = Path('styles.css').read_text().lower(); "
                        "assert 'styles.css' in page; assert 'font-family' in styles"
                        if parallel_fixture
                        else "assert '<h1' in page"
                    )
                ),
            ],
            fixture / ".adaptea" / "smoke-final-validation.log",
        )
        if not self.keep_fixture:
            await worktree_mgr.remove(integration)
        self.record(
            "Final test suite",
            all_merged and final_test.passed,
            "The deterministic final integration assertions passed."
            if final_test.passed
            else f"Final fixture assertions failed ({final_test.detail}); "
            "see smoke-final-validation.log.",
            layer="Adaptea",
        )
        self.announce("System test finished; collecting the final result…")

    async def _prepare_fixture(self, fixture: Path) -> None:
        (fixture / ".gitignore").write_text(".adaptea/\n", encoding="utf-8")
        (fixture / "README.md").write_text(
            "# Adaptea deterministic smoke fixture\n\nWorkers build a tiny Hello World page.\n",
            encoding="utf-8",
        )
        source_opencode = project_opencode_path(self.source_root)
        if source_opencode.is_file():
            shutil.copy2(source_opencode, fixture / source_opencode.name)
        capacity = self.source_root / ".adaptea" / "capacity.json"
        if capacity.is_file():
            target = fixture / ".adaptea" / "capacity.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(capacity, target)
        await git(fixture, "init", "-b", "main")
        await git(fixture, "add", ".")
        await git(
            fixture,
            "-c",
            "user.name=Adaptea Smoke Test",
            "-c",
            "user.email=smoke@localhost",
            "commit",
            "-m",
            "Create deterministic smoke fixture",
        )

    @staticmethod
    def _fixture_plan(*, parallel: bool = False) -> Plan:
        page = TaskSpec(
            id="page",
            title="Create the Hello World page",
            description=(
                "Create index.html with a valid HTML5 document, an English title, "
                "and a visible Hello World heading. Do not create any other files."
            ),
            acceptance_criteria=[
                "index.html starts with an HTML5 doctype",
                "The visible page contains Hello World",
            ],
            risk="low",
        )
        if not parallel:
            return Plan(
                goal="Build a tiny valid Hello World page through the complete Adaptea pipeline.",
                tasks=[page],
            )
        return Plan(
            goal=(
                "Build a tiny valid Hello World web page and verify the complete Adaptea pipeline."
            ),
            tasks=[
                page,
                TaskSpec(
                    id="styles",
                    title="Create minimal page styles",
                    description=(
                        "Create styles.css with a readable system font stack and a centered "
                        "main content layout. Do not modify index.html."
                    ),
                    acceptance_criteria=["styles.css defines font-family and layout rules"],
                    risk="low",
                ),
                TaskSpec(
                    id="connect-page",
                    title="Connect and verify the page",
                    description=(
                        "Add a stylesheet link for styles.css to index.html. Preserve the Hello "
                        "World heading and valid HTML5 structure."
                    ),
                    depends_on=["page", "styles"],
                    acceptance_criteria=[
                        "index.html links styles.css",
                        "The page still visibly contains Hello World",
                    ],
                    risk="low",
                ),
            ],
        )


def _concise_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0][:300]


async def run_mvp_smoke_test(
    root: Path,
    *,
    config: Config | None = None,
    progress: ProgressCallback | None = None,
    activity: ActivityCallback | None = None,
    keep_fixture: bool = False,
    scheduler_mode: Literal["adaptive", "fixed", "naive"] = "adaptive",
    concurrency: int | None = None,
    minimum_peak_workers: int = 1,
) -> SmokeTestResult:
    return await MVPSmokeTest(
        root,
        config,
        progress=progress,
        activity=activity,
        keep_fixture=keep_fixture,
        scheduler_mode=scheduler_mode,
        concurrency=concurrency,
        minimum_peak_workers=minimum_peak_workers,
    ).run()
