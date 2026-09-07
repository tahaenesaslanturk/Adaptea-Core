from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from adaptea.calibration.runner import CalibrationRunner
from adaptea.config import Config, FleetConfig, executable_stem, load_config
from adaptea.diagnostics.doctor import Check, run_doctor
from adaptea.inference import (
    InferenceBackendError,
    configured_model,
    create_inference_backend,
)
from adaptea.lmstudio.client import LMStudioClient, LMStudioError, select_loaded_model
from adaptea.lmstudio.models import LMModel
from adaptea.models import Plan, RunState, TelemetrySample, utc_now
from adaptea.planner.opencode import OpenCodePlanner
from adaptea.projects import ensure_project_configuration
from adaptea.runtime.controller import Orchestrator, create_run, prepare_resume
from adaptea.runtime.state import StateStore
from adaptea.security.approvals import ApprovalDecision, ApprovalRequest
from adaptea.workers.activity import Activity

RunMode = Literal["adaptive", "fixed", "naive"]


def model_download_guidance(config: Config) -> dict[str, object]:
    """The exact commands that add a model, for the backend actually selected.

    Adaptea needs the backend's CLI, not its desktop application, so the app should say
    how to download a model from that CLI rather than sending the user to a GUI it does
    not otherwise require.
    """
    from adaptea.config import AGENT_CONTEXT_LENGTH

    if config.inference.backend == "ollama":
        executable = executable_stem(config.ollama.executable)
        return {
            "backend": "ollama",
            "executable": executable,
            "browse_command": f"{executable} search",
            "download_command": f"{executable} pull <model>",
            "examples": [
                f"{executable} pull qwen2.5-coder:14b",
                f"{executable} pull devstral:24b",
            ],
            "list_command": f"{executable} list",
            "minimum_context": AGENT_CONTEXT_LENGTH,
        }
    if config.inference.backend == "llamacpp":
        return {
            "backend": "llamacpp",
            "executable": "hf",
            "browse_command": "hf download --help",
            "download_command": 'hf download <owner/repository> --include "*<quantization>*.gguf"',
            "examples": [
                'hf download bartowski/Qwen2.5-Coder-14B-Instruct-GGUF --include "*Q4_K_M*.gguf"',
                'hf download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF --include "*Q4_K_M*.gguf"',
            ],
            "list_command": "hf cache list",
            "minimum_context": AGENT_CONTEXT_LENGTH,
        }
    if config.inference.backend == "vllm":
        return {
            "backend": "vllm",
            "executable": "hf",
            "browse_command": "hf download --help",
            "download_command": "hf download <owner/repository>",
            "examples": [
                "hf download Qwen/Qwen2.5-Coder-14B-Instruct",
                "hf download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            ],
            "list_command": "hf cache list",
            "minimum_context": AGENT_CONTEXT_LENGTH,
        }
    executable = executable_stem(config.lmstudio.lms_executable)
    # Adaptea downloads into LM Studio's models folder itself rather than through
    # `lms get`, because `lms get` hands the transfer to LM Studio's service and nothing
    # Adaptea can do then stops it. That makes the Hub repository, not LM Studio's own
    # catalogue name, the thing to type — so say so.
    return {
        "backend": "lmstudio",
        "executable": "huggingface.co",
        "browse_command": "https://huggingface.co/models?library=gguf",
        "download_command": "<owner/repository>:<quantization>",
        "examples": [
            "lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M",
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
        ],
        "list_command": f"{executable} ls",
        "minimum_context": AGENT_CONTEXT_LENGTH,
    }


class ApplicationServices:
    """The shared application layer used by both CLI and Textual frontends."""

    def __init__(self, *, project_scoped_config: bool = False) -> None:
        self.active_orchestrators: dict[str, Orchestrator] = {}
        self.project_scoped_config = project_scoped_config

    def load_config(self, root: Path) -> Config:
        return load_config(
            root,
            explicit=root / "adaptea.toml" if self.project_scoped_config else None,
        )

    async def diagnose(self, root: Path) -> list[Check]:
        return await run_doctor(root, self.load_config(root))

    async def available_models(self, root: Path) -> tuple[list[LMModel], str | None]:
        config = self.load_config(root)
        async with create_inference_backend(config, timeout=10) as client:
            models = await client.models()
        return models, configured_model(config)

    async def lmstudio_health(self, root: Path) -> dict[str, object]:
        """Return a fast backend heartbeat; the legacy method name remains API-compatible."""
        config = self.load_config(root)
        planner = config.fleet.planner() if config.fleet.enabled else None
        configured = planner.model if planner else configured_model(config)
        try:
            async with create_inference_backend(config, timeout=2) as client:
                models = await client.models()
        except (InferenceBackendError, ValueError) as exc:
            return {
                "reachable": False,
                "backend": config.inference.backend,
                "error": str(exc),
                "configured_model": configured,
                "loaded_models": [],
                "selected_model": None,
            }
        selected = select_loaded_model(models, configured)
        return {
            "reachable": True,
            "backend": config.inference.backend,
            "error": None,
            "configured_model": configured,
            "loaded_models": [model.key for model in models if model.ready and model.type == "llm"],
            "selected_model": selected.key if selected else None,
        }

    async def fleet_status(self, root: Path) -> dict[str, object]:
        from adaptea.calibration.state import calibration_state
        from adaptea.fleet.calibration import profile_is_stale
        from adaptea.fleet.combinations import list_combinations
        from adaptea.fleet.discovery import discover_fleet
        from adaptea.fleet.lifecycle import available_memory_bytes
        from adaptea.fleet.recommendation import recommend_fleet

        config = self.load_config(root)
        inventory = await discover_fleet(root, config)
        available_memory = available_memory_bytes()

        # A status refresh is on the app's startup path. It used to run ``lms load
        # --estimate-only`` for every downloaded model, with a two-minute timeout per
        # model, even though the current UI does not consume the recommendation. The real
        # memory guard still runs for the one model the user actually saves and loads.
        recommendation = recommend_fleet(
            inventory.downloaded,
            available_memory_bytes=available_memory,
            estimated_memory_bytes={},
            max_loaded_instances=config.fleet.max_loaded_instances,
            memory_headroom_fraction=config.fleet.memory_headroom_fraction,
        )
        profile_path = root / ".adaptea" / "fleet.json"
        profile: dict[str, object] = {}
        if profile_path.is_file():
            value = json.loads(profile_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                profile = value
        configured_models = [item.model_dump(mode="json") for item in config.fleet.models]
        legacy_model = configured_model(config)
        if not configured_models and legacy_model:
            loaded = next(
                (
                    item
                    for item in inventory.instances
                    if item.model_key == legacy_model or item.instance_id == legacy_model
                ),
                None,
            )
            configured_models = [
                {
                    "name": "primary",
                    "model": loaded.model_key if loaded else legacy_model,
                    "tier": "strong",
                    "roles": ["planner", "worker", "reviewer"],
                    "instances": 1,
                    "context_length": loaded.context_length if loaded else None,
                    "parallel_limit": None,
                }
            ]
        combinations = list_combinations(root)
        return {
            "enabled": config.fleet.enabled,
            "model_download": model_download_guidance(config),
            "configured_models": configured_models,
            "available_memory_bytes": available_memory,
            "memory_budget_bytes": recommendation.memory_budget_bytes,
            "recommendation": recommendation.model_dump(mode="json"),
            "planner_model": next(
                (item.instance_id for item in inventory.instances if "planner" in item.roles),
                None,
            ),
            "reviewer_model": next(
                (item.instance_id for item in inventory.instances if "reviewer" in item.roles),
                next(
                    (item.instance_id for item in inventory.instances if "planner" in item.roles),
                    None,
                ),
            ),
            "downloaded_models": [item.model_dump(mode="json") for item in inventory.downloaded],
            "loaded_instances": [item.model_dump(mode="json") for item in inventory.instances],
            "fast_pool": [
                item.model_dump(mode="json")
                for item in inventory.instances
                if item.capability_tier == "fast"
            ],
            "strong_pool": [
                item.model_dump(mode="json")
                for item in inventory.instances
                if item.capability_tier == "strong"
            ],
            "topology": profile.get("recommended_topology", {}),
            "fleet_profile": profile,
            "calibration_stale": bool(profile) and profile_is_stale(profile, inventory),
            "calibration": calibration_state(root, config, inventory).as_dict(),
            "combinations": combinations["combinations"],
            "active_combination_id": combinations["active_id"],
        }

    async def activate_configured_fleet(self, root: Path) -> tuple[bool, str | None]:
        """Load newly configured instances when conservative memory checks allow it."""
        from adaptea.fleet.discovery import discover_fleet
        from adaptea.fleet.lifecycle import ensure_configured_instances

        try:
            config = self.load_config(root)
            if config.inference.backend != "lmstudio":
                return False, "Fleet lifecycle is currently supported only by LM Studio."
            inventory = await discover_fleet(root, config)
            changed = await ensure_configured_instances(root, config, inventory)
        except (LMStudioError, OSError, RuntimeError, ValueError) as exc:
            return False, str(exc)
        return changed, None

    async def load_model(
        self, root: Path, model_key: str, context_length: int | None = None
    ) -> dict[str, object]:
        """Load one downloaded model, so the app is where a model is turned on and off.

        Model availability was only reachable by saving a whole fleet, which made trying a
        second model a configuration edit rather than a switch. Memory safety is unchanged:
        the lifecycle manager still refuses a load its estimate cannot justify. What it does
        not do here is apply the configured instance ceiling, which describes what Adaptea
        may load unattended rather than what the user just asked for.
        """
        from adaptea.config import AGENT_CONTEXT_LENGTH
        from adaptea.fleet.discovery import discover_fleet
        from adaptea.fleet.lifecycle import InstanceLifecycleManager

        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            raise RuntimeError("Loading a single model is currently supported only by LM Studio.")
        inventory = await discover_fleet(root, config)
        existing = [item for item in inventory.instances if item.model_key == model_key]
        identifier = f"{model_key.split('/')[-1]}-{len(existing) + 1}"
        manager = InstanceLifecycleManager(config, inventory)
        instance = await manager.load(
            model_key,
            identifier,
            context_length=context_length or AGENT_CONTEXT_LENGTH,
            requested_by_user=True,
        )
        return {"instance_id": instance.instance_id, "model_key": instance.model_key}

    async def unload_model(self, root: Path, instance_id: str) -> str:
        """Unload one live instance by its id, refusing while a worker is using it."""
        from adaptea.fleet.discovery import discover_fleet
        from adaptea.fleet.lifecycle import InstanceLifecycleManager

        if self.active_orchestrators:
            raise RuntimeError("A run is active. Stop it before unloading a model.")
        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            raise RuntimeError("Unloading a single model is currently supported only by LM Studio.")
        inventory = await discover_fleet(root, config)
        instance = inventory.instance(instance_id)
        if instance is None:
            raise RuntimeError(f"No loaded LM Studio instance named {instance_id}.")
        await InstanceLifecycleManager(config, inventory).unload(instance)
        return instance_id

    async def unload_unconfigured_fleet_models(self, root: Path, fleet: FleetConfig) -> list[str]:
        """Unload live LM Studio instances explicitly removed from a saved fleet."""
        if self.active_orchestrators:
            raise RuntimeError(
                "A run is active. Stop it before removing a loaded model from the fleet."
            )
        from adaptea.fleet.discovery import discover_fleet

        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            raise RuntimeError("Fleet lifecycle is currently supported only by LM Studio.")
        inventory = await discover_fleet(root, config)
        desired = {item.model for item in fleet.models}
        previously_selected = (
            {item.model for item in config.fleet.models}
            if config.fleet.enabled and config.fleet.models
            else {model}
            if (model := configured_model(config))
            else set()
        )
        removed = [
            item
            for item in inventory.instances
            if item.model_key in previously_selected and item.model_key not in desired
        ]
        if not removed:
            return []
        async with LMStudioClient(
            config.lmstudio.base_url, config.lmstudio.api_token, timeout=120
        ) as client:
            for instance in removed:
                if instance.running_workers:
                    raise RuntimeError(
                        f"Refusing to unload {instance.instance_id}; it has an active worker."
                    )
                await client.unload(instance.instance_id)
        return [item.instance_id for item in removed]

    async def unload_selected_models(self, root: Path) -> list[str]:
        """Unload only the models selected by this project.

        LM Studio is machine-wide, so resetting one project must not unload unrelated
        models that another project or the user loaded independently.
        """
        if self.active_orchestrators:
            raise RuntimeError("A run is active. Stop it before resetting the model fleet.")
        from adaptea.fleet.discovery import discover_fleet

        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            return []
        selected = (
            {item.model for item in config.fleet.models}
            if config.fleet.enabled and config.fleet.models
            else {model}
            if (model := configured_model(config))
            else set()
        )
        if not selected:
            return []
        inventory = await discover_fleet(root, config)
        instances = [item for item in inventory.instances if item.model_key in selected]
        async with LMStudioClient(
            config.lmstudio.base_url, config.lmstudio.api_token, timeout=120
        ) as client:
            for instance in instances:
                if instance.running_workers:
                    raise RuntimeError(
                        f"Refusing to unload {instance.instance_id}; it has an active worker."
                    )
                await client.unload(instance.instance_id)
        return [item.instance_id for item in instances]

    async def unload_combination_models(self, root: Path, combination_id: str) -> list[str]:
        """Unload live LM Studio instances for models belonging to a specific combination."""
        if self.active_orchestrators:
            raise RuntimeError("A run is active. Stop it before unloading models.")
        from adaptea.fleet.combinations import combination_inputs
        from adaptea.fleet.discovery import discover_fleet

        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            return []
        try:
            target_fleet, _ = combination_inputs(root, combination_id)
        except KeyError:
            return []
        target_models = {item.model for item in target_fleet.models}
        if not target_models:
            return []
        inventory = await discover_fleet(root, config)
        instances = [item for item in inventory.instances if item.model_key in target_models]
        if not instances:
            return []
        async with LMStudioClient(
            config.lmstudio.base_url, config.lmstudio.api_token, timeout=120
        ) as client:
            for instance in instances:
                if instance.running_workers:
                    raise RuntimeError(
                        f"Refusing to unload {instance.instance_id}; it has an active worker."
                    )
                await client.unload(instance.instance_id)
        return [item.instance_id for item in instances]

    async def stop_models(self, root: Path, *, force: bool = False) -> list[str]:
        """Stop and unload loaded models for the active backend to free memory and VRAM."""
        config = self.load_config(root)
        unloaded: list[str] = []

        if config.inference.backend == "lmstudio":
            from adaptea.fleet.discovery import discover_fleet

            selected = (
                {item.model for item in config.fleet.models}
                if config.fleet.enabled and config.fleet.models
                else {model}
                if (model := configured_model(config))
                else set()
            )
            try:
                inventory = await discover_fleet(root, config)
            except Exception:
                inventory = None

            if inventory and inventory.instances:
                instances = (
                    [item for item in inventory.instances if item.model_key in selected]
                    if selected
                    else list(inventory.instances)
                )
                if not instances and inventory.instances:
                    instances = list(inventory.instances)
                async with LMStudioClient(
                    config.lmstudio.base_url, config.lmstudio.api_token, timeout=30
                ) as client:
                    for instance in instances:
                        if not force and instance.running_workers and self.active_orchestrators:
                            continue
                        try:
                            await client.unload(instance.instance_id)
                            unloaded.append(instance.instance_id)
                        except Exception:
                            pass

        elif config.inference.backend == "ollama":
            from adaptea.ollama.client import OllamaClient

            model = configured_model(config)
            async with OllamaClient(
                config.ollama.base_url, config.ollama.api_token, timeout=30
            ) as client:
                models_to_unload = [model] if model else []
                if not models_to_unload:
                    try:
                        all_models = await client.models()
                        models_to_unload = [m.key for m in all_models if m.ready]
                    except Exception:
                        pass
                for m in models_to_unload:
                    if m:
                        try:
                            await client.unload(m)
                            unloaded.append(m)
                        except Exception:
                            pass

        return unloaded

    async def ensure_ready_models(self, root: Path) -> bool:
        """Ensure required inference models or fleet instances are loaded and ready in LM Studio."""
        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            return False

        from adaptea.fleet.discovery import discover_fleet

        try:
            inventory = await discover_fleet(root, config)
        except Exception:
            return False

        if config.fleet.enabled and config.fleet.models:
            from adaptea.fleet.lifecycle import ensure_configured_instances

            try:
                return await ensure_configured_instances(root, config, inventory)
            except Exception:
                return False
        else:
            model_key = configured_model(config)
            if not model_key:
                return False
            is_loaded = any(
                item.model_key == model_key or item.instance_id == model_key
                for item in inventory.instances
            )
            if not is_loaded:
                import re

                from adaptea.fleet.lifecycle import InstanceLifecycleManager

                manager = InstanceLifecycleManager(config, inventory)
                model_name = model_key.rsplit("/", 1)[-1]
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
                identifier = f"adaptea-{safe_name or 'model'}-1"
                try:
                    await manager.load(model_key, identifier, requested_by_user=True)
                    return True
                except Exception:
                    return False
        return False

    async def plan(
        self,
        root: Path,
        goal: str,
        on_activity: Callable[[Activity], None] | None = None,
        previous: Plan | None = None,
    ) -> Plan:
        ensure_project_configuration(root)
        await self.ensure_ready_models(root)
        config = self.load_config(root)
        async with create_inference_backend(config, timeout=10) as client:
            models = await client.models()
            planner = config.fleet.planner() if config.fleet.enabled else None
            configured = planner.model if planner is not None else configured_model(config)
            selected = select_loaded_model(models, configured)
        if not selected:
            raise RuntimeError(
                "The project's planner model is not ready. Open Environment → Models, "
                "choose a primary model, and press Save; Adaptea will load and measure it."
            )
        planner_destination = selected.destination
        return await OpenCodePlanner(root, config, planner_destination, on_activity).plan(
            goal, previous=previous
        )

    def save_plan(self, root: Path, plan: Plan) -> Path:
        plan_id = f"plan-{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
        directory = root / ".adaptea" / "runs" / plan_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "plan.json"
        path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    async def create_run(
        self,
        root: Path,
        plan: Plan,
        mode: RunMode,
        max_agents: int,
        fixed_concurrency: int | None = None,
    ) -> RunState:
        ensure_project_configuration(root)
        await self.ensure_ready_models(root)
        return await create_run(
            root,
            self.load_config(root),
            plan,
            mode,
            max_agents,
            fixed_concurrency,
        )

    async def execute_run(
        self,
        root: Path,
        state: RunState,
        callback: Callable[[RunState, TelemetrySample | None, int], None] | None = None,
        activity: Callable[[str, dict[str, str]], None] | None = None,
        approval: Callable[[ApprovalRequest], None] | None = None,
    ) -> RunState:
        await self.ensure_ready_models(root)
        orchestrator = Orchestrator(
            root,
            self.load_config(root),
            state,
            status_callback=callback,
            activity_callback=activity,
            approval_callback=approval,
        )
        self.active_orchestrators[state.run_id] = orchestrator
        try:
            return await orchestrator.run()
        finally:
            self.active_orchestrators.pop(state.run_id, None)

    def set_admission(self, run_id: str, *, enabled: bool) -> bool:
        orchestrator = self.active_orchestrators.get(run_id)
        if orchestrator is None:
            return False
        if enabled:
            orchestrator.resume_admitting()
        else:
            orchestrator.stop_admitting()
        return True

    def pending_command_approvals(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Questions still waiting for an answer, across every live run by default."""
        orchestrators = (
            [self.active_orchestrators[run_id]]
            if run_id and run_id in self.active_orchestrators
            else []
            if run_id
            else list(self.active_orchestrators.values())
        )
        return [
            request.as_dict()
            for orchestrator in orchestrators
            for request in orchestrator.pending_command_approvals()
        ]

    def resolve_command_approval(
        self, request_id: str, decision: ApprovalDecision, run_id: str | None = None
    ) -> dict[str, Any]:
        """Answer one approval request, whichever live run raised it."""
        candidates = (
            [self.active_orchestrators[run_id]]
            if run_id and run_id in self.active_orchestrators
            else list(self.active_orchestrators.values())
        )
        for orchestrator in candidates:
            try:
                outcome = orchestrator.resolve_command_approval(request_id, decision)
            except KeyError:
                continue
            return {
                "request": outcome.request.as_dict(),
                "decision": outcome.decision,
                "approved": outcome.approved,
                "remembered": outcome.remembered,
                "config_path": outcome.config_path,
            }
        raise ValueError(
            "That command approval is no longer waiting for an answer. The run that "
            "asked for it has finished or it was already answered."
        )

    def abort_run(self, run_id: str) -> bool:
        orchestrator = self.active_orchestrators.get(run_id)
        if orchestrator is None:
            return False
        orchestrator.abort()
        return True

    def abort_all(self) -> int:
        aborted = 0
        for orchestrator in list(self.active_orchestrators.values()):
            orchestrator.abort()
            aborted += 1
        return aborted

    async def resume_run(
        self,
        root: Path,
        run_id: str,
        callback: Callable[[RunState, TelemetrySample | None, int], None] | None = None,
        activity: Callable[[str, dict[str, str]], None] | None = None,
        approval: Callable[[ApprovalRequest], None] | None = None,
        *,
        retry_failed: bool = False,
    ) -> RunState:
        store = StateStore(root / ".adaptea" / "runs" / run_id)
        if not store.path.exists():
            raise ValueError(f"Run not found: {run_id}")
        state = prepare_resume(store.load(), store.run_dir, retry_failed=retry_failed)
        store.save(state)
        return await self.execute_run(root, state, callback, activity, approval)

    async def calibrate(
        self,
        root: Path,
        *,
        quick: bool,
        max_agents: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            raise RuntimeError(
                "Direct capacity calibration currently requires LM Studio pressure telemetry. "
                "Ollama runs use the safe single-worker fallback."
            )
        if progress:
            progress("Connecting to LM Studio…")
        async with create_inference_backend(config, timeout=600) as client:
            result = await CalibrationRunner(root, config, client, progress).run(
                quick=quick, max_agents=max_agents
            )
        return result

    async def calibrate_fleet(
        self,
        root: Path,
        *,
        repetitions: int = 3,
        agent_validation: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        from adaptea.config import FleetModelConfig
        from adaptea.fleet.calibration import FleetCalibrationRunner
        from adaptea.fleet.discovery import discover_fleet
        from adaptea.fleet.models import TopologyCandidate
        from adaptea.smoke import run_mvp_smoke_test

        config = self.load_config(root)
        if config.inference.backend != "lmstudio":
            raise RuntimeError("Fleet calibration is currently supported only by LM Studio.")
        if not config.fleet.enabled:
            raise RuntimeError("Fleet is not configured. Configure a fleet before calibration.")
        if progress:
            progress("Discovering the configured LM Studio fleet…")
        inventory = await discover_fleet(root, config)

        async def validate_agent(candidate: TopologyCandidate) -> tuple[float, float, int]:
            candidate_config = config.model_copy(deep=True)
            planner = next(
                (item for item in candidate.instances if item.tier == "strong"),
                candidate.instances[0],
            )
            candidate_config.lmstudio.model = planner.model
            candidate_config.fleet.topology = "explicit"
            candidate_config.fleet.max_loaded_instances = max(
                candidate_config.fleet.max_loaded_instances, candidate.total_instances
            )
            candidate_config.fleet.models = [
                FleetModelConfig(
                    name=f"cal-{item.tier}-{index}",
                    model=item.model,
                    tier=item.tier,
                    roles=(
                        ["planner", "worker", "reviewer"]
                        if item.model == planner.model
                        else ["worker"]
                    ),
                    instances=item.count,
                    parallel_limit=item.workers_per_instance,
                )
                for index, item in enumerate(candidate.instances, 1)
            ]
            result = await run_mvp_smoke_test(
                root,
                config=candidate_config,
                scheduler_mode="fixed",
                concurrency=candidate.total_workers,
                minimum_peak_workers=min(2, candidate.total_workers),
            )
            return (
                1.0 if result.success else 0.0,
                result.duration_seconds,
                result.worker_retries,
            )

        async with create_inference_backend(config, timeout=600) as client:
            runner = FleetCalibrationRunner(
                root,
                config,
                inventory,
                client,
                agent_validator=validate_agent if agent_validation else None,
            )
            result = await runner.run(repetitions=repetitions)
        if progress:
            progress(f"Fleet calibration complete: {result}")
        return result


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows
