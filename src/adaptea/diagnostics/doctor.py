from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adaptea.config import AGENT_CONTEXT_LENGTH, MINIMUM_AGENT_CONTEXT_LENGTH, Config
from adaptea.diagnostics.system import DiagnosticSnapshot, SystemDiagnostics

# The setup order is a dependency order, not a preference: system tools have to exist
# before LM Studio can serve, a model has to be loaded before the project configuration
# means anything, and calibration is optional throughout. Grouping the checks this way is
# what lets the interface say which stage you are on.
SETUP_STAGES: tuple[str, ...] = ("System", "Runtime", "Model", "Project", "Optional")


@dataclass(slots=True)
class Check:
    level: str
    name: str
    detail: str
    stage: str = "System"
    # What the user should do when this check is not passing. `detail` says what is wrong;
    # `remedy` says what to do about it. Several checks used to answer "run adaptea setup",
    # which is useless advice to someone already looking at Setup.
    remedy: str | None = None


def checks_from_snapshot(snapshot: DiagnosticSnapshot, root: Path) -> list[Check]:
    is_lmstudio = snapshot.backend_kind == "lmstudio"
    backend = snapshot.backend_display_name
    checks = [
        Check("PASS", "Operating system", snapshot.operating_system, "System"),
        Check("PASS", "CPU / architecture", snapshot.architecture, "System"),
        Check(
            "PASS" if snapshot.python_ready else "FAIL",
            "Python",
            snapshot.python_version,
            "System",
            None
            if snapshot.python_ready
            else "Adaptea needs Python 3.12 or newer. Install it from python.org, then re-check.",
        ),
        Check(
            "PASS" if snapshot.git_executable else "FAIL",
            "Git",
            snapshot.git_version or "not found",
            "System",
            None
            if snapshot.git_executable
            else (
                "Workers commit inside isolated worktrees, so Git is required. "
                + (
                    "Install it using 'winget install Git.Git', with Chocolatey, or from https://git-scm.com/download/win."
                    if snapshot.operating_system.startswith("Windows")
                    else "On macOS run 'xcode-select --install'; Setup can install it for you when Homebrew is present."
                    if snapshot.operating_system.startswith("Darwin")
                    else "Install Git using your system package manager or from https://git-scm.com."
                )
            ),
        ),
        Check(
            "PASS" if snapshot.opencode_executable else "FAIL",
            "OpenCode",
            snapshot.opencode_version or "not installed",
            "System",
            None
            if snapshot.opencode_executable
            else (
                "OpenCode is the local coding worker. Choose Install OpenCode; "
                + (
                    "Adaptea uses winget, npm, choco, or scoop on Windows, and asks first."
                    if snapshot.operating_system.startswith("Windows")
                    else "Adaptea uses Homebrew or npm, whichever this machine already has, and asks first."
                    if snapshot.operating_system.startswith("Darwin")
                    else "Adaptea uses the official installer or npm, and asks first."
                )
            ),
        ),
        Check(
            "PASS" if snapshot.backend_executable else "WARN",
            "lms CLI"
            if is_lmstudio
            else "Ollama CLI"
            if snapshot.backend_kind == "ollama"
            else "llama-server CLI"
            if snapshot.backend_kind == "llamacpp"
            else "vLLM CLI",
            snapshot.backend_version or "not found",
            "System",
            None
            if snapshot.backend_executable
            else (
                (
                    "Open LM Studio once so it installs its bundled 'lms' CLI. Without it Adaptea "
                    "cannot start the server for you or read live pressure signals."
                    if snapshot.lmstudio_desktop
                    else "Download the official lms CLI (headless) via Safe Repairs, or install "
                    "LM Studio Desktop from https://lmstudio.ai/download."
                )
                if is_lmstudio
                else "Install Ollama from ollama.com/download or ensure its executable is on "
                "PATH. A separately running Ollama server can still be used."
                if snapshot.backend_kind == "ollama"
                else "Install llama.cpp or ensure llama-server is on PATH. A running llama-server "
                "instance can still be used directly."
                if snapshot.backend_kind == "llamacpp"
                else "Install vLLM (e.g. pip install vllm) or ensure vllm is on PATH."
            ),
        ),
    ]
    if snapshot.server_reachable:
        checks.append(Check("PASS", f"{backend} native API", "reachable", "Runtime"))
        ready = [model for model in snapshot.models if model.ready]
        # Something must exist to load before loading can be asked for. Nothing downloaded
        # is an environment problem; nothing loaded is one save away in Models.
        available = [model for model in snapshot.models if not model.ready]
        checks.append(
            Check(
                # Planning cannot make its first inference call without a loaded model.
                # Treating this as a warning let Work stay enabled and fail later with a
                # stack of planner errors instead of presenting the one required action.
                "PASS" if ready else "FAIL",
                "Loaded models" if is_lmstudio else "Available models",
                ", ".join(model.key for model in ready) or "none loaded",
                "Model",
                None
                if ready
                else (
                    "Assign a model in Environment -> Models and save; Adaptea loads it with "
                    f"at least {MINIMUM_AGENT_CONTEXT_LENGTH} context. A run needs one loaded, "
                    "setup does not."
                    if is_lmstudio and available
                    else "No local Ollama model was found. Pull one with 'ollama pull <model>', "
                    "then re-check."
                    if snapshot.backend_kind == "ollama"
                    else "No llama.cpp model was found. Start llama-server with a GGUF model, then re-check."
                    if snapshot.backend_kind == "llamacpp"
                    else "No vLLM model was found. Start vLLM with a model (e.g. 'vllm serve <model>'), then re-check."
                    if snapshot.backend_kind == "vllm"
                    else f"No model is downloaded in {backend}. Download one, then re-check."
                ),
            )
        )
        selected = snapshot.selected_model
        if selected:
            state_label = "available" if selected.ready else "not loaded"
            checks.append(
                Check(
                    "PASS" if selected.ready else "FAIL",
                    "Selected model",
                    f"{selected.key} ({selected.instance_id or state_label})",
                    "Model",
                    None
                    if selected.ready
                    else f"adaptea.toml names {selected.key}. Load it, or assign another model, "
                    "in Environment -> Models.",
                )
            )
            checks.append(
                Check(
                    "PASS",
                    "Concurrency management",
                    (
                        f"Adaptea manages demand within the backend ceiling ({selected.parallel})"
                        if selected.parallel is not None
                        else "Adaptea will use the safe ceiling reported by the backend"
                    ),
                    "Optional",
                )
            )
            context_length = selected.context_length or selected.max_context_length
            checks.append(
                Check(
                    (
                        "PASS"
                        if context_length and context_length >= MINIMUM_AGENT_CONTEXT_LENGTH
                        else "WARN"
                    ),
                    "Context length",
                    (
                        str(context_length)
                        if context_length and context_length >= MINIMUM_AGENT_CONTEXT_LENGTH
                        else (
                            f"{context_length} is too small for OpenCode's planner prompt; "
                            f"reload the model with at least {MINIMUM_AGENT_CONTEXT_LENGTH} "
                            f"({AGENT_CONTEXT_LENGTH} recommended)"
                            if context_length
                            else f"unknown; {AGENT_CONTEXT_LENGTH} recommended for OpenCode agents"
                        )
                    ),
                    "Model",
                    None
                    if context_length and context_length >= MINIMUM_AGENT_CONTEXT_LENGTH
                    else (
                        "In LM Studio open this model's load settings and raise Context Length "
                        f"to at least {MINIMUM_AGENT_CONTEXT_LENGTH}, then reload it."
                        if is_lmstudio
                        else "Create or select an Ollama model configured with sufficient "
                        f"num_ctx; at least {MINIMUM_AGENT_CONTEXT_LENGTH} is required."
                    ),
                )
            )
            normalized_format = selected.format.lower() if selected.format else None
            runtime_names = {
                "gguf": "GGUF / llama.cpp-compatible",
                "mlx": "MLX / Apple Silicon optimized",
            }
            runtime = (
                runtime_names.get(normalized_format, selected.format or "unknown")
                if normalized_format
                else "unknown"
            )
            level = "PASS" if normalized_format in {"gguf", "mlx"} else "WARN"
            if level == "WARN":
                runtime += "; useful continuous parallel prediction support is not confirmed"
            checks.append(
                Check(
                    level,
                    "Runtime / format",
                    runtime,
                    "Model",
                    None
                    if level == "PASS"
                    else (
                        "LM Studio documents continuous parallel requests for its GGUF engine. "
                        "On another runtime Adaptea cannot promise useful concurrency, so prefer "
                        "a GGUF build of this model if you want more than one worker."
                        if is_lmstudio
                        else "Ollama did not report a recognized local model format. Capacity "
                        "will remain conservative."
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    "FAIL",
                    "Selected model",
                    "no model is selected for this project",
                    "Model",
                    "Assign a model in Environment -> Models. A run needs one; finishing setup "
                    "does not.",
                )
            )
        checks.append(
            Check(
                "PASS" if snapshot.openai_api_usable else "FAIL",
                "OpenAI-compatible API",
                "/v1/models usable" if snapshot.openai_api_usable else "unavailable",
                "Runtime",
                None
                if snapshot.openai_api_usable
                else f"OpenCode talks to {backend} through /v1. The native API is up but that "
                "endpoint is not answering; confirm the configured endpoint.",
            )
        )
    else:
        checks.append(
            Check(
                "FAIL",
                f"{backend} server",
                snapshot.server_error or "not reachable",
                "Runtime",
                (
                    "Open LM Studio and start its local server (Developer -> Start Server). "
                    "Apply safe repairs can start it for you once the lms CLI is available."
                    if is_lmstudio
                    else "Start Ollama, usually with the desktop app or 'ollama serve', and "
                    "confirm the configured URL is reachable."
                    if snapshot.backend_kind == "ollama"
                    else "Start llama.cpp server (e.g. 'llama-server -m <model.gguf> --port 8080') "
                    "and confirm the configured endpoint is reachable."
                    if snapshot.backend_kind == "llamacpp"
                    else "Start vLLM server (e.g. 'vllm serve <model> --port 8000') and confirm "
                    "the configured endpoint is reachable."
                ),
            )
        )
    checks.extend(
        [
            Check(
                "PASS" if snapshot.opencode_configured else "FAIL",
                f"OpenCode → {backend}",
                "project-local provider configured"
                if snapshot.opencode_configured
                else "not configured for this project",
                "Project",
                None
                if snapshot.opencode_configured
                else "Choose Apply safe repairs. Adaptea writes the provider into this "
                "project's opencode.json only; your global OpenCode configuration is not "
                "touched.",
            ),
            Check(
                "PASS" if snapshot.adaptea_config else "FAIL",
                "Adaptea configuration",
                "adaptea.toml present" if snapshot.adaptea_config else "adaptea.toml missing",
                "Project",
                None
                if snapshot.adaptea_config
                else "Choose Apply safe repairs to write adaptea.toml into this project. It "
                f"records the {backend} endpoint, the selected model, and the worker limits.",
            ),
            Check(
                "PASS" if snapshot.git_repository else "WARN",
                "Git repository",
                str(root) if snapshot.git_repository else f"{root} is not a Git repository",
                "Project",
                None
                if snapshot.git_repository
                else "Every worker edits an isolated Git worktree, so this folder has to be a "
                "repository. Adaptea can initialize it and make the first commit for you.",
            ),
            Check(
                "PASS" if snapshot.telemetry_usable else "WARN",
                f"Live {backend} telemetry",
                (
                    "lms ps --json pressure signals usable"
                    if snapshot.telemetry_source == "lms_ps"
                    else (
                        "native API model telemetry usable; active workers and request stats "
                        "supplement it"
                    )
                )
                if snapshot.telemetry_usable
                else "unavailable; controller will use a conservative fallback",
                "Optional",
                None
                if snapshot.telemetry_usable
                else (
                    "Runs still work. With the lms CLI available the controller reads live "
                    "queue pressure instead of inferring it, so its targets react sooner."
                    if is_lmstudio
                    else "Ollama does not expose queue pressure through its public API. "
                    "Adaptive runs stay at the safe single-worker fallback."
                ),
            ),
            Check(
                "PASS" if snapshot.capacity_profile else "WARN",
                "Capacity profile",
                str(root / ".adaptea" / "capacity.json")
                if snapshot.capacity_profile
                else "not measured yet",
                "Optional",
                None
                if snapshot.capacity_profile
                else "Optional. Run Quick Calibration once after changing the model or the "
                "machine; until then Adaptea starts from a conservative target.",
            ),
        ]
    )
    return sort_by_stage(checks)


def sort_by_stage(checks: list[Check]) -> list[Check]:
    """Order checks by setup stage, stably within each stage."""
    order = {stage: index for index, stage in enumerate(SETUP_STAGES)}
    return sorted(checks, key=lambda check: order.get(check.stage, len(SETUP_STAGES)))


async def run_doctor(root: Path, config: Config) -> list[Check]:
    snapshot = await SystemDiagnostics(root, config).collect()
    checks = checks_from_snapshot(snapshot, root)
    checks.append(
        Check(
            "PASS",
            "Fleet support",
            "enabled" if config.fleet.enabled else "available; legacy single-model mode active",
            "Optional",
        )
    )
    if not snapshot.server_reachable:
        return sort_by_stage(checks)
    try:
        from adaptea.fleet.calibration import profile_is_stale
        from adaptea.fleet.discovery import discover_fleet

        inventory = await discover_fleet(root, config)
    except Exception as exc:
        checks.append(
            Check(
                "WARN",
                "Fleet discovery",
                str(exc).splitlines()[0],
                "Optional",
                "Single-model runs are unaffected. Fleet routing stays off until discovery "
                "succeeds.",
            )
        )
        return sort_by_stage(checks)
    checks.extend(
        [
            Check("PASS", "Downloaded models", str(len(inventory.downloaded)), "Optional"),
            Check("PASS", "Loaded instances", str(len(inventory.instances)), "Optional"),
            Check(
                "PASS",
                "Duplicate model instances",
                str(
                    sum(
                        max(0, count - 1)
                        for count in {
                            model: sum(
                                instance.model_key == model for instance in inventory.instances
                            )
                            for model in {item.model_key for item in inventory.instances}
                        }.values()
                    )
                ),
                "Optional",
            ),
        ]
    )
    if config.fleet.enabled:
        # What the user asked for, separately from what LM Studio currently holds.
        planner_assigned = any("planner" in item.roles for item in config.fleet.models)
        reviewer_assigned = any(
            "reviewer" in item.roles or "planner" in item.roles for item in config.fleet.models
        )
        strong_assigned = any(item.tier == "strong" for item in config.fleet.models)
        planner = next((item for item in inventory.instances if "planner" in item.roles), None)
        reviewer = next((item for item in inventory.instances if "reviewer" in item.roles), planner)
        fast = [item for item in inventory.instances if item.capability_tier == "fast"]
        strong = [item for item in inventory.instances if item.capability_tier == "strong"]
        checks.extend(
            [
                Check(
                    "PASS" if planner else "WARN" if planner_assigned else "FAIL",
                    "Planner model",
                    planner.instance_id
                    if planner
                    else "assigned but not loaded"
                    if planner_assigned
                    else "no model has the planner role",
                    "Model",
                    None
                    if planner
                    else "Open Environment -> Models and save; LM Studio loads the assigned "
                    "planner. A run needs it loaded, finishing setup does not."
                    if planner_assigned
                    else "Open Environment -> Models and give one model the planner role.",
                ),
                Check(
                    "PASS" if reviewer else "WARN" if reviewer_assigned else "FAIL",
                    "Reviewer model",
                    (
                        reviewer.instance_id
                        + (" (planner fallback)" if reviewer is planner else "")
                        if reviewer
                        else "assigned but not loaded"
                        if reviewer_assigned
                        else "no model has the reviewer or planner role"
                    ),
                    "Model",
                    None
                    if reviewer
                    else "Save in Environment -> Models so the assigned reviewer loads."
                    if reviewer_assigned
                    else "Assign the reviewer role in Environment -> Models. Without it no "
                    "validated diff can be approved for merge.",
                ),
                Check(
                    "PASS" if fast else "WARN",
                    "Fast worker pool",
                    f"{len(fast)} loaded instance(s)" if fast else "not configured/loaded",
                    "Optional",
                    None
                    if fast
                    else "Routing falls back to the strong pool for every task, which is "
                    "correct but slower on simple work.",
                ),
                Check(
                    "PASS" if strong else "WARN" if strong_assigned else "FAIL",
                    "Strong worker pool",
                    f"{len(strong)} loaded instance(s)"
                    if strong
                    else "assigned but not loaded"
                    if strong_assigned
                    else "not configured",
                    "Model",
                    None
                    if strong
                    else "Save in Environment -> Models so LM Studio loads the assigned strong "
                    "model. A run needs it loaded, finishing setup does not."
                    if strong_assigned
                    else "Fleet mode is on but no strong model is assigned. Assign one in "
                    "Environment -> Models, or turn fleet mode off to use the single model.",
                ),
            ]
        )
        profile_path = root / ".adaptea" / "fleet.json"
        if profile_path.is_file():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                stale = profile_is_stale(profile, inventory)
            except (OSError, ValueError):
                stale = True
            checks.append(
                Check(
                    "WARN" if stale else "PASS",
                    "Fleet profile",
                    "model/load configuration changed; recalibration recommended"
                    if stale
                    else "calibrated and current",
                    "Optional",
                    "Run Fleet calibration in Environment -> Calibration to measure the new "
                    "topology."
                    if stale
                    else None,
                )
            )
        else:
            checks.append(
                Check(
                    "WARN",
                    "Fleet profile",
                    "not measured yet",
                    "Optional",
                    "Optional. Run Fleet calibration in Environment -> Calibration after "
                    "changing the strong/fast topology.",
                )
            )
    return sort_by_stage(checks)
