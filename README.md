<p align="center">
  <img src="assets/adaptea-logo.svg" alt="Adaptea Logo" width="220">
</p>

# Adaptea Core (`adaptea-core`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/website-adaptea.dev-blue.svg)](https://adaptea.dev)

**Adaptive admission controller and multi-agent orchestrator for local coding models.**

Adaptea Core plans software tasks, decomposes them into parallel dependency graphs, executes subtasks across local coding agents ([OpenCode](https://opencode.ai)), and dynamically adapts concurrency and model routing based on live hardware capacity and inference pressure.

> [!NOTE]
> **Adaptea Core** is the open-source Python engine, CLI, and terminal interface (`adaptea`). It provides the core algorithms, Git worktree isolation, deterministic test validation, dynamic admission control, and multi-model fleet orchestration.
>
> For the graphical desktop application (macOS DMG / Windows installer), visit the [Adaptea Desktop repository](https://github.com/tahaenesaslanturk/Adaptea) or [adaptea.dev](https://adaptea.dev).

---

## Features

- **Adaptive Admission Control**: Dynamically regulates worker concurrency based on real-time inference latency and queue pressure, preventing token thrashing and memory exhaustion.
- **Multi-Backend Support**: Native drivers for LM Studio, Ollama, llama.cpp (`llama-server`), and vLLM without intermediate proxy layers.
- **Git Worktree Isolation**: Spawns each subtask in an isolated Git worktree branch, ensuring parallel execution never pollutes the main working directory.
- **Deterministic Validation**: Automatically detects and executes test suites (`pytest`, `npm test`, or custom verification commands), merging only passing task branches into the integration branch.
- **Adaptive Model Fleet**: Routes tasks across heterogeneous local models, delegating high-complexity architectural planning to strong models while routing routine coding subtasks to fast models.
- **Non-Interactive Security Guardrails**: Enforces project-level command execution policies (safe test and read operations allowed by default, mutations require explicit approvals, destructive actions blocked).
- **Reproducible Benchmarking**: Provides automated test harnesses to benchmark Serial, Fixed, Naive, and Adaptive execution modes against identical repository baselines.
- **Terminal Interface**: Includes an interactive CLI, setup assistant (`adaptea setup`), diagnostic health checker (`adaptea doctor`), and full-screen terminal dashboard (`adaptea tui`).

---

## Supported Inference Backends

Adaptea Core communicates directly with local inference servers using typed backend drivers:

| Backend | Default Endpoint | Health & Discovery | Concurrency & Telemetry Mechanism |
|---|---|---|---|
| **LM Studio** | `http://127.0.0.1:1234` | Native `/api/v1/models` & `lms ps` | Full adaptive admission using real-time queue pressure and latency telemetry (`lms log stream`) |
| **Ollama** | `http://127.0.0.1:11434` | `/api/tags`, `/api/ps`, `/api/show` | Multi-path binary discovery, automated daemon launch, and safe concurrency bounds |
| **llama.cpp** | `http://127.0.0.1:8080` | `/props`, `/health` | Native slot-level allocation tracking (`/slots`) and context limits via `llama-server` |
| **vLLM** | `http://127.0.0.1:8000` | `/v1/models`, `/health` | Real-time Prometheus metrics scraping (`/metrics`) for KV-cache saturation and queue monitoring |

---

## Installation

### Prerequisites
- **Python**: Version 3.12 or newer
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- **Git**: Installed and available on `PATH`
- **Coding Agent**: [OpenCode](https://opencode.ai)
- **Local Inference**: At least one local inference server (LM Studio, Ollama, llama.cpp, or vLLM)

### Option 1: Standalone Terminal Installer (macOS & Linux)
Installs an isolated Python runtime and places the `adaptea` CLI in `~/.local/bin`:
```bash
curl -LsSf https://adaptea.dev/install.sh | sh
```

### Option 2: Install via `uv` or `pip`
```bash
# Using uv (recommended)
uv tool install adaptea-core

# Or using pip
pip install adaptea-core
```

### Option 3: Install from Source
```bash
git clone https://github.com/tahaenesaslanturk/Adaptea-Core.git
cd Adaptea-Core
uv sync
```

---

## Quickstart & Common Workflows

### 1. Interactive Setup Wizard
Diagnose your system, detect running backends, select models, and configure OpenCode project-locally:
```bash
adaptea setup
```
To automatically resolve missing configurations without prompt interruptions:
```bash
adaptea setup --auto
```

### 2. System Diagnostic Health Check
Verify backend availability, port bindings, loaded context lengths (32k+ token requirement), and CLI tools:
```bash
adaptea doctor
```

### 3. End-to-End Pipeline Smoke Test
Run an automated end-to-end verification that checks Git -> Backend -> Model -> OpenCode -> Adaptea in a temporary fixture:
```bash
adaptea smoke-test
```

### 4. Running a Software Goal
Provide a natural-language engineering goal from within any Git repository:
```bash
adaptea run "Add user authentication with signup, password hashing, JWT tokens, and unit tests"
```

### 5. Inspecting or Editing the Plan First
Decompose the goal into an editable dependency graph (`plan.json`) before starting execution:
```bash
# Generate the plan
adaptea plan "Add user authentication"

# Execute a reviewed plan
adaptea run --plan .adaptea/runs/<plan-id>/plan.json
```

### 6. Full-Screen Terminal Dashboard (TUI)
Launch the interactive Textual interface to manage projects, review plans, and monitor live worker trees:
```bash
adaptea tui
```

### 7. Benchmarking Scheduler Performance
Compare Adaptive scheduling against Serial and Fixed Concurrency modes over reproducible iterations:
```bash
adaptea benchmark --plan plan.json --repetitions 3 --fixed-concurrency 2 --max-agents 8
adaptea report
```

### 8. CLI Reference Summary
```bash
adaptea                                    # Launch interactive terminal shell (default)
adaptea tui                                # Launch full-screen live dashboard
adaptea doctor                             # Verify system health and context sizes
adaptea setup                              # Interactive environment configurator
adaptea smoke-test                         # Shared MVP validation test
adaptea run "<goal>"                       # Execute autonomous coding plan
adaptea plan "<goal>"                      # Generate dependency plan without executing
adaptea status <run-id>                    # Inspect status of an active or past run
adaptea resume <run-id>                    # Safely recover an interrupted run
adaptea calibrate                          # Measure machine concurrency envelope
adaptea fleet list                         # List downloaded models and active instances
adaptea fleet status                       # Show multi-model pool allocations
adaptea fleet configure                    # Configure fast/strong model tiers
adaptea command-policy                     # Display worker command security categories
adaptea approve-command "<cmd>"            # Whitelist a command pattern for current project
```

---

## How Scheduling & Admission Control Work

Spawning multiple autonomous coding workers simultaneously can quickly saturate local machine resources. When multiple workers send concurrent token requests to local inference servers, memory bandwidth saturates and queue latencies increase significantly.

### Dynamic Fan-Out Formula
Adaptea Core regulates worker admissions dynamically:

$$\text{fan\_out}(t) = \min(\text{admission\_target}(t), \text{ready\_tasks}(t), \text{user\_ceiling}, \text{backend\_parallel\_limit})$$

- **Non-Preemptive**: Running workers are never terminated when capacity tightens; they finish naturally while the admission target scales back for subsequent waves.
- **Hysteresis and Cooldown**: Uses moving statistical windows to prevent thrashing admission targets under bursty decoding phases.
- **Context Length Requirement**: OpenCode worker prompts, tool schemas, and planning contracts require substantial context. Models must be loaded with at least **32,768** tokens (65,536 recommended). `adaptea doctor` flags sub-32k models as not ready.

### Bounded Failure Handling
Every task failure is classified into one of five categories:
1. `model_error`
2. `timeout`
3. `validation_failure`
4. `dependency_problem`
5. `infrastructure_error`

Each task is capped at **3 automatic retries** across all categories. In fleet mode, failed attempts on the fast tier automatically escalate to the strong tier. If a merge conflict persists after a clean retry, Adaptea creates an isolated `resolution-<task>` worktree with explicit manual steps rather than corrupting repository history.

---

## Configuration (`adaptea.toml`)

Adaptea Core reads project configuration from `adaptea.toml` in the repository root. Generate one with `adaptea setup` or configure it manually:

```toml
[project]
# Explicit command run to validate each task's worktree.
# When omitted, Adaptea automatically detects pytest or npm test.
test_command = ["uv", "run", "pytest"]
no_tests_is_failure = false

[inference]
# Selected backend: "lmstudio", "ollama", "llamacpp", or "vllm"
backend = "lmstudio"

[lmstudio]
base_url = "http://127.0.0.1:1234"
model = "qwen2.5-coder-32b-instruct"

[worker]
max_agents = 4
timeout_seconds = 600
planner_timeout_seconds = 1800

[worker.command_security]
# Approved non-default commands that workers are allowed to execute
approved_commands = [
  "npm install zod",
  "uv add httpx",
]

[fleet]
enabled = false
topology = "auto"
max_loaded_instances = 2

[fleet.routing]
high_complexity = "strong"
low_complexity = "fast"
medium_complexity = "auto"
escalate_failed_fast_task = true
```

---

## Development & Contributing

Contributions are welcome. Please consult the [CONTRIBUTING.md](CONTRIBUTING.md) guide before submitting pull requests.

### Setting Up Development Environment
```bash
git clone https://github.com/tahaenesaslanturk/Adaptea-Core.git
cd Adaptea-Core
uv sync
```

### Running Quality Checks
```bash
# Code formatting check
uv run ruff format --check .

# Linting
uv run ruff check .

# Static type checking
uv run python -m mypy

# Test suite
uv run python -m pytest
```

---

## License

Adaptea Core is open-source software licensed under the [Apache License 2.0](LICENSE).
