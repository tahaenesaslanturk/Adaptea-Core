# Adaptea decision log

## Product name

“Adaptea” describes a controller that adapts worker admission to changing local inference pressure. It is not
a coding-agent implementation.

## One V1 model backend

LM Studio is the only V1 inference backend so capability discovery, authentication, telemetry, calibration,
and user guidance remain testable. The native `/api/v1` model and chat APIs are used by Adaptea; OpenCode
uses LM Studio's OpenAI-compatible endpoint. No direct llama.cpp, MLX, Ollama, or cloud integration exists.

The GGUF/llama.cpp runtime is preferred because LM Studio currently documents Max Concurrent Predictions and
continuous batching for that engine. Other runtimes are reported honestly and warned about; equivalent
parallel behavior is not assumed.

LM Studio remains the only backend, but it is no longer restricted to one destination. A fleet destination
is a loaded instance ID, separate from its downloaded model key. User configuration assigns planner/worker
roles and strong/fast tiers. Routing uses planner-provided complexity, risk, and preferred tier plus measured
instance pressure; it does not rank intelligence from parameter count.

Model topology changes slowly: a run establishes its measured/configured topology, then adapts worker
admission within that stable topology. Workers are pinned to one instance for an attempt and are never moved
because of scheduler pressure. A failed fast-pool validation may make one new strong-pool attempt; this is a
bounded deterministic escalation, not online learning.

## OpenCode workers

OpenCode supplies the actual tool-using planning and coding loops. Adaptea supplies orchestration, isolation,
validation, and admission control. Project-local ephemeral configuration prevents changes to the user's
global OpenCode settings. A worker can issue many inference requests, so worker count is not request count.

## Setup and configuration ownership

First-run setup is a diagnose-repair-recheck loop, not a second orchestration system. `setup` and `doctor`
consume the same diagnostic snapshot so a green setup state has the same meaning as a passing doctor check.
Calibration is deliberately excluded from readiness because it is a workload measurement, not a software
prerequisite.

Setup may execute trusted local package-manager commands, but it always displays the exact command and
official source first. Remote installer scripts and potentially large model downloads require explicit
confirmation even in automatic mode. Commands use argument-array subprocesses; the only shell evaluation is
an approved official remote installer pipeline whose contents and source were shown to the user.

OpenCode configuration remains project-local. Setup merges only the LM Studio provider and selected model,
supports the stable and `opencode2` schemas, preserves unrelated settings, and backs up an existing file.
The TOML updater replaces only Adaptea-owned keys while retaining unrelated sections and comments where
practical. This makes repeated setup runs idempotent and avoids taking ownership of a user's broader tool
configuration.

## Admission-only control

Local inference pressure can change after workers start. Killing a healthy worker discards expensive progress
and can leave a worktree in an uncertain state, so target reductions affect only future admissions. Moving
windows, consecutive pressure/health samples, and cooldown provide understandable hysteresis. Decisions are
persisted with their observed signals.

## Measurement honesty

LM Studio's public API does not promise live KV-cache pressure. Adaptea records native chat measurements,
tolerantly observed `lms ps` state, log-stream statistics, and unknown values distinctly. Unknown values are
never converted into percentages or estimates.

## Calibration

An unmeasured warm-up is followed by seeded, interleaved concurrency trials to reduce laptop order and
thermal bias. Requests record actual generated work and short outputs are retried once, then retained as
invalid rather than silently compared. The heterogeneous workload switches request shape within the same
calibration. Medians feed a conservative starting profile; runtime adaptation may move away from it.

Fleet calibration treats continuous batching, same-model replication, and heterogeneous fast/strong fleets
as different topology families. Its candidate set is bounded by the configured instance and worker ceilings,
but agent-level validation includes one leading candidate from every family present. This prevents two nearly
identical high-throughput batching variants from consuming the validation budget while replication or a mixed
fleet goes untested. Failed repetitions lower an explicit measurement-success rate; a fast but unstable
candidate cannot outrank a repeatable one with equivalent validation quality.

## Integration and recovery

Each admitted task owns a branch/worktree. Deterministic project validation—not process success or model
self-report—gates integration. A dedicated integration worktree serializes merges. A conflict is aborted and
retried once from the current integration head. Atomic state files allow resume while preserving merged
tasks. Independent tasks continue after unrelated failures.

## Benchmark interpretation

Primary metrics are end-to-end wall time and deterministic pass rate. At least three repetitions and medians
are required for formal comparisons. “Oracle fixed” means the best observed fixed concurrency among tested
values, not a theoretical optimum. Laptop thermals, model behavior, and generated-token volume remain
limitations.

## Cross-platform constraints

The core runtime uses `pathlib`, argument-array subprocesses, Python file replacement, and Git's CLI without
shell pipelines, Unix signals for normal control, `/tmp`, symlink assumptions, or platform-only locking.
The explicitly confirmed official installers are the only setup actions that evaluate a shell pipeline.
Branch and worktree names are sanitized for Windows as well as macOS.
