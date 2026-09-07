from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptea.config import Config
from adaptea.lmstudio.lms_cli import normalize_subprocess_command
from adaptea.models import TaskSpec, utc_now
from adaptea.planner.opencode import opencode_config
from adaptea.runtime.events import append_jsonl
from adaptea.security.commands import decide_command, extract_shell_commands, policy_document
from adaptea.validation import resolve_test_command
from adaptea.workers.activity import Activity, relative_activity
from adaptea.workers.process import communicate_with_activity


@dataclass(slots=True)
class WorkerResult:
    exit_code: int
    started_at: str
    ended_at: str
    wall_seconds: float
    session_id: str | None
    summary: str


def worker_prompt(
    goal: str,
    task: TaskSpec,
    dependency_summaries: list[str],
    test_command: list[str],
    retry_context: str | None = None,
    plan_context: list[str] | None = None,
) -> str:
    validator = (
        json.dumps(test_command)
        if test_command
        else "none detected; this project has no automated test suite, so verify your "
        "change directly"
    )
    plan_overview = "\n".join(f"  {line}" for line in plan_context) if plan_context else "  none"
    return f"""You are one Adaptea coding worker in an isolated Git worktree.
Implement the specific task completely. You may inspect and edit only this worktree.
Do not modify other
repositories or global configuration. Run focused tests. Do not merely describe changes.

Read before you write. This is a real repository, not a blank page:
- Before editing any file, read it. Before creating one, list the directory it belongs in
  and read a neighbouring file of the same kind.
- Search the repository for every symbol, module, config key, or command you intend to
  use, and confirm it exists and takes the arguments you are about to pass. Never assume
  an API, a dependency, a script, or a file path — verify it in the tree.
- Follow the conventions already in this repository: its naming, its layout, its imports,
  its error handling, its test style, its comment density. Match the surrounding code
  rather than introducing a style it does not use.
- The files hint below is a hint, not the answer. Check it, and look wider when it is
  incomplete or wrong.
- Coordinate with other tasks in the plan: focus strictly on your task scope and do not
  duplicate changes handled by sibling tasks.
- If what you find contradicts the task description, trust the repository and say so in
  your summary.

Tool usage rules:
- Always use relative file paths from the worktree root (e.g. `src/app.py`).
- The `read` tool is strictly for reading individual files. NEVER call `read` on a directory or the root path.
- To inspect directory structure or discover files, use `glob`, `grep`, or safe shell commands (e.g. `ls`, `find`).
- Do not call `read` on files that do not exist yet (e.g. when scaffolding or creating new files from scratch); verify file existence with `glob` or write them directly.

Overall goal: {goal}

Plan overview (all tasks in this run):
{plan_overview}

Task: {task.title}
Description: {task.description}
Acceptance criteria: {json.dumps(task.acceptance_criteria)}
Files hint: {json.dumps(task.files_hint)}
Dependency summaries: {json.dumps(dependency_summaries)}
Project validator: {validator}
Retry context: {retry_context or "none"}

Command security:
- Safe-default inspection, package install, and test commands run automatically.
- Any other command is denied on first use and sent to the user as an approval request.
  An approval applies from your next attempt onward, not to the call that was denied.
- Host-wide destructive commands remain blocked and are never offered for approval.
- Use the built-in read, glob, and grep tools for file inspection.
- Run a denied command once, then continue with built-in read/edit tools. Do not retry it
  in a loop and do not work around the policy; name the command in your summary so the
  user can see what the task still needs.

Finish by summarizing files changed and tests run.
Adaptea will independently validate and commit the work.
"""


def _session_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            for key in ("sessionID", "session_id", "sessionId"):
                value = event.get(key)
                if isinstance(value, str):
                    return value
    return None


class OpenCodeWorker:
    def __init__(
        self,
        config: Config,
        model: str,
        selectable_models: list[str] | None = None,
        on_activity: Callable[[str, Activity], None] | None = None,
        on_blocked_command: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.selectable_models = selectable_models
        #: Reports each tool call with the task it belongs to, so a running worker shows
        #: the file it is editing instead of an indeterminate spinner.
        self.on_activity = on_activity
        #: Reports (task id, command) for a call the policy refused, so the run can ask
        #: the user about it rather than only writing it to an artifact nobody reads.
        self.on_blocked_command = on_blocked_command

    async def run(
        self,
        worktree: Path,
        artifact_dir: Path,
        goal: str,
        task: TaskSpec,
        dependency_summaries: list[str],
        retry_context: str | None = None,
        progress: Callable[[dict[str, str]], None] | None = None,
        plan_context: list[str] | None = None,
    ) -> WorkerResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt = worker_prompt(
            goal,
            task,
            dependency_summaries,
            resolve_test_command(worktree, self.config.project.test_command),
            retry_context,
            plan_context=plan_context,
        )
        environment = os.environ.copy()
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            opencode_config(self.config, self.model, self.selectable_models, secure_worker=True)
        )
        policy = policy_document(self.config.worker.command_security)
        (artifact_dir / "command-security-policy.json").write_text(
            json.dumps(policy, indent=2) + "\n", encoding="utf-8"
        )
        started_at = utc_now()
        clock = time.perf_counter()
        spawn_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            spawn_kwargs["start_new_session"] = True
        command = normalize_subprocess_command(
            [
                self.config.worker.executable,
                "run",
                "--pure",
                "--model",
                f"adaptea/{self.model}",
                "--format",
                "json",
                "--dir",
                str(worktree),
                prompt,
            ]
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=worktree,
            env=environment,
            # A coding worker is non-interactive; inheriting the desktop core's long-lived
            # stdin pipe makes OpenCode wait on it instead of starting work.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **spawn_kwargs,
        )
        # The worker's stdout is read as it arrives rather than buffered until exit.
        # `communicate()` withheld every step of a multi-minute task until it finished,
        # which left the desktop with nothing to show but a "running" badge.
        events_path = artifact_dir / "worker-events.jsonl"
        events_path.write_text("", encoding="utf-8")
        activity_log = artifact_dir / "activity.jsonl"

        def record_line(line: bytes) -> None:
            text = line.decode(errors="replace").strip()
            if not text:
                return
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

        def observe(activity: Activity) -> None:
            nonlocal progress
            activity = relative_activity(activity, worktree)
            append_jsonl(
                activity_log,
                {
                    "timestamp": utc_now(),
                    "kind": activity.kind,
                    "tool": activity.tool,
                    "text": activity.text,
                    "total_tokens": activity.total_tokens,
                },
            )
            if self.on_activity is not None:
                self.on_activity(task.id, activity)
            if activity.kind == "tool-blocked" and activity.command and self.on_blocked_command:
                # A blocked command is the one activity the user can act on, so it is
                # raised as a question immediately rather than after the attempt ends.
                try:
                    self.on_blocked_command(task.id, activity.command)
                except Exception:  # noqa: BLE001 - a worker never fails on telemetry
                    pass
            if progress is None:
                return
            # A worker must never fail because the desktop is not listening.
            try:
                progress(
                    {
                        "kind": activity.kind,
                        "title": activity.text,
                        "detail": f"{activity.total_tokens:,} tokens"
                        if activity.total_tokens is not None
                        else "",
                    }
                )
            except Exception:  # noqa: BLE001 - progress is best-effort telemetry
                progress = None

        try:
            output = await communicate_with_activity(
                process,
                timeout=self.config.worker.timeout_seconds,
                on_activity=observe,
                on_line=record_line,
                workspace=worktree,
            )
        except asyncio.CancelledError:
            self._record_command_decisions(artifact_dir, "")
            raise
        stdout_bytes, stderr_bytes = output.stdout, output.stderr
        if output.timed_out:
            stderr_bytes += (
                f"\nOpenCode worker timed out after {self.config.worker.timeout_seconds} seconds."
            ).encode()
            returncode = 124
        else:
            returncode = process.returncode or 0
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        (artifact_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (artifact_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        decisions = self._record_command_decisions(artifact_dir, stdout)
        summary = stdout[-4000:].strip()
        result = WorkerResult(
            exit_code=returncode,
            started_at=started_at,
            ended_at=utc_now(),
            wall_seconds=time.perf_counter() - clock,
            session_id=_session_id(stdout),
            summary=summary,
        )
        (artifact_dir / "summary.json").write_text(
            json.dumps(
                result.__dict__
                if hasattr(result, "__dict__")
                else {
                    "exit_code": result.exit_code,
                    "started_at": result.started_at,
                    "ended_at": result.ended_at,
                    "wall_seconds": result.wall_seconds,
                    "session_id": result.session_id,
                    "summary": result.summary,
                    "command_security": {
                        "observed_commands": len(decisions),
                        "allowed": sum(item["action"] == "allow" for item in decisions),
                        "denied": sum(item["action"] == "deny" for item in decisions),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def _record_command_decisions(self, artifact_dir: Path, output: str) -> list[dict[str, Any]]:
        path = artifact_dir / "command-security-decisions.jsonl"
        decisions = [
            decide_command(command, self.config.worker.command_security).as_dict()
            for command in extract_shell_commands(output)
        ]
        path.touch(exist_ok=True)
        for decision in decisions:
            append_jsonl(path, decision)
        return decisions
