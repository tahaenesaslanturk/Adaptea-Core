from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from adaptea.config import Config, executable_stem
from adaptea.inference import inference_connection
from adaptea.lmstudio.lms_cli import normalize_subprocess_command
from adaptea.models import Plan
from adaptea.planner.prompts import planner_prompt
from adaptea.security.commands import command_permission_config
from adaptea.workers.activity import Activity
from adaptea.workers.process import communicate_with_activity


def opencode_config(
    config: Config,
    model: str,
    selectable_models: list[str] | None = None,
    *,
    secure_worker: bool = False,
) -> dict[str, Any]:
    connection = inference_connection(config)
    model_catalog = {
        identifier: {"name": identifier}
        for identifier in dict.fromkeys([model, *(selectable_models or [])])
    }
    v2 = executable_stem(config.worker.executable) == "opencode2"
    if v2:
        settings: dict[str, Any] = {"baseURL": connection.openai_base_url}
        if connection.fallback_api_key:
            settings["apiKey"] = connection.fallback_api_key
        provider = {
            "package": "@opencode-ai/ai/providers/openai-compatible",
            "name": connection.display_name,
            "settings": settings,
            "models": model_catalog,
        }
        if connection.api_token_environment:
            provider["env"] = [connection.api_token_environment]
        result: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "providers": {"adaptea": provider},
        }
        if secure_worker:
            result.update(command_permission_config(config.worker.command_security, v2=True))
        return result
    options: dict[str, Any] = {"baseURL": connection.openai_base_url}
    if connection.api_token_environment:
        options["apiKey"] = f"{{env:{connection.api_token_environment}}}"
    elif connection.fallback_api_key:
        options["apiKey"] = connection.fallback_api_key
    result = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "adaptea": {
                "npm": "@ai-sdk/openai-compatible",
                "name": connection.display_name,
                "options": options,
                "models": model_catalog,
            }
        },
    }
    if secure_worker:
        result.update(command_permission_config(config.worker.command_security, v2=False))
    return result


def extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [text]
    fragments: list[str] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            for key in ("text", "content", "message", "output"):
                if isinstance(event.get(key), str):
                    candidates.append(event[key])
                    fragments.append(event[key])
            part = event.get("part")
            if isinstance(part, dict):
                for key in ("text", "content"):
                    if isinstance(part.get(key), str):
                        candidates.append(part[key])
                        fragments.append(part[key])
    if fragments:
        candidates.append("".join(fragments))
    for candidate in reversed(candidates):
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "tasks" in value:
                return value
    raise ValueError("planner output did not contain a JSON plan object")


class OpenCodePlanner:
    def __init__(
        self,
        root: Path,
        config: Config,
        model: str,
        on_activity: Callable[[Activity], None] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.model = model
        #: Reports each tool call the planner makes, so a long plan is visibly working.
        self.on_activity = on_activity

    async def plan(self, goal: str, retries: int = 2, previous: Plan | None = None) -> Plan:
        error: str | None = None
        for _attempt in range(retries + 1):
            prompt = planner_prompt(goal, error, previous)
            stdout, stderr, code = await self._invoke(prompt)
            if code != 0:
                detail = (stderr.strip() or stdout.strip())[-1000:]
                error = f"OpenCode exited {code}: {detail}"
                continue
            try:
                return Plan.model_validate(extract_json(stdout))
            except (ValidationError, ValueError) as exc:
                error = str(exc)
        raise RuntimeError(f"planner failed after {retries + 1} attempts: {error}")

    async def _invoke(self, prompt: str) -> tuple[str, str, int]:
        environment = os.environ.copy()
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            opencode_config(self.config, self.model)
        )
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
                "--agent",
                "plan",
                "--format",
                "json",
                "--dir",
                str(self.root),
                prompt,
            ]
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.root,
            env=environment,
            # Never inherit the parent's stdin. Under the desktop app the core's stdin is a
            # pipe Tauri holds open for the whole session, and OpenCode blocks on it instead
            # of issuing its first request — the plan only moved when the app was quit.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **spawn_kwargs,
        )
        result = await communicate_with_activity(
            process,
            timeout=self.config.worker.planner_timeout,
            on_activity=self.on_activity,
            workspace=self.root,
        )
        stderr = result.stderr
        if result.timed_out:
            stderr += (
                f"\nOpenCode planner timed out after {self.config.worker.planner_timeout} seconds."
            ).encode()
            return result.stdout.decode(errors="replace"), stderr.decode(errors="replace"), 124
        return (
            result.stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            process.returncode or 0,
        )


async def verify_opencode_models(config: Config, destinations: list[str], root: Path) -> None:
    """Verify generated fleet destinations through OpenCode's official model catalog command."""
    if not destinations:
        raise RuntimeError("No fleet model destinations were provided to OpenCode.")
    environment = os.environ.copy()
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        opencode_config(config, destinations[0], destinations)
    )
    command = normalize_subprocess_command(
        [
            config.worker.executable,
            "models",
            "adaptea",
        ]
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(
            "OpenCode could not list the project-local Adaptea model catalog: "
            + stderr.decode(errors="replace")[-500:].strip()
        )
    available = set(stdout.decode(errors="replace").split())
    missing = [
        destination
        for destination in destinations
        if f"adaptea/{destination}" not in available and destination not in available
    ]
    if missing:
        raise RuntimeError(f"OpenCode did not expose configured fleet destinations: {missing}")
