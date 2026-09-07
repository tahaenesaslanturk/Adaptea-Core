from __future__ import annotations

import asyncio
import json
from pathlib import Path

from adaptea.workers.activity import Activity, relative_activity, summarize_event, summarize_line
from adaptea.workers.process import communicate_with_activity


def test_a_tool_call_reads_as_the_action_it_performed() -> None:
    event = {
        "type": "tool_use",
        "part": {
            "type": "tool",
            "tool": "bash",
            "state": {"status": "completed", "title": "ls -la", "input": {"command": "ls -la"}},
        },
    }
    activity = summarize_event(event)
    assert activity is not None
    assert activity.text == "Running ls -la"
    assert activity.tool == "bash"


def test_a_write_names_the_file_rather_than_the_tool() -> None:
    event = {
        "part": {
            "type": "tool",
            "tool": "write",
            "state": {
                "status": "completed",
                "title": "index.html",
                "input": {"filePath": "index.html"},
            },
        }
    }
    activity = summarize_event(event)
    assert activity is not None and activity.text == "Writing index.html"


def test_an_unknown_tool_degrades_instead_of_disappearing() -> None:
    activity = summarize_event(
        {"part": {"type": "tool", "tool": "sqlquery", "state": {"title": "select 1"}}}
    )
    assert activity is not None and activity.text == "Running sqlquery select 1"


def test_a_failed_tool_call_says_so() -> None:
    activity = summarize_event(
        {"part": {"type": "tool", "tool": "bash", "state": {"status": "error", "title": "pytest"}}}
    )
    assert activity is not None and activity.text == "Tool call could not complete: Running pytest"


def test_a_command_denied_by_worker_policy_is_reported_as_blocked_not_failed() -> None:
    activity = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "error",
                    "input": {"command": 'grep -E "(vercel|next\\.config)" .gitignore'},
                    "error": "The user has specified a rule which prevents this tool call.",
                },
            }
        }
    )

    assert activity is not None
    assert activity.kind == "tool-blocked"
    assert activity.text.startswith("Blocked by command safety: grep -E")


def test_an_invalid_tool_call_names_the_model_mistake_without_saying_it_is_running() -> None:
    activity = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "invalid",
                "state": {
                    "status": "completed",
                    "title": "Invalid Tool",
                    "input": {"tool": "run", "error": "unavailable"},
                },
            }
        }
    )

    assert activity is not None
    assert activity.kind == "tool-error"
    assert activity.text == "Model tool call rejected: 'run' is unavailable"


def test_a_step_carries_the_token_total_it_reported() -> None:
    activity = summarize_event({"part": {"type": "step-finish", "tokens": {"total": 7658}}})
    assert activity is not None and activity.total_tokens == 7658


def test_output_that_describes_no_action_produces_none() -> None:
    assert summarize_event({"part": {"type": "text", "text": "\n\n"}}) is None
    assert summarize_line("not json at all") is None
    assert summarize_line("") is None


def _stream(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


class _Process:
    returncode = 0

    def __init__(self, stdout: bytes) -> None:
        self.stdout = _stream(stdout)
        self.stderr = _stream(b"")

    async def wait(self) -> int:
        return self.returncode


async def test_activity_is_reported_while_output_arrives_and_bytes_are_preserved() -> None:
    lines = [
        json.dumps({"part": {"type": "tool", "tool": "read", "state": {"title": "src/app.py"}}}),
        # A single JSON line can carry a whole tool output and exceed asyncio's line limit.
        json.dumps({"part": {"type": "tool", "tool": "bash", "state": {"title": "x" * 80_000}}}),
    ]
    payload = ("\n".join(lines) + "\n").encode()
    seen: list[str] = []
    result = await communicate_with_activity(
        _Process(payload),  # type: ignore[arg-type]
        timeout=5,
        on_activity=lambda activity: seen.append(activity.text),
    )
    assert result.stdout == payload
    assert result.timed_out is False
    assert seen[0] == "Reading src/app.py"
    assert len(seen) == 2


def test_an_unwrapped_tool_event_is_read_the_same_way() -> None:
    """Not every OpenCode build wraps stream events in `part`.

    The shape is not a stable contract, so an unwrapped event has to narrate rather than
    fall through and leave the run silent.
    """
    activity = summarize_event({"tool": "bash", "input": {"command": "git status --short"}})

    assert activity is not None
    assert activity.text == "Running git status --short"
    assert activity.tool == "bash"


def test_an_event_that_names_no_tool_produces_no_activity() -> None:
    assert summarize_event({"sessionID": "session-safe"}) is None


def test_worker_activity_hides_the_internal_adaptea_worktree_path() -> None:
    worktree = Path("/Users/ada/project/.adaptea/worktrees/run-1/task-a1")
    activity = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "write",
                "state": {
                    "status": "error",
                    "title": f"{worktree}/src/app.py",
                },
            }
        }
    )

    assert activity is not None
    displayed = relative_activity(activity, worktree)
    assert displayed.text == "Tool call could not complete: Writing src/app.py"
    assert ".adaptea" not in displayed.text

    # When the model attempts to read the directory root without a trailing slash
    root_activity = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "read",
                "state": {
                    "status": "error",
                    "title": str(worktree),
                },
            }
        }
    )
    assert root_activity is not None
    displayed_root = relative_activity(root_activity, worktree)
    assert displayed_root.text == "Tool call could not complete: Reading ."
    assert ".adaptea" not in displayed_root.text


def test_blocked_command_extracted_from_title_or_args() -> None:
    activity_title = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "error",
                    "title": "npm install lodash",
                    "error": "permission denied by policy",
                },
            }
        }
    )
    assert activity_title is not None
    assert activity_title.kind == "tool-blocked"
    assert activity_title.command == "npm install lodash"

    activity_args = summarize_event(
        {
            "part": {
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "error",
                    "input": {"args": ["npx", "vitest", "run"]},
                    "error": "The user has specified a rule which prevents this tool call.",
                },
            }
        }
    )
    assert activity_args is not None
    assert activity_args.kind == "tool-blocked"
    assert activity_args.command == "npx vitest run"


def test_worker_activity_relativizes_long_paths_before_truncation() -> None:
    worktree = Path(
        "/Users/tahaaslanturk/Developer/Adaptea/heloworld/Helo/.adaptea/worktrees/run-2026-09-06-82c7607b4b/init-scaffold-a1"
    )
    long_file_path = f"{worktree}/index.html"
    assert len(long_file_path) > 90

    # When workspace is provided, it must be relativized before truncation
    event = {
        "type": "tool_use",
        "part": {
            "type": "tool",
            "tool": "read",
            "state": {
                "status": "error",
                "input": {"filePath": long_file_path},
                "error": f"File not found: {long_file_path}",
            },
        },
    }
    activity = summarize_event(event, workspace=worktree)
    assert activity is not None
    assert activity.text == "File not found: index.html"
    assert ".adaptea" not in activity.text


def test_worker_activity_relative_activity_cleans_already_truncated_paths() -> None:
    worktree = Path(
        "/Users/tahaaslanturk/Developer/Adaptea/heloworld/Helo/.adaptea/worktrees/run-2026-09-06-82c7607b4b/init-scaffold-a1"
    )
    # Simulate an activity where the path was already truncated with an ellipsis
    truncated_activity = Activity(
        kind="tool-error",
        text="Tool call could not complete: Reading /Users/tahaaslanturk/Developer/Adaptea/heloworld/Helo/.adaptea/worktrees/run-2026-09-06-8…",
        tool="read",
    )
    cleaned = relative_activity(truncated_activity, worktree)
    assert ".adaptea" not in cleaned.text
    assert cleaned.text == "Tool call could not complete: Reading ."
