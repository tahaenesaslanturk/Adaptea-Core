"""Turn OpenCode's event stream into one readable line of what is happening now.

A local planning call can run for many minutes and a coding task for longer still. A
spinner during that time says only that the process has not exited, which is exactly the
information the user already had. OpenCode emits a structured event per step and per tool
call, so the honest alternative is to report the tool it actually invoked.

Nothing here interprets or judges the work: an event that does not describe an action
produces no activity rather than a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How each OpenCode tool reads as an action. Unknown tools keep their own name, so a new
#: OpenCode tool degrades to "running <tool>" instead of disappearing from the stream.
_VERBS: dict[str, str] = {
    "bash": "Running",
    "read": "Reading",
    "write": "Writing",
    "edit": "Editing",
    "patch": "Editing",
    "grep": "Searching",
    "glob": "Finding",
    "list": "Listing",
    "webfetch": "Fetching",
    "todowrite": "Planning",
    "todoread": "Reviewing plan",
    "task": "Delegating",
}

_MAXIMUM_DETAIL = 90


@dataclass(frozen=True, slots=True)
class Activity:
    """One observed action, already phrased for display."""

    kind: str
    text: str
    tool: str | None = None
    #: Cumulative tokens reported by the step that produced this, when OpenCode reports it.
    total_tokens: int | None = None
    #: The untruncated shell command behind this activity, when the event carries one.
    #: A blocked call is only actionable if the exact command can be shown to the user
    #: and turned into an approval, so it is kept apart from the display text.
    command: str | None = None


def _shorten(value: str) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= _MAXIMUM_DETAIL:
        return collapsed
    return collapsed[: _MAXIMUM_DETAIL - 1] + "…"


def _relativize_path_str(value: str, workspace: Path | None) -> str:
    if not workspace:
        return value
    text = value.replace("\\", "/")
    roots = {
        str(workspace),
        str(workspace.resolve()),
        str(workspace.parent),
        str(workspace.parent.resolve()),
    }
    for root in sorted(roots, key=len, reverse=True):
        normalized = root.replace("\\", "/").rstrip("/")
        marker_slash = normalized + "/"
        if marker_slash in text:
            return text.replace(marker_slash, "")
        if normalized in text:
            return text.replace(normalized, ".")
    return text


def _detail(tool: str, *sources: dict[str, Any], workspace: Path | None = None) -> str:
    for source in sources:
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            rel = _relativize_path_str(title, workspace) if workspace else title
            return _shorten(rel)
    for source in sources:
        payload = source.get("input")
        if isinstance(payload, dict):
            for key in ("command", "filePath", "file_path", "path", "pattern", "query"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    rel = _relativize_path_str(value, workspace) if workspace else value
                    return _shorten(rel)
    return tool


def _command(*sources: dict[str, Any]) -> str | None:
    """The exact shell command an event describes, unshortened, or None."""
    for source in sources:
        payload = source.get("input")
        if isinstance(payload, dict):
            for key in ("command", "cmd", "script", "query"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return " ".join(value.split())
            args = payload.get("args")
            if isinstance(args, list) and args:
                joined = " ".join(str(a) for a in args)
                if joined.strip():
                    return " ".join(joined.split())
        for key in ("command", "cmd", "script"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            return " ".join(title.split())
    return None


def summarize_event(event: Any, workspace: Path | None = None) -> Activity | None:
    """Describe one OpenCode JSONL event, or return None when it describes no action."""
    if not isinstance(event, dict):
        return None
    # OpenCode wraps stream events in `part`, but not every build does and the shape is
    # not a stable contract. Read an unwrapped event the same way rather than reporting
    # nothing at all, which is what the user had before.
    part = event.get("part")
    node: dict[str, Any] = part if isinstance(part, dict) else event
    kind = node.get("type")
    named_tool = node.get("tool")

    if kind == "tool" or (kind is None and isinstance(named_tool, str)):
        tool = named_tool if isinstance(named_tool, str) else "tool"
        raw_state = node.get("state")
        state: dict[str, Any] = raw_state if isinstance(raw_state, dict) else {}
        status = state.get("status")
        if tool == "invalid":
            raw_input = state.get("input")
            invalid_input = raw_input if isinstance(raw_input, dict) else {}
            requested = invalid_input.get("tool")
            detail = (
                f"'{requested}' is unavailable"
                if isinstance(requested, str) and requested
                else "the requested tool is unavailable"
            )
            return Activity(
                kind="tool-error",
                text=f"Model tool call rejected: {detail}",
                tool=tool,
            )
        raw_error = state.get("error") or node.get("error")
        error_text = str(raw_error).lower() if raw_error is not None else ""
        is_blocked = status == "error" and any(
            phrase in error_text
            for phrase in (
                "specified a rule which prevents",
                "rule prevents",
                "permission denied",
                "blocked by command safety",
                "not allowed by configuration",
                "denied by rule",
                "command is blocked",
                "requires approval",
                "permission is denied",
                "operation not permitted",
            )
        )
        if is_blocked:
            extracted_cmd: str | None = _command(state, node) or _detail(
                tool, state, node, workspace=workspace
            )
            if extracted_cmd == tool:
                extracted_cmd = None
            return Activity(
                kind="tool-blocked",
                text=f"Blocked by command safety: {_detail(tool, state, node, workspace=workspace)}",
                tool=tool,
                command=extracted_cmd,
            )
        # A completed call is reported in the past tense so a long-running tool is
        # distinguishable from one that already returned.
        verb = _VERBS.get(tool, f"Running {tool}")
        detail = _detail(tool, state, node, workspace=workspace)
        if status == "error":
            if "file not found" in error_text or "enoent" in error_text:
                return Activity(
                    kind="tool-error",
                    text=f"File not found: {detail}",
                    tool=tool,
                )
            return Activity(
                kind="tool-error",
                text=f"Tool call could not complete: {verb} {detail}",
                tool=tool,
            )
        return Activity(kind="tool", text=f"{verb} {detail}", tool=tool)

    if kind == "step-finish":
        tokens = node.get("tokens")
        total = tokens.get("total") if isinstance(tokens, dict) else None
        return Activity(
            kind="step",
            text="Thinking",
            total_tokens=total if isinstance(total, int) else None,
        )

    if kind == "reasoning":
        return Activity(kind="reasoning", text="Reasoning")

    return None


def summarize_line(line: str, workspace: Path | None = None) -> Activity | None:
    """Parse one stdout line. Non-JSON output is progress noise, not an error."""
    text = line.strip()
    if not text or not text.startswith("{"):
        return None
    try:
        return summarize_event(json.loads(text), workspace=workspace)
    except ValueError:
        return None


def relative_activity(activity: Activity, workspace: Path) -> Activity:
    """Hide an isolated worktree's implementation path from user-facing activity.

    OpenCode commonly reports an absolute file path even though it is correctly scoped
    to its worktree. Showing ``.adaptea/worktrees/...`` made that look like a runtime
    metadata edit and made otherwise useful paths unreadable. Paths outside the worktree
    are deliberately left untouched so a genuinely out-of-scope attempt stays visible.
    """
    import re

    # OpenCode and Python may disagree on separators when a test or subprocess describes
    # a non-native path. Normalizing only for the comparison also handles a Windows
    # workspace reported as ``C:\\...`` or ``\\Users\\...`` without hiding a genuinely
    # out-of-scope path.
    text = activity.text.replace("\\", "/")
    roots = {
        str(workspace),
        str(workspace.resolve()),
        str(workspace.parent),
        str(workspace.parent.resolve()),
    }
    changed = False
    for root in sorted(roots, key=len, reverse=True):
        normalized = root.replace("\\", "/").rstrip("/")
        marker_slash = normalized + "/"
        if marker_slash in text:
            text = text.replace(marker_slash, "")
            changed = True
            break
        elif normalized in text:
            text = text.replace(normalized, ".")
            changed = True
            break

    if not changed:
        # Check if text contains a truncated worktree path ending in ellipsis or internal .adaptea/worktrees/
        worktree_pattern = re.compile(
            r"/(?:[^\s\"']+/)?\.adaptea/worktrees/[^/\s\"']+(?:/[^/\s\"']+)?(?:/(?P<rel>[^\s\"']*))?"
        )
        match = worktree_pattern.search(text)
        if match:
            rel = match.group("rel")
            if rel and not rel.endswith("…"):
                text = text[: match.start()] + rel + text[match.end() :]
            elif rel and rel.endswith("…"):
                cleaned_rel = rel.rstrip("…").strip("/")
                text = text[: match.start()] + (cleaned_rel or ".") + text[match.end() :]
            else:
                text = text[: match.start()] + "." + text[match.end() :]
            changed = True
        elif ".adaptea/worktrees/" in text:
            # Fallback for paths truncated before the task directory
            sub_text = re.sub(r"/[^\s\"']*\.adaptea/worktrees/[^\s\"']*", ".", text)
            if sub_text != text:
                text = sub_text
                changed = True

    if not changed:
        return activity
    return Activity(
        kind=activity.kind,
        text=text,
        tool=activity.tool,
        total_tokens=activity.total_tokens,
        command=activity.command,
    )
