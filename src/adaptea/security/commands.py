from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from adaptea.config import CommandSecurityConfig
from adaptea.models import utc_now

CommandCategory = Literal["safe_default", "approval_required", "blocked"]
CommandAction = Literal["allow", "deny"]

# These commands inspect the worktree or run common project checks. Anything not on this
# deliberately narrow list requires an explicit project-local approval.
SAFE_COMMAND_PATTERNS: tuple[str, ...] = (
    "pwd",
    "ls",
    "ls *",
    "cat *",
    "head *",
    "tail *",
    "which *",
    "echo *",
    "find *",
    "grep *",
    "rg *",
    "wc *",
    "sort *",
    "uniq *",
    "diff *",
    "stat *",
    "file *",
    "basename *",
    "dirname *",
    "realpath *",
    "readlink *",
    "tree *",
    "du *",
    "df *",
    "date",
    "date *",
    "env",
    "printenv *",
    "cut *",
    "awk *",
    "sed *",
    "sed -n *",
    "jq *",
    # Creating a directory or an empty file changes no content and is what a worker
    # needs constantly to lay out a new module. Everything that writes into them is
    # still the edit tool or an approval away.
    "mkdir *",
    "touch *",
    "git status",
    "git status *",
    "git diff",
    "git diff *",
    "git log",
    "git log *",
    "git show",
    "git show *",
    "git rev-parse",
    "git rev-parse *",
    "git ls-files",
    "git ls-files *",
    "git grep *",
    "git blame *",
    "git shortlog *",
    "git describe *",
    "git remote",
    "git remote -v",
    "git remote -v *",
    "git stash list",
    "git stash list *",
    "git worktree list",
    "git worktree list *",
    "git config --get *",
    "git config --list",
    "git config --list *",
    "git add *",
    "git commit *",
    "git branch",
    "git branch *",
    "git checkout",
    "git checkout *",
    "git restore",
    "git restore *",
    "git --version",
    "git --version *",
    "node *",
    "node -e *",
    "node -v",
    "node -v *",
    "node --version",
    "node --version *",
    "npx *",
    "npm -v",
    "npm -v *",
    "npm --version",
    "npm --version *",
    "npm list *",
    "npm ls *",
    "npm view *",
    "npm info *",
    "python",
    "python *",
    "python -c *",
    "python -v",
    "python -v *",
    "python -V",
    "python -V *",
    "python --version",
    "python --version *",
    "python3",
    "python3 *",
    "python3 -c *",
    "python3 -v",
    "python3 -v *",
    "python3 -V",
    "python3 -V *",
    "pip list *",
    "pip show *",
    "pip --version",
    "pip --version *",
    "pip3 list *",
    "pip3 show *",
    "pip3 --version",
    "pip3 --version *",
    "python3 --version",
    "python3 --version *",
    "pytest *",
    "python -m pytest *",
    "python3 -m pytest *",
    "uv run pytest *",
    "uv run ruff check *",
    "uv run mypy *",
    "ruff check *",
    "ruff format --check *",
    "mypy *",
    "eslint *",
    "tsc *",
    "tsc --noEmit *",
    "uv --version",
    "uv --version *",
    "npm test *",
    "npm run test *",
    "npm run lint *",
    "npm run typecheck *",
    "npm run build *",
    "pnpm test *",
    "pnpm run test *",
    "pnpm run lint *",
    "pnpm run typecheck *",
    "pnpm run build *",
    "pnpm -v",
    "pnpm -v *",
    "pnpm --version",
    "pnpm --version *",
    "pnpm list *",
    "pnpm ls *",
    "yarn test *",
    "yarn run test *",
    "yarn run lint *",
    "yarn run typecheck *",
    "yarn run build *",
    "yarn -v",
    "yarn -v *",
    "yarn --version",
    "yarn --version *",
    "yarn list *",
    "bun test *",
    "bun run test *",
    "bun run lint *",
    "bun run typecheck *",
    "bun run build *",
    "bun -v",
    "bun -v *",
    "bun --version",
    "bun --version *",
    "cargo test *",
    "cargo check *",
    "cargo clippy *",
    "cargo build *",
    "cargo fmt --check *",
    "cargo --version",
    "cargo --version *",
    "rustc --version",
    "rustc --version *",
    "go test *",
    "go build *",
    "go vet *",
    "go version",
    "go version *",
    "dotnet test *",
    "make test *",
    "make check *",
    "make lint *",
)

# These patterns remain denied even if an approval entry happens to match. They target
# host-wide privilege changes, disks, catastrophic deletion, forced remote history changes,
# and remote-script execution. Ordinary rm/git reset/package-install commands fall into the
# approval-required category instead, so a user can approve an exact, scoped command.
BLOCKED_COMMAND_PATTERNS: tuple[str, ...] = (
    "sudo *",
    "doas *",
    "su *",
    "shutdown *",
    "reboot *",
    "halt *",
    "poweroff *",
    "mkfs *",
    "fdisk *",
    "parted *",
    "diskutil erase*",
    "format *",
    "dd *",
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~*",
    "rm -rf $home*",
    "rm -fr /",
    "rm -fr /*",
    "rm --recursive --force /",
    "rm --recursive --force /*",
    "remove-item *-recurse*-force*",
    "git push --force *",
    "git push -f *",
    "curl *|*sh*",
    "curl *|*bash*",
    "wget *|*sh*",
    "wget *|*bash*",
    "powershell *invoke-expression*",
    "powershell *iex *",
    "chmod -r 777 /*",
    "chown -r * /*",
)

# A broad safe pattern such as ``git status *`` must not silently authorize a second
# command. These patterns are placed after safe defaults and before explicit approvals,
# so shell composition requires an exact, reviewed project approval.
COMPOUND_COMMAND_PATTERNS: tuple[str, ...] = (
    "*;*",
    "*&&*",
    "*||*",
    "*|*",
    "*>*",
    "*<*",
    "*$(*",
    "*`*",
    "*\n*",
)

# Safe compound patterns explicitly permitted for common tool inspection workflows
SAFE_COMPOUND_PATTERNS: tuple[str, ...] = (
    "node * && npm *",
    "node * && pnpm *",
    "node * && yarn *",
    "node * && bun *",
    "npm * && node *",
    "git status * && git diff *",
    "git add * && git commit *",
    "git add * && git status *",
    "which * && * --version*",
    "which * && * -v*",
)


@dataclass(frozen=True, slots=True)
class CommandDecision:
    timestamp: str
    command: str
    category: CommandCategory
    action: CommandAction
    matched_pattern: str | None
    explicit_user_approval: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matches(command: str, pattern: str) -> bool:
    if "\n" in pattern:
        return "\n" in command
    normalized = " ".join(command.strip().split()).lower()
    candidate = " ".join(pattern.strip().split()).lower()
    if candidate.endswith(" *") and normalized == candidate[:-2]:
        return True
    return fnmatch.fnmatchcase(normalized, candidate)


def _first_match(command: str, patterns: tuple[str, ...] | list[str]) -> str | None:
    return next((pattern for pattern in patterns if _matches(command, pattern)), None)


def _is_safe_pipeline(command: str) -> bool:
    """Allow safe commands piped to output filters (e.g. pnpm install critters 2>&1 | tail -30)."""
    dangerous_chain = ("*&&*", "*||*", "*;*", "*$(*", "*`*", "*\n*")
    if any(_matches(command, pat) for pat in dangerous_chain):
        return False
    parts = [p.strip() for p in command.split("|")]
    if len(parts) <= 1:
        return False

    def clean_part(p: str) -> str:
        return " ".join(
            p.replace("2>&1", "")
            .replace("1>&2", "")
            .replace("2>/dev/null", "")
            .replace(">/dev/null", "")
            .strip()
            .split()
        )

    return all(
        bool(_first_match(clean_part(part), SAFE_COMMAND_PATTERNS))
        for part in parts
        if clean_part(part)
    )


def _is_safe_compound(command: str, security: CommandSecurityConfig | None = None) -> bool:
    """Allow safe commands chained together (e.g. node --version && npm --version)."""
    dangerous_substitutions = ("*$(*", "*`*", "*\n*", "*>*", "*<*")
    if any(_matches(command, pat) for pat in dangerous_substitutions):
        return False
    if not any(sep in command for sep in ("&&", ";", "|")):
        return False
    import re

    parts = [p.strip() for p in re.split(r"&&|;|\|", command) if p.strip()]
    if len(parts) <= 1:
        return False

    def clean_part(p: str) -> str:
        return " ".join(
            p.replace("2>&1", "")
            .replace("1>&2", "")
            .replace("2>/dev/null", "")
            .replace(">/dev/null", "")
            .strip()
            .split()
        )

    for part in parts:
        cleaned = clean_part(part)
        if not cleaned:
            continue
        if _first_match(cleaned, _effective_blocked_patterns()):
            return False
        is_safe = bool(_first_match(cleaned, SAFE_COMMAND_PATTERNS))
        is_approved = bool(security and _first_match(cleaned, security.approved_commands))
        if not (is_safe or is_approved):
            return False
    return True


def _is_unquoted_compound(command: str) -> bool:
    """Check if command contains real unquoted compound shell operators: ;, &&, ||, |, >, <, $(, `"""
    in_single = False
    in_double = False
    escape = False
    i = 0
    n = len(command)
    while i < n:
        char = command[i]
        if escape:
            escape = False
            i += 1
            continue
        if char == "\\":
            escape = True
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single and not in_double:
            if char in (";", "\n", "`"):
                return True
            if char in (">", "<"):
                return True
            if char == "|" and (i + 1 < n and command[i + 1] == "|"):
                return True
            if char == "|":
                return True
            if char == "&" and (i + 1 < n and command[i + 1] == "&"):
                return True
            if char == "$" and (i + 1 < n and command[i + 1] == "("):
                return True
        i += 1
    return False


def _effective_blocked_patterns() -> tuple[str, ...]:
    """Match blocked operations even when hidden behind a shell prefix."""
    return tuple(
        dict.fromkeys(
            pattern
            for base in BLOCKED_COMMAND_PATTERNS
            for pattern in (base, base if base.startswith("*") else f"*{base}")
        )
    )


def _permission_patterns(patterns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Include the zero-argument spelling for patterns written with a trailing wildcard."""
    return tuple(
        dict.fromkeys(
            variant
            for pattern in patterns
            for variant in ((pattern[:-2], pattern) if pattern.endswith(" *") else (pattern,))
        )
    )


def decide_command(command: str, security: CommandSecurityConfig) -> CommandDecision:
    normalized = " ".join(command.strip().split())
    blocked = _first_match(normalized, _effective_blocked_patterns())
    if blocked:
        return CommandDecision(
            timestamp=utc_now(),
            command=normalized,
            category="blocked",
            action="deny",
            matched_pattern=blocked,
            explicit_user_approval=False,
            reason="blocked commands cannot be enabled by a worker approval",
        )
    is_compound = _is_unquoted_compound(command)
    is_safe_pipe = _is_safe_pipeline(command)
    is_safe_comp = _is_safe_compound(command, security)
    safe = (
        _first_match(normalized, SAFE_COMMAND_PATTERNS)
        if (not is_compound or is_safe_pipe or is_safe_comp)
        else None
    )
    if safe or is_safe_pipe or is_safe_comp:
        return CommandDecision(
            timestamp=utc_now(),
            command=normalized,
            category="safe_default",
            action="allow",
            matched_pattern=safe or ("safe_pipeline" if is_safe_pipe else "safe_compound"),
            explicit_user_approval=False,
            reason="matched the conservative safe-default command list",
        )
    approved = _first_match(normalized, security.approved_commands)
    if approved:
        return CommandDecision(
            timestamp=utc_now(),
            command=normalized,
            category="approval_required",
            action="allow",
            matched_pattern=approved,
            explicit_user_approval=True,
            reason="matched an explicit project-local user approval",
        )
    matched_compound = _first_match(command, COMPOUND_COMMAND_PATTERNS) if is_compound else None
    return CommandDecision(
        timestamp=utc_now(),
        command=normalized,
        category="approval_required",
        action="deny",
        matched_pattern=matched_compound,
        explicit_user_approval=False,
        reason="no explicit project-local user approval matched",
    )


def policy_document(security: CommandSecurityConfig) -> dict[str, Any]:
    return {
        "policy_version": 1,
        "enforcement": (
            "OpenCode shell permissions; unknown and approval-required commands are denied "
            "in the non-interactive worker unless explicitly approved"
        ),
        "categories": {
            "safe_default": {
                "effect": "allow",
                "patterns": list(SAFE_COMMAND_PATTERNS),
            },
            "approval_required": {
                "effect": "deny_unless_explicitly_approved",
                "approved_patterns": security.approved_commands,
                "compound_shell_patterns": list(COMPOUND_COMMAND_PATTERNS),
            },
            "blocked": {
                "effect": "deny_even_when_an_approval_matches",
                "patterns": list(BLOCKED_COMMAND_PATTERNS),
            },
        },
        "external_directory": "deny",
    }


def command_permission_config(security: CommandSecurityConfig, *, v2: bool) -> dict[str, Any]:
    if v2:
        rules: list[dict[str, str]] = [{"action": "shell", "resource": "*", "effect": "deny"}]
        rules.extend(
            {"action": "shell", "resource": pattern, "effect": "allow"}
            for pattern in _permission_patterns(SAFE_COMMAND_PATTERNS)
        )
        rules.extend(
            {"action": "shell", "resource": pattern, "effect": "deny"}
            for pattern in COMPOUND_COMMAND_PATTERNS
        )
        rules.extend(
            {"action": "shell", "resource": pattern, "effect": "allow"}
            for pattern in _permission_patterns(SAFE_COMPOUND_PATTERNS)
        )
        rules.extend(
            {"action": "shell", "resource": pattern, "effect": "allow"}
            for pattern in _permission_patterns(security.approved_commands)
        )
        rules.extend(
            {"action": "shell", "resource": pattern, "effect": "deny"}
            for pattern in _permission_patterns(_effective_blocked_patterns())
        )
        rules.append({"action": "external_directory", "resource": "*", "effect": "deny"})
        return {"permissions": rules}

    bash: dict[str, str] = {"*": "deny"}
    bash.update(dict.fromkeys(_permission_patterns(SAFE_COMMAND_PATTERNS), "allow"))
    bash.update(dict.fromkeys(COMPOUND_COMMAND_PATTERNS, "deny"))
    bash.update(dict.fromkeys(_permission_patterns(SAFE_COMPOUND_PATTERNS), "allow"))
    bash.update(dict.fromkeys(_permission_patterns(security.approved_commands), "allow"))
    bash.update(dict.fromkeys(_permission_patterns(_effective_blocked_patterns()), "deny"))
    return {"permission": {"bash": bash, "external_directory": "deny"}}


def extract_shell_commands(output: str) -> list[str]:
    commands: list[str] = []

    def visit(value: Any, shell_context: bool = False) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, shell_context)
            return
        if not isinstance(value, dict):
            return
        names = [value.get(key) for key in ("tool", "tool_name", "toolName", "name")]
        current = shell_context or any(
            isinstance(name, str) and name.lower() in {"bash", "shell"} for name in names
        )
        for key, item in value.items():
            if current and key.lower() in {"command", "cmd"} and isinstance(item, str):
                command = " ".join(item.strip().split())
                if command:
                    commands.append(command)
            elif isinstance(item, dict | list):
                visit(item, current)

    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        visit(event)
    return commands
