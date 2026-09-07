from __future__ import annotations

import asyncio
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from adaptea.config import load_config
from adaptea.git.repository import ADAPTEA_GIT_AUTHOR_EMAIL, ADAPTEA_GIT_AUTHOR_NAME, git


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    path: Path
    name: str
    is_git: bool
    git_branch: str | None
    git_status: str
    project_type: str
    has_config: bool
    # The desktop must never give up on a model-backed command before the core does,
    # so it reads the core's own planning deadline from here rather than assuming one.
    planner_timeout_seconds: int = 1800


@dataclass(frozen=True, slots=True)
class RecentProject:
    path: str
    opened_at: str


def user_state_directory(
    system: str | None = None, environment: dict[str, str] | None = None
) -> Path:
    """Return a platform-native per-user state directory without assuming Unix paths."""
    env = os.environ if environment is None else environment
    if override := env.get("ADAPTEA_STATE_DIR"):
        return Path(override).expanduser()
    operating_system = system or platform.system()
    if operating_system == "Windows":
        base = env.get("LOCALAPPDATA") or env.get("APPDATA")
        return Path(base) / "Adaptea" if base else Path.home() / "Adaptea"
    if operating_system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Adaptea"
    base = env.get("XDG_STATE_HOME")
    return Path(base) / "adaptea" if base else Path.home() / ".local" / "state" / "adaptea"


class RecentProjects:
    def __init__(self, path: Path | None = None, *, limit: int = 12) -> None:
        self.path = path or user_state_directory() / "recent-projects.json"
        self.limit = limit

    def load(self) -> list[RecentProject]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(value, list):
            return []
        projects: list[RecentProject] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            raw_path = row.get("path")
            opened_at = row.get("opened_at")
            if not isinstance(raw_path, str) or not isinstance(opened_at, str):
                continue
            candidate = Path(raw_path).expanduser()
            if candidate.is_dir():
                projects.append(RecentProject(str(candidate), opened_at))
        return projects[: self.limit]

    def remember(self, project: Path) -> None:
        resolved = project.expanduser().resolve()
        rows = [row for row in self.load() if Path(row.path) != resolved]
        rows.insert(0, RecentProject(str(resolved), datetime.now(UTC).isoformat()))
        self._save(rows)

    def forget(self, project: Path) -> bool:
        resolved = project.expanduser().resolve()
        rows = self.load()
        remaining = [row for row in rows if Path(row.path).resolve() != resolved]
        if len(remaining) == len(rows):
            return False
        self._save(remaining)
        return True

    def _save(self, rows: list[RecentProject]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([asdict(row) for row in rows[: self.limit]], indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def filesystem_roots(
    system: str | None = None, environment: dict[str, str] | None = None
) -> list[Path]:
    operating_system = system or platform.system()
    env = os.environ if environment is None else environment
    if operating_system == "Windows":
        listdrives = getattr(os, "listdrives", None)
        if listdrives is not None:
            return [Path(drive) for drive in listdrives()]
        roots = [Path(f"{letter}:\\") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        existing = [root for root in roots if root.exists()]
        if existing:
            return existing
        home = env.get("USERPROFILE")
        if home:
            anchor = Path(home).anchor
            if anchor:
                return [Path(anchor)]
        return [Path("C:\\")]
    return [Path("/")]


def child_directories(path: Path) -> list[Path]:
    try:
        return sorted(
            (item for item in path.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        return []


def detect_project_type(root: Path) -> str:
    markers = (
        ("pyproject.toml", "Python"),
        ("package.json", "JavaScript / TypeScript"),
        ("Cargo.toml", "Rust"),
        ("go.mod", "Go"),
        ("pom.xml", "Java / Maven"),
        ("build.gradle", "Java / Gradle"),
        ("Package.swift", "Swift"),
        ("*.sln", ".NET"),
    )
    for marker, project_type in markers:
        if any(root.glob(marker)):
            return project_type
    return "Empty / unrecognized" if not any(root.iterdir()) else "General"


def ensure_project_configuration(root: Path) -> None:
    """Ensure project .gitignore ignores .adaptea/ and auto-create opencode.json if missing."""
    ensure_adaptea_ignored(root)
    opencode_json = root / "opencode.json"
    opencode_jsonc = root / "opencode.jsonc"
    if not opencode_json.is_file() and not opencode_jsonc.is_file():
        from adaptea.setup.configuration import merge_opencode_config

        config = load_config(root)
        backend = config.inference.backend
        model = config.lmstudio.model if backend == "lmstudio" else config.ollama.model
        base_url = config.lmstudio.base_url if backend == "lmstudio" else config.ollama.base_url
        try:
            merge_opencode_config(
                root,
                config.worker.executable,
                model or "qwen2.5-coder-14b",
                base_url,
                backend=backend,
            )
        except Exception:
            pass


async def inspect_project(root: Path) -> ProjectInfo:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Project folder does not exist: {resolved}")
    ensure_project_configuration(resolved)
    probe = await git(resolved, "rev-parse", "--show-toplevel", check=False)
    is_git = probe.returncode == 0
    branch: str | None = None
    status = "Not a Git repository"
    if is_git:
        branch_result, dirty = await asyncio.gather(
            git(resolved, "branch", "--show-current", check=False),
            git(resolved, "status", "--porcelain", check=False),
        )
        branch = branch_result.stdout.strip() or "detached HEAD"
        status = "Changes present" if dirty.stdout.strip() else "Clean"
    return ProjectInfo(
        path=resolved,
        name=resolved.name or str(resolved),
        is_git=is_git,
        git_branch=branch,
        git_status=status,
        project_type=detect_project_type(resolved),
        has_config=(resolved / "adaptea.toml").is_file(),
        planner_timeout_seconds=load_config(resolved).worker.planner_timeout,
    )


def validate_project_name(name: str) -> str:
    value = name.strip()
    if not value or value in {".", ".."}:
        raise ValueError("Enter a project name.")
    if Path(value).name != value or any(separator in value for separator in ("/", "\\")):
        raise ValueError("Project name must be one folder name, not a path.")
    if re.search(r'[<>:"|?*\x00-\x1f]', value):
        raise ValueError("Project name contains characters unsupported on Windows.")
    if value.rstrip(". ") != value:
        raise ValueError("Project name cannot end with a dot or space.")
    return value


def ensure_adaptea_ignored(root: Path) -> bool:
    """Keep project-local runtime state out of source control.

    Returns true when the ignore file changed. Existing project rules and comments are
    preserved verbatim; this only appends the one repository-wide runtime rule.
    """
    ignore = root / ".gitignore"
    existing = ignore.read_text(encoding="utf-8") if ignore.is_file() else ""
    rules = {
        line.strip()
        for line in existing.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if rules.intersection({".adaptea", ".adaptea/", "/.adaptea", "/.adaptea/"}):
        return False
    separator = "" if not existing or existing.endswith("\n") else "\n"
    ignore.write_text(existing + separator + ".adaptea/\n", encoding="utf-8")
    return True


async def initialize_repository(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / ".git").exists():
        ensure_adaptea_ignored(root)
        return
    result = await git(root, "init", "-b", "main", check=False)
    if result.returncode != 0:
        result = await git(root, "init", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Git initialization failed: {result.stderr.strip()}")
    ensure_project_configuration(root)
    await git(root, "add", ".")
    commit = await git(
        root,
        "-c",
        f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
        "-c",
        f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
        "commit",
        "--allow-empty",
        "-m",
        "Initialize project",
        check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"Initial Git commit failed: {commit.stderr.strip()}")


async def create_project(parent: Path, name: str) -> ProjectInfo:
    project_name = validate_project_name(name)
    location = parent.expanduser().resolve()
    if not location.is_dir():
        raise ValueError(f"Parent folder does not exist: {location}")
    root = location / project_name
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Project folder already exists and is not empty: {root}")
    root.mkdir(exist_ok=True)
    await initialize_repository(root)
    return await inspect_project(root)
