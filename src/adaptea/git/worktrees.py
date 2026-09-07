from __future__ import annotations

import asyncio
from pathlib import Path

from adaptea.git.repository import git, sanitize_branch


class WorktreeManager:
    def __init__(self, repository: Path, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id
        self.root = repository / ".adaptea" / "worktrees" / run_id
        self.integration_path = self.root / "integration"
        self.integration_branch = sanitize_branch(f"adaptea/{run_id}/integration", 100)
        self._lock = asyncio.Lock()

    async def create_integration(self) -> Path:
        async with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.integration_path.exists():
                return self.integration_path
            branch_exists = (
                await git(
                    self.repository,
                    "show-ref",
                    "--verify",
                    f"refs/heads/{self.integration_branch}",
                    check=False,
                )
            ).returncode == 0
            args = ["worktree", "add"]
            if branch_exists:
                args.extend([str(self.integration_path), self.integration_branch])
            else:
                args.extend(["-b", self.integration_branch, str(self.integration_path), "HEAD"])
            await git(self.repository, *args)
            return self.integration_path

    async def create_task(self, task_id: str, attempt: int) -> tuple[Path, str]:
        async with self._lock:
            safe = sanitize_branch(task_id)
            branch = sanitize_branch(f"adaptea/{self.run_id}/{safe}-a{attempt}", 100)
            path = self.root / f"{safe}-a{attempt}"
            if path.exists():
                return path, branch
            await git(
                self.repository,
                "worktree",
                "add",
                "-b",
                branch,
                str(path),
                self.integration_branch,
            )
            return path, branch

    def resolution_spec(self, task_id: str, attempt: int) -> tuple[Path, str]:
        safe = sanitize_branch(task_id)
        branch = sanitize_branch(f"adaptea/{self.run_id}/resolve-{safe}-a{attempt}", 100)
        path = self.root / f"resolution-{safe}-a{attempt}"
        return path, branch

    async def create_resolution(self, task_id: str, attempt: int) -> tuple[Path, str]:
        """Create a dedicated branch/worktree based on the clean integration branch."""
        async with self._lock:
            path, branch = self.resolution_spec(task_id, attempt)
            if path.exists():
                return path, branch
            branch_exists = (
                await git(
                    self.repository,
                    "show-ref",
                    "--verify",
                    f"refs/heads/{branch}",
                    check=False,
                )
            ).returncode == 0
            args = ["worktree", "add"]
            if branch_exists:
                args.extend([str(path), branch])
            else:
                args.extend(["-b", branch, str(path), self.integration_branch])
            await git(self.repository, *args)
            return path, branch

    async def remove(self, path: Path) -> None:
        async with self._lock:
            if path.exists():
                await git(self.repository, "worktree", "remove", "--force", str(path), check=False)
            await git(self.repository, "worktree", "prune", check=False)
            # Git removes the checkout itself but leaves Adaptea's run container behind.
            # Drop only empty directories; failed/manual-resolution worktrees remain intact.
            for directory in (path.parent, self.root):
                try:
                    directory.rmdir()
                except OSError:
                    pass
