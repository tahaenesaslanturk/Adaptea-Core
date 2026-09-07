from __future__ import annotations

from pathlib import Path

from adaptea.git.repository import (
    ADAPTEA_GIT_AUTHOR_EMAIL,
    ADAPTEA_GIT_AUTHOR_NAME,
    GitResult,
    git,
)


async def commit_worker_changes(worktree: Path, task_id: str) -> str:
    await git(worktree, "add", "--all")
    changed = await git(worktree, "diff", "--cached", "--quiet", check=False)
    if changed.returncode == 0:
        # A no-op task can still be valid if upstream already satisfied it.
        return (await git(worktree, "rev-parse", "HEAD")).stdout.strip()
    await git(
        worktree,
        "-c",
        f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
        "-c",
        f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
        "commit",
        "-m",
        f"adaptea: complete {task_id}",
    )
    return (await git(worktree, "rev-parse", "HEAD")).stdout.strip()


async def merge_branch(integration: Path, branch: str) -> GitResult:
    result = await git(
        integration,
        "-c",
        f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
        "-c",
        f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
        "merge",
        "--no-ff",
        "--no-edit",
        branch,
        check=False,
    )
    if result.returncode != 0:
        unmerged = await git(
            integration,
            "diff",
            "--name-only",
            "--diff-filter=U",
            check=False,
        )
        result.conflicting_files = sorted(
            {line.strip() for line in unmerged.stdout.splitlines() if line.strip()}
        )
        await git(integration, "merge", "--abort", check=False)
    return result


async def create_squashed_commit(
    repository: Path,
    integration_branch: str,
    parent_commit: str,
    message: str,
) -> str:
    """Collapse a completed integration branch into one user-facing commit.

    Workers still commit and merge independently on Adaptea's private branches so runs
    remain resumable and auditable. The project branch should not inherit that machinery,
    though: users asked for one change and should see one commit after the whole run passes.
    ``commit-tree`` lets us reuse the reviewed final tree with the project's current HEAD as
    its sole parent, without checking out or rewriting either internal branch.
    """
    ancestor = await git(
        repository,
        "merge-base",
        "--is-ancestor",
        parent_commit,
        integration_branch,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "The project branch advanced independently after this run started; "
            "the reviewed result cannot be squashed onto it safely."
        )

    changed = await git(
        repository,
        "diff",
        "--quiet",
        parent_commit,
        integration_branch,
        check=False,
    )
    if changed.returncode == 0:
        return parent_commit
    if changed.returncode != 1:
        raise RuntimeError("Git could not compare the completed run with the project branch.")

    tree = (await git(repository, "rev-parse", f"{integration_branch}^{{tree}}")).stdout.strip()
    commit = await git(
        repository,
        "-c",
        f"user.name={ADAPTEA_GIT_AUTHOR_NAME}",
        "-c",
        f"user.email={ADAPTEA_GIT_AUTHOR_EMAIL}",
        "commit-tree",
        tree,
        "-p",
        parent_commit,
        "-m",
        message,
    )
    return commit.stdout.strip()
