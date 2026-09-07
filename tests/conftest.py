"""Test-wide isolation for Adaptea's per-user state.

The inference environment, the saved model combinations and the recent-projects list all
live in one per-user directory. Without this, running the suite would read and rewrite the
developer's own Adaptea configuration — and, worse, tests would see each other's state
through it, which is exactly the kind of shared mutable path that makes a suite pass or
fail depending on what ran before it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_state_directory(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    state = tmp_path_factory.mktemp("adaptea-state")
    monkeypatch.setenv("ADAPTEA_STATE_DIR", str(state))
    yield state
