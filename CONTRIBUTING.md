# Contributing to Adaptea Core

Thank you for your interest in contributing to **Adaptea Core**! Adaptea Core is an open-source adaptive admission controller and multi-agent orchestrator for local coding models.

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`
- Git

### Initializing the environment

```bash
git clone https://github.com/tahaenesaslanturk/Adaptea-Core.git
cd Adaptea-Core
uv sync --locked
```

## Running Tests & Quality Checks

Before submitting a Pull Request, ensure all linters, type checks, and test suites pass:

```bash
# Code formatting check
uv run ruff format --check .

# Linting
uv run ruff check .

# Static type checking
uv run python -m mypy

# Test suite
uv run python -m pytest
```

To automatically format your code:
```bash
uv run ruff format .
uv run ruff check --fix .
```

## Code Guidelines

- **Architecture Boundary**: Adaptea Core is a headless engine. It provides the Python API (`ApplicationServices`), CLI/TUI (`adaptea`), and JSON-RPC / NDJSON stdio server (`adaptea-core`) for frontend clients (like the Adaptea Desktop app).
- **Type Safety**: All code must pass `mypy --strict` with zero type errors.
- **Async Execution**: Subprocesses and HTTP requests should be non-blocking using `asyncio` and `httpx`.
- **Git Worktree Isolation**: Changes to workers should respect isolated Git worktrees and deterministic validation gating.

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Commit your changes with clear, semantic commit messages.
3. Verify all tests pass locally.
4. Open a Pull Request on GitHub with a description of the problem solved and test coverage.
