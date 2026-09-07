from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test, package the Python core, and build the native Adaptea desktop app."
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()
    if not args.skip_tests:
        run("uv", "run", "ruff", "format", "--check", ".")
        run("uv", "run", "ruff", "check", ".")
        # Invoke Python tools through uv's active interpreter instead of their generated
        # console scripts. Those scripts contain absolute shebangs and stop working when
        # a checkout (and its existing .venv) is moved to a different directory.
        run("uv", "run", "python", "-m", "mypy")
        run("uv", "run", "python", "-m", "pytest")
        run("npm", "run", "lint", cwd=DESKTOP)
        run("npm", "run", "test", cwd=DESKTOP)
    run("npm", "run", "build", cwd=DESKTOP)
    if args.frontend_only:
        return
    if platform.system() == "Darwin":
        run("npm", "run", "build:mac", cwd=DESKTOP)
    elif platform.system() == "Windows":
        run("uv", "run", "python", "scripts/build_sidecar.py")
        run("npm", "run", "tauri", "--", "build", "--bundles", "nsis,msi", cwd=DESKTOP)
    else:
        raise RuntimeError("Desktop installers are currently scoped to macOS and Windows.")


if __name__ == "__main__":
    main()
