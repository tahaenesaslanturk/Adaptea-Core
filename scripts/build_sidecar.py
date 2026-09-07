from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
BINARIES = DESKTOP / "src-tauri" / "binaries"


def target_triple() -> str:
    rustc = shutil.which("rustc")
    if rustc:
        result = subprocess.run(
            [rustc, "--print", "host-tuple"], check=True, capture_output=True, text=True
        )
        value = result.stdout.strip()
        if value:
            return value
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "aarch64-apple-darwin" if machine in {"arm64", "aarch64"} else "x86_64-apple-darwin"
    if system == "Windows":
        return (
            "aarch64-pc-windows-msvc"
            if machine in {"arm64", "aarch64"}
            else "x86_64-pc-windows-msvc"
        )
    if system == "Linux":
        return (
            "aarch64-unknown-linux-gnu"
            if machine in {"arm64", "aarch64"}
            else "x86_64-unknown-linux-gnu"
        )
    raise RuntimeError(f"Unsupported sidecar platform: {system} {machine}")


def destination(triple: str) -> Path:
    suffix = ".exe" if "windows" in triple else ""
    return BINARIES / f"adaptea-core-{triple}{suffix}"


def placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".exe":
        path.touch()
    else:
        path.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Prepared development sidecar placeholder: {path}")


def build(path: Path) -> None:
    output = ROOT / "build" / "desktop-sidecar"
    work = ROOT / "build" / "pyinstaller"
    spec = ROOT / "build" / "pyinstaller-spec"
    for directory in (output, work, spec, path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "adaptea-core",
        "--paths",
        str(ROOT / "src"),
        "--distpath",
        str(output),
        "--workpath",
        str(work),
        "--specpath",
        str(spec),
    ]
    if platform.system() == "Darwin":
        command.extend(
            [
                "--osx-entitlements-file",
                str(DESKTOP / "src-tauri" / "Entitlements.plist"),
            ]
        )
    command.append(str(ROOT / "scripts" / "adaptea_core_entry.py"))
    subprocess.run(command, cwd=ROOT, check=True)
    built = output / ("adaptea-core.exe" if os.name == "nt" else "adaptea-core")
    if not built.is_file():
        raise RuntimeError(f"PyInstaller did not produce {built}")
    shutil.copy2(built, path)
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Built Tauri sidecar: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Adaptea's target-named Tauri sidecar.")
    parser.add_argument(
        "--dev-placeholder",
        action="store_true",
        help=(
            "Create only the target-named placeholder; debug Tauri invokes Python source directly."
        ),
    )
    args = parser.parse_args()
    path = destination(target_triple())
    placeholder(path) if args.dev_placeholder else build(path)


if __name__ == "__main__":
    main()
