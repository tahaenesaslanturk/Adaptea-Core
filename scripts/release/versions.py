"""Keep every version source in the repository in agreement.

Adaptea states its version in five places — the Python package, its ``__init__``, the
desktop package manifest, the Tauri bundle config, and the Rust crate. A release that
ships them out of step produces installers whose filename, About box, and update feed
disagree, so this module is the single place that reads and writes all of them.

No secrets are involved; this is pure repository metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A release version is plain semver. Tauri rejects pre-release suffixes in bundle
# versions on Windows, so the repository refuses them everywhere rather than failing
# halfway through a release.
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, slots=True)
class VersionSource:
    name: str
    path: Path
    version: str | None


def _json_version(path: Path, *keys: str) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)  # type: ignore[assignment]
    return data if isinstance(data, str) else None


def _regex_version(path: Path, pattern: re.Pattern[str]) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = pattern.search(text)
    return match.group(1) if match else None


_PYPROJECT = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
_DUNDER = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
# Only the first [package] version in Cargo.toml, not a dependency's.
_CARGO = re.compile(r'^\[package\][^\[]*?^version\s*=\s*"([^"]+)"', re.M | re.S)


def collect(root: Path = ROOT) -> list[VersionSource]:
    sources = [
        VersionSource(
            "pyproject.toml",
            root / "pyproject.toml",
            _regex_version(root / "pyproject.toml", _PYPROJECT),
        ),
        VersionSource(
            "adaptea.__version__",
            root / "src/adaptea/__init__.py",
            _regex_version(root / "src/adaptea/__init__.py", _DUNDER),
        ),
    ]
    if (root / "desktop").exists():
        sources.extend(
            [
                VersionSource(
                    "desktop/package.json",
                    root / "desktop/package.json",
                    _json_version(root / "desktop/package.json", "version"),
                ),
                VersionSource(
                    "tauri.conf.json",
                    root / "desktop/src-tauri/tauri.conf.json",
                    _json_version(root / "desktop/src-tauri/tauri.conf.json", "version"),
                ),
                VersionSource(
                    "src-tauri/Cargo.toml",
                    root / "desktop/src-tauri/Cargo.toml",
                    _regex_version(root / "desktop/src-tauri/Cargo.toml", _CARGO),
                ),
            ]
        )
    return sources


def disagreements(sources: list[VersionSource]) -> dict[str, list[str]]:
    """Group source names by the version they declare; more than one group is a failure."""
    grouped: dict[str, list[str]] = {}
    for source in sources:
        grouped.setdefault(source.version or "<unreadable>", []).append(source.name)
    return grouped


def set_version(version: str, root: Path = ROOT) -> list[str]:
    """Write `version` into every source. Returns the files that changed."""
    if not SEMVER.match(version):
        raise ValueError(f"{version!r} is not a plain MAJOR.MINOR.PATCH version")
    changed: list[str] = []
    for source in collect(root):
        if source.version == version or not source.path.is_file():
            continue
        text = source.path.read_text(encoding="utf-8")
        if source.path.suffix == ".json":
            # Rewrite only the top-level "version" line so formatting elsewhere survives.
            updated = re.sub(r'("version"\s*:\s*)"[^"]+"', rf'\1"{version}"', text, count=1)
        elif source.name == "adaptea.__version__":
            updated = _DUNDER.sub(f'__version__ = "{version}"', text, count=1)
        elif source.name == "src-tauri/Cargo.toml":
            match = _CARGO.search(text)
            if not match:
                continue
            start, end = match.span(1)
            updated = text[:start] + version + text[end:]
        else:
            updated = _PYPROJECT.sub(f'version = "{version}"', text, count=1)
        if updated != text:
            source.path.write_text(updated, encoding="utf-8")
            changed.append(source.name)
    return changed
