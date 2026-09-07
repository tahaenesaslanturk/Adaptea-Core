"""Set the release version everywhere it is written by hand.

Five files carry the version and every one of them has to agree: the Python package
reports it, the desktop app shows it in Settings, and Tauri stamps the bundle and the
updater feed from its own copy. Bumping four of five ships an installer whose name,
about box, and update check disagree, so this writes all of them in one pass. The
lockfiles are not touched; `uv sync` and the next cargo build regenerate those.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Each entry is the file, a pattern matching only its own version line, and the
# replacement. Anchored patterns keep a dependency that happens to share the number from
# being rewritten along with it.
EDITS: tuple[tuple[Path, str, str], ...] = (
    (ROOT / "pyproject.toml", r'(?m)^version = "[^"]+"$', 'version = "{version}"'),
    (ROOT / "src/adaptea/__init__.py", r'(?m)^__version__ = "[^"]+"$', '__version__ = "{version}"'),
    (ROOT / "desktop/package.json", r'(?m)^  "version": "[^"]+",$', '  "version": "{version}",'),
    (
        ROOT / "desktop/src-tauri/tauri.conf.json",
        r'(?m)^  "version": "[^"]+",$',
        '  "version": "{version}",',
    ),
    (ROOT / "desktop/src-tauri/Cargo.toml", r'(?m)^version = "[^"]+"$', 'version = "{version}"'),
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: uv run python scripts/set_version.py <version>", file=sys.stderr)
        return 2
    version = argv[1].lstrip("v")
    # Tauri rejects anything that is not major.minor.patch, so a typo is caught here
    # rather than after a full build.
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(f"'{version}' is not a major.minor.patch version.", file=sys.stderr)
        return 2
    for path, pattern, template in EDITS:
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(pattern, template.format(version=version), text, count=1)
        if count != 1:
            print(f"No version line found in {path.relative_to(ROOT)}.", file=sys.stderr)
            return 1
        path.write_text(updated, encoding="utf-8")
        print(f"{path.relative_to(ROOT)} → {version}")
    print("\nLockfiles follow on the next `uv sync` and cargo build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
