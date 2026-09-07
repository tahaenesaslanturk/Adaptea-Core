"""Configure the signed update feed at release time, without storing keys in the repo.

Tauri verifies every update package against a public key held in tauri.conf.json. The key
is public, but committing it implies a matching private key belongs to this repository,
and it also silently enables the updater for local builds. Instead the committed config
carries no updater block at all, and a release job injects one from the environment just
before `tauri build`, then restores the file.

This never generates a key pair. Run `npx tauri signer generate` yourself, keep the
private key in your CI secret store, and expose the public half as
TAURI_SIGNING_PUBLIC_KEY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UpdaterSettings:
    public_key: str
    endpoints: tuple[str, ...]

    def validate(self) -> None:
        if not self.public_key.strip():
            raise ValueError("TAURI_SIGNING_PUBLIC_KEY is empty; run 'tauri signer generate' first")
        if not self.endpoints:
            raise ValueError("at least one update endpoint is required")
        for endpoint in self.endpoints:
            # An update feed served over plain HTTP can be swapped in transit. The
            # signature check would still catch a forged package, but the version and
            # release notes the user is shown would not be trustworthy.
            if not endpoint.startswith("https://"):
                raise ValueError(f"update endpoint must use https: {endpoint}")


def apply(config_path: Path, settings: UpdaterSettings) -> str:
    """Write the updater block into tauri.conf.json. Returns the original text."""
    settings.validate()
    original = config_path.read_text(encoding="utf-8")
    config = json.loads(original)
    config.setdefault("plugins", {})["updater"] = {
        "pubkey": settings.public_key.strip(),
        "endpoints": list(settings.endpoints),
        # Only install packages this key signed; the default, restated so it is visible.
        "windows": {"installMode": "passive"},
    }
    config.setdefault("bundle", {})["createUpdaterArtifacts"] = True
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return original


def restore(config_path: Path, original: str) -> None:
    """Put the committed configuration back, so no key survives in the working tree."""
    config_path.write_text(original, encoding="utf-8")
