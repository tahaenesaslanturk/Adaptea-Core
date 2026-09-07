"""Describe and validate the signing/notarization environment.

Every credential is read from the environment. Nothing here creates a certificate, a
key, or a keychain entry, and nothing prints a secret: values are reported only as
present/absent, so this is safe to run in CI logs and to paste into a bug report.

macOS and Windows each have two independent concerns:

  * signing        — proves who built the artifact
  * notarization   — Apple's malware scan, required before Gatekeeper will open a
                     download without the "unidentified developer" refusal

Local development needs neither. With no variables set, Tauri falls back to the ad-hoc
signature it already used, so `npm run tauri:dev` and `npm run build:mac` keep working
exactly as before.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Credential:
    name: str
    purpose: str
    #: Alternative variables that satisfy the same requirement.
    alternatives: tuple[str, ...] = ()

    def satisfied(self, environment: dict[str, str]) -> bool:
        return any(environment.get(key, "").strip() for key in (self.name, *self.alternatives))


@dataclass(frozen=True, slots=True)
class SigningProfile:
    platform: str
    signing: tuple[Credential, ...]
    notarization: tuple[Credential, ...] = ()
    tools: tuple[str, ...] = field(default=())


MACOS = SigningProfile(
    platform="macOS",
    signing=(
        Credential(
            "APPLE_SIGNING_IDENTITY",
            "Developer ID Application certificate name, "
            "e.g. 'Developer ID Application: Acme (TEAMID)'",
        ),
        Credential(
            "APPLE_CERTIFICATE",
            "base64 of the .p12 certificate; CI imports it into a temporary keychain",
        ),
        Credential("APPLE_CERTIFICATE_PASSWORD", "password for that .p12"),
    ),
    notarization=(
        # Tauri accepts either an App Store Connect API key or an Apple ID + app password.
        Credential("APPLE_API_KEY", "App Store Connect key id", ("APPLE_ID",)),
        Credential("APPLE_API_ISSUER", "App Store Connect issuer id", ("APPLE_PASSWORD",)),
        Credential("APPLE_API_KEY_PATH", "path to the .p8 key file", ("APPLE_TEAM_ID",)),
    ),
    tools=("codesign", "xcrun"),
)

WINDOWS = SigningProfile(
    platform="Windows",
    signing=(
        Credential("WINDOWS_CERTIFICATE", "base64 of the code-signing .pfx"),
        Credential("WINDOWS_CERTIFICATE_PASSWORD", "password for that .pfx"),
    ),
    tools=("signtool",),
)

#: Public key for verifying update packages. Public by definition — but still injected at
#: release time rather than committed, so the repository never implies a matching private
#: key exists somewhere in it.
UPDATER = Credential("TAURI_SIGNING_PUBLIC_KEY", "updater public key from 'tauri signer generate'")
UPDATER_PRIVATE = Credential("TAURI_SIGNING_PRIVATE_KEY", "updater private key, CI secret only")


@dataclass(frozen=True, slots=True)
class ProfileReport:
    platform: str
    signing_ready: bool
    notarization_ready: bool
    missing_signing: list[str]
    missing_notarization: list[str]
    missing_tools: list[str]

    @property
    def distributable(self) -> bool:
        """Whether an artifact built with this environment can be opened by someone else."""
        if self.platform == "macOS":
            return self.signing_ready and self.notarization_ready
        return self.signing_ready


def inspect(profile: SigningProfile, environment: dict[str, str] | None = None) -> ProfileReport:
    env = dict(os.environ if environment is None else environment)
    missing_signing = [c.name for c in profile.signing if not c.satisfied(env)]
    missing_notarization = [c.name for c in profile.notarization if not c.satisfied(env)]
    missing_tools = [tool for tool in profile.tools if shutil.which(tool) is None]
    return ProfileReport(
        platform=profile.platform,
        signing_ready=not missing_signing,
        notarization_ready=not profile.notarization or not missing_notarization,
        missing_signing=missing_signing,
        missing_notarization=missing_notarization,
        missing_tools=missing_tools,
    )


def updater_ready(environment: dict[str, str] | None = None) -> bool:
    env = dict(os.environ if environment is None else environment)
    return UPDATER.satisfied(env) and UPDATER_PRIVATE.satisfied(env)
