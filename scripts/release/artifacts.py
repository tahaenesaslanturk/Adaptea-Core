"""Verify what a release build actually produced.

A green build is not a shippable release. These checks run against the bundle directory
after `tauri build` and answer the questions that otherwise only surface once a user
double-clicks the download:

  * is every expected installer present, and does its name carry the release version?
  * does the .app contain the Python sidecar, or did the bundle ship without a core?
  * is the code signature intact, and is it a real Developer ID rather than ad-hoc?
  * is the notarization ticket stapled, so Gatekeeper opens it offline?
  * do the recorded checksums match what is on disk?

Everything is read-only. Nothing here signs, notarizes, uploads, or mutates a build.
"""

from __future__ import annotations

import hashlib
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Finding:
    level: str  # PASS | WARN | FAIL
    name: str
    detail: str

    @property
    def blocking(self) -> bool:
        return self.level == "FAIL"


def _run(*command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        # Cross-platform verification can inspect the opposite platform's incomplete
        # bundle in tests and diagnostics. A missing codesign or PowerShell executable is
        # a failed probe to report, not a reason for the verifier itself to crash.
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_document(paths: list[Path]) -> str:
    """A sha256sum-compatible manifest, so a downloader can verify without our tooling."""
    return "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(paths, key=lambda p: p.name))


_LATEST_ASSET_NAMES = {
    "macos": "Adaptea-macos-arm64.dmg",
    "windows": "Adaptea-windows-x64-setup.exe",
}


def prepare_latest_alias(bundle: Path, version: str, platform: str) -> tuple[Path, Path]:
    """Copy a versioned installer to the stable name used by the website.

    GitHub's ``releases/latest/download`` URL only stays stable when the asset name does.
    Release artifacts keep their versioned names for traceability; this additional byte-for-byte
    alias lets the website point at the newest published release without a site deployment.
    """
    if platform not in _LATEST_ASSET_NAMES:
        raise ValueError(f"unsupported release platform: {platform}")
    pattern = "dmg/*.dmg" if platform == "macos" else "nsis/*.exe"
    candidates = [path for path in bundle.glob(pattern) if version in path.name]
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise ValueError(
            f"expected exactly one {platform} installer carrying {version}; found {names}"
        )

    destination = bundle / _LATEST_ASSET_NAMES[platform]
    shutil.copy2(candidates[0], destination)
    checksum = destination.with_name(f"{destination.name}.sha256")
    checksum.write_text(checksum_document([destination]), encoding="utf-8")
    return destination, checksum


def verify_checksums(document: str, directory: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line in document.splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition("  ")
        candidate = directory / name.strip()
        if not candidate.is_file():
            findings.append(
                Finding("FAIL", f"checksum {name.strip()}", "listed but missing on disk")
            )
            continue
        actual = sha256(candidate)
        findings.append(
            Finding("PASS", f"checksum {name.strip()}", "matches")
            if actual == expected.strip()
            else Finding(
                "FAIL",
                f"checksum {name.strip()}",
                f"expected {expected.strip()[:16]}…, got {actual[:16]}…",
            )
        )
    return findings


def verify_version_naming(paths: list[Path], version: str) -> list[Finding]:
    """Confirm the release version is present, and that nothing older is shipping with it.

    A bundle directory is reused between builds, so installers from previous versions
    survive there. Uploading the directory wholesale would publish all of them, and the
    newest file is not necessarily the one a download page links to.
    """
    current = [path for path in paths if version in path.name]
    stale = [path for path in paths if version not in path.name]
    findings: list[Finding] = []
    if current:
        findings.append(
            Finding(
                "PASS",
                "release artifacts",
                f"{len(current)} carrying {version}: "
                + ", ".join(sorted(path.name for path in current)),
            )
        )
    else:
        findings.append(
            Finding(
                "FAIL", "release artifacts", f"no installer in this directory carries {version}"
            )
        )
    if stale:
        findings.append(
            Finding(
                "FAIL",
                "stale artifacts",
                "left over from an earlier build and would be published alongside this "
                f"release: {', '.join(sorted(path.name for path in stale))}. Build into a "
                "clean bundle directory.",
            )
        )
    return findings


def verify_macos_app(app: Path, *, expect_developer_id: bool) -> list[Finding]:
    findings: list[Finding] = []
    if not app.is_dir():
        return [Finding("FAIL", "app bundle", f"{app} is missing")]

    # The sidecar is the Python core. Without it the app launches to a dead bridge.
    sidecar = app / "Contents" / "MacOS" / "adaptea-core"
    findings.append(
        Finding("PASS", "sidecar", "adaptea-core present and executable")
        if sidecar.is_file() and sidecar.stat().st_mode & 0o111
        else Finding("FAIL", "sidecar", "adaptea-core missing or not executable")
    )

    info = app / "Contents" / "Info.plist"
    if info.is_file():
        try:
            plist = plistlib.loads(info.read_bytes())
            findings.append(
                Finding("PASS", "bundle version", str(plist.get("CFBundleShortVersionString")))
            )
            minimum = plist.get("LSMinimumSystemVersion")
            findings.append(
                Finding("PASS", "minimum macOS", str(minimum))
                if minimum
                else Finding("WARN", "minimum macOS", "not declared")
            )
        except (plistlib.InvalidFileException, ValueError) as exc:
            findings.append(Finding("FAIL", "Info.plist", f"unreadable: {exc}"))
    else:
        findings.append(Finding("FAIL", "Info.plist", "missing"))

    signature = _run("codesign", "--verify", "--deep", "--strict", str(app))
    if expect_developer_id:
        findings.append(
            Finding("PASS", "code signature", "valid on disk")
            if signature.returncode == 0
            else Finding(
                "FAIL",
                "code signature",
                signature.stderr.strip().splitlines()[-1]
                if signature.stderr.strip()
                else "invalid",
            )
        )
    else:
        findings.append(
            Finding("PASS", "code signature", "valid on disk")
            if signature.returncode == 0
            else Finding(
                "WARN",
                "code signature",
                "unsigned or unverified (expected on CI runner without signing cert)",
            )
        )

    authority = _run("codesign", "--display", "--verbose=2", str(app))
    text = authority.stderr + authority.stdout
    adhoc = "Signature=adhoc" in text
    developer_id = "Developer ID Application" in text
    if expect_developer_id:
        findings.append(
            Finding("PASS", "signing identity", "Developer ID Application")
            if developer_id
            else Finding(
                "FAIL", "signing identity", "ad-hoc signature; this build cannot be distributed"
            )
        )
    else:
        findings.append(
            Finding("WARN", "signing identity", "ad-hoc; fine locally, not distributable")
            if adhoc or not developer_id
            else Finding("PASS", "signing identity", "Developer ID Application")
        )

    staple = _run("xcrun", "stapler", "validate", str(app))
    if expect_developer_id:
        findings.append(
            Finding("PASS", "notarization", "ticket stapled")
            if staple.returncode == 0
            else Finding(
                "FAIL", "notarization", "no stapled ticket; Gatekeeper will refuse this download"
            )
        )
    else:
        findings.append(
            Finding("WARN", "notarization", "not notarized; expected for a local build")
        )
    return findings


def verify_windows_installer(installer: Path, *, expect_signature: bool) -> list[Finding]:
    findings: list[Finding] = []
    if not installer.is_file():
        return [Finding("FAIL", "installer", f"{installer} is missing")]
    findings.append(
        Finding("PASS", "installer", f"{installer.name} ({installer.stat().st_size // 1024} KiB)")
    )
    probe = _run(
        "powershell",
        "-NoProfile",
        "-Command",
        f"(Get-AuthenticodeSignature '{installer}').Status",
    )
    status = probe.stdout.strip() or "Unavailable"
    if expect_signature:
        findings.append(
            Finding("PASS", "authenticode", status)
            if status == "Valid"
            else Finding("FAIL", "authenticode", f"signature status is {status}")
        )
    else:
        findings.append(Finding("WARN", "authenticode", f"unsigned local build ({status})"))
    return findings
