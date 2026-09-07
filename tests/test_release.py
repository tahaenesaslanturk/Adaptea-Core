"""Tests for the release tooling: version consistency, signing preflight, verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release import __main__ as release_cli
from scripts.release import artifacts, signing, updater, versions

# --- version sources ---------------------------------------------------------------


def _repo(tmp_path: Path, version: str = "1.2.3") -> Path:
    (tmp_path / "src/adaptea").mkdir(parents=True)
    (tmp_path / "desktop/src-tauri").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "adaptea"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "src/adaptea/__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "desktop/package.json").write_text(
        json.dumps({"name": "adaptea-desktop", "version": version}, indent=2), encoding="utf-8"
    )
    (tmp_path / "desktop/src-tauri/tauri.conf.json").write_text(
        json.dumps({"productName": "Adaptea", "version": version}, indent=2), encoding="utf-8"
    )
    (tmp_path / "desktop/src-tauri/Cargo.toml").write_text(
        f'[package]\nname = "adaptea-desktop"\nversion = "{version}"\n\n'
        '[dependencies]\ntauri = { version = "2" }\n',
        encoding="utf-8",
    )
    return tmp_path


def test_every_version_source_is_discovered(tmp_path: Path) -> None:
    sources = versions.collect(_repo(tmp_path))
    assert {source.name for source in sources} == {
        "pyproject.toml",
        "adaptea.__version__",
        "desktop/package.json",
        "tauri.conf.json",
        "src-tauri/Cargo.toml",
    }
    assert all(source.version == "1.2.3" for source in sources), sources


def test_the_real_repository_agrees_with_itself() -> None:
    """A release built from disagreeing sources ships mismatched installers."""
    grouped = versions.disagreements(versions.collect())
    assert len(grouped) == 1, grouped
    assert "<unreadable>" not in grouped


def test_setting_a_version_updates_every_source(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    changed = versions.set_version("2.0.0", root)
    assert len(changed) == 5
    assert all(source.version == "2.0.0" for source in versions.collect(root))
    # Idempotent: a second run has nothing left to change.
    assert versions.set_version("2.0.0", root) == []


def test_cargo_dependency_versions_are_not_mistaken_for_the_package_version(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    versions.set_version("3.1.4", root)
    cargo = (root / "desktop/src-tauri/Cargo.toml").read_text()
    assert 'version = "3.1.4"' in cargo
    assert 'tauri = { version = "2" }' in cargo, "the dependency pin must survive"


@pytest.mark.parametrize("bad", ["0.2.0-rc1", "1.2", "v1.2.3", "", "1.2.3.4"])
def test_only_plain_semver_is_accepted(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        versions.set_version(bad, _repo(tmp_path))


# --- signing preflight -------------------------------------------------------------


def test_an_unconfigured_machine_is_reported_as_local_only() -> None:
    report = signing.inspect(signing.MACOS, {})
    assert not report.signing_ready
    assert not report.distributable
    assert "APPLE_SIGNING_IDENTITY" in report.missing_signing


def test_macos_needs_notarization_as_well_as_a_signature() -> None:
    signed_only = {
        "APPLE_SIGNING_IDENTITY": "Developer ID Application: Acme (TEAM)",
        "APPLE_CERTIFICATE": "base64",
        "APPLE_CERTIFICATE_PASSWORD": "pw",
    }
    report = signing.inspect(signing.MACOS, signed_only)
    assert report.signing_ready
    # Signed but not notarized still trips Gatekeeper on a downloaded copy.
    assert not report.notarization_ready
    assert not report.distributable


def test_either_notarization_credential_family_satisfies_macos() -> None:
    apple_id = {
        "APPLE_SIGNING_IDENTITY": "Developer ID Application: Acme (TEAM)",
        "APPLE_CERTIFICATE": "base64",
        "APPLE_CERTIFICATE_PASSWORD": "pw",
        "APPLE_ID": "release@example.com",
        "APPLE_PASSWORD": "app-specific",
        "APPLE_TEAM_ID": "TEAM",
    }
    assert signing.inspect(signing.MACOS, apple_id).distributable


def test_windows_is_distributable_on_a_signature_alone() -> None:
    report = signing.inspect(
        signing.WINDOWS, {"WINDOWS_CERTIFICATE": "b64", "WINDOWS_CERTIFICATE_PASSWORD": "pw"}
    )
    assert report.distributable


def test_preflight_never_exposes_credential_values() -> None:
    """Preflight output goes into CI logs, so it may name variables but never read them."""
    secret = "S3CRET-VALUE-DO-NOT-LEAK"
    report = signing.inspect(signing.WINDOWS, {"WINDOWS_CERTIFICATE": secret})
    assert secret not in repr(report)
    assert secret not in " ".join(report.missing_signing)


def test_updater_requires_both_halves_of_the_key() -> None:
    assert not signing.updater_ready({"TAURI_SIGNING_PUBLIC_KEY": "pub"})
    assert signing.updater_ready(
        {"TAURI_SIGNING_PUBLIC_KEY": "pub", "TAURI_SIGNING_PRIVATE_KEY": "priv"}
    )


# --- artifact verification ---------------------------------------------------------


def test_stale_installers_from_an_earlier_build_are_a_blocking_finding(tmp_path: Path) -> None:
    current = tmp_path / "Adaptea_0.2.0_aarch64.dmg"
    stale = tmp_path / "Adaptea_0.1.9_aarch64.dmg"
    for path in (current, stale):
        path.write_bytes(b"x")

    findings = artifacts.verify_version_naming([current, stale], "0.2.0")
    stale_finding = next(f for f in findings if f.name == "stale artifacts")
    assert stale_finding.blocking
    assert "0.1.9" in stale_finding.detail
    assert any(f.name == "release artifacts" and f.level == "PASS" for f in findings)


def test_a_directory_without_the_release_version_fails(tmp_path: Path) -> None:
    older = tmp_path / "Adaptea_0.1.9_aarch64.dmg"
    older.write_bytes(b"x")
    findings = artifacts.verify_version_naming([older], "0.2.0")
    assert any(f.name == "release artifacts" and f.blocking for f in findings)


def test_a_missing_app_bundle_is_reported_rather_than_raising(tmp_path: Path) -> None:
    findings = artifacts.verify_macos_app(tmp_path / "absent.app", expect_developer_id=True)
    assert findings and findings[0].blocking


def test_an_app_without_the_sidecar_is_blocking(tmp_path: Path) -> None:
    app = tmp_path / "Adaptea.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    findings = artifacts.verify_macos_app(tmp_path / "Adaptea.app", expect_developer_id=False)
    sidecar = next(f for f in findings if f.name == "sidecar")
    assert sidecar.blocking, "an app with no core opens to a dead bridge"


def test_checksums_round_trip_and_detect_tampering(tmp_path: Path) -> None:
    installer = tmp_path / "Adaptea_0.2.0_aarch64.dmg"
    installer.write_bytes(b"original bytes")
    document = artifacts.checksum_document([installer])
    assert artifacts.verify_checksums(document, tmp_path)[0].level == "PASS"

    installer.write_bytes(b"tampered bytes")
    tampered = artifacts.verify_checksums(document, tmp_path)[0]
    assert tampered.blocking


def test_checksum_document_is_sha256sum_compatible(tmp_path: Path) -> None:
    installer = tmp_path / "Adaptea_0.2.0_aarch64.dmg"
    installer.write_bytes(b"payload")
    line = artifacts.checksum_document([installer]).rstrip("\n")
    digest, separator, name = line.partition("  ")
    assert len(digest) == 64 and separator == "  " and name == installer.name


@pytest.mark.parametrize(
    ("platform", "directory", "source_name", "alias_name"),
    [
        ("macos", "dmg", "Adaptea_1.2.3_aarch64.dmg", "Adaptea-macos-arm64.dmg"),
        (
            "windows",
            "nsis",
            "Adaptea_1.2.3_x64-setup.exe",
            "Adaptea-windows-x64-setup.exe",
        ),
    ],
)
def test_latest_alias_has_a_stable_name_and_matching_checksum(
    tmp_path: Path,
    platform: str,
    directory: str,
    source_name: str,
    alias_name: str,
) -> None:
    source = tmp_path / directory / source_name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"release bytes")

    alias, checksum = artifacts.prepare_latest_alias(tmp_path, "1.2.3", platform)

    assert alias.name == alias_name
    assert alias.read_bytes() == source.read_bytes()
    assert artifacts.verify_checksums(checksum.read_text(), tmp_path)[0].level == "PASS"


def test_latest_alias_refuses_an_ambiguous_release_directory(tmp_path: Path) -> None:
    directory = tmp_path / "dmg"
    directory.mkdir()
    (directory / "Adaptea_1.2.3_aarch64.dmg").write_bytes(b"arm")
    (directory / "Adaptea_1.2.3_universal.dmg").write_bytes(b"universal")

    with pytest.raises(ValueError, match="exactly one"):
        artifacts.prepare_latest_alias(tmp_path, "1.2.3", "macos")


# --- updater configuration ---------------------------------------------------------


def test_updater_injection_is_reversible_so_no_key_survives(tmp_path: Path) -> None:
    config = tmp_path / "tauri.conf.json"
    committed = json.dumps({"version": "0.2.0", "bundle": {"active": True}}, indent=2) + "\n"
    config.write_text(committed, encoding="utf-8")

    original = updater.apply(
        config, updater.UpdaterSettings("dW50cnVzdGVkAAA=", ("https://example.com/latest.json",))
    )
    injected = json.loads(config.read_text())
    assert injected["plugins"]["updater"]["pubkey"] == "dW50cnVzdGVkAAA="
    assert injected["bundle"]["createUpdaterArtifacts"] is True

    updater.restore(config, original)
    assert config.read_text() == committed
    assert "pubkey" not in config.read_text()


def test_release_cli_injects_a_signed_https_update_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "tauri.conf.json"
    config.write_text('{"bundle": {}, "plugins": {}}', encoding="utf-8")
    monkeypatch.setenv("TAURI_SIGNING_PUBLIC_KEY", "dW50cnVzdGVkAAA=")
    monkeypatch.setenv("TAURI_UPDATE_ENDPOINT", "https://example.com/latest.json")

    assert release_cli.main(["configure-updater", "--config", str(config)]) == 0
    injected = json.loads(config.read_text(encoding="utf-8"))
    assert injected["bundle"]["createUpdaterArtifacts"] is True
    assert injected["plugins"]["updater"]["endpoints"] == ["https://example.com/latest.json"]


def test_update_endpoints_must_be_https() -> None:
    with pytest.raises(ValueError, match="https"):
        updater.UpdaterSettings("key", ("http://example.com/latest.json",)).validate()


def test_an_empty_public_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="tauri signer generate"):
        updater.UpdaterSettings("  ", ("https://example.com/latest.json",)).validate()


def test_the_committed_configuration_ships_no_updater_key() -> None:
    """The repository must never imply a private signing key lives in it."""
    tauri_conf = Path("desktop/src-tauri/tauri.conf.json")
    if not tauri_conf.exists():
        pytest.skip("Desktop directory not present in standalone core repository")
    config = json.loads(tauri_conf.read_text(encoding="utf-8"))
    assert "updater" not in config.get("plugins", {})
    assert config["bundle"].get("createUpdaterArtifacts") is not True
