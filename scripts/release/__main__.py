"""Command line for Adaptea's release checks.

uv run python -m scripts.release versions            # all version sources agree?
uv run python -m scripts.release versions --set 0.2.0
uv run python -m scripts.release preflight           # is this machine able to sign?
uv run python -m scripts.release verify --bundle DIR --version 0.2.0 [--distributable]
uv run python -m scripts.release checksums --bundle DIR
uv run python -m scripts.release latest-alias --bundle DIR --version 0.2.0 --platform macos
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.release import artifacts, signing, updater, versions
from scripts.release.artifacts import Finding

_MARK = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL"}


def _report(findings: list[Finding]) -> int:
    for finding in findings:
        print(f"  [{_MARK[finding.level]}] {finding.name}: {finding.detail}")
    blocking = [finding for finding in findings if finding.blocking]
    if blocking:
        print(f"\n{len(blocking)} blocking problem(s).")
        return 1
    print("\nNo blocking problems.")
    return 0


def _cmd_versions(args: argparse.Namespace) -> int:
    if args.set:
        changed = versions.set_version(args.set)
        print(f"Set {args.set} in: {', '.join(changed) or 'nothing (already current)'}")
        return 0
    sources = versions.collect()
    for source in sources:
        print(f"  {source.name:26} {source.version or '<unreadable>'}")
    grouped = versions.disagreements(sources)
    if len(grouped) == 1 and "<unreadable>" not in grouped:
        print(f"\nAll sources agree on {next(iter(grouped))}.")
        return 0
    print("\nVersion sources disagree:")
    for version, names in grouped.items():
        print(f"  {version}: {', '.join(names)}")
    return 1


def _cmd_preflight(_args: argparse.Namespace) -> int:
    exit_code = 0
    for profile in (signing.MACOS, signing.WINDOWS):
        report = signing.inspect(profile)
        print(f"{profile.platform}:")
        print(f"  signing        {'ready' if report.signing_ready else 'not configured'}")
        if profile.notarization:
            print(f"  notarization   {'ready' if report.notarization_ready else 'not configured'}")
        for name in report.missing_signing + report.missing_notarization:
            purpose = next(
                (c.purpose for c in (*profile.signing, *profile.notarization) if c.name == name), ""
            )
            print(f"    missing {name} — {purpose}")
        if report.missing_tools:
            print(f"    missing tools: {', '.join(report.missing_tools)}")
        if not report.distributable:
            if profile.platform == "Windows":
                print("  → unsigned preview only; SmartScreen will warn and policy may block it")
            else:
                print("  → local development only; artifacts will not open on another machine")
    print(f"\nUpdate signing: {'configured' if signing.updater_ready() else 'not configured'}")
    if os.environ.get("ADAPTEA_REQUIRE_UPDATER") == "1" and not signing.updater_ready():
        print("\nADAPTEA_REQUIRE_UPDATER=1 but update signing keys are incomplete.")
        exit_code = 1
    if os.environ.get("ADAPTEA_REQUIRE_SIGNING") == "1":
        current = signing.MACOS if sys.platform == "darwin" else signing.WINDOWS
        if not signing.inspect(current).distributable:
            print(f"\nADAPTEA_REQUIRE_SIGNING=1 but {current.platform} credentials are incomplete.")
            exit_code = 1
    return exit_code


def _cmd_configure_updater(args: argparse.Namespace) -> int:
    public_key = os.environ.get("TAURI_SIGNING_PUBLIC_KEY", "")
    endpoint = os.environ.get("TAURI_UPDATE_ENDPOINT", "")
    if not public_key or not endpoint:
        print("TAURI_SIGNING_PUBLIC_KEY and TAURI_UPDATE_ENDPOINT are required.")
        return 1
    try:
        updater.apply(
            Path(args.config),
            updater.UpdaterSettings(public_key=public_key, endpoints=(endpoint,)),
        )
    except (OSError, ValueError) as error:
        print(f"Could not configure updater: {error}")
        return 1
    print(f"Configured signed HTTPS updater in {args.config}.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.is_dir():
        print(f"Bundle directory not found: {bundle}")
        return 1
    installers = [
        path
        for pattern in ("dmg/*.dmg", "nsis/*.exe", "msi/*.msi")
        for path in bundle.glob(pattern)
    ]
    findings: list[Finding] = []
    if installers:
        findings += artifacts.verify_version_naming(installers, args.version)
    else:
        findings.append(Finding("FAIL", "installers", f"no dmg/exe/msi under {bundle}"))
    for app in bundle.glob("macos/*.app"):
        findings += artifacts.verify_macos_app(app, expect_developer_id=args.distributable)
    for installer in bundle.glob("nsis/*.exe"):
        findings += artifacts.verify_windows_installer(
            installer, expect_signature=args.distributable
        )
    print(f"Verifying {bundle}\n")
    return _report(findings)


def _cmd_checksums(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    installers = [
        path
        for pattern in ("dmg/*.dmg", "nsis/*.exe", "msi/*.msi")
        for path in bundle.glob(pattern)
    ]
    if not installers:
        print(f"No installers under {bundle}")
        return 1
    document = artifacts.checksum_document(installers)
    destination = Path(args.output) if args.output else bundle / "SHA256SUMS"
    destination.write_text(document, encoding="utf-8")
    print(document, end="")
    print(f"\nWrote {destination}")
    return 0


def _cmd_latest_alias(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    try:
        installer, checksum = artifacts.prepare_latest_alias(bundle, args.version, args.platform)
    except (OSError, ValueError) as error:
        print(f"Could not prepare latest-release alias: {error}")
        return 1
    print(installer)
    print(checksum)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.release", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    versions_parser = sub.add_parser("versions", help="check or set every version source")
    versions_parser.add_argument("--set", metavar="X.Y.Z", help="write this version everywhere")
    versions_parser.set_defaults(handler=_cmd_versions)

    preflight = sub.add_parser("preflight", help="report signing/notarization readiness")
    preflight.set_defaults(handler=_cmd_preflight)

    configure_updater = sub.add_parser(
        "configure-updater", help="inject the signed update feed for a release build"
    )
    configure_updater.add_argument("--config", default="desktop/src-tauri/tauri.conf.json")
    configure_updater.set_defaults(handler=_cmd_configure_updater)

    verify = sub.add_parser("verify", help="verify built installers")
    verify.add_argument("--bundle", required=True, help="path to target/release/bundle")
    verify.add_argument("--version", required=True)
    verify.add_argument(
        "--distributable",
        action="store_true",
        help="require a Developer ID signature and a stapled notarization ticket",
    )
    verify.set_defaults(handler=_cmd_verify)

    checksums = sub.add_parser("checksums", help="write SHA256SUMS for the installers")
    checksums.add_argument("--bundle", required=True)
    checksums.add_argument("--output")
    checksums.set_defaults(handler=_cmd_checksums)

    latest_alias = sub.add_parser(
        "latest-alias", help="copy a versioned installer to its stable latest-release name"
    )
    latest_alias.add_argument("--bundle", required=True)
    latest_alias.add_argument("--version", required=True)
    latest_alias.add_argument("--platform", choices=("macos", "windows"), required=True)
    latest_alias.set_defaults(handler=_cmd_latest_alias)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
