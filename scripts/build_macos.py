from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
TARGET = DESKTOP / "src-tauri" / "target" / "release" / "bundle"
TAURI_TARGET = DESKTOP / "src-tauri" / "target"
TAURI_SOURCE = DESKTOP / "src-tauri"
TARGET_ROOT_MARKER = TAURI_TARGET / ".adaptea-source-root"
APP = TARGET / "macos" / "Adaptea.app"
ARCH = "aarch64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
TAURI_CONFIG = json.loads((DESKTOP / "src-tauri" / "tauri.conf.json").read_text())
VERSION = str(TAURI_CONFIG["version"])
DMG = TARGET / "dmg" / f"Adaptea_{VERSION}_{ARCH}.dmg"


ICON = DESKTOP / "src-tauri" / "icons" / "icon.icns"
# The disk image gets its own icon rather than the application's. The app icon is a filled
# rounded square, and Finder draws it on the DMG as a dark tile that covers the window's own
# background; this one is the bare mark on transparency, inset so it does not run to the
# edge. `sips` and `iconutil` both ship with macOS, so no extra tooling is required.
VOLUME_ICON_SOURCE = ROOT / "assets" / "adaptea-volume.svg"
ICONSET_SIZES: tuple[tuple[int, str], ...] = (
    (16, "16x16"),
    (32, "16x16@2x"),
    (32, "32x32"),
    (64, "32x32@2x"),
    (128, "128x128"),
    (256, "128x128@2x"),
    (256, "256x256"),
    (512, "256x256@2x"),
    (512, "512x512"),
    (1024, "512x512@2x"),
)

# Finder shows a folder's custom icon only when its FinderInfo carries kHasCustomIcon
# (0x0400 in the flags at offset 8). SetFile would set it, but that needs full Xcode;
# `xattr` ships with macOS itself, so the DMG gets its icon on any build machine.
CUSTOM_ICON_FINDER_INFO = "0000000000000000" + "0400" + "0" * 44


# `npm run tauri` finds the Tauri CLI in the desktop package's own bin directory, so a
# checkout that has never installed its dependencies cannot package at all.
TAURI_CLI = DESKTOP / "node_modules" / ".bin" / "tauri"


def run(*command: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def mark_custom_icon(target: Path) -> bool:
    """Set kHasCustomIcon so Finder actually draws `.VolumeIcon.icns`."""
    try:
        subprocess.run(
            ["xattr", "-wx", "com.apple.FinderInfo", CUSTOM_ICON_FINDER_INFO, str(target)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"! Could not mark the volume icon as custom: {exc}")
        return False
    return True


def volume_icon(workspace: Path) -> Path | None:
    """Render the transparent volume icon, falling back to the application icon.

    The icon is cosmetic, so a machine whose `sips` cannot read the SVG still gets a
    working DMG — with the old app-icon tile rather than no icon at all.
    """
    if not VOLUME_ICON_SOURCE.is_file():
        return ICON if ICON.is_file() else None
    iconset = workspace / "Adaptea.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    icns = workspace / "AdapteaVolume.icns"
    try:
        for pixels, name in ICONSET_SIZES:
            subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    "png",
                    "--resampleHeightWidth",
                    str(pixels),
                    str(pixels),
                    str(VOLUME_ICON_SOURCE),
                    "--out",
                    str(iconset / f"icon_{name}.png"),
                ],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"! Could not render {VOLUME_ICON_SOURCE.name} ({exc}); using the app icon.")
        return ICON if ICON.is_file() else None
    return icns


def build_dmg(staging: Path, icon: Path | None) -> None:
    """Write the compressed DMG, giving the volume the Adaptea icon when possible.

    `hdiutil create -srcfolder` copies the files but not the folder's FinderInfo, so the
    custom-icon flag has to be set on a mounted read-write image and only then compressed.
    The icon is cosmetic: any failure along that path falls back to a plain, working DMG
    rather than failing the release build.
    """
    if icon is None:
        print("! No volume icon available; building a DMG with the generic icon.")
        plain_dmg(staging)
        return
    shutil.copy2(icon, staging / ".VolumeIcon.icns")
    writable = staging.parent / "adaptea-rw.dmg"
    mountpoint = staging.parent / "mnt"
    mountpoint.mkdir()
    try:
        run(
            "hdiutil",
            "create",
            "-volname",
            "Adaptea",
            "-srcfolder",
            str(staging),
            "-ov",
            "-format",
            "UDRW",
            str(writable),
        )
        run("hdiutil", "attach", str(writable), "-nobrowse", "-mountpoint", str(mountpoint))
    except subprocess.CalledProcessError as exc:
        print(f"! Could not prepare the writable image ({exc}); building a plain DMG.")
        plain_dmg(staging)
        return
    marked = mark_custom_icon(mountpoint)
    subprocess.run(["hdiutil", "detach", str(mountpoint), "-quiet"], check=False)
    if not marked:
        plain_dmg(staging)
        return
    run("hdiutil", "convert", str(writable), "-format", "UDZO", "-ov", "-o", str(DMG))
    print(f"+ volume icon: {icon.name}")


def plain_dmg(staging: Path) -> None:
    run(
        "hdiutil",
        "create",
        "-volname",
        "Adaptea",
        "-srcfolder",
        str(staging),
        "-ov",
        "-format",
        "UDZO",
        str(DMG),
    )


def ensure_desktop_dependencies() -> None:
    """Install the desktop package when the checkout has never done so.

    Packaging is meant to be one command from a clean tree, and a fresh clone or a new
    Git worktree has no `desktop/node_modules`. Without this the build failed several
    steps later inside npm, reporting only that `tauri` could not be found — which names
    neither the missing step nor the command that fixes it. The lockfile is installed
    exactly as CI installs it, so this adds nothing a release build would not already do.
    """
    if TAURI_CLI.exists():
        return
    print(f"+ desktop dependencies are missing; installing {DESKTOP / 'package-lock.json'}")
    run("npm", "ci", cwd=DESKTOP)
    if not TAURI_CLI.exists():
        raise RuntimeError(f"npm ci did not provide the Tauri CLI at {TAURI_CLI}")


def ensure_tauri_cache_matches_checkout() -> None:
    """Discard generated Rust metadata after the checkout has moved.

    Cargo and Tauri build-script output embeds absolute paths. Reusing ``target`` after
    moving the repository can therefore make an otherwise unchanged build look for
    generated permissions in the checkout's old location. A small ignored marker keeps
    incremental builds fast while making the first build after a move self-healing.
    """
    source_root = str(TAURI_SOURCE.resolve())
    previous_root = None
    if TARGET_ROOT_MARKER.is_file():
        previous_root = TARGET_ROOT_MARKER.read_text(encoding="utf-8").strip()
    elif TAURI_TARGET.is_dir() and any(TAURI_TARGET.iterdir()):
        # Existing caches predate the marker and may already contain absolute paths.
        previous_root = "unknown"

    if previous_root is not None and previous_root != source_root:
        print(f"+ Tauri build cache belongs to {previous_root}; rebuilding it for {source_root}")
        shutil.rmtree(TAURI_TARGET)

    TAURI_TARGET.mkdir(parents=True, exist_ok=True)
    TARGET_ROOT_MARKER.write_text(source_root + "\n", encoding="utf-8")


def find_keychain_identity() -> str | None:
    if identity := os.environ.get("APPLE_SIGNING_IDENTITY"):
        return identity
    try:
        proc = subprocess.run(
            ["security", "find-identity", "-v", "-p", "codesigning"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if "Developer ID Application:" in line:
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1:
                    return line[start + 1 : end]
    except Exception:
        pass
    return None


def notarize_dmg(dmg_path: Path) -> None:
    """Submit DMG to Apple Notary service and staple the ticket if credentials are present."""
    notary_args: list[str] | None = None

    for profile in ["notarytool-profile", "AC_PASSWORD", "adaptea"]:
        try:
            proc = subprocess.run(
                ["xcrun", "notarytool", "history", "--keychain-profile", profile],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                notary_args = ["--keychain-profile", profile]
                print(f"+ Found Notarytool keychain profile: '{profile}'")
                break
        except Exception:
            pass

    if not notary_args:
        apple_id = os.environ.get("APPLE_ID")
        apple_password = os.environ.get("APPLE_PASSWORD")
        apple_team_id = os.environ.get("APPLE_TEAM_ID", "C6R68ZKRSM")
        if apple_id and apple_password:
            notary_args = [
                "--apple-id",
                apple_id,
                "--password",
                apple_password,
                "--team-id",
                apple_team_id,
            ]
            print("+ Using Notarytool credentials from environment variables.")

    if not notary_args:
        print(
            "! Skipping DMG notarization (no notary keychain profile or APPLE_ID/APPLE_PASSWORD env vars found)."
        )
        return

    print(f"+ Submitting {dmg_path.name} to Apple Notary service (this takes ~30-60s)...")
    try:
        subprocess.run(
            ["xcrun", "notarytool", "submit", str(dmg_path), *notary_args, "--wait"],
            check=True,
        )
        print(f"+ Notarization accepted! Stapling ticket to {dmg_path.name}...")
        subprocess.run(
            ["xcrun", "stapler", "staple", str(dmg_path)],
            check=True,
        )
        print(f"✓ Successfully notarized and stapled {dmg_path.name}!")
    except Exception as exc:
        print(f"! Warning: Notarization error: {exc}")


def main() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("The macOS bundle must be built on macOS.")
    ensure_desktop_dependencies()
    ensure_tauri_cache_matches_checkout()
    signing_identity = find_keychain_identity()
    if signing_identity:
        os.environ["APPLE_SIGNING_IDENTITY"] = signing_identity
        print(f"+ Using signing identity from keychain/env: {signing_identity}")
    run("npm", "run", "sidecar", cwd=DESKTOP)
    run("npm", "run", "tauri", "--", "build", "--bundles", "app", cwd=DESKTOP)
    if not APP.is_dir():
        raise RuntimeError(f"Tauri did not produce {APP}")
    if signing_identity:
        print(f"+ Signing internal binaries and {APP.name} with Developer ID...")
        entitlements = DESKTOP / "src-tauri" / "Entitlements.plist"
        for binary in (APP / "Contents" / "MacOS").iterdir():
            if binary.is_file():
                subprocess.run(
                    [
                        "codesign",
                        "--force",
                        "--options",
                        "runtime",
                        "--sign",
                        signing_identity,
                        "--timestamp",
                        str(binary),
                    ],
                    check=True,
                )
        subprocess.run(
            [
                "codesign",
                "--force",
                "--deep",
                "--options",
                "runtime",
                "--entitlements",
                str(entitlements),
                "--sign",
                signing_identity,
                "--timestamp",
                str(APP),
            ],
            check=True,
        )
    DMG.parent.mkdir(parents=True, exist_ok=True)
    if DMG.exists():
        DMG.unlink()
    with tempfile.TemporaryDirectory(prefix="adaptea-dmg-") as raw:
        staging = Path(raw) / "Adaptea"
        staging.mkdir()
        shutil.copytree(APP, staging / APP.name, symlinks=True)
        os.symlink("/Applications", staging / "Applications")
        build_dmg(staging, volume_icon(Path(raw)))
    if signing_identity:
        print(f"+ Signing DMG with identity: {signing_identity}")
        try:
            subprocess.run(
                ["codesign", "--force", "--sign", signing_identity, "--timestamp", str(DMG)],
                check=True,
            )
        except Exception as exc:
            print(f"! Warning: Failed to sign DMG: {exc}")
        notarize_dmg(DMG)
    print(f"Built macOS app: {APP}")
    print(f"Built macOS DMG: {DMG}")


if __name__ == "__main__":
    main()
