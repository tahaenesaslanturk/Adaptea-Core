# Releasing Adaptea

The public beta ships a signed and notarized Apple-silicon macOS `.dmg` plus an unsigned
x64 Windows NSIS preview. Windows can be built and shared without buying a certificate,
but SmartScreen shows an unknown-publisher warning and managed PCs may block it. This
document is the runbook for turning a commit into beta artifacts without confusing the
zero-cost Windows preview with a trusted Windows release.

Nothing here requires a certificate to be present in the repository, and no step writes a
key into the working tree. Local development builds keep working with no configuration at
all. The macOS release refuses to ship without its signing credentials; Windows signing is
optional while the artifact is explicitly labelled as an unsigned preview.

## The one-paragraph version

Set the version everywhere, tag it, and let CI build. CI refuses to start if the version
sources disagree, refuses the macOS build if signing is incomplete, and refuses to publish
stale or malformed artifacts. Windows verification reports its intentionally missing
Authenticode signature as a warning.

## 1. Set the version

Five files declare the version: `pyproject.toml`, `src/adaptea/__init__.py`,
`desktop/package.json`, `desktop/src-tauri/tauri.conf.json`, and
`desktop/src-tauri/Cargo.toml`. Set them together:

```bash
uv run python -m scripts.release versions --set 0.8.0
```

Check at any time:

```bash
uv run python -m scripts.release versions
```

Pre-release suffixes (`0.2.0-rc1`) are rejected so every package and update-feed version
uses the same plain `MAJOR.MINOR.PATCH` value.

## 2. Confirm your signing environment

```bash
uv run python -m scripts.release preflight
```

It reports whether signing, notarization, and update signing are configured, and names any
variable that is missing along with what it is for. It never prints a value, so its output
is safe to paste into an issue.

With nothing configured it says *"local development only"*. That is the correct state for a
development machine.

### macOS

| Variable | Purpose |
| --- | --- |
| `APPLE_SIGNING_IDENTITY` | Developer ID Application certificate name |
| `APPLE_CERTIFICATE` | base64 of the `.p12`; CI imports it into a temporary keychain |
| `APPLE_CERTIFICATE_PASSWORD` | password for that `.p12` |
| `APPLE_API_KEY` + `APPLE_API_ISSUER` + `APPLE_API_KEY_PATH` | App Store Connect key for notarization |

An Apple ID alternative (`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) also satisfies
notarization; preflight accepts either.

Both halves matter. A signed but un-notarized app still gets *"Apple cannot check it for
malicious software"* on any machine that downloaded it, because the quarantine attribute
triggers Gatekeeper. Only a stapled notarization ticket clears that offline.

### Windows preview

No Windows signing secret is required for the beta preview. CI builds an x64 NSIS
installer on `windows-latest` and labels both the artifact and release notes as unsigned.
GitHub-hosted runner use comes from the account's included Actions allowance; standard
runners are unlimited if the repository is public. A future trusted build can provide
`WINDOWS_CERTIFICATE` and `WINDOWS_CERTIFICATE_PASSWORD` and enable the existing strict
verification path.

## 3. Tag and let CI build

```bash
git tag v0.8.0 && git push origin v0.8.0
```

`.github/workflows/release.yml` then:

1. **version-gate** — fails if the five version sources disagree, or if the tag does not
   match what the repository declares.
2. **build** — requires signing for macOS and produces the explicitly unsigned Windows
   preview in a second matrix job.
3. **verify** — uses strict distributable checks for macOS and warning-level Authenticode
   checks for Windows.
4. **checksums** — writes one platform-named SHA-256 manifest per job.
5. **stable aliases** — copies each verified installer to a version-independent asset name used
   by the website's `/releases/latest/download/…` links and writes a matching checksum.
6. **draft release** — uploads the DMG, NSIS installer, stable aliases, signed updater archives,
   generated `latest.json`, and checksums to a GitHub draft. A human publishes it after review.

## 4. What verification actually checks

```bash
uv run python -m scripts.release verify \
  --bundle desktop/src-tauri/target/release/bundle \
  --version 0.8.0 --distributable
```

* every expected installer exists and its filename carries the release version
* **no stale installer from an earlier build is sitting in the same directory** — a reused
  bundle directory will happily publish three versions at once
* the `.app` contains the `adaptea-core` sidecar and it is executable; without it the app
  opens to a dead bridge
* `codesign --verify --deep --strict` passes
* the signature is a real Developer ID, not ad-hoc
* a notarization ticket is stapled

Without `--distributable` signature findings are warnings rather than failures. The
Windows preview deliberately uses this mode; its missing Authenticode signature must stay
visible in the job log and release notes.

Verify a published download the way a user would:

```bash
shasum -a 256 -c SHA256SUMS
spctl --assess --type execute --verbose /Applications/Adaptea.app
```

## 5. Updates

The updater is **off in the committed configuration**, and its public key is not stored in
the repository. That is deliberate: a committed key implies a matching private key belongs
here, and an always-on updater would point development builds at a production feed.

To enable it for a release, generate a key pair yourself — this repository will not do it
for you:

```bash
cd desktop && npx tauri signer generate -w ~/.tauri/adaptea.key
```

Keep the private key in your CI secret store as `TAURI_SIGNING_PRIVATE_KEY`. Expose the
public half as `TAURI_SIGNING_PUBLIC_KEY`. Configure the HTTPS feed as the repository
variable `TAURI_UPDATE_ENDPOINT`. The release job validates both keys and injects the
public key and endpoint just before building; the same operation can be run locally with:

```bash
TAURI_SIGNING_PUBLIC_KEY="$(cat updater.key.pub)" \
TAURI_UPDATE_ENDPOINT="https://github.com/tahaenesaslanturk/Adaptea/releases/latest/download/latest.json" \
uv run python -m scripts.release configure-updater
```

For the GitHub-hosted beta, set `TAURI_UPDATE_ENDPOINT` to that same URL as a repository
variable. The release action creates `latest.json` in a draft; the stable endpoint starts
serving it only when the draft is promoted. Beta is a product label here, not GitHub's
“prerelease” flag, because GitHub excludes prereleases from `/releases/latest`.

The committed configuration remains updater-free, so development builds never contact a
production feed. CI workspaces are ephemeral; for a local release, restore
`desktop/src-tauri/tauri.conf.json` after the build.

Endpoints must be `https`. The signature check would still catch a forged package over
plain HTTP, but the version and release notes shown to the user would not be trustworthy.

## What is deliberately not automated

* **Key generation.** Creating a signing key is a decision about custody, not a build step.
* **Publishing.** The workflow prepares a complete draft; promoting it to a public release
  is a human action.
* **Certificate renewal.** Preflight tells you when credentials are missing, not when they
  are about to expire.
