# Desktop public beta readiness

The primary public beta targets Apple-silicon macOS 12 or newer. CI also produces an x64
Windows NSIS preview without requiring a paid Windows certificate. Intel macOS remains a
separate support decision: either add and test an `x86_64-apple-darwin` artifact before the
beta announcement or say clearly that the macOS beta requires Apple silicon.

## Implemented foundations

- The release build is Developer ID signed, notarized, stapled, and checked before upload.
- Update packages are signed independently and the production HTTPS feed is injected only
  in release CI. Development builds never contact it.
- Installed builds check for updates on launch. Users can choose automatic verified
  download/install or manual checks. An installed update waits for a user-controlled app
  restart, so active local agents are never killed.
- Every project chooses what happens after a reviewed run: keep Adaptea's integration
  branch, create one local commit, or create the commit and non-force push it to `origin`.
  Remote rejection leaves the local commit intact.
- Standard GitHub-hosted Windows runners use the repository account's included Actions
  allowance (and are unlimited if the repository becomes public). The Windows installer
  itself needs no paid service to build or host.

## Windows preview and the zero-cost boundary

The Windows x64 NSIS installer is intentionally unsigned. It can be built, attached to a
GitHub release, downloaded, and installed without paying Microsoft or a certificate
authority. This is suitable for informed beta testers, not a polished mass-market launch:

- Microsoft Defender SmartScreen shows “Windows protected your PC” and identifies the
  publisher as unknown. A tester must choose **More info → Run anyway**.
- Smart App Control or an organization policy may block the installer with no bypass.
- Each unsigned release starts with no transferable publisher reputation.

Do not tell Windows testers that the build is signed or broadly trusted. Add Authenticode
later when the beta justifies its recurring certificate/service cost; the existing release
preflight already understands the required Windows credentials.

## Required before inviting public testers

1. Add the Apple certificate and notarization credentials listed in `docs/releasing.md` to
   repository secrets and pass the distributable release job.
2. Generate the Tauri updater key pair, store the private half only in CI, and set
   `TAURI_UPDATE_ENDPOINT` to the GitHub latest-release URL documented in the release
   runbook. CI will attach `latest.json` and its signed macOS archive to the draft.
3. Install the downloaded DMG on a clean, non-developer Mac. Separately install the NSIS
   preview on clean Windows 11 and record the exact SmartScreen path. Verify first launch,
   sidecar startup, model-provider setup, one completed run, all three Git completion
   actions, and an update from the previous beta build on both platforms.
4. Publish a privacy notice, support/contact route, known-issues page, and deletion policy
   before collecting any diagnostics outside the user's Mac.
5. Decide whether Apple silicon is the stated beta limit or add a separately signed and
   tested Intel artifact. Do not advertise generic “macOS support” while shipping only
   arm64.
6. Back up the updater private key and Apple signing credentials in controlled storage,
   document rotation/revocation ownership, and keep release promotion a human approval.

## Corpus and diagnostics policy

Do not upload prompts, repository paths, source code, diffs, worker logs, or model output in
the first public build. They can contain credentials and private code even when a project
is public. A consent toggle without a working data contract is not meaningful consent.

Start with an opt-in, metrics-only beta channel after these are fixed:

- a published event schema containing app version, operating system/architecture, inference backend,
  scheduler, task counts, terminal outcomes, duration buckets, and coarse failure codes;
- no stable machine, user, repository, branch, commit, path, prompt, or source identifier;
- visible preview/export of the exact payload, a retention window, deletion contact, and a
  server endpoint that rejects fields outside the allowlist;
- upload off by default and independent from update checks;
- secret-scanning and size limits before persistence, plus integration tests proving
  disallowed fields never leave the process.

Only consider a richer training corpus later, under separate explicit consent and terms.
It should have per-run review, redaction, provenance/license metadata, withdrawal support,
and must never be implied by accepting beta updates.

## Release sequence

1. Finish the secrets and update host.
2. Cut an internal signed build and exercise the clean-Mac checklist.
3. Publish a small invite-only beta and test one real update cycle.
4. Fix launch, updater, and Git-publication failures before expanding access.
5. Add opt-in metrics only after the privacy notice, schema, endpoint, and deletion path are
   all live.
