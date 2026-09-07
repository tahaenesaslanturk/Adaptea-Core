# Desktop architecture

Adaptea Desktop is an additional frontend over the existing Python engine. It does not implement planning,
scheduling, fleet routing, setup, calibration, Git, validation, or merge policy in React.

```text
React + TypeScript
    │ typed commands / events
    ▼
Tauri 2 Rust process host
    │ versioned NDJSON over stdin/stdout
    ▼
PyInstaller `adaptea-core-<target-triple>` sidecar
    │
    ▼
ApplicationServices → existing planner, scheduler, fleet, setup, smoke, history, and reports
```

## Protocol

Protocol version 1 uses one JSON value per line. Commands have `protocol`, `id`, `type: "command"`,
`command`, and `payload`. Responses echo the ID and contain either `data` or a structured error with
`code`, `message`, and `retryable`. Events have `type: "event"`, an event name, and data.

The bridge validates every command, bounds frontend-to-Rust messages to 2 MiB, supports request timeouts and
cancellation, and refuses a clean shutdown while an orchestrator is active. Run state remains durable in
the repository if the UI exits. The frontend receives no arbitrary shell permission; privileged behavior is
limited to explicit Rust/Python commands and narrowly scoped dialog/notification capabilities.

## Development and packaging

Debug builds start the repository virtual-environment Python directly for fast iteration. Release builds
use Tauri `externalBin` and a PyInstaller one-file sidecar named with the Rust target triple. See
`scripts/build_sidecar.py`, `scripts/build_desktop.py`, and `.github/workflows/desktop-build.yml`.
