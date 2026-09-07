from pathlib import Path

from adaptea.diagnostics.doctor import SETUP_STAGES, checks_from_snapshot
from adaptea.diagnostics.system import DiagnosticSnapshot, LocalModel


def ready_snapshot() -> DiagnosticSnapshot:
    model = LocalModel(
        key="coder",
        loaded=True,
        instance_id="coder@1",
        context_length=8192,
        max_context_length=32768,
        parallel=4,
        format="gguf",
    )
    return DiagnosticSnapshot(
        operating_system="Darwin 25",
        architecture="arm64",
        python_version="3.12.8",
        python_ready=True,
        git_executable="/usr/bin/git",
        git_version="git version 2.50",
        opencode_executable="/usr/local/bin/opencode",
        opencode_version="opencode 1.0",
        lms_executable="/usr/local/bin/lms",
        lms_version="lms 0.1",
        lmstudio_desktop=Path("/Applications/LM Studio.app"),
        server_reachable=True,
        server_error=None,
        native_api_usable=True,
        openai_api_usable=True,
        models=[model],
        selected_model=model,
        telemetry_usable=True,
        telemetry_source="lms_ps",
        git_repository=True,
        capacity_profile=False,
        adaptea_config=True,
        opencode_configured=True,
    )


def test_doctor_reports_shared_snapshot_capabilities(tmp_path: Path) -> None:
    checks = checks_from_snapshot(ready_snapshot(), tmp_path)
    by_name = {check.name: check for check in checks}
    assert by_name["LM Studio native API"].level == "PASS"
    assert by_name["Selected model"].detail.startswith("coder")
    assert by_name["Concurrency management"].detail.endswith("(4)")
    assert by_name["Context length"].level == "WARN"
    assert by_name["Context length"].detail == (
        "8192 is too small for OpenCode's planner prompt; reload the model with at "
        "least 32768 (65536 recommended)"
    )
    assert by_name["Runtime / format"].level == "PASS"
    assert by_name["OpenAI-compatible API"].level == "PASS"
    assert by_name["Capacity profile"].level == "WARN"


def test_doctor_accepts_mlx_and_explains_native_telemetry(tmp_path: Path) -> None:
    snapshot = ready_snapshot()
    assert snapshot.selected_model is not None
    snapshot.selected_model.format = "mlx"
    snapshot.telemetry_source = "native"

    checks = {check.name: check for check in checks_from_snapshot(snapshot, tmp_path)}

    assert checks["Runtime / format"].level == "PASS"
    assert checks["Runtime / format"].detail == "MLX / Apple Silicon optimized"
    assert checks["Live LM Studio telemetry"].level == "PASS"
    assert "native API" in checks["Live LM Studio telemetry"].detail
    # Guidance now lives in `remedy`; `detail` states the condition only.
    assert checks["Capacity profile"].detail == "not measured yet"
    assert "Quick Calibration" in (checks["Capacity profile"].remedy or "")


def _fresh_snapshot(**overrides: object) -> DiagnosticSnapshot:
    """A machine where nothing is installed and LM Studio is not running."""
    base: dict[str, object] = dict(
        operating_system="Darwin 25.5.0",
        architecture="arm64",
        python_version="3.12.0",
        python_ready=True,
        git_executable=None,
        git_version=None,
        opencode_executable=None,
        opencode_version=None,
        lms_executable=None,
        lms_version=None,
        lmstudio_desktop=None,
        server_reachable=False,
        server_error="connection refused",
        native_api_usable=False,
        openai_api_usable=False,
    )
    base.update(overrides)
    return DiagnosticSnapshot(**base)  # type: ignore[arg-type]


def test_checks_are_ordered_by_setup_stage(tmp_path: Path) -> None:
    """The order a first-time user reads is the order the pieces actually depend on."""
    checks = checks_from_snapshot(_fresh_snapshot(), tmp_path)
    seen = [SETUP_STAGES.index(check.stage) for check in checks]
    assert seen == sorted(seen), [(c.stage, c.name) for c in checks]

    stage_of = {check.name: check.stage for check in checks}
    assert stage_of["Git"] == "System"
    assert stage_of["OpenCode"] == "System"
    assert stage_of["LM Studio server"] == "Runtime"
    assert stage_of["Adaptea configuration"] == "Project"
    assert stage_of["Capacity profile"] == "Optional"


def test_every_unsatisfied_check_says_what_to_do(tmp_path: Path) -> None:
    """A blocker with no remedy leaves the user stuck, which is the whole complaint."""
    checks = checks_from_snapshot(_fresh_snapshot(), tmp_path)
    unmet = [check for check in checks if check.level != "PASS"]
    assert unmet, "the fresh-machine fixture should have unmet checks"
    missing = [check.name for check in unmet if not (check.remedy or "").strip()]
    assert not missing, f"no remedy for: {missing}"


def test_remedies_never_send_the_user_back_to_setup(tmp_path: Path) -> None:
    """ "run adaptea setup" is useless advice to someone already looking at Setup."""
    checks = checks_from_snapshot(_fresh_snapshot(), tmp_path)
    for check in checks:
        text = f"{check.detail} {check.remedy or ''}".lower()
        assert "run adaptea setup" not in text, check.name


def test_passing_checks_carry_no_remedy(tmp_path: Path) -> None:
    checks = checks_from_snapshot(_fresh_snapshot(python_ready=True), tmp_path)
    assert all(check.remedy is None for check in checks if check.level == "PASS")


def test_project_stage_reports_a_missing_git_repository(tmp_path: Path) -> None:
    checks = checks_from_snapshot(_fresh_snapshot(git_repository=False), tmp_path)
    repository = next(check for check in checks if check.name == "Git repository")
    assert repository.level == "WARN"
    assert str(tmp_path) in repository.detail
    assert "worktree" in (repository.remedy or "")


def test_a_downloaded_but_unloaded_model_blocks_work_until_save_loads_it(
    tmp_path: Path,
) -> None:
    """Work cannot plan until the selected model is actually inference-ready."""
    snapshot = ready_snapshot()
    assert snapshot.selected_model is not None
    snapshot.selected_model.loaded = False
    snapshot.selected_model.instance_id = None

    checks = {check.name: check for check in checks_from_snapshot(snapshot, tmp_path)}

    assert checks["Loaded models"].level == "FAIL"
    assert checks["Loaded models"].detail == "none loaded"
    assert checks["Selected model"].level == "FAIL"
    assert "Environment" in (checks["Selected model"].remedy or "")
    assert [
        check
        for check in checks_from_snapshot(snapshot, tmp_path)
        if check.level == "FAIL" and check.stage == "Model"
    ]


def test_nothing_downloaded_is_still_a_real_problem(tmp_path: Path) -> None:
    """There must be something to load before loading can be asked for."""
    snapshot = ready_snapshot()
    snapshot.models = []
    snapshot.selected_model = None

    checks = {check.name: check for check in checks_from_snapshot(snapshot, tmp_path)}

    assert checks["Loaded models"].level == "FAIL"
    assert checks["Selected model"].level == "FAIL"


def test_doctor_windows_remedies_show_windows_tools(tmp_path: Path) -> None:
    snapshot = ready_snapshot()
    snapshot.operating_system = "Windows 11"
    snapshot.git_executable = None
    snapshot.opencode_executable = None

    checks = {check.name: check for check in checks_from_snapshot(snapshot, tmp_path)}

    assert "winget" in (checks["Git"].remedy or "")
    assert "xcode-select" not in (checks["Git"].remedy or "")
    assert "winget" in (checks["OpenCode"].remedy or "")
    assert "Homebrew" not in (checks["OpenCode"].remedy or "")
