from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any


def latest_calibration(root: Path) -> Path | None:
    parent = root / ".adaptea" / "calibration"
    if not parent.exists():
        return None
    candidates = [path for path in parent.iterdir() if (path / "aggregate.json").exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def benchmark_summary(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in run_summaries:
        label = str(row["configuration"])
        grouped.setdefault(label, []).append(row)
    aggregate: dict[str, Any] = {}
    for label, rows in grouped.items():
        durations = [
            float(row.get("total_duration_seconds", row.get("wall_seconds", 0))) for row in rows
        ]
        successes = [float(row.get("success_rate", row.get("pass_rate", 0))) for row in rows]
        aggregate[label] = {
            "repetitions": len(rows),
            "median_total_duration_seconds": statistics.median(durations),
            "median_wall_seconds": statistics.median(durations),
            "median_success_rate": statistics.median(successes),
            "median_pass_rate": statistics.median(successes),
            "median_retry_count": _optional_median(rows, "retry_count"),
            "median_tasks_per_second": statistics.median(
                float(row.get("tasks_per_second", 0)) for row in rows
            ),
        }
    fixed = {key: value for key, value in aggregate.items() if key.startswith("fixed-c")}
    if fixed:
        passing = {key: value for key, value in fixed.items() if value["median_pass_rate"] == 1.0}
        pool = passing or fixed
        oracle = min(pool, key=lambda key: pool[key]["median_wall_seconds"])
        aggregate["oracle_fixed"] = {
            "definition": "best observed fixed concurrency among tested values",
            "configuration": oracle,
            **pool[oracle],
        }
    aggregate["formal_comparison"] = bool(grouped) and all(
        len(rows) >= 3 for rows in grouped.values()
    )
    return aggregate


def write_benchmark_csv(path: Path, aggregate: dict[str, Any]) -> None:
    if isinstance(aggregate.get("medians"), dict) and isinstance(aggregate.get("runs"), list):
        _write_full_benchmark_csv(path, aggregate)
        return
    rows = [
        {"configuration": key, **value}
        for key, value in aggregate.items()
        if isinstance(value, dict)
    ]
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_benchmark_html(path: Path, aggregate: dict[str, Any]) -> None:
    if isinstance(aggregate.get("medians"), dict) and isinstance(aggregate.get("runs"), list):
        _write_full_benchmark_html(path, aggregate)
        return
    rows = [
        (key, value)
        for key, value in aggregate.items()
        if isinstance(value, dict) and "median_wall_seconds" in value
    ]
    maximum = max((float(value["median_wall_seconds"]) for _, value in rows), default=1.0)
    bars = []
    for label, value in rows:
        wall = float(value["median_wall_seconds"])
        width = wall / maximum * 100 if maximum else 0
        bars.append(
            f'<div class="row"><span>{html.escape(label)}</span>'
            f'<div class="bar" style="width:{width:.1f}%"></div>'
            f"<b>{wall:.2f}s; pass={float(value['median_pass_rate']):.2f}</b></div>"
        )
    document = f"""<!doctype html><meta charset="utf-8"><title>Adaptea benchmark</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:3rem auto}}
.row{{display:grid;grid-template-columns:10rem 1fr 14rem;gap:1rem;margin:.8rem}}
.bar{{background:#69b3a2;height:1.5rem}}
</style><h1>Adaptea benchmark comparison</h1>{"".join(bars)}
<p>Shorter wall time is better only when deterministic pass rate remains acceptable.</p>"""
    path.write_text(document, encoding="utf-8")


def _optional_median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if isinstance((value := row.get(key)), int | float) and not isinstance(value, bool)
    ]
    return statistics.median(values) if values else None


def _flatten_pressure(value: object, *, median: bool = False) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    if median:
        keys = (
            "telemetry_coverage_runs",
            "median_queue_pressure_ratio",
            "median_peak_queued_predictions",
            "median_busy_ratio",
            "median_ttft_seconds",
            "median_tokens_per_second",
            "median_peak_workers",
            "median_average_worker_utilization",
        )
        return {key: value.get(key) for key in keys if key in value}
    fields = {
        "queue_pressure_ratio": "queue_pressure_ratio",
        "peak_queued_predictions": "peak_queued_predictions",
        "busy_ratio": "busy_ratio",
        "median_ttft_seconds": "median_ttft_seconds",
        "median_tokens_per_second": "median_tokens_per_second",
        "peak_workers": "peak_workers",
        "average_worker_utilization": "average_worker_utilization",
        "telemetry_coverage_runs": "telemetry_coverage_runs",
    }
    return {target: value.get(source) for source, target in fields.items() if source in value}


def _write_full_benchmark_csv(path: Path, report: dict[str, Any]) -> None:
    rows: list[dict[str, object]] = []
    for run in report["runs"]:
        if not isinstance(run, dict):
            continue
        rows.append(
            {
                "row_type": "run",
                "mode": run.get("mode"),
                "repetition": run.get("repetition"),
                "sequence": run.get("sequence"),
                "source_commit": run.get("source_commit"),
                "plan_sha256": run.get("plan_sha256"),
                "run_id": run.get("run_id"),
                "final_target_concurrency": run.get("final_target_concurrency"),
                "parallel_limit": run.get("parallel_limit"),
                "total_duration_seconds": run.get("total_duration_seconds"),
                "success_rate": run.get("success_rate"),
                "retry_count": run.get("retry_count"),
                "error": run.get("error"),
                **_flatten_pressure(run.get("resource_pressure")),
            }
        )
    medians = report["medians"]
    for mode, value in medians.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "row_type": "median",
                "mode": mode,
                "repetitions": value.get("repetitions"),
                "total_duration_seconds": value.get("median_total_duration_seconds"),
                "success_rate": value.get("median_success_rate"),
                "retry_count": value.get("median_retry_count"),
                **_flatten_pressure(value.get("resource_pressure"), median=True),
            }
        )
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: object, suffix: str = "") -> str:
    if value is None:
        return "not observed"
    if isinstance(value, int | float):
        return f"{float(value):.3f}{suffix}"
    return html.escape(str(value))


def _write_full_benchmark_html(path: Path, report: dict[str, Any]) -> None:
    medians = report["medians"]
    rows: list[str] = []
    for mode in ("serial", "fixed", "naive", "adaptive"):
        value = medians.get(mode, {})
        if not isinstance(value, dict):
            continue
        pressure = value.get("resource_pressure", {})
        if not isinstance(pressure, dict):
            pressure = {}
        success = value.get("median_success_rate")
        success_text = (
            f"{float(success) * 100:.1f}%" if isinstance(success, int | float) else "not observed"
        )
        queue = pressure.get("median_queue_pressure_ratio")
        queue_text = (
            f"{float(queue) * 100:.1f}%" if isinstance(queue, int | float) else "not observed"
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(mode.title())}</th>"
            f"<td>{value.get('repetitions', 0)}</td>"
            f"<td>{_format_metric(value.get('median_total_duration_seconds'), 's')}</td>"
            f"<td>{success_text}</td>"
            f"<td>{_format_metric(value.get('median_retry_count'))}</td>"
            f"<td>{queue_text}</td>"
            f"<td>{_format_metric(pressure.get('median_peak_queued_predictions'))}</td>"
            f"<td>{_format_metric(pressure.get('median_peak_workers'))}</td>"
            f"<td>{pressure.get('telemetry_coverage_runs', 0)}/{value.get('repetitions', 0)}</td>"
            "</tr>"
        )
    errors = [
        run for run in report["runs"] if isinstance(run, dict) and isinstance(run.get("error"), str)
    ]
    error_section = ""
    if errors:
        error_section = (
            "<h2>Run errors</h2><ul>"
            + "".join(
                f"<li>{html.escape(str(run.get('mode')))} r{run.get('repetition')}: "
                f"{html.escape(str(run.get('error')))}</li>"
                for run in errors
            )
            + "</ul>"
        )
    title = html.escape(str(report.get("benchmark_id", "Adaptea benchmark")))
    commit = html.escape(str(report.get("source_commit", "unknown")))
    plan_hash = html.escape(str(report.get("plan_sha256", "unknown")))
    formal = "yes" if report.get("formal_comparison") is True else "no"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
:root{{--ink:#101426;--muted:#5f6782;--line:#dce1f2;--accent:#175bc4;--wash:#f5f7ff}}
body{{font:15px system-ui,sans-serif;color:var(--ink);max-width:1180px;
margin:3rem auto;padding:0 1rem}}
h1{{margin-bottom:.25rem}}p{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;margin:2rem 0}}
th,td{{border:1px solid var(--line);padding:.7rem;text-align:right}}
th:first-child{{text-align:left}}thead th{{background:var(--wash)}}
code{{word-break:break-all}}.method{{border-left:4px solid var(--accent);
padding:1rem;background:var(--wash)}}
</style></head><body><h1>{title}</h1>
<p>Formal comparison: <strong>{formal}</strong></p>
<div class="method">All modes used commit <code>{commit}</code> and plan <code>{plan_hash}</code>.
Runs were sequential, seeded, and round-interleaved. Values below are medians;
missing telemetry is shown as “not observed” rather than estimated.</div>
<table><thead><tr><th>Mode</th><th>Runs</th><th>Total duration</th><th>Success</th>
<th>Retries</th><th>Queue pressure</th><th>Peak queue</th><th>Peak workers</th><th>Telemetry</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table>{error_section}</body></html>"""
    path.write_text(document, encoding="utf-8")


def read_report(root: Path, calibration: Path | None = None) -> tuple[Path, dict[str, Any]]:
    directory = calibration or latest_calibration(root)
    if not directory:
        raise FileNotFoundError("no calibration report found; run adaptea calibrate")
    path = directory / "aggregate.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid report: {path}")
    return directory, value
