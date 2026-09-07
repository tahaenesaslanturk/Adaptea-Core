from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("valid", False)]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in valid:
        grouped[int(sample["concurrency"])].append(sample)
    results: dict[str, Any] = {}
    for concurrency, rows in sorted(grouped.items()):
        latencies = [float(row["latency_seconds"]) for row in rows]
        ttfts = [float(row["ttft_seconds"]) for row in rows if row.get("ttft_seconds") is not None]
        speeds = [
            float(row["tokens_per_second"])
            for row in rows
            if row.get("tokens_per_second") is not None
        ]
        batch_durations = [
            float(row["batch_wall_seconds"])
            for row in rows
            if row.get("batch_wall_seconds") is not None
        ]
        total_input = sum(int(row.get("input_tokens") or 0) for row in rows)
        total_output = sum(int(row.get("output_tokens") or 0) for row in rows)
        total_wall = sum(batch_durations) / concurrency if batch_durations else sum(latencies)
        results[str(concurrency)] = {
            "valid_requests": len(rows),
            "invalid_requests": sum(
                1 for row in samples if row["concurrency"] == concurrency and not row.get("valid")
            ),
            "median_wall_seconds": statistics.median(batch_durations) if batch_durations else None,
            "requests_per_second": len(rows) / total_wall if total_wall else None,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "input_tokens_per_second": total_input / total_wall if total_wall else None,
            "generation_tokens_per_second": statistics.median(speeds) if speeds else None,
            "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
            "p95_ttft_seconds": percentile(ttfts, 0.95),
            "median_latency_seconds": statistics.median(latencies),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "queued_request_observations": sum(
                1 for row in rows if (row.get("queued_predictions") or 0) > 0
            ),
        }
    return results


def recommend(aggregate: dict[str, Any]) -> tuple[int, int]:
    if not aggregate:
        return 1, 1
    rows = {int(key): value for key, value in aggregate.items()}
    throughput = {
        concurrency: float(row.get("requests_per_second") or 0) for concurrency, row in rows.items()
    }
    best = max(throughput, key=throughput.get)  # type: ignore[arg-type]
    best_rate = throughput[best]
    # Prefer the smallest concurrency within 90% of observed peak throughput.
    recommended = min(
        (concurrency for concurrency, rate in throughput.items() if rate >= best_rate * 0.9),
        default=best,
    )
    safe = max(
        (
            concurrency
            for concurrency, row in rows.items()
            if int(row.get("invalid_requests") or 0) == 0
            and throughput[concurrency] >= best_rate * 0.75
        ),
        default=recommended,
    )
    return recommended, safe
