from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    prompt: str
    max_output_tokens: int
    minimum_output_ratio: float = 0.35


def workloads(quick: bool = False) -> list[Workload]:
    output = 48 if quick else 192
    repository = "\n".join(
        f"module_{index}.py: def transform_{index}(value): return value + {index}"
        for index in range(40 if quick else 180)
    )
    return [
        Workload(
            "prefill-heavy",
            "UNIQUE-PREFILL-17 Review this repository listing and produce a concise numbered "
            "refactor "
            f"plan with exactly 12 numbered items.\n{repository}",
            output,
            0.25,
        ),
        Workload(
            "decode-heavy",
            "UNIQUE-DECODE-29 Write a numbered implementation checklist with exactly 24 concise "
            "steps for adding a transactional job queue. Do not stop before item 24.",
            output * 2,
            0.45,
        ),
        Workload(
            "mixed",
            "UNIQUE-MIXED-43 Diagnose race conditions in a Python async worker pool and give "
            "exactly "
            "16 numbered fixes with brief code-oriented explanations. Do not stop early.\n"
            + repository[: len(repository) // 3],
            output,
            0.35,
        ),
        Workload(
            "heterogeneous-prefill",
            "UNIQUE-HET-A-71 Phase A: inspect the code-like input and list exactly 12 risks.\n"
            + repository,
            output,
            0.25,
        ),
        Workload(
            "heterogeneous-decode",
            "UNIQUE-HET-B-89 Phase B: produce exactly 24 numbered test cases for a scheduler. "
            "Do not stop before item 24.",
            output * 2,
            0.45,
        ),
    ]
