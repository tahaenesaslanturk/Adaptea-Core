from __future__ import annotations


def normalize_title(value: str) -> str:
    return " ".join(value.strip().split())
