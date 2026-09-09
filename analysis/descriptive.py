"""Descriptive statistics engine. Pure pandas/numpy -- no LLM involvement."""
from __future__ import annotations

import pandas as pd

from models.schemas import StatResult


def describe_variable(df: pd.DataFrame, col: str) -> StatResult:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    details = {}
    if len(s):
        mean = float(s.mean())
        std = float(s.std()) if len(s) > 1 else 0.0
        details = {
            "mean": round(mean, 4),
            "median": round(float(s.median()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "std": round(std, 4),
            "cv": round(std / mean, 4) if mean not in (0, None) else None,
            "n": int(len(s)),
        }
    return StatResult(label=f"Descriptive statistics for {col}", variable=col, stat_type="descriptive",
                       value=details.get("mean"), details=details)


def describe_all(df: pd.DataFrame, columns: list[str]) -> list[StatResult]:
    return [describe_variable(df, c) for c in columns if c in df.columns]
