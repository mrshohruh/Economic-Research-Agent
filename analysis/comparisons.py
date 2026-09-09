"""Comparison analyses: current vs. previous period/year/historical average,
and pre/post structural-break comparisons."""
from __future__ import annotations

import pandas as pd


def current_vs_historical_average(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    current = float(s.iloc[-1])
    hist_avg = float(s.iloc[:-1].mean()) if len(s) > 1 else current
    return {"current": current, "historical_average": hist_avg,
            "difference": current - hist_avg,
            "pct_difference": (current - hist_avg) / hist_avg * 100 if hist_avg else None}


def period_comparison(s: pd.Series, split_date: str) -> dict:
    split = pd.to_datetime(split_date)
    before = s[s.index < split].dropna()
    after = s[s.index >= split].dropna()
    if before.empty or after.empty:
        return {}
    return {
        "before_mean": float(before.mean()), "after_mean": float(after.mean()),
        "difference": float(after.mean() - before.mean()),
        "before_n": int(len(before)), "after_n": int(len(after)),
    }
