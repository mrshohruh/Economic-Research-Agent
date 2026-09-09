"""Trend detection: linear trend slope over sub-periods, direction changes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def linear_trend(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 3:
        return {}
    x = np.arange(len(s))
    slope, intercept, r, p, se = stats.linregress(x, s.values)
    return {"slope": float(slope), "r_squared": float(r ** 2), "p_value": float(p)}


def compare_recent_vs_prior_trend(s: pd.Series, split_frac: float = 0.5) -> dict:
    s = s.dropna()
    if len(s) < 6:
        return {}
    cut = max(3, int(len(s) * split_frac))
    first, second = s.iloc[:cut], s.iloc[cut:]
    t1, t2 = linear_trend(first), linear_trend(second)
    if not t1 or not t2:
        return {}
    return {
        "prior_slope": t1["slope"], "recent_slope": t2["slope"],
        "trend_reversed": np.sign(t1["slope"]) != np.sign(t2["slope"]) and abs(t2["slope"]) > 1e-9,
        "trend_accelerated": abs(t2["slope"]) > abs(t1["slope"]) * 1.3,
        "trend_decelerated": abs(t2["slope"]) < abs(t1["slope"]) * 0.7,
    }
