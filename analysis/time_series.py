"""Time-series analysis engine: MoM/QoQ/YoY changes, rolling averages,
cumulative changes, trend/turning-point detection, volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.schemas import StatResult

FREQ_LAG = {"monthly": {"mom": 1, "yoy": 12, "qoq": 3}, "quarterly": {"qoq": 1, "yoy": 4},
            "annual": {"yoy": 1}, "daily": {}, "weekly": {}}


def indexed_series(s: pd.Series, base_value: float | None = None) -> pd.Series:
    base = base_value if base_value is not None else s.dropna().iloc[0]
    if not base:
        return s * np.nan
    return s / base * 100.0


def pct_change_lag(s: pd.Series, lag: int) -> pd.Series:
    return s.pct_change(periods=lag) * 100.0


def compute_change_series(df: pd.DataFrame, date_col: str, col: str, frequency: str | None) -> dict[str, pd.Series]:
    work = df[[date_col, col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    s = pd.to_numeric(work[col], errors="coerce")
    s.index = work[date_col]

    out: dict[str, pd.Series] = {}
    lags = FREQ_LAG.get(frequency or "", {})
    if "mom" in lags:
        out["mom_pct"] = pct_change_lag(s, lags["mom"])
    if "qoq" in lags:
        out["qoq_pct"] = pct_change_lag(s, lags["qoq"])
    if "yoy" in lags:
        out["yoy_pct"] = pct_change_lag(s, lags["yoy"])
    out["level"] = s
    out["rolling_avg_3"] = s.rolling(3, min_periods=2).mean()
    out["cumulative_change_pct"] = (s / s.dropna().iloc[0] - 1) * 100.0 if s.dropna().shape[0] else s
    return out


def latest_change_stats(df: pd.DataFrame, date_col: str, col: str, frequency: str | None,
                         label_prefix: str = "") -> list[StatResult]:
    series = compute_change_series(df, date_col, col, frequency)
    results: list[StatResult] = []
    s = series["level"].dropna()
    if s.empty:
        return results

    latest_date = s.index[-1]
    latest_val = float(s.iloc[-1])
    results.append(StatResult(
        label=f"{label_prefix}{col} — latest value", variable=col, stat_type="latest_level",
        value=latest_val, period=str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date),
    ))

    for key in ("mom_pct", "qoq_pct", "yoy_pct"):
        if key in series:
            sub = series[key].dropna()
            if not sub.empty:
                results.append(StatResult(
                    label=f"{label_prefix}{col} — {key.split('_')[0].upper()} change",
                    variable=col, stat_type=key, value=round(float(sub.iloc[-1]), 3), unit="pp/%",
                    period=str(sub.index[-1].date()),
                    details={"previous_value": round(float(s.iloc[-2]), 4) if len(s) > 1 else None},
                ))

    return results


def detect_turning_points(s: pd.Series, min_run: int = 2) -> list[dict]:
    """Detect sign changes in the trend (local min/max) of a smoothed series."""
    s = s.dropna()
    if len(s) < 2 * min_run + 1:
        return []
    smooth = s.rolling(min_run, min_periods=1).mean()
    diffs = np.sign(smooth.diff().fillna(0))
    turning = []
    for i in range(1, len(diffs) - 1):
        if diffs.iloc[i] != 0 and diffs.iloc[i] != diffs.iloc[i - 1] and diffs.iloc[i - 1] != 0:
            turning.append({"date": str(smooth.index[i]), "value": float(s.iloc[i]),
                             "type": "peak" if diffs.iloc[i] < 0 else "trough"})
    return turning


def volatility(s: pd.Series, window: int = 6) -> pd.Series:
    return s.pct_change().rolling(window).std() * 100.0
