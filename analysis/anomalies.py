"""Anomaly detection: unusual increases/decreases, outliers vs. historical
distribution of period-over-period changes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_anomalies(change_series: pd.Series, threshold: float = 2.0) -> pd.DataFrame:
    s = change_series.dropna()
    if len(s) < 5:
        return pd.DataFrame(columns=["date", "value", "zscore"])
    mu, sd = s.mean(), s.std()
    if not sd:
        return pd.DataFrame(columns=["date", "value", "zscore"])
    z = (s - mu) / sd
    flagged = z[abs(z) >= threshold]
    return pd.DataFrame({"date": flagged.index, "value": s.loc[flagged.index].values, "zscore": flagged.values})


def latest_is_anomalous(change_series: pd.Series, threshold: float = 2.0) -> dict | None:
    s = change_series.dropna()
    if len(s) < 6:
        return None
    hist = s.iloc[:-1]
    mu, sd = hist.mean(), hist.std()
    if not sd:
        return None
    latest = s.iloc[-1]
    z = (latest - mu) / sd
    if abs(z) >= threshold:
        return {"value": float(latest), "zscore": float(z), "historical_mean": float(mu), "historical_std": float(sd)}
    return None
