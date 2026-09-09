"""Relationship analyses: correlation matrix, lead/lag correlation, simple
OLS regression where mathematically justified. No test is run "for show" --
callers should only surface results that are relevant to the research topic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(min_periods=5)


def pairwise_correlation(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    joint = pd.concat([a, b], axis=1).dropna()
    if len(joint) < 5:
        return {}
    r, p = stats.pearsonr(joint[col_a], joint[col_b])
    return {"r": float(r), "p_value": float(p), "n": int(len(joint))}


def lead_lag_correlation(df: pd.DataFrame, col_a: str, col_b: str, max_lag: int = 6) -> dict:
    """Correlation of col_a with col_b shifted by k periods (k>0: b lags a)."""
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    best = {"lag": 0, "r": 0.0}
    for lag in range(-max_lag, max_lag + 1):
        shifted = b.shift(lag)
        joint = pd.concat([a, shifted], axis=1).dropna()
        if len(joint) < 5:
            continue
        r = joint.iloc[:, 0].corr(joint.iloc[:, 1])
        if pd.notna(r) and abs(r) > abs(best["r"]):
            best = {"lag": lag, "r": float(r)}
    return best


def simple_regression(df: pd.DataFrame, y_col: str, x_col: str) -> dict:
    y = pd.to_numeric(df[y_col], errors="coerce")
    x = pd.to_numeric(df[x_col], errors="coerce")
    joint = pd.concat([y, x], axis=1).dropna()
    if len(joint) < 8:
        return {}
    slope, intercept, r, p, se = stats.linregress(joint[x_col], joint[y_col])
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r ** 2),
            "p_value": float(p), "n": int(len(joint))}
