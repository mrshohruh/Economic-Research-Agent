"""Headline indicators computed from a single time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingest.profiler import PERIODS_PER_YEAR, infer_grain


def _f(value: Any) -> float | None:
    """Coerce to a plain float, mapping NaN/inf to None so JSON stays valid."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else round(out, 4)


@dataclass
class SeriesSummary:
    metric: str
    role: str = "value"
    unit: str = ""
    observations: int = 0
    grain: str = "unknown"
    start: str | None = None
    end: str | None = None
    first_value: float | None = None
    latest: float | None = None
    previous: float | None = None
    period_change_pct: float | None = None
    yoy_pct: float | None = None
    yoy_prev_pct: float | None = None
    ytd_pct: float | None = None
    total_change_pct: float | None = None
    cagr_pct: float | None = None
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_date: str | None = None
    max_date: str | None = None
    std: float | None = None
    volatility_pct: float | None = None
    pct_from_peak: float | None = None
    direction: str = "flat"
    momentum: str = "stable"
    # Rates and inflation are already percentages: a "percent change of a percent"
    # is misleading, so their movement is also expressed in percentage points.
    change_unit: str = "%"
    yoy_pp: float | None = None
    period_change_pp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def summarise_series(
    series: pd.Series,
    metric: str,
    role: str = "value",
    unit: str = "",
) -> SeriesSummary:
    """Compute levels, growth rates and dispersion for one indicator."""
    s = pd.Series(series).dropna().sort_index().astype(float)
    summary = SeriesSummary(metric=metric, role=role, unit=unit, observations=len(s))
    if s.empty:
        return summary

    grain = infer_grain(pd.Series(s.index)) if len(s) > 1 else "single period"
    summary.grain = grain
    per_year = PERIODS_PER_YEAR.get(grain, 12)

    summary.start = _iso(s.index[0])
    summary.end = _iso(s.index[-1])
    summary.first_value = _f(s.iloc[0])
    summary.latest = _f(s.iloc[-1])
    summary.mean = _f(s.mean())
    summary.median = _f(s.median())
    summary.minimum = _f(s.min())
    summary.maximum = _f(s.max())
    summary.min_date = _iso(s.idxmin())
    summary.max_date = _iso(s.idxmax())
    summary.std = _f(s.std())

    if len(s) > 1:
        summary.previous = _f(s.iloc[-2])
        summary.period_change_pct = _pct(s.iloc[-1], s.iloc[-2])

    if s.iloc[0] not in (0, None) and np.isfinite(s.iloc[0]):
        summary.total_change_pct = _pct(s.iloc[-1], s.iloc[0])

    # Year on year, using the actual calendar offset rather than a fixed lag.
    summary.yoy_pct = _yoy(s, 0)
    summary.yoy_prev_pct = _yoy(s, 1)

    # Year to date against the last observation of the previous year.
    if isinstance(s.index, pd.DatetimeIndex) and len(s) > 1:
        last_year = s.index[-1].year - 1
        prior = s[s.index.year == last_year]
        if not prior.empty:
            summary.ytd_pct = _pct(s.iloc[-1], prior.iloc[-1])

    years = _span_years(s, per_year)
    if years >= 0.75 and s.iloc[0] > 0 and s.iloc[-1] > 0:
        summary.cagr_pct = _f(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100)

    returns = s.pct_change().dropna()
    if len(returns) > 2:
        summary.volatility_pct = _f(returns.std() * np.sqrt(per_year) * 100)

    if summary.maximum:
        summary.pct_from_peak = _pct(s.iloc[-1], s.max())

    if role in {"rate", "inflation"}:
        summary.change_unit = "pp"
        summary.yoy_pp = _level_change(s, years=1)
        if len(s) > 1:
            summary.period_change_pp = _f(s.iloc[-1] - s.iloc[-2])

    summary.direction = _direction(summary.yoy_pct if summary.yoy_pct is not None else summary.total_change_pct)
    summary.momentum = _momentum(summary.yoy_pct, summary.yoy_prev_pct)
    return summary


def _level_change(s: pd.Series, years: int = 1) -> float | None:
    """Absolute change against the same period `years` ago, in the series' own units."""
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 2:
        return None
    target = s.index[-1] - pd.DateOffset(years=years)
    window = s[(s.index >= target - pd.Timedelta(days=20)) & (s.index <= target + pd.Timedelta(days=20))]
    if window.empty:
        return None
    base = window.iloc[(np.abs(window.index - target)).argmin()]
    return _f(s.iloc[-1] - base)


def _yoy(s: pd.Series, periods_back: int) -> float | None:
    """Percent change vs the same period one year earlier."""
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 2:
        return None
    idx = len(s) - 1 - periods_back
    if idx < 0:
        return None
    current_date = s.index[idx]
    target = current_date - pd.DateOffset(years=1)
    window = s[(s.index >= target - pd.Timedelta(days=20)) & (s.index <= target + pd.Timedelta(days=20))]
    if window.empty:
        return None
    base = window.iloc[(np.abs(window.index - target)).argmin()]
    return _pct(s.iloc[idx], base)


def _pct(current: float, base: float) -> float | None:
    if base in (0, None) or not np.isfinite(base) or not np.isfinite(current):
        return None
    return _f((current / base - 1) * 100)


def _span_years(s: pd.Series, per_year: int) -> float:
    if isinstance(s.index, pd.DatetimeIndex):
        return max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    return max((len(s) - 1) / per_year, 1e-9)


def _direction(change: float | None) -> str:
    if change is None:
        return "flat"
    if change > 15:
        return "rising sharply"
    if change > 3:
        return "rising"
    if change < -15:
        return "falling sharply"
    if change < -3:
        return "falling"
    return "broadly flat"


def _momentum(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "stable"
    delta = current - previous
    if delta > 2:
        return "accelerating"
    if delta < -2:
        return "decelerating"
    return "stable"


def _iso(value: Any) -> str | None:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value) if value is not None else None


# ---------------------------------------------------------------------------
@dataclass
class DerivedIndicator:
    """A metric the agent computed itself rather than reading from the file."""

    name: str
    description: str
    series: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    unit: str = ""


def rebase_index(series: pd.Series, base_value: float = 100.0) -> pd.Series:
    s = pd.Series(series).dropna().astype(float)
    if s.empty or s.iloc[0] == 0:
        return pd.Series(dtype=float)
    return s / s.iloc[0] * base_value


def real_terms(nominal: pd.Series, inflation_index: pd.Series) -> pd.Series:
    """Deflate a nominal series by a price index or a CPI inflation rate series."""
    nominal = pd.Series(nominal).dropna().astype(float)
    infl = pd.Series(inflation_index).dropna().astype(float)
    if nominal.empty or infl.empty:
        return pd.Series(dtype=float)

    joined = pd.concat([nominal.rename("nom"), infl.rename("infl")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    # If the series looks like a rate (single digit percentages), chain it into an index.
    if joined["infl"].abs().median() < 60:
        factors = (1 + joined["infl"] / 100.0).cumprod()
        deflator = factors / factors.iloc[0]
    else:
        deflator = joined["infl"] / joined["infl"].iloc[0]

    return (joined["nom"] / deflator).rename("real")


def affordability_ratio(price: pd.Series, income: pd.Series, months: int = 12) -> pd.Series:
    """Years of gross income needed to buy the average dwelling."""
    joined = pd.concat(
        [pd.Series(price).rename("price"), pd.Series(income).rename("income")], axis=1
    ).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    annual_income = joined["income"] * months
    ratio = joined["price"] / annual_income.replace(0, np.nan)
    return ratio.dropna().rename("affordability_years")


def contribution_table(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Share of the latest total, and each group's contribution to the change."""
    if frame.empty or group_col not in frame.columns:
        return pd.DataFrame()

    latest_date = frame["date"].max()
    first_date = frame["date"].min()
    latest = frame[frame["date"] == latest_date].groupby(group_col)[value_col].sum()
    first = frame[frame["date"] == first_date].groupby(group_col)[value_col].sum()

    total_latest = latest.sum()
    total_change = total_latest - first.sum()

    out = pd.DataFrame(
        {
            "latest": latest,
            "first": first.reindex(latest.index),
            "share_pct": latest / total_latest * 100 if total_latest else np.nan,
        }
    )
    out["change"] = out["latest"] - out["first"]
    out["change_pct"] = np.where(out["first"] != 0, (out["latest"] / out["first"] - 1) * 100, np.nan)
    if total_change:
        out["contribution_pct"] = out["change"] / total_change * 100
    return out.sort_values("latest", ascending=False)
