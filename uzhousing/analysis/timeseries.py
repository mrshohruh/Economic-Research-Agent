"""Time-series structure: trend, seasonality, breaks, turning points, forecast."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..ingest.profiler import PANDAS_FREQ, PERIODS_PER_YEAR, infer_grain

LOGGER = logging.getLogger(__name__)


@dataclass
class TrendResult:
    slope_per_period: float | None = None
    slope_per_year_pct: float | None = None
    r_squared: float | None = None
    p_value: float | None = None
    significant: bool = False
    description: str = "insufficient data"


@dataclass
class Breakpoint:
    date: str
    before_mean: float
    after_mean: float
    shift_pct: float
    direction: str


@dataclass
class TurningPoint:
    date: str
    kind: str  # "peak" or "trough"
    value: float


@dataclass
class SeasonalityResult:
    detected: bool = False
    strength: float | None = None
    peak_period: str | None = None
    trough_period: str | None = None
    monthly_effect: dict[str, float] = field(default_factory=dict)


@dataclass
class Forecast:
    method: str = "none"
    horizon: int = 0
    points: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""


@dataclass
class TimeSeriesAnalysis:
    metric: str
    trend: TrendResult = field(default_factory=TrendResult)
    seasonality: SeasonalityResult = field(default_factory=SeasonalityResult)
    breakpoints: list[Breakpoint] = field(default_factory=list)
    turning_points: list[TurningPoint] = field(default_factory=list)
    forecast: Forecast = field(default_factory=Forecast)
    rolling_yoy: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "trend": vars(self.trend),
            "seasonality": {
                "detected": self.seasonality.detected,
                "strength": self.seasonality.strength,
                "peak_period": self.seasonality.peak_period,
                "trough_period": self.seasonality.trough_period,
            },
            "breakpoints": [vars(b) for b in self.breakpoints],
            "turning_points": [vars(t) for t in self.turning_points],
            "forecast": {
                "method": self.forecast.method,
                "horizon": self.forecast.horizon,
                "points": self.forecast.points,
                "note": self.forecast.note,
            },
        }


# ---------------------------------------------------------------------------
def analyse(series: pd.Series, metric: str, horizon: int | None = None) -> TimeSeriesAnalysis:
    s = _regularise(series)
    result = TimeSeriesAnalysis(metric=metric)
    if len(s) < 3:
        return result

    grain = infer_grain(pd.Series(s.index))
    per_year = PERIODS_PER_YEAR.get(grain, 12)

    result.trend = fit_trend(s, per_year)
    result.seasonality = detect_seasonality(s, per_year)
    result.breakpoints = detect_breakpoints(s)
    result.turning_points = detect_turning_points(s, per_year)
    result.rolling_yoy = year_on_year(s, per_year)
    result.forecast = forecast(s, per_year, horizon or max(2, per_year // 2))
    return result


def _regularise(series: pd.Series) -> pd.Series:
    """Sort, drop NaNs and snap to a regular frequency so models behave."""
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return s
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    grain = infer_grain(pd.Series(s.index))
    freq = PANDAS_FREQ.get(grain)
    if freq and len(s) > 3:
        try:
            resampled = s.resample(freq).mean()
            # Only accept the resample if it does not invent a lot of gaps.
            if resampled.isna().mean() < 0.25:
                s = resampled.interpolate(limit_direction="both")
        except Exception as exc:  # pragma: no cover - odd indexes
            LOGGER.debug("resample failed for %s: %s", s.name, exc)
    return s


# ---------------------------------------------------------------------------
def fit_trend(s: pd.Series, per_year: int) -> TrendResult:
    from scipy import stats

    if len(s) < 4:
        return TrendResult()

    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    reg = stats.linregress(x, y)

    mean_level = float(np.nanmean(y))
    slope_year_pct = (reg.slope * per_year / mean_level * 100) if mean_level else None

    significant = bool(reg.pvalue < 0.05)
    if not significant:
        desc = "no statistically significant linear trend"
    elif reg.slope > 0:
        desc = f"significant upward trend of about {abs(slope_year_pct):.1f}% of the average level per year"
    else:
        desc = f"significant downward trend of about {abs(slope_year_pct):.1f}% of the average level per year"

    return TrendResult(
        slope_per_period=round(float(reg.slope), 4),
        slope_per_year_pct=round(float(slope_year_pct), 2) if slope_year_pct is not None else None,
        r_squared=round(float(reg.rvalue**2), 4),
        p_value=round(float(reg.pvalue), 6),
        significant=significant,
        description=desc,
    )


def detect_seasonality(s: pd.Series, per_year: int) -> SeasonalityResult:
    result = SeasonalityResult()
    if per_year < 2 or len(s) < 2 * per_year or not isinstance(s.index, pd.DatetimeIndex):
        return result

    try:
        from statsmodels.tsa.seasonal import STL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(s, period=per_year, robust=True).fit()
        var_resid = float(np.nanvar(stl.resid))
        var_seasonal_resid = float(np.nanvar(stl.seasonal + stl.resid))
        strength = max(0.0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid else 0.0
        seasonal = pd.Series(stl.seasonal, index=s.index)
    except Exception as exc:  # pragma: no cover - statsmodels edge cases
        LOGGER.debug("STL failed: %s", exc)
        detrended = s - s.rolling(per_year, center=True, min_periods=1).mean()
        seasonal = detrended
        total_var = float(np.nanvar(s))
        strength = float(np.nanvar(detrended) / total_var) if total_var else 0.0

    label = "month" if per_year == 12 else "quarter"
    grouper = seasonal.index.month if per_year == 12 else seasonal.index.quarter
    effect = seasonal.groupby(grouper).mean()

    result.strength = round(float(strength), 3)
    result.detected = strength > 0.3 and len(effect) > 1
    if not effect.empty:
        result.peak_period = f"{label} {int(effect.idxmax())}"
        result.trough_period = f"{label} {int(effect.idxmin())}"
        result.monthly_effect = {str(int(k)): round(float(v), 3) for k, v in effect.items()}
    return result


def detect_breakpoints(s: pd.Series, max_breaks: int = 3, min_segment: int = 4) -> list[Breakpoint]:
    """Binary segmentation on mean shifts — flags where the level jumped."""
    if len(s) < 2 * min_segment + 1:
        return []

    values = s.to_numpy(dtype=float)
    segments = [(0, len(values))]
    breaks: list[int] = []

    for _ in range(max_breaks):
        best_gain, best_idx, best_seg = 0.0, None, None
        for seg in segments:
            start, end = seg
            if end - start < 2 * min_segment:
                continue
            block = values[start:end]
            total_ss = float(np.sum((block - block.mean()) ** 2))
            for i in range(min_segment, len(block) - min_segment):
                left, right = block[:i], block[i:]
                split_ss = float(
                    np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
                )
                gain = total_ss - split_ss
                if gain > best_gain:
                    best_gain, best_idx, best_seg = gain, start + i, seg
        if best_idx is None:
            break
        # Require the split to explain a meaningful share of the variance.
        block = values[best_seg[0] : best_seg[1]]
        total_ss = float(np.sum((block - block.mean()) ** 2))
        if total_ss <= 0 or best_gain / total_ss < 0.25:
            break
        breaks.append(best_idx)
        segments.remove(best_seg)
        segments.extend([(best_seg[0], best_idx), (best_idx, best_seg[1])])

    out: list[Breakpoint] = []
    for idx in sorted(breaks):
        before = float(np.mean(values[max(0, idx - 12) : idx]))
        after = float(np.mean(values[idx : idx + 12]))
        if before == 0:
            continue
        shift = (after / before - 1) * 100
        if abs(shift) < 3:
            continue
        out.append(
            Breakpoint(
                date=pd.Timestamp(s.index[idx]).date().isoformat(),
                before_mean=round(before, 2),
                after_mean=round(after, 2),
                shift_pct=round(shift, 2),
                direction="step up" if shift > 0 else "step down",
            )
        )
    return out


def detect_turning_points(s: pd.Series, per_year: int) -> list[TurningPoint]:
    """Local peaks and troughs in the smoothed level."""
    window = max(3, per_year // 2)
    if len(s) < 2 * window + 1:
        return []

    smooth = s.rolling(window, center=True, min_periods=1).mean()
    values = smooth.to_numpy(dtype=float)
    points: list[TurningPoint] = []
    for i in range(window, len(values) - window):
        left, right = values[i - window : i], values[i + 1 : i + 1 + window]
        if values[i] >= left.max() and values[i] >= right.max():
            points.append(TurningPoint(pd.Timestamp(s.index[i]).date().isoformat(), "peak", round(float(s.iloc[i]), 2)))
        elif values[i] <= left.min() and values[i] <= right.min():
            points.append(TurningPoint(pd.Timestamp(s.index[i]).date().isoformat(), "trough", round(float(s.iloc[i]), 2)))

    # Collapse runs of adjacent same-kind points.
    deduped: list[TurningPoint] = []
    for point in points:
        if deduped and deduped[-1].kind == point.kind:
            continue
        deduped.append(point)
    return deduped[-8:]


def year_on_year(s: pd.Series, per_year: int) -> pd.Series:
    if per_year < 2 or len(s) <= per_year:
        return pd.Series(dtype=float)
    return (s / s.shift(per_year) - 1).mul(100).dropna().rename("yoy_pct")


def forecast(s: pd.Series, per_year: int, horizon: int) -> Forecast:
    """Short projection: Holt-Winters where the history supports it, else a trend."""
    if len(s) < 6 or not isinstance(s.index, pd.DatetimeIndex):
        return Forecast(note="not enough history to project")

    grain = infer_grain(pd.Series(s.index))
    freq = PANDAS_FREQ.get(grain, "MS")
    horizon = max(1, min(horizon, per_year))

    values: np.ndarray | None = None
    method = ""
    note = ""

    if len(s) >= 2 * per_year + 4 and per_year > 1:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    s, trend="add", seasonal="add", seasonal_periods=per_year,
                    initialization_method="estimated",
                ).fit()
            values = np.asarray(model.forecast(horizon), dtype=float)
            method = "Holt-Winters (additive trend and seasonality)"
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Holt-Winters failed: %s", exc)

    if values is None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(s, trend="add", initialization_method="estimated").fit()
            values = np.asarray(model.forecast(horizon), dtype=float)
            method = "Holt linear trend"
        except Exception:
            from scipy import stats

            x = np.arange(len(s), dtype=float)
            reg = stats.linregress(x, s.to_numpy(dtype=float))
            values = reg.intercept + reg.slope * np.arange(len(s), len(s) + horizon, dtype=float)
            method = "ordinary least squares trend extrapolation"

    residual_std = float(np.nanstd(s.diff().dropna())) or 0.0
    future_index = pd.date_range(start=s.index[-1], periods=horizon + 1, freq=freq)[1:]

    points = []
    for step, (date, value) in enumerate(zip(future_index, values), start=1):
        band = 1.96 * residual_std * np.sqrt(step)
        points.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "forecast": round(float(value), 2),
                "lower": round(float(value - band), 2),
                "upper": round(float(value + band), 2),
            }
        )

    note = note or (
        "Projection is a statistical extrapolation of the supplied history only. "
        "It does not incorporate announced policy measures."
    )
    return Forecast(method=method, horizon=horizon, points=points, note=note)
