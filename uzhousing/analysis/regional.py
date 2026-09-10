"""Cross-regional and cross-segment comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupComparison:
    metric: str
    group_field: str
    table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    leaders: list[dict[str, Any]] = field(default_factory=list)
    laggards: list[dict[str, Any]] = field(default_factory=list)
    dispersion_pct: float | None = None
    spread_ratio: float | None = None
    convergence: str = "unclear"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "group_field": self.group_field,
            "leaders": self.leaders,
            "laggards": self.laggards,
            "dispersion_pct": self.dispersion_pct,
            "spread_ratio": self.spread_ratio,
            "convergence": self.convergence,
            "note": self.note,
        }


def compare_groups(
    tidy: pd.DataFrame,
    metric: str,
    group_field: str = "region",
    role: str = "value",
    top_n: int = 5,
) -> GroupComparison | None:
    """Rank groups by latest level and growth, and measure how far apart they are."""
    if tidy.empty or group_field not in tidy.columns:
        return None

    sub = tidy[tidy["metric"] == metric].dropna(subset=["value"])
    if sub.empty or sub[group_field].nunique() < 2:
        return None

    agg = "sum" if role in {"volume", "supply", "population"} else "mean"
    panel = sub.pivot_table(index="date", columns=group_field, values="value", aggfunc=agg).sort_index()
    panel = panel.dropna(axis=1, how="all")
    if panel.shape[1] < 2:
        return None

    latest_row = panel.ffill().iloc[-1]
    first_row = panel.bfill().iloc[0]

    rows = []
    for group in panel.columns:
        series = panel[group].dropna()
        if series.empty:
            continue
        yoy = _yoy(panel[group])
        rows.append(
            {
                group_field: str(group),
                "latest": _r(latest_row.get(group)),
                "first": _r(first_row.get(group)),
                "change_pct": _pct(latest_row.get(group), first_row.get(group)),
                "yoy_pct": yoy,
                "peak": _r(series.max()),
                "from_peak_pct": _pct(series.iloc[-1], series.max()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return None

    total = table["latest"].sum()
    if role in {"volume", "supply", "population"} and total:
        table["share_pct"] = (table["latest"] / total * 100).round(2)

    sort_key = "yoy_pct" if table["yoy_pct"].notna().sum() >= 2 else "change_pct"
    ranked = table.sort_values(sort_key, ascending=False, na_position="last")

    comparison = GroupComparison(metric=metric, group_field=group_field, table=ranked)
    comparison.leaders = ranked.head(top_n).to_dict("records")
    comparison.laggards = ranked.tail(top_n).iloc[::-1].to_dict("records")

    levels = latest_row.dropna()
    if len(levels) >= 2 and levels.mean():
        comparison.dispersion_pct = round(float(levels.std() / levels.mean() * 100), 2)
        if levels.min() > 0:
            comparison.spread_ratio = round(float(levels.max() / levels.min()), 2)

    comparison.convergence = _convergence(panel)
    comparison.note = (
        f"{panel.shape[1]} {group_field}s compared across {panel.shape[0]} periods; "
        f"ranked by {'year-on-year growth' if sort_key == 'yoy_pct' else 'change over the full sample'}."
    )
    return comparison


def _convergence(panel: pd.DataFrame) -> str:
    """Is the cross-sectional spread narrowing (sigma convergence) or widening?"""
    filled = panel.ffill()
    if filled.shape[0] < 4 or filled.shape[1] < 3:
        return "unclear"
    cv = filled.std(axis=1) / filled.mean(axis=1)
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cv) < 4:
        return "unclear"
    first_half = float(cv.iloc[: len(cv) // 2].mean())
    second_half = float(cv.iloc[len(cv) // 2 :].mean())
    if not first_half:
        return "unclear"
    change = (second_half / first_half - 1) * 100
    if change < -8:
        return "converging (regional gaps narrowing)"
    if change > 8:
        return "diverging (regional gaps widening)"
    return "stable (gaps roughly unchanged)"


def concentration(tidy: pd.DataFrame, metric: str, group_field: str = "region") -> dict[str, Any]:
    """Herfindahl-style concentration of the latest period."""
    sub = tidy[tidy["metric"] == metric].dropna(subset=["value"])
    if sub.empty or group_field not in sub.columns:
        return {}
    latest = sub[sub["date"] == sub["date"].max()]
    totals = latest.groupby(group_field)["value"].sum()
    total = totals.sum()
    if total <= 0 or len(totals) < 2:
        return {}
    shares = totals / total
    top = shares.sort_values(ascending=False)
    return {
        "hhi": round(float((shares**2).sum() * 10000), 1),
        "top_group": str(top.index[0]),
        "top_share_pct": round(float(top.iloc[0] * 100), 2),
        "top3_share_pct": round(float(top.head(3).sum() * 100), 2),
        "groups": int(len(totals)),
    }


def _yoy(series: pd.Series) -> float | None:
    s = series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 2:
        return None
    target = s.index[-1] - pd.DateOffset(years=1)
    window = s[(s.index >= target - pd.Timedelta(days=25)) & (s.index <= target + pd.Timedelta(days=25))]
    if window.empty:
        return None
    base = float(window.iloc[np.abs(window.index - target).argmin()])
    return _pct(s.iloc[-1], base)


def _pct(current: Any, base: Any) -> float | None:
    try:
        current, base = float(current), float(base)
    except (TypeError, ValueError):
        return None
    if base == 0 or not np.isfinite(base) or not np.isfinite(current):
        return None
    return round((current / base - 1) * 100, 2)


def _r(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, 2) if np.isfinite(out) else None
