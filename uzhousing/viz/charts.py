"""Chart factory.

Every builder returns a :class:`Figure` that carries both the rendered PNG and
the underlying numbers. The report always prints the numbers next to or below
the picture, which is also what satisfies the palette's relief rule for the
lower-contrast hues.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import theme

LOGGER = logging.getLogger(__name__)


@dataclass
class Figure:
    id: str
    title: str
    path: Path
    caption: str = ""
    source_note: str = ""
    kind: str = ""
    table: pd.DataFrame | None = field(default=None, repr=False)
    table_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "path": str(self.path),
            "caption": self.caption,
            "kind": self.kind,
        }


class ChartBuilder:
    """Creates figures into a directory, numbering them as it goes."""

    def __init__(self, output_dir: Path, source_note: str = "") -> None:
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.source_note = source_note
        self.figures: list[Figure] = []
        theme.apply_style()

    # -- internals ------------------------------------------------------
    def _next_id(self) -> str:
        return f"F{len(self.figures) + 1}"

    def _save(self, fig, fid: str, title: str, caption: str, kind: str,
              table: pd.DataFrame | None = None, table_title: str = "") -> Figure:
        path = self.dir / f"{fid}_{kind}.png"
        fig.savefig(path)
        plt.close(fig)
        record = Figure(
            id=fid, title=title, path=path, caption=caption,
            source_note=self.source_note, kind=kind,
            table=table, table_title=table_title,
        )
        self.figures.append(record)
        return record

    @staticmethod
    def _time_axis(ax, index: pd.Index) -> None:
        if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            return
        span_days = (index[-1] - index[0]).days
        if span_days > 365 * 8:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days > 365 * 2:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days > 200:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # ------------------------------------------------------------------
    def trend_line(
        self,
        series: pd.Series,
        label: str,
        unit: str = "",
        events: Sequence[dict[str, Any]] | None = None,
        show_trend: bool = True,
    ) -> Figure | None:
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 3:
            return None

        fig, ax = plt.subplots()
        color = theme.CATEGORICAL[0]
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=1.9, zorder=4)
        ax.fill_between(s.index, s.min() - (s.max() - s.min()) * 0.5, s.to_numpy(),
                        color=color, alpha=0.07, zorder=1)

        if show_trend and len(s) >= 6:
            x = np.arange(len(s), dtype=float)
            slope, intercept = np.polyfit(x, s.to_numpy(dtype=float), 1)
            ax.plot(s.index, intercept + slope * x, color=theme.INK_MUTED,
                    linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
            quarter = len(s) // 4
            ax.annotate("linear trend", xy=(s.index[quarter], intercept + slope * quarter),
                        xytext=(6, -15), textcoords="offset points",
                        fontsize=7.5, color=theme.INK_MUTED, ha="left",
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": theme.SURFACE,
                              "edgecolor": "none"})

        latest = s.iloc[-1]
        ax.scatter([s.index[-1]], [latest], s=34, color=color, zorder=5,
                   edgecolor=theme.SURFACE, linewidth=1.6)
        theme.direct_label(ax, s.index[-1], latest, theme.compact(latest), color)

        ax.yaxis.set_major_formatter(theme.smart_formatter(s.to_numpy()))
        self._time_axis(ax, s.index)
        # Extra headroom so the event markers sit inside the plot, clear of the title.
        theme.nice_ylim(ax, s.to_numpy(), pad=0.08)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.10)
        marked = self._mark_events(ax, events, s)

        span = f"{s.index[0]:%b %Y} – {s.index[-1]:%b %Y}" if isinstance(s.index, pd.DatetimeIndex) else ""
        theme.finish(ax)
        theme.title_block(fig, f"{label} over time", f"{unit or 'level'} · {span}")

        change = (latest / s.iloc[0] - 1) * 100 if s.iloc[0] else float("nan")
        caption = (
            f"{label} moved from {theme.compact(s.iloc[0])} to {theme.compact(latest)} "
            f"over the sample, a change of {change:+.1f}%."
        )
        if marked:
            caption += f" Vertical markers show {marked} dated policy or macro event(s)."

        table = s.rename(label).to_frame()
        table.index = [self._fmt_date(i) for i in table.index]
        return self._save(fig, self._next_id(), f"{label} over time", caption, "trend",
                          table=self._thin(table), table_title=f"{label}: selected observations")

    def _mark_events(self, ax, events: Sequence[dict[str, Any]] | None, s: pd.Series) -> int:
        if not events or not isinstance(s.index, pd.DatetimeIndex):
            return 0
        lo, hi = s.index[0], s.index[-1]
        count = 0
        for event in events:
            try:
                when = pd.Timestamp(str(event.get("date"))[:10])
            except Exception:
                continue
            if not (lo <= when <= hi):
                continue
            count += 1
            ax.axvline(when, color=theme.INK_MUTED, linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
            # Inside the plot, against the top edge: outside the axes it would
            # collide with the title block.
            ax.annotate(
                str(count),
                xy=(when, 0.985), xycoords=("data", "axes fraction"),
                fontsize=7, color=theme.INK_SECONDARY, ha="center", va="top",
                bbox={"boxstyle": "circle,pad=0.22", "facecolor": theme.SURFACE,
                      "edgecolor": theme.BASELINE, "linewidth": 0.6},
                zorder=6,
            )
            if count >= 8:
                break
        return count

    # ------------------------------------------------------------------
    def yoy_bars(self, yoy: pd.Series, label: str) -> Figure | None:
        s = pd.Series(yoy).dropna().astype(float)
        if len(s) < 3:
            return None

        fig, ax = plt.subplots()
        colors = [theme.POSITIVE if v >= 0 else theme.NEGATIVE for v in s]
        positions = mdates.date2num(s.index) if isinstance(s.index, pd.DatetimeIndex) else np.arange(len(s))
        width = self._bar_width(positions)
        theme.rounded_bars(ax, positions, s.to_numpy(), colors, width=width)

        theme.zero_line(ax)
        mean = float(s.mean())
        ax.axhline(mean, color=theme.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.annotate(f"sample average {mean:+.1f}%", xy=(0.995, mean), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", fontsize=7.5,
                    color=theme.INK_SECONDARY, ha="right", va="bottom")

        if isinstance(s.index, pd.DatetimeIndex):
            ax.xaxis_date()
            self._time_axis(ax, s.index)
        else:
            ax.set_xticks(positions[:: max(1, len(s) // 10)])
            ax.set_xticklabels([str(i) for i in s.index[:: max(1, len(s) // 10)]], rotation=0)

        ax.yaxis.set_major_formatter(theme.smart_formatter(s.to_numpy(), percent=True))
        ax.set_xlim(positions.min() - width, positions.max() + width)
        theme.finish(ax)
        theme.title_block(
            fig, f"{label}: year-on-year growth",
            "Blue = growth, red = decline · % change on the same period a year earlier",
        )

        latest = s.iloc[-1]
        negatives = int((s < 0).sum())
        caption = (
            f"Year-on-year growth in {label} was {latest:+.1f}% in the latest period, "
            f"against a sample average of {mean:+.1f}%. "
            f"{negatives} of {len(s)} periods recorded a fall."
        )
        table = s.rename("yoy_pct").to_frame()
        table.index = [self._fmt_date(i) for i in table.index]
        return self._save(fig, self._next_id(), f"{label}: year-on-year growth", caption, "yoy",
                          table=self._thin(table), table_title=f"{label}: year-on-year growth (%)")

    @staticmethod
    def _bar_width(positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 20.0
        gap = float(np.median(np.diff(np.sort(positions))))
        return max(gap * 0.72, gap - 2.0) if gap > 0 else 0.72

    # ------------------------------------------------------------------
    def region_index_lines(self, panel: pd.DataFrame, label: str, top_n: int = 4,
                           group_name: str = "region") -> Figure | None:
        """Rebase each group to 100 at the start so different levels share one axis."""
        data = panel.dropna(axis=1, how="all").ffill()
        if data.shape[1] < 2 or data.shape[0] < 3:
            return None

        latest = data.iloc[-1].sort_values(ascending=False)
        keep = list(latest.head(top_n).index)
        rebased = pd.DataFrame(
            {col: data[col] / data[col].dropna().iloc[0] * 100 for col in keep if data[col].dropna().any()}
        )
        if rebased.empty:
            return None

        fig, ax = plt.subplots()
        colors = theme.series_colors(len(rebased.columns))
        names = [str(c) for c in rebased.columns]
        for (name, series), color in zip(rebased.items(), colors):
            ax.plot(series.index, series.to_numpy(), color=color, linewidth=1.8,
                    zorder=4, label=theme.shorten(str(name)))

        ax.axhline(100, color=theme.BASELINE, linewidth=1.0, zorder=2)
        self._time_axis(ax, rebased.index)
        ax.yaxis.set_major_formatter(theme.smart_formatter(rebased.to_numpy().ravel()))
        ax.set_xlim(rebased.index[0], rebased.index[-1] + (rebased.index[-1] - rebased.index[0]) * 0.13)
        theme.nice_ylim(ax, rebased.to_numpy().ravel())

        # Labels are placed after the limits are settled, and pushed apart so
        # two series that finish close together stay readable.
        low, high = ax.get_ylim()
        ends = [float(rebased[c].iloc[-1]) for c in rebased.columns]
        placed = theme.declutter(ends, (high - low) * 0.062)
        for name, color, end, y in zip(names, colors, ends, placed):
            theme.direct_label(ax, rebased.index[-1], y, theme.shorten(name, 15), color)

        theme.finish(ax, legend=True, legend_cols=min(4, len(rebased.columns)))
        theme.title_block(fig, f"{label} by {group_name}, indexed to 100",
                          f"Each {group_name} rebased to 100 at the start of the sample")

        final = rebased.iloc[-1].sort_values(ascending=False)
        caption = (
            f"Indexing removes level differences so growth is comparable. "
            f"{final.index[0]} grew fastest ({final.iloc[0]:.0f} vs a base of 100); "
            f"{final.index[-1]} grew slowest ({final.iloc[-1]:.0f})."
        )
        table = rebased.round(1)
        table.index = [self._fmt_date(i) for i in table.index]
        return self._save(fig, self._next_id(), f"{label} by {group_name}, indexed", caption, "index",
                          table=self._thin(table), table_title=f"{label} index (start of sample = 100)")

    # ------------------------------------------------------------------
    def ranking_bars(self, table: pd.DataFrame, value_col: str, group_col: str,
                     label: str, unit: str = "", top_n: int = 12) -> Figure | None:
        data = table.dropna(subset=[value_col]).sort_values(value_col, ascending=True).tail(top_n)
        if len(data) < 2:
            return None

        fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.34 * len(data) + 1.3)))
        positions = np.arange(len(data), dtype=float)
        theme.rounded_bars(ax, positions, data[value_col].to_numpy(dtype=float),
                           theme.CATEGORICAL[0], width=0.66, horizontal=True)

        ax.set_yticks(positions)
        ax.set_yticklabels([theme.shorten(v, 26) for v in data[group_col]], fontsize=8.5, color=theme.INK_SECONDARY)
        ax.set_ylim(-0.7, len(data) - 0.3)
        theme.value_labels(ax, positions, data[value_col].to_numpy(dtype=float), horizontal=True)

        ax.grid(axis="x", color=theme.GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(theme.smart_formatter(data[value_col].to_numpy()))
        peak = float(np.nanmax(np.abs(data[value_col].to_numpy(dtype=float))))
        ax.set_xlim(0, peak * 1.18)
        theme.finish(ax)
        theme.title_block(fig, f"{label} by {group_col}, latest period",
                          unit or "latest observed level")

        top, bottom = data.iloc[-1], data.iloc[0]
        ratio = (top[value_col] / bottom[value_col]) if bottom[value_col] else float("nan")
        caption = (
            f"{top[group_col]} records the highest reading for {label} at "
            f"{theme.compact(top[value_col])}, {ratio:.1f} times the lowest "
            f"({bottom[group_col]}, {theme.compact(bottom[value_col])})."
        )
        return self._save(fig, self._next_id(), f"{label} by {group_col}", caption, "ranking",
                          table=data.iloc[::-1].round(2), table_title=f"{label} by {group_col}")

    # ------------------------------------------------------------------
    def growth_ranking(self, table: pd.DataFrame, value_col: str, group_col: str,
                       label: str, top_n: int = 12) -> Figure | None:
        data = table.dropna(subset=[value_col]).sort_values(value_col, ascending=True)
        if len(data) < 2:
            return None
        if len(data) > top_n:  # keep the extremes, drop the undifferentiated middle
            half = top_n // 2
            data = pd.concat([data.head(half), data.tail(top_n - half)])

        fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.34 * len(data) + 1.3)))
        values = data[value_col].to_numpy(dtype=float)
        colors = [theme.POSITIVE if v >= 0 else theme.NEGATIVE for v in values]
        positions = np.arange(len(data), dtype=float)
        theme.rounded_bars(ax, positions, values, colors, width=0.66, horizontal=True)

        theme.zero_line(ax, horizontal=False)
        ax.set_yticks(positions)
        ax.set_yticklabels([theme.shorten(v, 26) for v in data[group_col]], fontsize=8.5, color=theme.INK_SECONDARY)
        ax.set_ylim(-0.7, len(data) - 0.3)
        theme.value_labels(ax, positions, values, horizontal=True, fmt=lambda v: f"{v:+.1f}%")

        ax.grid(axis="x", color=theme.GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
        # Anchor at zero when every value shares a sign, so the bars are not
        # floating off a baseline that sits in empty space.
        pad = max(abs(values.min()), abs(values.max())) * 0.28
        left = min(values.min() - pad, 0.0) if values.min() < 0 else 0.0
        right = max(values.max() + pad, 0.0) if values.max() > 0 else 0.0
        ax.set_xlim(left, right)
        ax.xaxis.set_major_formatter(theme.smart_formatter(values, percent=True))
        theme.finish(ax)
        theme.title_block(fig, f"{label}: growth by {group_col}",
                          "Blue = growth, red = decline · % change")

        gainers = int((values > 0).sum())
        caption = (
            f"{gainers} of {len(values)} {group_col}s recorded growth. "
            f"The range runs from {values.min():+.1f}% ({data.iloc[0][group_col]}) "
            f"to {values.max():+.1f}% ({data.iloc[-1][group_col]})."
        )
        # Only the ranked column, so this table does not repeat the levels table above.
        columns = [group_col, value_col] + [c for c in ("latest",) if c in data.columns and c != value_col]
        return self._save(fig, self._next_id(), f"{label}: growth by {group_col}", caption, "growth_rank",
                          table=data[columns].iloc[::-1].round(2),
                          table_title=f"{label} growth by {group_col} (%)")

    # ------------------------------------------------------------------
    def volume_bars(self, series: pd.Series, label: str, unit: str = "") -> Figure | None:
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 4:
            return None

        fig, ax = plt.subplots()
        positions = mdates.date2num(s.index) if isinstance(s.index, pd.DatetimeIndex) else np.arange(len(s))
        width = self._bar_width(positions)
        theme.rounded_bars(ax, positions, s.to_numpy(), theme.CATEGORICAL[0], width=width)

        window = min(12, max(3, len(s) // 4))
        moving = s.rolling(window, min_periods=2).mean()
        ax.plot(positions, moving.to_numpy(), color=theme.CATEGORICAL[1], linewidth=1.9,
                zorder=5, label=f"{window}-period moving average")
        ax.plot([], [], color=theme.CATEGORICAL[0], linewidth=6, solid_capstyle="butt", label=label)

        if isinstance(s.index, pd.DatetimeIndex):
            ax.xaxis_date()
            self._time_axis(ax, s.index)
        ax.yaxis.set_major_formatter(theme.smart_formatter(s.to_numpy()))
        ax.set_xlim(positions.min() - width, positions.max() + width)
        ax.set_ylim(0, float(s.max()) * 1.12)
        theme.finish(ax, legend=True, legend_cols=2)
        theme.title_block(fig, f"{label} per period", unit or "level per period")

        caption = (
            f"{label} peaked at {theme.compact(s.max())} ({self._fmt_date(s.idxmax())}) and "
            f"bottomed at {theme.compact(s.min())} ({self._fmt_date(s.idxmin())}). "
            f"The moving average smooths short-term noise to show the underlying run rate."
        )
        table = pd.DataFrame({label: s, f"{window}-period MA": moving.round(2)})
        table.index = [self._fmt_date(i) for i in table.index]
        return self._save(fig, self._next_id(), f"{label} per period", caption, "volume",
                          table=self._thin(table), table_title=f"{label} per period")

    # ------------------------------------------------------------------
    def seasonality_heatmap(self, series: pd.Series, label: str,
                            detected: bool | None = None) -> Figure | None:
        s = pd.Series(series).dropna().astype(float)
        if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 24:
            return None
        if s.index.to_period("M").nunique() < len(s) * 0.8:
            return None

        frame = pd.DataFrame({"year": s.index.year, "month": s.index.month, "value": s.to_numpy()})
        # Show deviation from each year's own mean, so the seasonal shape is visible
        # regardless of the trend in levels.
        frame["deviation"] = frame.groupby("year")["value"].transform(lambda g: (g / g.mean() - 1) * 100)
        grid = frame.pivot_table(index="year", columns="month", values="deviation", aggfunc="mean")
        if grid.shape[0] < 2 or grid.shape[1] < 4:
            return None

        fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.36 * grid.shape[0] + 1.5)))
        limit = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
        mesh = ax.imshow(grid.to_numpy(), cmap=theme.DIVERGING_CMAP, aspect="auto",
                         vmin=-limit, vmax=limit, interpolation="nearest")

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([month_names[m - 1] for m in grid.columns], fontsize=8)
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([str(y) for y in grid.index], fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.iloc[i, j]
                if pd.isna(value):
                    continue
                shade = theme.SURFACE if abs(value) > limit * 0.6 else theme.INK_PRIMARY
                text = "0" if abs(value) < 0.5 else f"{value:+.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=shade)

        bar = fig.colorbar(mesh, ax=ax, fraction=0.028, pad=0.02)
        bar.set_label("% deviation from the year's average", fontsize=8, color=theme.INK_SECONDARY)
        bar.ax.tick_params(labelsize=7.5, length=0, colors=theme.INK_MUTED)
        bar.outline.set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        heading = (
            f"{label}: seasonal pattern by month" if detected
            else f"{label}: within-year profile by month"
        )
        theme.title_block(fig, heading,
                          "Each cell is that month's deviation from its own year's average")

        monthly = grid.mean()
        caption = (
            f"On average {label} runs strongest in "
            f"{month_names[int(monthly.idxmax()) - 1]} ({monthly.max():+.1f}% vs the annual mean) "
            f"and weakest in {month_names[int(monthly.idxmin()) - 1]} ({monthly.min():+.1f}%)."
        )
        if detected is False:
            caption += (
                " Seasonal decomposition found no repeating seasonal component in this series, so "
                "the pattern above is mostly the underlying trend running through each year rather "
                "than a true time-of-year effect."
            )
        elif detected:
            caption += " Seasonal decomposition confirms this pattern repeats from year to year."

        display = grid.round(1)
        display.columns = [month_names[m - 1] for m in grid.columns]
        return self._save(fig, self._next_id(), heading, caption, "seasonality",
                          table=display, table_title=f"{label}: % deviation from the year's average")

    # ------------------------------------------------------------------
    def driver_panels(self, target: pd.Series, target_label: str,
                      driver: pd.Series, driver_label: str,
                      correlation: float | None = None) -> Figure | None:
        """Two stacked panels on a shared time axis — never a second y-scale."""
        t = pd.Series(target).dropna().astype(float)
        d = pd.Series(driver).dropna().astype(float)
        if len(t) < 4 or len(d) < 4:
            return None
        lo = max(t.index.min(), d.index.min())
        hi = min(t.index.max(), d.index.max())
        t, d = t[(t.index >= lo) & (t.index <= hi)], d[(d.index >= lo) & (d.index <= hi)]
        if len(t) < 4 or len(d) < 4:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True,
                                 gridspec_kw={"hspace": 0.32})
        for ax, series, label, color in (
            (axes[0], t, target_label, theme.CATEGORICAL[0]),
            (axes[1], d, driver_label, theme.CATEGORICAL[1]),
        ):
            ax.plot(series.index, series.to_numpy(), color=color, linewidth=1.9, zorder=4)
            ax.scatter([series.index[-1]], [series.iloc[-1]], s=30, color=color, zorder=5,
                       edgecolor=theme.SURFACE, linewidth=1.5)
            ax.yaxis.set_major_formatter(theme.smart_formatter(series.to_numpy()))
            theme.nice_ylim(ax, series.to_numpy())
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(theme.BASELINE)
            ax.annotate(label, xy=(0, 1.03), xycoords="axes fraction", fontsize=9,
                        fontweight="600", color=color, ha="left", va="bottom")

        self._time_axis(axes[1], d.index)
        note = f" · correlation of growth rates r = {correlation:+.2f}" if correlation is not None else ""
        theme.title_block(fig, f"{target_label} against {driver_label}",
                          f"Separate panels, each on its own scale{note}")

        caption = (
            f"{target_label} and {driver_label} are shown on separate panels sharing one time axis; "
            "putting two different scales on one axis would exaggerate or hide the relationship."
        )
        if correlation is not None:
            caption += f" Growth rates co-move with a correlation of {correlation:+.2f}."

        joined = pd.concat([t.rename(target_label), d.rename(driver_label)], axis=1).dropna()
        joined.index = [self._fmt_date(i) for i in joined.index]
        return self._save(fig, self._next_id(), f"{target_label} against {driver_label}",
                          caption, "driver", table=self._thin(joined.round(2)),
                          table_title=f"{target_label} and {driver_label}")

    # ------------------------------------------------------------------
    def correlation_heatmap(self, matrix: pd.DataFrame, title: str = "Correlation of growth rates") -> Figure | None:
        data = matrix.dropna(how="all").dropna(axis=1, how="all")
        if data.shape[0] < 3:
            return None
        data = data.iloc[:8, :8]
        n = data.shape[0]

        # Only the lower triangle carries information: the matrix is symmetric and
        # the diagonal is 1 by construction, so both are masked out.
        values = data.to_numpy(dtype=float).copy()
        mask = np.triu(np.ones_like(values, dtype=bool))
        values[mask] = np.nan

        fig, ax = plt.subplots(figsize=(6.6, max(2.9, 0.46 * n + 1.5)))
        cmap = theme.DIVERGING_CMAP.with_extremes(bad=theme.SURFACE)
        mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1,
                         aspect="auto", interpolation="nearest")

        labels = [theme.shorten(str(c), 24) for c in data.columns]
        ax.set_xticks(range(n - 1))
        ax.set_xticklabels(labels[: n - 1], rotation=32, ha="right", fontsize=7.5,
                           color=theme.INK_SECONDARY)
        ax.set_yticks(range(1, n))
        ax.set_yticklabels(labels[1:], fontsize=7.5, color=theme.INK_SECONDARY)
        ax.set_xlim(-0.5, n - 1.5)
        ax.set_ylim(n - 0.5, 0.5)
        ax.grid(False)
        ax.tick_params(length=0)

        for i in range(n):
            for j in range(n):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                shade = theme.SURFACE if abs(value) > 0.62 else theme.INK_PRIMARY
                weight = "600" if abs(value) >= 0.5 else "normal"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=7.5, color=shade, fontweight=weight)

        bar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("Pearson r", fontsize=8, color=theme.INK_SECONDARY)
        bar.set_ticks([-1, -0.5, 0, 0.5, 1])
        bar.ax.tick_params(labelsize=7.5, length=0, colors=theme.INK_MUTED)
        bar.outline.set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        theme.title_block(fig, title,
                          "Blue = move together, red = move oppositely, grey = no relationship")

        strongest = _strongest_pair(data)
        caption = (
            "Correlations are computed on growth rates rather than levels, because two "
            "independently trending series will correlate strongly in levels for no meaningful "
            "reason. Only the lower triangle is shown; the matrix is symmetric."
        )
        if strongest:
            caption += f" The strongest relationship is {strongest}."
        return self._save(fig, self._next_id(), title, caption, "correlation",
                          table=data.round(2), table_title="Correlation matrix (growth rates)")

    # ------------------------------------------------------------------
    def forecast_chart(self, history: pd.Series, points: list[dict[str, Any]],
                       label: str, method: str = "") -> Figure | None:
        s = pd.Series(history).dropna().astype(float)
        if len(s) < 6 or not points:
            return None

        future_index = pd.to_datetime([p["date"] for p in points])
        central = np.array([p["forecast"] for p in points], dtype=float)
        lower = np.array([p.get("lower", p["forecast"]) for p in points], dtype=float)
        upper = np.array([p.get("upper", p["forecast"]) for p in points], dtype=float)

        # Join the projection to the last actual so the line is continuous.
        link_x = np.concatenate([[s.index[-1]], future_index])
        link_y = np.concatenate([[s.iloc[-1]], central])

        fig, ax = plt.subplots()
        ax.plot(s.index, s.to_numpy(), color=theme.CATEGORICAL[0], linewidth=1.9,
                zorder=4, label="Observed")
        ax.fill_between(future_index, lower, upper, color=theme.CATEGORICAL[1], alpha=0.16, zorder=2)
        ax.plot(link_x, link_y, color=theme.CATEGORICAL[1], linewidth=1.9,
                linestyle=(0, (5, 3)), zorder=4, label="Projection (95% band)")

        ax.scatter([future_index[-1]], [central[-1]], s=30, color=theme.CATEGORICAL[1],
                   zorder=5, edgecolor=theme.SURFACE, linewidth=1.5)
        theme.direct_label(ax, future_index[-1], central[-1], theme.compact(central[-1]), theme.CATEGORICAL[1])

        ax.yaxis.set_major_formatter(theme.smart_formatter(np.concatenate([s.to_numpy(), upper])))
        self._time_axis(ax, pd.DatetimeIndex(list(s.index) + list(future_index)))
        theme.finish(ax, legend=True, legend_cols=2)
        theme.title_block(fig, f"{label}: projection", method or "statistical extrapolation")

        change = (central[-1] / s.iloc[-1] - 1) * 100 if s.iloc[-1] else float("nan")
        caption = (
            f"Extrapolating the observed history {len(points)} period(s) ahead points to "
            f"{theme.compact(central[-1])} ({change:+.1f}% on the latest reading). "
            "The band widens with the horizon and reflects historical variability only — "
            "it excludes policy changes and shocks."
        )
        table = pd.DataFrame(points).set_index("date")
        return self._save(fig, self._next_id(), f"{label}: projection", caption, "forecast",
                          table=table, table_title=f"{label}: projected values")

    # ------------------------------------------------------------------
    def segment_bars(self, tidy: pd.DataFrame, metric: str, label: str) -> Figure | None:
        sub = tidy[(tidy["metric"] == metric)].dropna(subset=["value"])
        if sub.empty or "segment" not in sub.columns or sub["segment"].nunique() < 2:
            return None

        latest = sub[sub["date"] == sub["date"].max()]
        grouped = latest.groupby("segment")["value"].mean().sort_values(ascending=False)
        if len(grouped) < 2:
            return None
        if len(grouped) > 8:
            head = grouped.head(7)
            grouped = pd.concat([head, pd.Series({"Other": grouped.iloc[7:].mean()})])

        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        positions = np.arange(len(grouped), dtype=float)
        theme.rounded_bars(ax, positions, grouped.to_numpy(), theme.CATEGORICAL[0], width=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([theme.shorten(str(s), 16) for s in grouped.index], fontsize=8.5,
                           color=theme.INK_SECONDARY)
        theme.value_labels(ax, positions, grouped.to_numpy())
        ax.yaxis.set_major_formatter(theme.smart_formatter(grouped.to_numpy()))
        ax.set_ylim(0, float(grouped.max()) * 1.16)
        theme.finish(ax)
        theme.title_block(fig, f"{label} by segment, latest period",
                          f"As at {self._fmt_date(sub['date'].max())}")

        spread = grouped.max() / grouped.min() if grouped.min() else float("nan")
        caption = (
            f"{grouped.index[0]} carries the highest reading for {label} at "
            f"{theme.compact(grouped.iloc[0])}, {spread:.1f} times the lowest segment "
            f"({grouped.index[-1]})."
        )
        return self._save(fig, self._next_id(), f"{label} by segment", caption, "segment",
                          table=grouped.round(2).to_frame(label), table_title=f"{label} by segment")

    # ------------------------------------------------------------------
    def policy_timeline(self, events: list[dict[str, Any]], title: str = "Policy and macro timeline") -> Figure | None:
        dated = []
        for event in events:
            try:
                dated.append((pd.Timestamp(str(event.get("date"))[:10]), event))
            except Exception:
                continue
        if len(dated) < 2:
            return None
        dated.sort(key=lambda pair: pair[0])
        dated = dated[-14:]

        direction_color = {
            "positive": theme.POSITIVE,
            "negative": theme.NEGATIVE,
            "mixed": theme.CATEGORICAL[3],
            "neutral": theme.INK_MUTED,
        }

        fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.5 * len(dated) + 1.4)))
        for i, (when, event) in enumerate(dated):
            color = direction_color.get(str(event.get("direction", "neutral")).lower(), theme.INK_MUTED)
            ax.plot([when, when], [i - 0.3, i + 0.16], color=color, linewidth=2.4, solid_capstyle="round")
            ax.scatter([when], [i - 0.3], s=30, color=color, zorder=4,
                       edgecolor=theme.SURFACE, linewidth=1.4)
            # Labels sit above their marker rather than to the right of it, so the
            # time axis does not have to be padded out into empty years.
            ax.annotate(
                f"{when:%b %Y} · {theme.shorten(str(event.get('title', '')), 58)}",
                xy=(when, i + 0.2), xytext=(-3, 0), textcoords="offset points",
                fontsize=8, color=theme.INK_SECONDARY, va="bottom", ha="left",
                annotation_clip=False,
            )

        ax.set_yticks([])
        ax.set_ylim(-0.75, len(dated) - 0.35)
        first, last = dated[0][0], dated[-1][0]
        span = (last - first) if last > first else pd.Timedelta(days=400)
        ax.set_xlim(first - span * 0.05, last + span * 0.08)
        self._time_axis(ax, pd.DatetimeIndex([first, last]))
        ax.grid(axis="x", color=theme.GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)

        handles = [
            plt.Line2D([], [], color=col, linewidth=2.4, label=name.capitalize())
            for name, col in direction_color.items()
            if any(str(e.get("direction", "neutral")).lower() == name for _, e in dated)
        ]
        if len(handles) > 1:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.08),
                      ncol=len(handles), frameon=False)
        theme.finish(ax)
        theme.title_block(fig, title,
                          "Colour shows the expected direction of effect on housing prices")

        caption = (
            f"{len(dated)} dated measures and macro events across the sample period, coloured by "
            "the direction of their expected effect on housing prices."
        )
        table = pd.DataFrame(
            [
                {
                    "Date": f"{when:%Y-%m-%d}",
                    "Event": str(event.get("title", "")),
                    "Type": str(event.get("category", "")),
                    "Expected effect": str(event.get("direction", "")),
                }
                for when, event in dated
            ]
        )
        return self._save(fig, self._next_id(), title, caption, "timeline",
                          table=table, table_title="Policy and macro events plotted above")

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_date(value: Any) -> str:
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return str(value)
        return f"{ts:%Y-%m}" if ts.day == 1 else f"{ts:%Y-%m-%d}"

    @staticmethod
    def _thin(table: pd.DataFrame, max_rows: int = 24) -> pd.DataFrame:
        """Keep report tables readable: sample evenly, always keeping both ends."""
        if len(table) <= max_rows:
            return table.round(2)
        # Round the step up, or the sample still exceeds max_rows and the document
        # truncates it a second time.
        step = max(1, math.ceil(len(table) / (max_rows - 1)))
        sampled = table.iloc[::step]
        if not sampled.index.equals(table.index[-1:]) and table.index[-1] not in sampled.index:
            sampled = pd.concat([sampled, table.iloc[[-1]]])
        return sampled.round(2)


def _strongest_pair(matrix: pd.DataFrame) -> str:
    """Describe the largest off-diagonal correlation, for the figure caption."""
    values = matrix.to_numpy(dtype=float).copy()
    if values.shape[0] < 2:
        return ""
    np.fill_diagonal(values, np.nan)
    values[np.triu_indices_from(values)] = np.nan
    if not np.isfinite(values).any():
        return ""
    i, j = np.unravel_index(np.nanargmax(np.abs(values)), values.shape)
    value = float(values[i, j])
    direction = "moving together" if value > 0 else "moving in opposite directions"
    return f"{matrix.index[i]} and {matrix.columns[j]}, {direction} at r = {value:+.2f}"
