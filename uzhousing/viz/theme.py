"""Chart styling: palette, matplotlib defaults and drawing helpers.

Charts are rendered as PNGs for a printed Word document, so only the light-mode
palette is used. Colours come from the reference data-visualisation palette
unchanged; the slot ORDER is the colour-vision-deficiency safety mechanism, so
hues are assigned in fixed order and never cycled.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a server / CI box

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter

# --- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- categorical slots, in fixed order --------------------------------------
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Only the first three slots clear the all-pairs gate; forms that put every
# series against every other (scatter, small multiples) must not exceed this.
ALL_PAIRS_CAP = 3

# --- sequential (magnitude) -------------------------------------------------
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("uz_seq", SEQUENTIAL_STEPS)

# --- diverging (polarity): blue <-> red with a neutral grey midpoint ---------
# Positive is blue and negative is red everywhere in this system, matching the
# growth bars, so the two ends must be ordered low -> high as red -> blue.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "uz_div", ["#9e2423", "#e34948", "#f0a9a8", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"]
)
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL_FILL = "#f0efec"

# --- status (reserved; never used as a series colour) -----------------------
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

FIGSIZE = (7.0, 3.7)
DPI = 200


def apply_style() -> None:
    """Set the global matplotlib defaults once per process."""
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.4,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
        }
    )


def series_colors(n: int, all_pairs: bool = False) -> list[str]:
    """Hues in fixed slot order. Never cycles: callers must fold the tail into 'Other'."""
    cap = ALL_PAIRS_CAP if all_pairs else len(CATEGORICAL)
    return CATEGORICAL[: min(n, cap)]


def finish(ax, ylabel: str = "", legend: bool = False, legend_cols: int = 4) -> None:
    """Apply the shared chrome: recessive axes and an optional legend below the plot.

    Titles are NOT set here — they are drawn by :func:`title_block` in figure
    coordinates after layout, which is the only way to guarantee the title and
    subtitle never land on top of each other or on the plot.
    """
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=legend_cols, frameon=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


TITLE_INCHES = 0.26
SUBTITLE_INCHES = 0.21


def title_block(fig, title: str, subtitle: str = "", x: float = 0.008) -> None:
    """Lay the figure out, reserve headroom, then draw the title above the plot.

    Call this instead of ``fig.tight_layout()``. Because the header is measured in
    inches and converted to figure fractions, it works identically for a short
    two-panel figure and a tall ranking chart.
    """
    try:
        with warnings.catch_warnings():
            # Colorbar axes are not tight_layout-compatible; the warning is noise
            # because the explicit headroom below is what actually positions things.
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    except Exception:  # pragma: no cover - other odd layouts
        pass
    if not title:
        return

    height = float(fig.get_size_inches()[1]) or 1.0
    header = TITLE_INCHES + (SUBTITLE_INCHES if subtitle else 0.0) + 0.10
    top = max(0.55, 1.0 - header / height)
    fig.subplots_adjust(top=top)

    y = 1.0 - 0.05 / height
    fig.text(x, y, title, ha="left", va="top",
             fontsize=11, fontweight="600", color=INK_PRIMARY)
    if subtitle:
        fig.text(x, y - TITLE_INCHES / height, subtitle, ha="left", va="top",
                 fontsize=8.5, color=INK_SECONDARY)


def declutter(values: Sequence[float], min_gap: float) -> list[float]:
    """Nudge end-of-line labels apart so they never overlap.

    Returns a y position for each input value, in the same order, preserving the
    original ranking but enforcing at least ``min_gap`` between neighbours.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    for position, index in enumerate(order):
        if position == 0:
            continue
        previous = adjusted[order[position - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    return adjusted


def zero_line(ax, horizontal: bool = True) -> None:
    if horizontal:
        ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
    else:
        ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)


# ---------------------------------------------------------------------------
# Rounded data-end bars
# ---------------------------------------------------------------------------
def rounded_bars(
    ax,
    positions: Sequence[float],
    values: Sequence[float],
    colors: Sequence[str] | str,
    width: float = 0.72,
    horizontal: bool = False,
    radius_frac: float = 0.28,
    zorder: int = 3,
) -> list[PathPatch]:
    """Draw bars whose data end is rounded and whose baseline end is square."""
    if isinstance(colors, str):
        colors = [colors] * len(values)

    finite = [abs(float(v)) for v in values if v is not None and np.isfinite(v)]
    scale = max(finite) if finite else 1.0
    patches = []

    for pos, value, color in zip(positions, values, colors):
        if value is None or not np.isfinite(value):
            continue
        value = float(value)
        thickness = width
        # Radius is capped so short bars do not turn into lozenges.
        radius = min(thickness * radius_frac, abs(value) * 0.45)
        if scale:
            radius = min(radius, scale * 0.04)

        if horizontal:
            path = _rounded_path(0, pos - thickness / 2, value, thickness, radius, along_x=True)
        else:
            path = _rounded_path(pos - thickness / 2, 0, thickness, value, radius, along_x=False)

        patch = PathPatch(path, facecolor=color, edgecolor="none", zorder=zorder)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def _rounded_path(x: float, y: float, w: float, h: float, r: float, along_x: bool) -> MplPath:
    """Rectangle with the two corners at the far (data) end rounded."""
    k = 0.5523  # circular arc via cubic Bezier
    verts: list[tuple[float, float]] = []
    codes: list[int] = []

    def move(px, py):
        verts.append((px, py)); codes.append(MplPath.MOVETO)

    def line(px, py):
        verts.append((px, py)); codes.append(MplPath.LINETO)

    def curve(c1, c2, end):
        verts.extend([c1, c2, end]); codes.extend([MplPath.CURVE4] * 3)

    if along_x:
        sign = 1.0 if w >= 0 else -1.0
        length, r = abs(w), min(r, abs(w) / 2, abs(h) / 2)
        x0, x1 = x, x + sign * length
        y0, y1 = y, y + h
        move(x0, y0)
        line(x1 - sign * r, y0)
        curve((x1 - sign * r * (1 - k), y0), (x1, y0 + r * (1 - k)), (x1, y0 + r))
        line(x1, y1 - r)
        curve((x1, y1 - r * (1 - k)), (x1 - sign * r * (1 - k), y1), (x1 - sign * r, y1))
        line(x0, y1)
    else:
        sign = 1.0 if h >= 0 else -1.0
        height, r = abs(h), min(r, abs(h) / 2, abs(w) / 2)
        x0, x1 = x, x + w
        y0, y1 = y, y + sign * height
        move(x0, y0)
        line(x0, y1 - sign * r)
        curve((x0, y1 - sign * r * (1 - k)), (x0 + r * (1 - k), y1), (x0 + r, y1))
        line(x1 - r, y1)
        curve((x1 - r * (1 - k), y1), (x1, y1 - sign * r * (1 - k)), (x1, y1 - sign * r))
        line(x1, y0)

    codes.append(MplPath.CLOSEPOLY)
    verts.append(verts[0])
    return MplPath(verts, codes)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def compact(value: float, decimals: int = 1) -> str:
    """1_234_567 -> '1.2M'."""
    if value is None or not np.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{value / threshold:,.{decimals}f}{suffix}"
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.{decimals}f}"
    return f"{value:,.2f}"


def smart_formatter(values: Iterable[float], percent: bool = False) -> FuncFormatter:
    """Axis formatter that picks a sensible scale for the data's magnitude."""
    finite = [abs(float(v)) for v in values if v is not None and np.isfinite(float(v))]
    peak = max(finite) if finite else 0.0
    if percent:
        return FuncFormatter(lambda v, _: f"{v:,.0f}%" if abs(v) >= 10 else f"{v:,.1f}%")
    if peak >= 1000:
        return FuncFormatter(lambda v, _: compact(v, 0 if peak >= 1e6 else 1))
    if peak >= 10:
        return FuncFormatter(lambda v, _: f"{v:,.0f}")
    return FuncFormatter(lambda v, _: f"{v:,.2f}")


def direct_label(ax, x, y, text: str, color: str, dx: float = 6, dy: float = 0, weight: str = "600") -> None:
    """Label a series at its end so identity never depends on colour alone."""
    ax.annotate(
        text,
        xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        color=color, fontsize=8.5, fontweight=weight, va="center", ha="left",
        annotation_clip=False,
    )


def value_labels(ax, positions, values, horizontal: bool = False, fmt=None, color: str = INK_SECONDARY) -> None:
    """Selective value labels — used where a chart's colours sit below 3:1 contrast."""
    fmt = fmt or (lambda v: compact(v))
    for pos, value in zip(positions, values):
        if value is None or not np.isfinite(value):
            continue
        if horizontal:
            offset = 5 if value >= 0 else -5
            ax.annotate(fmt(value), xy=(value, pos), xytext=(offset, 0), textcoords="offset points",
                        va="center", ha="left" if value >= 0 else "right", fontsize=8, color=color)
        else:
            offset = 5 if value >= 0 else -5
            ax.annotate(fmt(value), xy=(pos, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=8, color=color)


def shorten(text: str, limit: int = 22) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def nice_ylim(ax, values: Sequence[float], pad: float = 0.08) -> None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    if math.isclose(lo, hi):
        lo, hi = lo - abs(lo or 1) * 0.1, hi + abs(hi or 1) * 0.1
    span = hi - lo
    ax.set_ylim(lo - span * pad, hi + span * pad)
