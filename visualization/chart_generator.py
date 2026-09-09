"""
Stage 11: Figure generation with matplotlib. Every figure carries a title,
labeled axes with units, a legend where needed, a source note, and the
observation period. Axes are never truncated in a way that would exaggerate
a change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import FIGURES_DIR
from data.ingestion import SheetData
from models.schemas import FigurePlan

plt.rcParams.update({
    "figure.figsize": (9, 5), "figure.dpi": 140, "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})

SOURCE_NOTE = "Source: uploaded dataset; computed by AI Economic Research Agent."


def _finalize(ax, fig, plan: FigurePlan, unit_label: str, path: Path):
    ax.set_title(plan.title)
    ax.set_xlabel("Period")
    ax.set_ylabel(unit_label)
    if len(ax.get_lines()) > 1 or len(ax.patches) and plan.fig_type in ("bar", "stacked_bar", "growth_bar"):
        ax.legend(loc="best", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, f"{SOURCE_NOTE} Period: {plan.period or 'full sample'}.", fontsize=7.5, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figure(plan: FigurePlan, sd: SheetData) -> Path:
    path = FIGURES_DIR / f"{plan.id}.png"
    df = sd.df.copy()
    date_col = sd.date_column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    fig, ax = plt.subplots()
    unit_label = "Value"

    if plan.fig_type == "line":
        for v in plan.variables:
            if v in df.columns:
                ax.plot(df[date_col], pd.to_numeric(df[v], errors="coerce"), label=v, linewidth=1.8)
        unit_label = plan.variables[0] if len(plan.variables) == 1 else "Value"

    elif plan.fig_type == "growth_bar":
        v = plan.variables[0]
        s = pd.to_numeric(df[v], errors="coerce")
        s.index = df[date_col]
        yoy = s.pct_change(12) * 100 if len(s) > 12 else s.pct_change() * 100
        colors = ["#c0392b" if x < 0 else "#2471a3" for x in yoy.fillna(0)]
        ax.bar(yoy.index, yoy.values, color=colors, width=20)
        unit_label = "YoY change (%)"
        ax.axhline(0, color="black", linewidth=0.8)

    elif plan.fig_type == "indexed_line":
        for v in plan.variables:
            if v in df.columns:
                s = pd.to_numeric(df[v], errors="coerce")
                base = s.dropna().iloc[0] if s.dropna().shape[0] else None
                idx = (s / base * 100.0) if base else s
                ax.plot(df[date_col], idx, label=f"{v} (index)", linewidth=1.8)
        unit_label = "Index (base period = 100)"

    elif plan.fig_type == "dual_axis" and len(plan.variables) >= 2:
        tail = df.tail(24)
        v1, v2 = plan.variables[0], plan.variables[1]
        ax.plot(tail[date_col], pd.to_numeric(tail[v1], errors="coerce"), color="#2471a3", label=v1, linewidth=1.8)
        ax.set_ylabel(v1, color="#2471a3")
        ax2 = ax.twinx()
        ax2.plot(tail[date_col], pd.to_numeric(tail[v2], errors="coerce"), color="#c0392b", label=v2, linewidth=1.8)
        ax2.set_ylabel(v2, color="#c0392b")
        ax2.spines["top"].set_visible(False)
        unit_label = v1

    else:  # generic fallback
        for v in plan.variables:
            if v in df.columns:
                ax.plot(df[date_col], pd.to_numeric(df[v], errors="coerce"), label=v)

    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30)

    _finalize(ax, fig, plan, unit_label, path)
    return path
