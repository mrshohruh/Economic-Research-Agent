"""
Stage 10: Table generation with pandas. Tables are always recomputed from
the source data -- numbers are never hand-typed into the report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import descriptive, time_series
from config.settings import TABLES_DIR
from data.ingestion import SheetData
from models.schemas import Finding, TablePlan


def build_table(plan: TablePlan, sd: SheetData, findings: list[Finding]) -> pd.DataFrame:
    if plan.table_type == "latest_indicators":
        return _latest_indicators_table(sd, plan.variables)
    if plan.table_type == "summary_statistics":
        return _summary_statistics_table(sd, plan.variables)
    if plan.table_type == "findings_table":
        return _findings_table(findings)
    return pd.DataFrame()


def _latest_indicators_table(sd: SheetData, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if not isinstance(v, str) or v not in sd.df.columns:
            continue
        stats = time_series.latest_change_stats(sd.df, sd.date_column, v, sd.frequency)
        row = {"Variable": v}
        for s in stats:
            if s.stat_type == "latest_level":
                row["Latest value"] = round(s.value, 3) if s.value is not None else None
                row["Latest period"] = s.period
            elif s.stat_type in ("mom_pct", "qoq_pct", "yoy_pct"):
                row[s.stat_type.split("_")[0].upper() + " change (%)"] = s.value
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_statistics_table(sd: SheetData, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if not isinstance(v, str) or v not in sd.df.columns:
            continue
        r = descriptive.describe_variable(sd.df, v)
        rows.append({"Variable": v, **r.details})
    return pd.DataFrame(rows)


def _findings_table(findings: list[Finding]) -> pd.DataFrame:
    unusual = [f for f in findings if f.unusual][:8]
    rows = [{
        "Finding": f.finding, "Variables": ", ".join(f.variables), "Importance": f.importance.capitalize(),
        "Direction": f.direction, "Magnitude": f.magnitude,
    } for f in unusual]
    return pd.DataFrame(rows)


def save_table(plan: TablePlan, df: pd.DataFrame) -> Path:
    path = TABLES_DIR / f"{plan.id}.csv"
    df.to_csv(path, index=False)
    return path
