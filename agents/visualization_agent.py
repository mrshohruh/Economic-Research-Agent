"""
Stage 9-12: Visualization Agent. Orchestrates the visualization planner and
generators, then writes the grounded economic analysis text for each table
and figure (never "Figure 1 shows X declining" without explaining why it
matters).
"""
from __future__ import annotations

import pandas as pd

from agents.economist_agent import write_visual_analysis
from data.ingestion import SheetData
from models.schemas import Finding, FigurePlan, TablePlan
from visualization import chart_generator, table_generator, visualization_planner


def build_tables(topic: str, sd: SheetData, findings: list[Finding]) -> dict[str, tuple[TablePlan, pd.DataFrame]]:
    plans = visualization_planner.plan_tables(topic, sd, findings)
    out: dict[str, tuple[TablePlan, pd.DataFrame]] = {}
    for plan in plans:
        df = table_generator.build_table(plan, sd, findings)
        if df.empty:
            continue
        path = table_generator.save_table(plan, df)
        plan.file_path = str(path)
        plan.dataframe_json = df.to_json(orient="records")
        key_numbers = _summarize_df(df)
        plan.analysis_text = write_visual_analysis(plan.title, plan.purpose, key_numbers, topic)
        out[plan.id] = (plan, df)
    return out


def build_figures(topic: str, sd: SheetData, findings: list[Finding]) -> dict[str, FigurePlan]:
    plans = visualization_planner.plan_figures(topic, sd, findings)
    out: dict[str, FigurePlan] = {}
    for plan in plans:
        try:
            path = chart_generator.generate_figure(plan, sd)
        except Exception:
            continue
        plan.file_path = str(path)
        related = [f for f in findings if f.id in plan.related_finding_ids]
        key_numbers = "; ".join(f.finding for f in related) or "See underlying data."
        plan.analysis_text = write_visual_analysis(plan.title, plan.purpose, key_numbers, topic)
        out[plan.id] = plan
    return out


def _summarize_df(df: pd.DataFrame, max_rows: int = 6) -> str:
    parts = []
    for _, row in df.head(max_rows).iterrows():
        parts.append(", ".join(f"{c}={row[c]}" for c in df.columns if pd.notna(row[c])))
    return " | ".join(parts)
