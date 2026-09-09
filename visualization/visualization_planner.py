"""
Stage 9: Visualization Planner. Decides which tables and figures are
actually needed to communicate the important findings -- not a fixed set of
default charts. Selection is driven by the ranked findings and the dataset.
"""
from __future__ import annotations

from data.ingestion import SheetData
from models.schemas import Finding, FigurePlan, TablePlan


def plan_tables(topic: str, sd: SheetData, findings: list[Finding]) -> list[TablePlan]:
    plans: list[TablePlan] = []
    top_vars = _top_variables(findings, sd.numeric_columns)

    if top_vars:
        plans.append(TablePlan(
            id="T1", title="Latest Indicators and Recent Changes", table_type="latest_indicators",
            variables=top_vars, period="latest available", purpose="Show current levels and MoM/QoQ/YoY changes "
            "for the variables most relevant to the research topic.",
        ))
    if sd.date_column and len(sd.df) > 12 and top_vars:
        plans.append(TablePlan(
            id="T2", title="Summary Statistics Over the Sample Period", table_type="summary_statistics",
            variables=top_vars, period="full sample", purpose="Provide the descriptive baseline (mean, "
            "std. dev., min, max) against which recent changes can be judged.",
        ))
    unusual = [f for f in findings if f.unusual][:5]
    if unusual:
        plans.append(TablePlan(
            id="T3", title="Economically Important Changes Identified in the Data", table_type="findings_table",
            variables=list({v for f in unusual for v in f.variables}), period="various",
            purpose="Summarize the statistically unusual or economically significant movements detected by the "
            "quantitative analysis engine, ranked by importance.",
        ))
    return plans


def plan_figures(topic: str, sd: SheetData, findings: list[Finding]) -> list[FigurePlan]:
    plans: list[FigurePlan] = []
    top_vars = _top_variables(findings, sd.numeric_columns)
    if not sd.date_column or not top_vars:
        return plans

    fid = 0
    fid += 1
    plans.append(FigurePlan(
        id=f"V{fid}", title=f"{', '.join(top_vars[:3])} Over Time", fig_type="line",
        variables=top_vars[:3], period="full sample",
        purpose="Show the level and trajectory of the key variables identified by the research topic.",
        reason="A time-series line chart is the clearest way to communicate the trend and turning points that "
               "motivate the report's findings.",
        related_finding_ids=[f.id for f in findings if any(v in top_vars[:3] for v in f.variables)][:5],
    ))

    yoy_relevant = [f for f in findings if any("yoy" in (s.stat_type or "") for s in f.supporting_stats)]
    if yoy_relevant and top_vars:
        fid += 1
        plans.append(FigurePlan(
            id=f"V{fid}", title=f"Year-on-Year Growth: {top_vars[0]}", fig_type="growth_bar",
            variables=[top_vars[0]], period="full sample",
            purpose="Visualize the pace of year-on-year change to highlight acceleration/deceleration.",
            reason="Growth-rate charts make turning points and unusual changes visually salient.",
            related_finding_ids=[f.id for f in yoy_relevant][:5],
        ))

    co_moves = [f for f in findings if f.co_movements]
    if co_moves:
        fid += 1
        pair_vars = list(dict.fromkeys(co_moves[0].variables))[:2]
        plans.append(FigurePlan(
            id=f"V{fid}", title=f"{' vs. '.join(pair_vars)}: Indexed Comparison", fig_type="indexed_line",
            variables=pair_vars, period="full sample",
            purpose="Illustrate the co-movement (or divergence) between two economically related variables.",
            reason="Indexing both series to a common base makes relative movements comparable despite "
                   "different units/scales.",
            related_finding_ids=[co_moves[0].id],
        ))

    if len(top_vars) >= 2:
        fid += 1
        plans.append(FigurePlan(
            id=f"V{fid}", title=f"{top_vars[0]} vs. {top_vars[1]}: Recent Period Detail", fig_type="dual_axis",
            variables=top_vars[:2], period="last 24 observations",
            purpose="Zoom in on the most recent period to make the latest dynamics legible.",
            reason="A dual-axis recent-period chart complements the full-sample chart by showing near-term detail "
                   "obscured by long-run scale.",
            related_finding_ids=[f.id for f in findings if f.importance == "high"][:5],
        ))

    return plans


def _top_variables(findings: list[Finding], available: list[str], max_n: int = 4) -> list[str]:
    ordered: list[str] = []
    for f in findings:
        for v in f.variables:
            if v in available and v not in ordered:
                ordered.append(v)
    if not ordered:
        ordered = list(available[:max_n])
    return ordered[:max_n]
