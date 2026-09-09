"""
Stage 3-4: Quantitative Analyst agent. Computes every relevant statistic in
Python (never asks the LLM to do arithmetic) and then identifies the
economically important changes in the data -- the central intelligence of
the system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from analysis import anomalies, comparisons, descriptive, relationships, time_series, trends
from data.ingestion import SheetData
from models.schemas import Finding, StatResult

STOPWORDS = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "is", "are", "recent",
             "analyze", "analysis", "identify", "main", "factors", "behind", "what", "why", "how"}


def _topic_keywords(topic: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", topic.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _topic_relevance(col: str, keywords: set[str]) -> float:
    name = col.lower()
    hits = sum(1 for k in keywords if k in name or name in k)
    return min(1.0, hits * 0.5)


@dataclass
class QuantResult:
    stats: list[StatResult]
    findings: list[Finding]
    correlation_matrix: pd.DataFrame | None


def run_quantitative_analysis(sd: SheetData, topic: str) -> QuantResult:
    df, date_col, freq = sd.df, sd.date_column, sd.frequency
    numeric_cols = sd.numeric_columns
    keywords = _topic_keywords(topic)

    all_stats: list[StatResult] = []
    all_stats += descriptive.describe_all(df, numeric_cols)

    findings: list[Finding] = []
    fid = 0

    series_cache: dict[str, dict[str, pd.Series]] = {}

    for col in numeric_cols:
        if not date_col:
            break
        change_stats = time_series.latest_change_stats(df, date_col, col, freq)
        all_stats += change_stats
        series = time_series.compute_change_series(df, date_col, col, freq)
        series_cache[col] = series

        relevance = _topic_relevance(col, keywords)

        # 1) Anomalous latest change
        for key in ("yoy_pct", "mom_pct", "qoq_pct"):
            if key not in series:
                continue
            anomaly = anomalies.latest_is_anomalous(series[key])
            if anomaly:
                fid += 1
                direction = "increase" if anomaly["value"] > anomaly["historical_mean"] else "decrease"
                findings.append(Finding(
                    id=f"F{fid}",
                    finding=f"{col}: latest {key.split('_')[0].upper()} change of {anomaly['value']:.2f}% is "
                            f"statistically unusual (z={anomaly['zscore']:.2f} vs. historical pattern).",
                    importance="high" if abs(anomaly["zscore"]) >= 2.5 or relevance > 0 else "medium",
                    variables=[col], magnitude=f"{anomaly['value']:.2f}%", direction=direction, unusual=True,
                    supporting_stats=[s for s in change_stats if s.stat_type == key],
                ))

        # 2) Trend reversal / acceleration
        tr = trends.compare_recent_vs_prior_trend(series["level"])
        if tr:
            if tr.get("trend_reversed"):
                fid += 1
                findings.append(Finding(
                    id=f"F{fid}", finding=f"{col}: the trend direction reversed between the earlier and more "
                                           f"recent parts of the sample.",
                    importance="high" if relevance > 0 else "medium",
                    variables=[col], direction="reversal", unusual=True,
                ))
            elif tr.get("trend_accelerated"):
                fid += 1
                findings.append(Finding(
                    id=f"F{fid}", finding=f"{col}: the trend has accelerated in the more recent part of the sample "
                                           f"relative to the earlier period.",
                    importance="medium" if relevance > 0 else "low",
                    variables=[col], direction="acceleration",
                ))

        # 3) Current vs historical average
        cmp = comparisons.current_vs_historical_average(series["level"])
        if cmp and cmp.get("pct_difference") is not None and abs(cmp["pct_difference"]) > 15:
            fid += 1
            findings.append(Finding(
                id=f"F{fid}",
                finding=f"{col}: the latest value ({cmp['current']:.2f}) deviates from its historical average "
                        f"({cmp['historical_average']:.2f}) by {cmp['pct_difference']:.1f}%.",
                importance="medium" if relevance > 0 else "low",
                variables=[col], magnitude=f"{cmp['pct_difference']:.1f}%",
                direction="above" if cmp["difference"] > 0 else "below",
            ))

    # 4) Co-movements: correlation between topic-relevant variables and others
    corr_df = None
    if len(numeric_cols) >= 2:
        corr_df = relationships.correlation_matrix(df, numeric_cols)
        relevant_cols = [c for c in numeric_cols if _topic_relevance(c, keywords) > 0] or numeric_cols[:1]
        for rc in relevant_cols:
            if rc not in corr_df.columns:
                continue
            for other in numeric_cols:
                if other == rc or other not in corr_df.columns:
                    continue
                r = corr_df.loc[rc, other]
                if pd.notna(r) and abs(r) >= 0.6:
                    fid += 1
                    co_dir = "moves together with" if r > 0 else "moves opposite to"
                    findings.append(Finding(
                        id=f"F{fid}",
                        finding=f"{rc} {co_dir} {other} over the sample period (correlation r={r:.2f}).",
                        importance="medium", variables=[rc, other],
                        co_movements=[other], possible_explanations=[],
                        supporting_stats=[StatResult(label=f"Correlation {rc} vs {other}", variable=rc,
                                                      stat_type="correlation", value=round(float(r), 3))],
                    ))

    # Rank: high > medium > low, then unusual first, then topic relevance via variable name match
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.importance], not f.unusual,
                                  -_topic_relevance(f.variables[0] if f.variables else "", keywords)))

    return QuantResult(stats=all_stats, findings=findings, correlation_matrix=corr_df)
