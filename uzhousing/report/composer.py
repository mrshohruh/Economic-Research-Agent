"""Assemble the finished Word report from every piece the pipeline produced."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..viz.charts import Figure
from .docx_builder import INK_MUTED, INK_SECONDARY, ReportDocument
from .narrative import Narrative

LOGGER = logging.getLogger(__name__)

HEADINGS: dict[str, dict[str, str]] = {
    "en": {
        "contents": "Contents",
        "exec": "1. Executive summary",
        "findings": "Key findings",
        "data": "2. Data and methodology",
        "current": "3. Current state of the market",
        "trends": "4. Historical trends and turning points",
        "regional": "5. Regional and segment analysis",
        "drivers": "6. What is driving the market",
        "policy": "7. Policy environment and its transmission",
        "macro": "8. Macroeconomic and external context",
        "outlook": "9. Outlook",
        "recs": "10. Recommendations",
        "risks": "11. Risks",
        "limits": "12. Limitations and data quality",
        "sources": "13. Sources",
        "appendix": "Appendix A. Supporting evidence",
    },
    "ru": {
        "contents": "Содержание",
        "exec": "1. Краткое резюме",
        "findings": "Ключевые выводы",
        "data": "2. Данные и методология",
        "current": "3. Текущее состояние рынка",
        "trends": "4. Исторические тенденции и точки перелома",
        "regional": "5. Региональный и сегментный анализ",
        "drivers": "6. Факторы, определяющие рынок",
        "policy": "7. Политика и механизмы её влияния",
        "macro": "8. Макроэкономический и внешний контекст",
        "outlook": "9. Прогноз",
        "recs": "10. Рекомендации",
        "risks": "11. Риски",
        "limits": "12. Ограничения и качество данных",
        "sources": "13. Источники",
        "appendix": "Приложение А. Подтверждающие материалы",
    },
    "uz": {
        "contents": "Mundarija",
        "exec": "1. Qisqacha xulosa",
        "findings": "Asosiy topilmalar",
        "data": "2. Ma'lumotlar va metodologiya",
        "current": "3. Bozorning hozirgi holati",
        "trends": "4. Tarixiy tendensiyalar va burilish nuqtalari",
        "regional": "5. Hududiy va segment tahlili",
        "drivers": "6. Bozorni harakatga keltiruvchi omillar",
        "policy": "7. Siyosat muhiti va uning ta'siri",
        "macro": "8. Makroiqtisodiy va tashqi kontekst",
        "outlook": "9. Istiqbol",
        "recs": "10. Tavsiyalar",
        "risks": "11. Xavflar",
        "limits": "12. Cheklovlar va ma'lumot sifati",
        "sources": "13. Manbalar",
        "appendix": "A ilova. Qo'shimcha dalillar",
    },
}


def compose(
    *,
    output_path: Path,
    title: str,
    subtitle: str,
    analysis: Any,
    narrative: Narrative,
    research: Any,
    figures: list[Figure],
    settings: Any,
    run_meta: dict[str, str],
    language: str = "en",
) -> Path:
    """Write the report and return the saved path."""
    labels = HEADINGS.get(language, HEADINGS["en"])
    doc = ReportDocument(title=title, subtitle=subtitle)
    by_kind = _index_figures(figures)
    brief = analysis.brief

    # ---- cover & contents ---------------------------------------------
    doc.cover(run_meta)
    doc.toc(labels["contents"])
    doc.footer(title)

    # ---- 1. executive summary -----------------------------------------
    doc.heading(labels["exec"], 1)
    doc.paragraphs(narrative.executive_summary)
    if narrative.key_findings:
        doc.heading(labels["findings"], 2)
        doc.bullets(narrative.key_findings)
    _place(doc, by_kind, "trend")

    doc.page_break()

    # ---- 2. data and methodology --------------------------------------
    doc.heading(labels["data"], 1)
    _data_section(doc, analysis, research, settings, narrative)

    # ---- 3. current state ---------------------------------------------
    doc.heading(labels["current"], 1)
    doc.paragraphs(narrative.current_situation)

    section = getattr(analysis, "cross_section", None)
    if section is not None and section.available:
        _price_level_section(doc, section, by_kind)

    summary_table = _metric_summary_table(analysis)
    if summary_table is not None:
        doc.table(summary_table, title="Indicator dashboard: latest levels and growth",
                  include_index=False, max_rows=20,
                  note="Growth figures are computed from the supplied data. "
                       "CAGR is the compound annual rate across the whole sample.")
    _place(doc, by_kind, "segment")
    _place(doc, by_kind, "volume")

    # ---- 4. historical trends ------------------------------------------
    doc.heading(labels["trends"], 1)
    doc.paragraphs(narrative.historical_trends)
    _place(doc, by_kind, "yoy")
    breaks = _breakpoint_table(analysis)
    if breaks is not None:
        doc.table(breaks, title="Detected structural breaks in the headline series",
                  include_index=False,
                  note="Breaks are found by binary segmentation on the mean level. A break marks "
                       "where the level shifted, not necessarily why.")
    _place(doc, by_kind, "seasonality")

    # ---- 5. regional ----------------------------------------------------
    if narrative.regional_analysis or by_kind.get("index") or by_kind.get("ranking") or section:
        doc.heading(labels["regional"], 1)
        doc.paragraphs(narrative.regional_analysis)
        if section is not None and section.available:
            _regional_price_section(doc, section, by_kind)
        _place(doc, by_kind, "index")
        _place(doc, by_kind, "ranking")
        _place(doc, by_kind, "growth_rank")

    # ---- 6. drivers -----------------------------------------------------
    doc.heading(labels["drivers"], 1)
    doc.paragraphs(narrative.drivers)
    _place(doc, by_kind, "correlation")
    for figure in by_kind.get("driver", []):
        _emit(doc, figure)
    for key in ("drivers", "activity_drivers"):
        block = analysis.brief.get(key) or {}
        links = _driver_table(block)
        if links is None:
            continue
        target = block.get("target_label") or block.get("target")
        doc.table(
            links, title=f"Candidate drivers of {target}, ranked by association",
            include_index=False,
            note=(
                f"Computed on {block.get('basis', 'growth rates')}. “Best lag” is how many periods "
                "the driver must be shifted forward to maximise the correlation — a positive value "
                f"means it moves before {target}. " + str(block.get("basis_note", ""))
            ).strip(),
        )
        regression = _regression_table(block)
        if regression is not None:
            doc.table(regression, title=f"Multivariate regression on {target} growth",
                      include_index=False,
                      note=(block.get("regression") or {}).get("note", ""))

    # ---- 7. policy -------------------------------------------------------
    doc.heading(labels["policy"], 1)
    doc.paragraphs(narrative.policy_analysis)
    _place(doc, by_kind, "timeline")
    policy = _policy_table(research)
    if policy is not None:
        doc.table(policy, title="Policy and macro measures affecting the housing market",
                  include_index=False, max_rows=24,
                  note="Confidence reflects how well the entry is evidenced. "
                       "“Local record” entries come from knowledge/policy_events.json; "
                       "“web research” entries were retrieved during this run and are cited in Sources.")

    # ---- 8. macro --------------------------------------------------------
    doc.heading(labels["macro"], 1)
    doc.paragraphs(narrative.macro_context)
    macro = _macro_table(research)
    if macro is not None:
        doc.table(macro, title="Macroeconomic factors and their housing-market channel",
                  include_index=False)

    # ---- 9. outlook ------------------------------------------------------
    doc.heading(labels["outlook"], 1)
    doc.paragraphs(narrative.outlook)
    _place(doc, by_kind, "forecast")

    # ---- 10. recommendations ---------------------------------------------
    doc.heading(labels["recs"], 1)
    recommendations = _recommendation_table(narrative)
    if recommendations is not None:
        doc.para(
            "Each recommendation names the body best placed to act, the action itself, and the "
            "evidence in this report that motivates it.",
            color=INK_SECONDARY,
        )
        doc.table(recommendations, title="Recommended actions", include_index=False, max_rows=20)
    else:
        doc.para("No recommendations were generated for this run.")

    # ---- 11. risks --------------------------------------------------------
    doc.heading(labels["risks"], 1)
    doc.bullets(narrative.risks)

    # ---- 12. limitations --------------------------------------------------
    doc.heading(labels["limits"], 1)
    doc.bullets(narrative.limitations)

    # ---- 13. sources -------------------------------------------------------
    doc.heading(labels["sources"], 1)
    _sources_section(doc, research, brief)

    # ---- appendix ----------------------------------------------------------
    remaining = [f for figures_of_kind in by_kind.values() for f in figures_of_kind if not f.__dict__.get("_placed")]
    themes = getattr(research, "themes", [])
    if remaining or themes:
        doc.page_break()
        doc.heading(labels["appendix"], 1)
        for figure in remaining:
            _emit(doc, figure)
        for theme in themes:
            doc.heading(theme.title, 2)
            if theme.summary:
                doc.para(theme.summary)
            if theme.points:
                doc.bullets(theme.points)
            for source in theme.sources[:4]:
                doc.hyperlink(source.get("title", source.get("url", "")), source.get("url", ""), prefix="→ ")

    return doc.save(output_path)


# ---------------------------------------------------------------------------
def _index_figures(figures: list[Figure]) -> dict[str, list[Figure]]:
    out: dict[str, list[Figure]] = {}
    for figure in figures:
        out.setdefault(figure.kind, []).append(figure)
    return out


def _place(doc: ReportDocument, by_kind: dict[str, list[Figure]], kind: str, limit: int = 1,
           with_table: bool = True) -> None:
    for figure in by_kind.get(kind, [])[:limit]:
        _emit(doc, figure, with_table=with_table)


# Kinds whose underlying table is a ranked list, so the row numbers add nothing.
_NO_INDEX_KINDS = {
    "ranking", "growth_rank", "timeline", "forecast", "distribution", "category",
    "cs_region_level", "cs_region_sqm", "cs_region_coverage", "cs_city_level",
}


def _emit(doc: ReportDocument, figure: Figure, with_table: bool = True) -> None:
    if figure.__dict__.get("_placed"):
        return
    figure.__dict__["_placed"] = True
    doc.figure(figure.path, caption=figure.caption, title=figure.title)
    if with_table and figure.table is not None and len(figure.table):
        doc.table(figure.table, title=figure.table_title or figure.title,
                  include_index=figure.kind not in _NO_INDEX_KINDS,
                  index_label="Period" if figure.kind in {"trend", "yoy", "volume", "index", "driver"} else "",
                  max_rows=26)


# ---------------------------------------------------------------------------
# Cross-sectional sections (property microdata)
# ---------------------------------------------------------------------------
def _price_level_section(doc: ReportDocument, section: Any, by_kind: dict[str, list[Figure]]) -> None:
    """What a property costs: the level, the spread, and what moves it."""
    overall = section.overall
    label = section.price_label
    money = _money_unit(section)

    doc.heading(f"What a property costs: {label}", 2)
    doc.para(
        f"The sample holds {overall.get('listings', 0):,} individual adverts. "
        f"The median {label} is {money}{overall.get('median', 0):,.0f}"
        + (
            f", or {money}{overall['median_per_sqm']:,.2f} per square metre"
            if overall.get("median_per_sqm") is not None
            else ""
        )
        + f", on a median floor area of {overall.get('median_area_sqm', 0):,.0f} m². "
        + (
            f"The mean of {money}{overall.get('mean', 0):,.0f} sits "
            f"{overall['skew_mean_over_median_pct']:.0f}% above the median, which is the "
            "signature of a right-skewed market: a thin band of high-end property pulls the "
            "average above what a typical household faces. The median is used throughout this "
            "report for that reason."
            if overall.get("skew_mean_over_median_pct")
            else ""
        )
    )
    if section.currency_note:
        doc.para(section.currency_note, size=9, color=INK_SECONDARY)

    doc.table(
        _overall_table(overall, label, money),
        title=f"{label.capitalize()}: distribution across the whole sample",
        include_index=False,
        note="Percentiles are computed directly from the listing records. Half of all adverts "
             "fall between the 25th and 75th percentile.",
    )
    _place(doc, by_kind, "distribution")

    for figure in by_kind.get("category", []):
        _emit(doc, figure)


def _regional_price_section(doc: ReportDocument, section: Any, by_kind: dict[str, list[Figure]]) -> None:
    """Where it is expensive — the question the regional section exists to answer."""
    region = section.by_region
    if region is None or region.table.empty:
        return

    label = section.price_label
    money = _money_unit(section)
    top, bottom = region.most_expensive, region.cheapest
    field_name = region.field_name

    doc.heading(f"Which {field_name} is most expensive", 2)
    doc.para(
        f"{top.get(field_name)} is the most expensive {field_name} in the sample, with a median "
        f"{label} of {money}{top.get('median', 0):,.0f}"
        + (
            f" ({money}{top['median_per_sqm']:,.2f} per m²)"
            if top.get("median_per_sqm") is not None
            else ""
        )
        + f". The cheapest is {bottom.get(field_name)} at {money}{bottom.get('median', 0):,.0f}"
        + (
            f" ({money}{bottom['median_per_sqm']:,.2f} per m²)"
            if bottom.get("median_per_sqm") is not None
            else ""
        )
        + (
            f", so the most expensive {field_name} runs {region.spread_ratio:.2f} times the cheapest. "
            if region.spread_ratio
            else ". "
        )
        + (
            f"Across regions the coefficient of variation of the median is {region.dispersion_pct:.1f}%, "
            "which measures how unequal the country's markets are in a single number."
            if region.dispersion_pct is not None
            else ""
        )
    )

    # The charts and this one table say the same thing, so the figures are
    # emitted without repeating their own copy of the numbers underneath.
    _place(doc, by_kind, "cs_region_level", with_table=False)
    doc.table(
        _region_table(region.table, field_name, label, money),
        title=f"{label.capitalize()} by {field_name}, ranked most to least expensive",
        include_index=False,
        max_rows=30,
        note=(
            "Median and mean are computed from the individual adverts in each group. "
            "“vs national” compares each group's median with the median of the whole sample. "
            "Groups marked thin hold too few adverts to rank confidently."
        ),
    )
    _place(doc, by_kind, "cs_region_sqm", with_table=False)
    _place(doc, by_kind, "cs_region_coverage", with_table=False)

    city = section.by_city
    if city is not None and not city.table.empty and len(city.table) >= 3:
        _place(doc, by_kind, "cs_city_level", with_table=False)
        doc.table(
            _region_table(city.table, city.field_name, label, money),
            title=f"{label.capitalize()} by city, best-covered cities",
            include_index=False,
            max_rows=20,
            note="Restricted to cities with enough adverts to support a median.",
        )


def _money_unit(section: Any) -> str:
    return "$" if "USD" in (section.currency_note or "") or "dollar" in (section.currency_note or "") else ""


def _overall_table(overall: dict[str, Any], label: str, money: str) -> pd.DataFrame:
    rows = [
        ("Listings in the sample", f"{overall.get('listings', 0):,}"),
        ("Median", f"{money}{overall.get('median', 0):,.0f}"),
        ("Mean", f"{money}{overall.get('mean', 0):,.0f}"),
        ("25th percentile", f"{money}{overall.get('p25', 0):,.0f}"),
        ("75th percentile", f"{money}{overall.get('p75', 0):,.0f}"),
        ("90th percentile", f"{money}{overall.get('p90', 0):,.0f}"),
        ("Lowest advert", f"{money}{overall.get('minimum', 0):,.0f}"),
        ("Highest advert", f"{money}{overall.get('maximum', 0):,.0f}"),
    ]
    if overall.get("median_per_sqm") is not None:
        rows.append(("Median per m²", f"{money}{overall['median_per_sqm']:,.2f}"))
    if overall.get("median_area_sqm") is not None:
        rows.append(("Median floor area", f"{overall['median_area_sqm']:,.0f} m²"))
    if overall.get("median_rooms") is not None:
        rows.append(("Median rooms", f"{overall['median_rooms']:,.0f}"))
    return pd.DataFrame(rows, columns=["Statistic", label.capitalize()])


def _region_table(table: pd.DataFrame, field_name: str, label: str, money: str) -> pd.DataFrame:
    """Rename the computed columns into something a reader can scan."""
    out = pd.DataFrame()
    out[field_name.capitalize()] = table[field_name].astype(str)
    out["Listings"] = table["listings"].astype(int)
    out["Share %"] = table.get("share_of_listings_pct")
    out[f"Median {label} ({money.strip() or 'level'})"] = table["median"]
    out["Mean"] = table["mean"]
    if "median_per_sqm" in table.columns:
        out["Median per m²"] = table["median_per_sqm"]
    if "median_area_sqm" in table.columns:
        out["Median area m²"] = table["median_area_sqm"]
    out["vs national %"] = table.get("vs_national_pct")
    out["Sample"] = table.get("sample")
    return out


# ---------------------------------------------------------------------------
def _data_section(doc: ReportDocument, analysis: Any, research: Any, settings: Any,
                  narrative: Narrative) -> None:
    understanding = analysis.understanding
    profile = understanding.primary_profile
    dataset = understanding.dataset

    doc.para(
        f"The analysis is built on {dataset.total_rows:,} rows loaded from a {dataset.kind} source. "
        f"The agent profiled {len(dataset.tables)} table(s) and selected “{profile.name}” as the "
        f"centrepiece because it carries a time dimension, "
        f"{len(profile.metric_cols)} measured indicator(s)"
        + (f" and a {profile.region_col} breakdown" if profile.region_col else "")
        + f". Observations run at {profile.grain} frequency."
    )

    overview = pd.DataFrame(
        [
            {
                "Table": name,
                "Rows": p.rows,
                "Columns": len(p.columns),
                "Time column": p.date_col or "—",
                "Frequency": p.grain,
                "Indicators": len(p.metric_cols),
                "Selected": "yes" if name == understanding.primary else "no",
            }
            for name, p in understanding.profiles.items()
        ]
    )
    doc.table(overview, title="Tables found in the source", include_index=False)

    roles = pd.DataFrame(
        [
            {
                "Column": c.name,
                "Detected role": c.role,
                "Unit": c.unit or "—",
                "Missing %": c.missing_pct,
                "Distinct": c.unique,
                "How it was identified": c.reason,
            }
            for c in profile.columns
        ]
    )
    doc.table(roles, title=f"Column roles inferred for “{profile.name}”", include_index=False, max_rows=28)

    doc.heading("Method", 2)
    method_steps = [
        "Ingestion: the file is parsed (JSON, SQL script, SQLite, CSV or Excel), column names are "
        "normalised, and numeric text such as “1 234,5” or “12%” is coerced to numbers.",
        "Understanding: every column is assigned a semantic role — date, region, segment, price, "
        "transaction volume, mortgage, rate, income and so on — from multilingual name patterns and "
        "value checks"
        + (", then reviewed and corrected by the language model." if understanding.llm_reviewed else "."),
        "Statistics: levels, period and year-on-year growth, CAGR, volatility and drawdown from peak; "
        "a linear trend test; STL seasonal decomposition; binary-segmentation break detection; "
        "turning points; and a Holt-Winters projection.",
        "Attribution: every other indicator is tested against the headline series on growth rates, "
        "with lead/lag scanning, followed by a multivariate OLS regression on the strongest candidates.",
        "Context: policy and macro research is gathered from the web"
        + (" and synthesised by the language model" if getattr(research, "llm_used", False) else "")
        + ", and merged with the local policy record. Sources are cited in section 13.",
        "Reporting: figures, tables and prose are assembled into this document. Every figure is "
        "accompanied by the numbers behind it.",
    ]
    doc.bullets(method_steps)

    engine = pd.DataFrame(
        [
            {"Component": "Narrative writer", "Mode": narrative.generated_by},
            {"Component": "Schema review", "Mode": "language model" if understanding.llm_reviewed else "heuristics only"},
            {"Component": "Policy research", "Mode": "web + local record" if getattr(research, "web_used", False) else "local record only"},
            {"Component": "Research synthesis", "Mode": "language model" if getattr(research, "llm_used", False) else "evidence digest"},
            {"Component": "Report language", "Mode": settings.language},
        ]
    )
    doc.table(engine, title="How this run was produced", include_index=False)

    notes = list(analysis.notes) + list(getattr(research, "notes", []))
    if notes:
        doc.heading("Data quality notes", 2)
        doc.bullets(notes)


# ---------------------------------------------------------------------------
def _metric_summary_table(analysis: Any) -> pd.DataFrame | None:
    rows = []
    for item in analysis.brief.get("metrics", []):
        is_rate = item.get("change_unit") == "pp"
        rows.append(
            {
                "Indicator": item.get("label"),
                "Latest": item.get("latest"),
                "As at": item.get("end"),
                # Rates move in percentage points; everything else in percent.
                "Change vs a year ago": item.get("yoy_pp") if is_rate else item.get("yoy_pct"),
                "Unit": "pp" if is_rate else "%",
                "Change over sample %": None if is_rate else item.get("total_change_pct"),
                "CAGR %": None if is_rate else item.get("cagr_pct"),
                "Peak": item.get("maximum"),
                "From peak %": item.get("pct_from_peak"),
                "Direction": item.get("direction"),
            }
        )
    return pd.DataFrame(rows) if rows else None


def _breakpoint_table(analysis: Any) -> pd.DataFrame | None:
    points = (analysis.brief.get("timeseries") or {}).get("breakpoints") or []
    if not points:
        return None
    return pd.DataFrame(
        [
            {
                "Date": p.get("date"),
                "Type": p.get("direction"),
                "Average before": p.get("before_mean"),
                "Average after": p.get("after_mean"),
                "Shift %": p.get("shift_pct"),
            }
            for p in points
        ]
    )


def _driver_table(block: dict[str, Any]) -> pd.DataFrame | None:
    links = block.get("links") or []
    if not links:
        return None
    return pd.DataFrame(
        [
            {
                "Driver": link.get("driver_label") or link.get("driver"),
                "Type": link.get("driver_role"),
                "Correlation": link.get("correlation"),
                "p-value": link.get("p_value"),
                "n": link.get("n"),
                "Best lag": link.get("best_lag"),
                "Lagged r": link.get("best_lag_correlation"),
                "Strength": link.get("strength"),
                "Significant": "yes" if link.get("significant") else "no",
            }
            for link in links[:14]
        ]
    )


def _regression_table(block: dict[str, Any]) -> pd.DataFrame | None:
    regression = block.get("regression") or {}
    coefficients = regression.get("coefficients") or []
    if not coefficients:
        return None
    rows = [
        {
            "Variable": c.get("variable"),
            "Coefficient": c.get("coefficient"),
            "Std. error": c.get("std_error"),
            "t": c.get("t_stat"),
            "p-value": c.get("p_value"),
            "Significant": "yes" if c.get("significant") else "no",
        }
        for c in coefficients
    ]
    rows.append(
        {
            "Variable": "Model fit",
            "Coefficient": regression.get("r_squared"),
            "Std. error": regression.get("adj_r_squared"),
            "t": regression.get("n"),
            "p-value": None,
            "Significant": "R² / adj. R² / n",
        }
    )
    return pd.DataFrame(rows)


def _policy_table(research: Any) -> pd.DataFrame | None:
    events = getattr(research, "policy_events", [])
    if not events:
        return None
    return pd.DataFrame(
        [
            {
                "Date": e.date,
                "Measure": e.title,
                "Type": e.category,
                "Expected effect on prices": e.direction,
                "Transmission to the housing market": e.expected_impact or e.summary,
                "Confidence": e.confidence,
                "Origin": e.origin,
            }
            for e in events
        ]
    )


def _macro_table(research: Any) -> pd.DataFrame | None:
    factors = getattr(research, "macro_factors", [])
    if not factors:
        return None
    return pd.DataFrame(
        [
            {
                "Factor": f.get("factor"),
                "Current state": f.get("current_state"),
                "Housing-market impact": f.get("housing_impact"),
                "Direction": f.get("direction"),
            }
            for f in factors
        ]
    )


def _recommendation_table(narrative: Narrative) -> pd.DataFrame | None:
    if not narrative.recommendations:
        return None
    order = {"high": 0, "medium": 1, "low": 2}
    items = sorted(narrative.recommendations, key=lambda r: order.get(r.priority, 1))
    return pd.DataFrame(
        [
            {
                "#": i,
                "Priority": r.priority.capitalize(),
                "Who should act": r.audience,
                "Recommended action": r.action,
                "Why (evidence in this report)": r.rationale,
                "Horizon": r.horizon or "—",
            }
            for i, r in enumerate(items, 1)
        ]
    )


def _sources_section(doc: ReportDocument, research: Any, brief: dict[str, Any]) -> None:
    sources = research.source_index() if hasattr(research, "source_index") else []
    if sources:
        doc.para(
            f"{len(sources)} sources were consulted during this run. Claims drawn from them are "
            "attributed in the policy and macro sections above.",
            color=INK_SECONDARY,
        )
        for i, source in enumerate(sources, 1):
            doc.hyperlink(source.get("title") or source.get("url", ""), source.get("url", ""),
                          prefix=f"[{i}] ")
    else:
        doc.para(
            "No external sources were retrieved for this run. The findings rest on the supplied "
            "dataset and the local policy record only.",
            color=INK_SECONDARY,
        )

    doc.heading("Dataset", 2)
    coverage = brief.get("coverage") or {}
    doc.para(
        f"Source: {coverage.get('source', 'user-supplied file')} · "
        f"{coverage.get('rows', 'n/a')} rows · {coverage.get('start', '?')} to {coverage.get('end', '?')} · "
        f"{coverage.get('grain', 'unknown')} frequency.",
        size=9, color=INK_SECONDARY, align="left",
    )
    doc.para(
        f"Report generated {datetime.now():%d %B %Y at %H:%M} by the Uzbekistan Housing Market "
        "Research Agent.",
        size=8, color=INK_MUTED, italic=True, align="left",
    )
