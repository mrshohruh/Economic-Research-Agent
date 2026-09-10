"""End-to-end orchestration: data in, Word report out."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .analysis import drivers as drivers_mod
from .analysis import metrics as metrics_mod
from .analysis import regional as regional_mod
from .analysis import timeseries as ts_mod
from .config import Settings
from .ingest import loader, profiler
from .ingest.profiler import PERIODS_PER_YEAR, Understanding
from .llm import LLM
from .report import composer
from .report.narrative import Narrative, write as write_narrative
from .research import context as research_mod
from .viz.charts import ChartBuilder, Figure

LOGGER = logging.getLogger(__name__)

# Which roles make the best headline indicator for a housing report, best first.
HEADLINE_PRIORITY = ["price_per_sqm", "price", "volume", "supply", "mortgage", "income", "value"]

# Roles worth their own chart treatment.
VOLUME_ROLES = {"volume", "supply"}

# How prominently each role is presented in the report's indicator dashboard.
METRIC_REPORT_ORDER = [
    "price_per_sqm", "price", "volume", "supply", "mortgage", "rate",
    "income", "inflation", "fx", "population", "area", "value",
]


@dataclass
class AnalysisResult:
    understanding: Understanding
    headline_metric: str = ""
    headline_label: str = ""
    summaries: dict[str, metrics_mod.SeriesSummary] = field(default_factory=dict)
    timeseries: ts_mod.TimeSeriesAnalysis | None = None
    regional: regional_mod.GroupComparison | None = None
    concentration: dict[str, Any] = field(default_factory=dict)
    drivers: drivers_mod.DriverAnalysis | None = None
    activity_metric: str = ""
    activity_label: str = ""
    activity_drivers: drivers_mod.DriverAnalysis | None = None
    panel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    brief: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    report_path: Path | None = None
    figures: list[Figure] = field(default_factory=list)
    analysis: AnalysisResult | None = None
    research: research_mod.ResearchFindings | None = None
    narrative: Narrative | None = None
    run_log: Path | None = None
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.report_path is not None and self.report_path.exists()


# ---------------------------------------------------------------------------
def run(
    *,
    settings: Settings | None = None,
    data_path: str | Path | None = None,
    raw_text: str | None = None,
    filename: str | None = None,
    connection_url: str | None = None,
    query: str | None = None,
    title: str = "",
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Load, analyse, research, chart and write the report."""
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    say = progress or (lambda message: LOGGER.info(message))
    started = datetime.now()
    result = RunResult()

    llm = LLM(settings.anthropic_api_key, settings.model)
    say(f"Language model: {llm.status}")

    # 1. Load ---------------------------------------------------------------
    say("Loading data...")
    dataset = loader.load(
        data_path, raw_text=raw_text, filename=filename,
        connection_url=connection_url, query=query,
    )
    say(f"Loaded {dataset.total_rows:,} rows across {len(dataset.tables)} table(s) from a {dataset.kind} source.")

    # 2. Understand ---------------------------------------------------------
    say("Working out what the columns mean...")
    understanding = profiler.understand(dataset, llm)
    profile = understanding.primary_profile
    say(
        f"Primary table “{understanding.primary}”: {profile.rows:,} rows, "
        f"{len(profile.metric_cols)} indicator(s), {profile.grain} frequency."
    )

    # 3. Analyse ------------------------------------------------------------
    say("Computing indicators, trends and relationships...")
    analysis = analyse(understanding)
    result.analysis = analysis
    if analysis.headline_metric:
        say(f"Headline indicator: {analysis.headline_label}")

    # 4. Charts -------------------------------------------------------------
    say("Drawing charts...")
    stamp = started.strftime("%Y%m%d_%H%M%S")
    figure_dir = settings.figures_dir / stamp
    builder = ChartBuilder(figure_dir, source_note=f"Source: {dataset.source}")

    # 5. Research -----------------------------------------------------------
    coverage = analysis.brief.get("coverage", {})
    research = research_mod.gather(
        llm,
        period_start=str(coverage.get("start", "")),
        period_end=str(coverage.get("end", "")),
        regions=understanding.regions[:8],
        data_highlights=_highlights(analysis),
        use_web=settings.web_research,
        results_per_query=settings.search_results_per_query,
        pages_to_read=settings.pages_to_read,
        policy_file=settings.policy_file,
        progress=say,
    )
    result.research = research
    analysis.brief["research"] = research.to_dict()
    say(
        f"Research: {len(research.policy_events)} policy events, "
        f"{len(research.sources)} sources, {len(research.themes)} themes."
    )

    build_figures(builder, analysis, research)
    result.figures = builder.figures
    say(f"Produced {len(builder.figures)} figures.")

    # 6. Narrative ----------------------------------------------------------
    say("Writing the report narrative...")
    narrative = write_narrative(analysis.brief, llm, settings.language)
    result.narrative = narrative
    say(f"Narrative written by: {narrative.generated_by}")

    # 7. Compose ------------------------------------------------------------
    say("Building the Word document...")
    report_title = title or "Uzbekistan Housing Market"
    subtitle = _subtitle(analysis)
    filename_stem = _slug(report_title)
    output_path = settings.reports_dir / f"{filename_stem}_{stamp}.docx"

    run_meta = {
        "Prepared": f"{started:%d %B %Y}",
        "Coverage": f"{coverage.get('start', '?')} to {coverage.get('end', '?')} ({coverage.get('grain', 'unknown')})",
        "Data source": str(dataset.source),
        "Observations": f"{dataset.total_rows:,} rows across {len(dataset.tables)} table(s)",
        "Headline indicator": analysis.headline_label or "—",
        "Analysis engine": narrative.generated_by,
        "Research": (
            f"{len(research.sources)} web sources, {len(research.policy_events)} policy events"
            if research.web_used
            else f"{len(research.policy_events)} policy events (local record only)"
        ),
    }

    result.report_path = composer.compose(
        output_path=output_path,
        title=report_title,
        subtitle=subtitle,
        analysis=analysis,
        narrative=narrative,
        research=research,
        figures=builder.figures,
        settings=settings,
        run_meta=run_meta,
        language=settings.language,
    )
    say(f"Report saved: {result.report_path}")

    # 8. Run log -------------------------------------------------------------
    result.duration_seconds = (datetime.now() - started).total_seconds()
    result.warnings = list(analysis.notes) + list(research.notes)
    result.run_log = _write_run_log(settings, stamp, result, analysis, narrative)
    return result


# ---------------------------------------------------------------------------
def analyse(understanding: Understanding) -> AnalysisResult:
    """All quantitative work, gathered into one result plus a JSON-ready brief."""
    result = AnalysisResult(understanding=understanding, notes=list(understanding.notes))
    tidy = understanding.tidy

    if tidy.empty:
        result.brief = {
            "coverage": _coverage(understanding, pd.Series(dtype=float)),
            "metrics": [],
            "notes": result.notes + ["No time series could be derived, so trend analysis was skipped."],
        }
        return result

    metric_names = understanding.metrics
    series_map: dict[str, pd.Series] = {}
    for metric in metric_names:
        series = understanding.national_series(metric)
        if len(series.dropna()) >= 3:
            series_map[metric] = series

    if not series_map:
        result.notes.append("Every indicator had fewer than three usable observations.")
        result.brief = {"coverage": _coverage(understanding, pd.Series(dtype=float)), "metrics": [], "notes": result.notes}
        return result

    result.headline_metric = _pick_headline(understanding, series_map)
    result.headline_label = _label(result.headline_metric)
    headline_series = series_map[result.headline_metric]

    for metric, series in series_map.items():
        result.summaries[metric] = metrics_mod.summarise_series(
            series, metric,
            role=understanding.role_for_metric(metric),
            unit=understanding.unit_for_metric(metric),
        )

    result.timeseries = ts_mod.analyse(headline_series, result.headline_metric)

    role = understanding.role_for_metric(result.headline_metric)
    if len(understanding.regions) > 1:
        result.regional = regional_mod.compare_groups(tidy, result.headline_metric, "region", role)
        result.concentration = regional_mod.concentration(tidy, result.headline_metric, "region")

    result.panel = drivers_mod.build_panel(series_map)
    roles = {metric: understanding.role_for_metric(metric) for metric in series_map}
    labels = {metric: _label(metric) for metric in series_map}
    per_year = PERIODS_PER_YEAR.get(understanding.primary_profile.grain, 12)
    result.drivers = drivers_mod.analyse_drivers(
        result.panel, result.headline_metric, roles,
        periods_per_year=per_year, target_label=result.headline_label, labels=labels,
    )

    # Prices and activity respond to different things — mortgage rates hit
    # transaction volumes long before they show up in prices — so the main
    # activity series gets its own attribution run.
    activity = _pick_activity(understanding, series_map, exclude=result.headline_metric)
    if activity:
        result.activity_metric = activity
        result.activity_label = _label(activity)
        result.activity_drivers = drivers_mod.analyse_drivers(
            result.panel, activity, roles,
            periods_per_year=per_year, target_label=result.activity_label, labels=labels,
        )

    result.brief = _build_brief(understanding, result, series_map)
    return result


def _pick_activity(
    understanding: Understanding, series_map: dict[str, pd.Series], exclude: str
) -> str | None:
    """The transaction / supply series that best represents market activity."""
    candidates = [
        metric for metric in series_map
        if metric != exclude and understanding.role_for_metric(metric) in VOLUME_ROLES
    ]
    if not candidates:
        return None
    order = {"volume": 0, "supply": 1}
    candidates.sort(key=lambda m: (order.get(understanding.role_for_metric(m), 2),
                                   -len(series_map[m].dropna())))
    return candidates[0]


def _pick_headline(understanding: Understanding, series_map: dict[str, pd.Series]) -> str:
    """Prefer a price series, then activity, then whatever has the longest history."""
    scored: list[tuple[int, int, str]] = []
    for metric in series_map:
        role = understanding.role_for_metric(metric)
        rank = HEADLINE_PRIORITY.index(role) if role in HEADLINE_PRIORITY else len(HEADLINE_PRIORITY)
        scored.append((rank, -len(series_map[metric].dropna()), metric))
    scored.sort()
    return scored[0][2]


def _build_brief(
    understanding: Understanding,
    result: AnalysisResult,
    series_map: dict[str, pd.Series],
) -> dict[str, Any]:
    headline_summary = result.summaries[result.headline_metric]

    # Order by how central the role is to a housing report, so the dashboard and
    # the key findings lead with prices and activity rather than average floor area.
    def rank(metric: str) -> tuple[int, str]:
        role = understanding.role_for_metric(metric)
        return (
            METRIC_REPORT_ORDER.index(role) if role in METRIC_REPORT_ORDER else len(METRIC_REPORT_ORDER),
            metric,
        )

    ordered = [result.headline_metric] + sorted(
        (m for m in result.summaries if m != result.headline_metric), key=rank
    )

    brief: dict[str, Any] = {
        "coverage": _coverage(understanding, series_map[result.headline_metric]),
        "headline": {**headline_summary.to_dict(), "label": result.headline_label},
        "metrics": [
            {**result.summaries[m].to_dict(), "label": _label(m)}
            for m in ordered
        ],
        "timeseries": result.timeseries.to_dict() if result.timeseries else {},
        "notes": result.notes,
    }

    if result.regional:
        brief["regional"] = {
            **result.regional.to_dict(),
            "concentration": result.concentration,
        }
    if result.drivers:
        brief["drivers"] = result.drivers.to_dict()
    if result.activity_drivers:
        brief["activity_drivers"] = result.activity_drivers.to_dict()

    segments = sorted(understanding.tidy["segment"].dropna().unique().tolist()) if "segment" in understanding.tidy else []
    if len(segments) > 1:
        brief["segments"] = segments[:20]
    return brief


def _coverage(understanding: Understanding, series: pd.Series) -> dict[str, Any]:
    profile = understanding.primary_profile
    index = series.dropna().index
    return {
        "source": understanding.dataset.source,
        "kind": understanding.dataset.kind,
        "rows": understanding.dataset.total_rows,
        "tables": len(understanding.dataset.tables),
        "primary_table": understanding.primary,
        "grain": profile.grain,
        "start": f"{index[0]:%Y-%m-%d}" if len(index) else (
            f"{profile.period_start:%Y-%m-%d}" if profile.period_start is not None else ""),
        "end": f"{index[-1]:%Y-%m-%d}" if len(index) else (
            f"{profile.period_end:%Y-%m-%d}" if profile.period_end is not None else ""),
        "periods": int(len(index)),
        "metrics": len(understanding.metrics),
        "regions": len(understanding.regions),
    }


def _highlights(analysis: AnalysisResult) -> str:
    """A compact text summary of the data, given to the research step for grounding."""
    brief = analysis.brief
    headline = brief.get("headline") or {}
    coverage = brief.get("coverage") or {}
    if not headline:
        return "The dataset contains no usable time series."

    lines = [
        f"Headline indicator: {headline.get('label')} ({headline.get('role')}), "
        f"{coverage.get('grain')} data from {coverage.get('start')} to {coverage.get('end')}.",
        f"Latest value {headline.get('latest')}, year-on-year {headline.get('yoy_pct')}%, "
        f"change over the whole sample {headline.get('total_change_pct')}%, "
        f"compound annual growth {headline.get('cagr_pct')}%.",
    ]
    for point in (brief.get("timeseries") or {}).get("breakpoints", [])[:3]:
        lines.append(
            f"Structural break around {point.get('date')}: level shifted {point.get('shift_pct')}% "
            f"({point.get('direction')})."
        )
    regional = brief.get("regional") or {}
    if regional.get("leaders"):
        field_name = regional.get("group_field", "region")
        lines.append(
            "Fastest-growing regions: "
            + ", ".join(f"{r.get(field_name)} ({r.get('change_pct')}%)" for r in regional["leaders"][:3])
            + f". Regional spread is {regional.get('convergence')}."
        )
    for key in ("drivers", "activity_drivers"):
        block = brief.get(key) or {}
        significant = [l for l in block.get("links", []) if l.get("significant")]
        if significant:
            lines.append(
                f"Significant co-movements with {block.get('target_label', block.get('target'))} "
                f"(on {block.get('basis', 'growth rates')}): "
                + "; ".join(
                    f"{l.get('driver_label') or l['driver']} (r = {l['correlation']}"
                    + (f", strongest at a {l['best_lag']}-period lead" if l.get("best_lag") else "")
                    + ")"
                    for l in significant[:4]
                )
                + "."
            )
    other = [m.get("label") for m in brief.get("metrics", [])[1:8]]
    if other:
        lines.append("Other indicators in the dataset: " + ", ".join(str(o) for o in other) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def build_figures(
    builder: ChartBuilder,
    analysis: AnalysisResult,
    research: research_mod.ResearchFindings,
) -> list[Figure]:
    """Render every chart the data can support. Individual failures are non-fatal."""
    understanding = analysis.understanding
    if not analysis.headline_metric:
        return builder.figures

    tidy = understanding.tidy
    headline = analysis.headline_metric
    label = analysis.headline_label
    unit = understanding.unit_for_metric(headline)
    series = understanding.national_series(headline)
    events = [e.to_dict() for e in research.policy_events]

    def attempt(name: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - chart-specific edge cases
            LOGGER.warning("chart '%s' failed: %s\n%s", name, exc, traceback.format_exc(limit=3))
            analysis.notes.append(f"The “{name}” chart could not be drawn ({exc}).")

    attempt("headline trend", builder.trend_line, series, label, unit, events)

    if analysis.timeseries is not None:
        yoy = analysis.timeseries.rolling_yoy
        if len(yoy):
            attempt("year-on-year growth", builder.yoy_bars, yoy, label)
        attempt("seasonality", builder.seasonality_heatmap, series, label,
                analysis.timeseries.seasonality.detected)

    # Regional views
    if len(understanding.regions) > 1:
        role = understanding.role_for_metric(headline)
        agg = "sum" if role in {"volume", "supply", "population"} else "mean"
        panel = tidy[tidy["metric"] == headline].pivot_table(
            index="date", columns="region", values="value", aggfunc=agg
        ).sort_index()
        attempt("regional index", builder.region_index_lines, panel, label, 4)
        if analysis.regional is not None and not analysis.regional.table.empty:
            table = analysis.regional.table
            attempt("regional ranking", builder.ranking_bars, table, "latest", "region", label, unit)
            growth_col = "yoy_pct" if table["yoy_pct"].notna().sum() >= 2 else "change_pct"
            attempt("regional growth ranking", builder.growth_ranking, table, growth_col, "region", label)

    # Activity series get their own bar chart.
    for metric, summary in analysis.summaries.items():
        if summary.role in VOLUME_ROLES and metric != headline:
            attempt(
                f"{_label(metric)} volumes",
                builder.volume_bars,
                understanding.national_series(metric), _label(metric), summary.unit,
            )
            break

    attempt("segment comparison", builder.segment_bars, tidy, headline, label)

    # Drivers
    if analysis.drivers is not None:
        if not analysis.drivers.correlation_matrix.empty and analysis.drivers.correlation_matrix.shape[0] >= 3:
            matrix = analysis.drivers.correlation_matrix.rename(index=_label, columns=_label)
            attempt("correlation matrix", builder.correlation_heatmap, matrix)
        for link in analysis.drivers.significant_links[:2]:
            attempt(
                f"{label} vs {link.driver}",
                builder.driver_panels,
                series, label,
                understanding.national_series(link.driver), _label(link.driver),
                link.correlation,
            )

    if analysis.activity_drivers is not None and analysis.activity_metric:
        activity_series = understanding.national_series(analysis.activity_metric)
        for link in analysis.activity_drivers.significant_links[:2]:
            attempt(
                f"{analysis.activity_label} vs {link.driver}",
                builder.driver_panels,
                activity_series, analysis.activity_label,
                understanding.national_series(link.driver), _label(link.driver),
                link.correlation,
            )

    if analysis.timeseries is not None and analysis.timeseries.forecast.points:
        attempt(
            "forecast", builder.forecast_chart, series,
            analysis.timeseries.forecast.points, label, analysis.timeseries.forecast.method,
        )

    if len(events) >= 2:
        attempt("policy timeline", builder.policy_timeline, events)

    return builder.figures


# ---------------------------------------------------------------------------
def _label(metric: str) -> str:
    text = str(metric).replace("_", " ").strip()
    fixes = {
        "pct": "%", "avg": "average", "med": "median", "sqm": "m²", "m2": "m²",
        "usd": "USD", "uzs": "UZS", "gdp": "GDP", "cpi": "CPI", "yoy": "YoY", "fx": "FX",
        "bn": "bn", "mn": "mn",
    }
    words = []
    for word in text.split():
        lowered = word.lower()
        words.append(fixes.get(lowered, word))
    text = " ".join(words)
    return text[:1].upper() + text[1:] if text else str(metric)


def _subtitle(analysis: AnalysisResult) -> str:
    coverage = analysis.brief.get("coverage") or {}
    start, end = coverage.get("start", ""), coverage.get("end", "")
    parts = []
    if start and end:
        parts.append(f"{start[:7]} – {end[:7]}")
    if analysis.headline_label:
        parts.append(analysis.headline_label)
    if coverage.get("regions", 0) > 1:
        parts.append(f"{coverage['regions']} regions")
    return "Trends, drivers, policy context and recommendations · " + " · ".join(parts) if parts else \
        "Trends, drivers, policy context and recommendations"


def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^\w\s-]", "", str(text)).strip().lower()
    return re.sub(r"[\s_-]+", "_", slug)[:60] or "housing_report"


def _write_run_log(
    settings: Settings,
    stamp: str,
    result: RunResult,
    analysis: AnalysisResult,
    narrative: Narrative,
) -> Path:
    path = settings.runs_dir / f"run_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "duration_seconds": round(result.duration_seconds, 2),
        "report": str(result.report_path) if result.report_path else None,
        "figures": [f.to_dict() for f in result.figures],
        "warnings": result.warnings,
        "narrative_engine": narrative.generated_by,
        "brief": analysis.brief,
        "narrative": narrative.to_dict(),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("could not write the run log: %s", exc)
    return path
