"""Turn the analysis brief into report prose.

Two paths produce the same :class:`Narrative` shape:

* ``write_with_llm`` - a housing economist persona writes the sections from the
  computed brief, and is explicitly forbidden from inventing numbers.
* ``write_fallback`` - deterministic prose assembled from the same numbers, used
  when there is no API key. It is terser but every sentence is still grounded in
  a computed statistic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM, LLMUnavailable

LOGGER = logging.getLogger(__name__)

SYSTEM = """You are a senior housing-market economist writing for the Ministry of Economy \
and Finance of Uzbekistan and for institutional investors. Your writing is precise, \
quantitative and free of filler.

Hard rules:
1. Every number you write must come from the brief. Never invent, round up, or extrapolate a figure.
2. Keep three kinds of statement visibly separate:
   (a) what the dataset shows — already computed for you in the brief, quote it exactly;
   (b) what external evidence shows — the policy events, macro factors and cited sources;
   (c) your interpretation joining the two. Interpretation is written as a plausible mechanism, \
not as a demonstrated cause: "mortgage lending expanded over the same period, which is a \
plausible channel through which credit growth supported demand" — never "mortgage policy caused \
prices to rise".
3. Correlation is not causation, and you say so where it matters.
4. Prefer specific mechanisms ("a 3pp rise in the policy rate raised mortgage servicing costs, \
which shows up as a fall in transaction volumes two quarters later") over vague statements \
("market conditions worsened").
5. If the evidence for something is thin, say the evidence is thin. Never pad.
6. Write in flowing professional prose. No bullet-point fragments inside paragraphs.
7. Call the measured quantity exactly what the brief calls it. If the brief reports a monthly \
asking rent, every sentence says rent — never "house price", "property price" or "sale price". \
If it reports asking prices from listings, say advertised asking prices, not transaction prices."""

LANGUAGE_NAMES = {"en": "English", "ru": "Russian", "uz": "Uzbek (Latin script)"}


@dataclass
class Recommendation:
    audience: str
    action: str
    rationale: str
    priority: str = "medium"
    horizon: str = ""


@dataclass
class Narrative:
    executive_summary: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    current_situation: list[str] = field(default_factory=list)
    historical_trends: list[str] = field(default_factory=list)
    regional_analysis: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    policy_analysis: list[str] = field(default_factory=list)
    macro_context: list[str] = field(default_factory=list)
    outlook: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_by: str = "deterministic template"

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in vars(self).items() if k != "recommendations"}
        data["recommendations"] = [vars(r) for r in self.recommendations]
        return data


# ---------------------------------------------------------------------------
def write(brief: dict[str, Any], llm: LLM | None, language: str = "en") -> Narrative:
    """Preferred entry point: try the LLM, fall back cleanly."""
    if llm is not None and llm.available:
        try:
            narrative = write_with_llm(brief, llm, language)
            if narrative.executive_summary or narrative.current_situation:
                return narrative
            LOGGER.warning("LLM returned an empty narrative; using the template writer")
        except LLMUnavailable as exc:
            LOGGER.warning("LLM narrative unavailable (%s); using the template writer", exc)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("LLM narrative failed (%s); using the template writer", exc)
    return write_fallback(brief)


# ---------------------------------------------------------------------------
def write_with_llm(brief: dict[str, Any], llm: LLM, language: str = "en") -> Narrative:
    language_name = LANGUAGE_NAMES.get(language, "English")
    payload = json.dumps(brief, ensure_ascii=False, indent=1, default=str)
    if len(payload) > 60000:
        payload = payload[:60000] + "\n... (brief truncated)"

    prompt = f"""Write a housing-market research report on Uzbekistan from the analysis brief below.

The brief contains (a) statistics computed directly from the analyst's dataset and
(b) policy and macro research gathered from the web with citations.

=== ANALYSIS BRIEF ===
{payload}
=== END BRIEF ===

Write every section in {language_name}.

Section guidance:
- executive_summary: 3-4 paragraphs a deputy minister could read alone and act on. Lead with the
  single most important quantitative finding.
- key_findings: 5-8 one-sentence findings, each containing a specific number from the brief.
- current_situation: where the market stands now. When the brief carries a "cross_section" block,
  this section leads with it: the median and mean price level, the gap between them and what that
  says about skew, the spread from the 25th to the 75th percentile, price per square metre, and how
  price varies with dwelling size and condition. State the sample size. Only then turn to growth.
- historical_trends: the shape of the whole sample. Name the turning points and structural breaks
  in the brief by date and say what changed at each. If the brief's notes warn that the time
  dimension comes from listing dates rather than a price index, say so plainly here and treat any
  apparent trend as provisional.
- regional_analysis: which region is most expensive and which is cheapest, by how much, and what
  the ratio between them is - take these from "cross_section.by_region" when it is present, naming
  the regions and quoting their medians. Then cover dispersion, the share of listings each region
  holds, and whether the level ranking survives when measured per square metre (a region can look
  cheap only because the dwellings advertised there are smaller). Flag any region the brief marks
  as a thin sample. Omit this section's content entirely (empty list) only if the brief has no
  regional breakdown at all.
- drivers: THE core section. For each major trend, give the reason. Use the correlation and
  regression results, the policy events, and the macro factors. Distinguish what the data shows
  from what you are inferring. Where the statistics are weak, say the attribution is judgement.
- policy_analysis: what each relevant measure was designed to do, how it transmits to prices,
  volumes or credit, and whether the data shows a response around its date.
- macro_context: inflation, the policy rate, incomes, remittances, FX, construction costs,
  demographics - only the ones the brief actually supports.
- outlook: what the projection implies, plus the qualitative factors it cannot capture.
- recommendations: 5-8 concrete, actionable items. Each names a specific audience
  (e.g. "Central Bank of Uzbekistan", "Ministry of Construction", "commercial bank lenders",
  "developers", "investors"), a specific action, and the evidence that motivates it.
- risks: 4-6 downside risks with the channel through which each would hit the market.
- limitations: honest data and method caveats, including anything flagged in the brief.

Return JSON of exactly this shape:
{{
  "executive_summary": ["paragraph", "..."],
  "key_findings": ["finding", "..."],
  "current_situation": ["paragraph", "..."],
  "historical_trends": ["paragraph", "..."],
  "regional_analysis": ["paragraph", "..."],
  "drivers": ["paragraph", "..."],
  "policy_analysis": ["paragraph", "..."],
  "macro_context": ["paragraph", "..."],
  "outlook": ["paragraph", "..."],
  "recommendations": [
    {{"audience": "...", "action": "...", "rationale": "...",
      "priority": "high|medium|low", "horizon": "immediate|6-12 months|long term"}}
  ],
  "risks": ["risk with its transmission channel", "..."],
  "limitations": ["caveat", "..."]
}}"""

    result = llm.complete_json(prompt, system=SYSTEM, max_tokens=16000)
    if not isinstance(result, dict):
        raise LLMUnavailable("narrative was not a JSON object")

    narrative = Narrative(generated_by=f"Claude ({llm.model})")
    for field_name in (
        "executive_summary", "key_findings", "current_situation", "historical_trends",
        "regional_analysis", "drivers", "policy_analysis", "macro_context",
        "outlook", "risks", "limitations",
    ):
        value = result.get(field_name) or []
        if isinstance(value, str):
            value = [value]
        setattr(narrative, field_name, [str(v).strip() for v in value if str(v).strip()])

    for item in result.get("recommendations", []) or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        narrative.recommendations.append(
            Recommendation(
                audience=str(item.get("audience", "Policy makers")).strip(),
                action=action,
                rationale=str(item.get("rationale", "")).strip(),
                priority=str(item.get("priority", "medium")).strip().lower(),
                horizon=str(item.get("horizon", "")).strip(),
            )
        )
    return narrative


# ---------------------------------------------------------------------------
# Deterministic writer
# ---------------------------------------------------------------------------
def write_fallback(brief: dict[str, Any]) -> Narrative:
    n = Narrative()
    headline = brief.get("headline") or {}
    metrics: list[dict[str, Any]] = brief.get("metrics") or []
    ts = brief.get("timeseries") or {}
    regional = brief.get("regional") or {}
    drivers = brief.get("drivers") or {}
    research = brief.get("research") or {}
    coverage = brief.get("coverage") or {}
    section = brief.get("cross_section") or {}

    label = headline.get("label", "the headline indicator")
    latest = _num(headline.get("latest"))
    yoy = _num(headline.get("yoy_pct"))
    cagr = _num(headline.get("cagr_pct"))
    total = _num(headline.get("total_change_pct"))
    start, end = coverage.get("start", "?"), coverage.get("end", "?")

    # -- executive summary ------------------------------------------------
    n.executive_summary.append(
        f"This report analyses the Uzbek housing market using {coverage.get('rows', 'the supplied')} "
        f"observations covering {start} to {end} at {coverage.get('grain', 'unknown')} frequency, "
        f"across {coverage.get('metrics', 0)} measured indicators"
        + (f" and {coverage.get('regions', 0)} regions." if coverage.get("regions", 0) > 1 else ".")
    )
    # Where the source is property microdata, the price level and its regional
    # spread are the report's substance and lead every relevant section.
    _write_cross_section(n, section)
    if latest is not None:
        sentence = f"{label} stands at {_fmt(latest)} in the latest period"
        if yoy is not None:
            sentence += f", {yoy:+.1f}% higher than a year earlier" if yoy >= 0 else f", {yoy:.1f}% below a year earlier"
        if total is not None:
            sentence += f", and {total:+.1f}% against the start of the sample"
        n.executive_summary.append(sentence + ".")
    if cagr is not None:
        wage = next((m for m in metrics if m.get("role") == "income"), None)
        wage_cagr = _num(wage.get("cagr_pct")) if wage else None
        comparison = ""
        if wage_cagr is not None:
            gap = cagr - wage_cagr
            comparison = (
                f" Over the same period {wage.get('label')} compounded at {wage_cagr:+.1f}% a year, "
                f"so prices ran {abs(gap):.1f} percentage points "
                + ("ahead of" if gap > 0 else "behind")
                + " incomes — the direct measure of whether affordability tightened or eased."
            )
        n.executive_summary.append(
            f"Compounded, that is average growth of {cagr:+.1f}% a year." + comparison
        )

    trend = (ts.get("trend") or {})
    if trend.get("description"):
        n.executive_summary.append(
            f"Statistically, the series shows a {trend['description']} "
            f"(R² = {trend.get('r_squared', 'n/a')}, p = {trend.get('p_value', 'n/a')})."
        )

    # -- key findings -----------------------------------------------------
    for metric in metrics[:6]:
        parts = [f"{metric.get('label', metric.get('metric'))} is at {_fmt(_num(metric.get('latest')))}"]
        parts.append(_yoy_phrase(metric))
        if metric.get("change_unit") != "pp" and _num(metric.get("cagr_pct")) is not None:
            parts.append(f"{_num(metric['cagr_pct']):+.1f}% a year compounded over the sample")
        n.key_findings.append(", ".join(p for p in parts if p) + ".")

    for point in (ts.get("breakpoints") or [])[:2]:
        n.key_findings.append(
            f"A structural {point.get('direction')} in {label} is detected around {point.get('date')}: "
            f"the average level shifted {_num(point.get('shift_pct')):+.1f}%."
        )
    if regional.get("dispersion_pct") is not None:
        n.key_findings.append(
            f"Regional dispersion in {label} stands at {regional['dispersion_pct']}% "
            f"(coefficient of variation); the pattern is {regional.get('convergence', 'unclear')}."
        )

    # -- current situation ------------------------------------------------
    if latest is not None:
        text = f"As at {end}, {label} is {_fmt(latest)}"
        if _num(headline.get("previous")) is not None:
            text += f", against {_fmt(_num(headline['previous']))} in the preceding period"
        text += ". "
        if yoy is not None:
            text += f"Year-on-year growth is {yoy:+.1f}%"
            prev_yoy = _num(headline.get("yoy_prev_pct"))
            if prev_yoy is not None:
                direction = "accelerating" if yoy > prev_yoy else "decelerating" if yoy < prev_yoy else "steady"
                text += f", {direction} from {prev_yoy:+.1f}% in the previous period"
            text += ". "
        if _num(headline.get("pct_from_peak")) is not None:
            gap = _num(headline["pct_from_peak"])
            text += (
                "The series is at its sample peak. "
                if abs(gap) < 0.5
                else f"It sits {abs(gap):.1f}% below the sample peak recorded in {headline.get('max_date', 'n/a')}. "
            )
        n.current_situation.append(text.strip())

    if _num(headline.get("volatility_pct")) is not None:
        n.current_situation.append(
            f"Annualised volatility of period-over-period changes is {_num(headline['volatility_pct']):.1f}%, "
            "which sets the bar for judging whether any single move is meaningful or noise."
        )

    for metric in metrics[1:4]:
        phrase = _yoy_phrase(metric)
        if not phrase:
            continue
        n.current_situation.append(
            f"{metric.get('label')} is {_fmt(_num(metric.get('latest')))}, {phrase}, "
            f"and is {metric.get('direction', 'flat')} with {metric.get('momentum', 'stable')} momentum."
        )

    # -- historical trends ------------------------------------------------
    if trend.get("description"):
        n.historical_trends.append(
            f"Across {start} to {end}, {label} shows a {trend['description']}. "
            f"The linear fit explains {_pct_of(trend.get('r_squared'))} of the variation, "
            f"with a p-value of {trend.get('p_value', 'n/a')}."
        )
    breaks = ts.get("breakpoints") or []
    if breaks:
        described = "; ".join(
            f"{b['date']} ({b['direction']}, {_num(b['shift_pct']):+.1f}% shift in the average level)"
            for b in breaks
        )
        n.historical_trends.append(
            f"Structural break detection flags {len(breaks)} point(s) where the level shifted: {described}. "
            "These dates are the natural places to look for a policy or macro trigger."
        )
    turns = ts.get("turning_points") or []
    if turns:
        n.historical_trends.append(
            "Smoothed turning points fall at "
            + ", ".join(f"{t['date']} ({t['kind']})" for t in turns[-5:])
            + ", which bounds the cycles visible in the sample."
        )
    season = ts.get("seasonality") or {}
    if season.get("detected"):
        n.historical_trends.append(
            f"A repeating seasonal pattern is present (strength {season.get('strength')}), peaking in "
            f"{season.get('peak_period')} and troughing in {season.get('trough_period')}. "
            "Period-on-period comparisons should therefore be made year on year, not against the previous period."
        )
    else:
        n.historical_trends.append(
            "No material seasonal pattern is detected, so period-on-period comparisons are not "
            "distorted by the time of year."
        )

    # -- regional ---------------------------------------------------------
    leaders = regional.get("leaders") or []
    laggards = regional.get("laggards") or []
    if leaders:
        field_name = regional.get("group_field", "region")
        # Quote the column the ranking was actually built on, or the "strongest"
        # regions get listed alongside a number that does not order them.
        ranked_by = regional.get("ranked_by", "change_pct")
        basis = "year-on-year growth" if ranked_by == "yoy_pct" else "change over the full sample"

        def named(records: list[dict[str, Any]]) -> str:
            return ", ".join(
                f"{r.get(field_name)} ({_num(r.get(ranked_by)):+.1f}%)"
                for r in records[:3]
                if _num(r.get(ranked_by)) is not None
            )

        strongest, weakest = named(leaders), named(laggards)
        if strongest:
            n.regional_analysis.append(
                f"Ranking {field_name}s by {basis}, the strongest are {strongest}"
                + (f". The weakest are {weakest}" if weakest else "")
                + "."
            )
    if regional.get("spread_ratio"):
        n.regional_analysis.append(
            f"The highest {regional.get('group_field', 'region')} sits {regional['spread_ratio']}× the lowest "
            f"in level terms, and the cross-sectional spread is {regional.get('convergence', 'unclear')}. "
            + (
                "A widening spread usually means demand is concentrating in a few urban centres while "
                "secondary markets stagnate."
                if "diverging" in str(regional.get("convergence", ""))
                else "A narrowing spread is consistent with growth spreading beyond the leading centres."
                if "converging" in str(regional.get("convergence", ""))
                else ""
            )
        )
    conc = regional.get("concentration") or {}
    if conc:
        n.regional_analysis.append(
            f"{conc.get('top_group')} alone accounts for {conc.get('top_share_pct')}% of the latest total "
            f"and the top three for {conc.get('top3_share_pct')}%, giving a Herfindahl index of {conc.get('hhi')}."
        )

    # -- drivers ----------------------------------------------------------
    links: list[dict[str, Any]] = []
    for block in (drivers, brief.get("activity_drivers") or {}):
        if not block:
            continue
        target = block.get("target_label") or block.get("target") or label
        candidates = block.get("links") or []
        strong = [l for l in candidates if l.get("significant")]
        # Copy so the brief is not mutated; remember which series each link explains.
        links.extend({**l, "target_of": target} for l in strong)

        n.drivers.append(
            f"Attribution for {target}, computed on {block.get('basis', 'growth rates')}: "
            + (
                f"of the {len(candidates)} candidate drivers tested, {len(strong)} show a "
                "statistically reliable association."
                if strong
                else "no candidate driver in this dataset shows a statistically reliable association "
                "at the 5% level. Either the sample is too short, or the forces moving this series "
                "are not measured in the supplied file, which pushes the explanatory weight onto the "
                "policy and macro evidence below."
            )
        )
        for link in strong[:4]:
            n.drivers.append(link.get("interpretation", ""))
        if block.get("basis_note"):
            n.drivers.append(block["basis_note"])

        regression = block.get("regression") or {}
        if regression.get("r_squared") is not None:
            significant = [
                c for c in regression.get("coefficients", [])
                if c.get("significant") and c.get("variable") != "const"
            ]
            n.drivers.append(
                f"A joint regression of {regression.get('formula')} explains "
                f"{_pct_of(regression.get('r_squared'))} of the variation in {target} "
                f"(adjusted R² = {regression.get('adj_r_squared')}, n = {regression.get('n')}). "
                + (
                    "Holding the others constant, "
                    + ", ".join(
                        f"a one-unit rise in {c['variable']} is associated with a {c['coefficient']:+.3f} "
                        f"unit change in {target} (p = {c['p_value']})"
                        for c in significant[:3]
                    )
                    + "."
                    if significant
                    else "No individual coefficient reaches significance, so the drivers cannot be "
                    "separated from one another in this sample."
                )
            )
        elif regression.get("note"):
            n.drivers.append(regression["note"])

    if not n.drivers:
        # An empty attribution section reads as an omission. Saying why nothing
        # could be tested is itself a finding, and it tells the reader exactly
        # how much weight the policy and macro sections below are carrying.
        n.drivers.append(
            "No candidate driver could be tested against "
            f"{label} within this dataset, because the source file measures prices and the "
            "physical characteristics of the properties themselves — it carries no interest "
            "rate, income, mortgage, construction-cost or credit series to correlate against. "
            "Any attribution in this report therefore rests on documented policy measures and "
            "published macroeconomic evidence, set alongside the computed price statistics, "
            "rather than on a relationship estimated from the file."
        )
        if section.get("by_region"):
            n.drivers.append(
                "What the data does establish is the size and structure of the differences: the "
                "regional, size and condition tables above quantify how much of the variation in "
                "price is associated with location, floor area and the state of the dwelling. "
                "Those are measured associations within a single snapshot, not estimates of how "
                "price would respond to a change in any of them."
            )

    # -- policy and macro from research ------------------------------------
    events = research.get("policy_events") or []
    if events:
        n.policy_analysis.append(
            f"{len(events)} policy and macro measures are on record for the period covered. "
            "Their expected transmission to the housing market is summarised in the policy table below."
        )
        for event in events[:6]:
            impact = event.get("expected_impact") or event.get("summary") or ""
            n.policy_analysis.append(
                f"{event.get('date')} — {event.get('title')} ({event.get('category', 'policy')}, "
                f"{event.get('confidence', 'medium')} confidence): {event.get('summary', '')} "
                f"Expected effect: {impact}"
            )
        aligned = _align_events_to_breaks(events, breaks)
        if aligned:
            n.policy_analysis.append(
                "Two independent pieces of evidence line up: "
                + "; ".join(aligned)
                + ". Timing coincidence is suggestive, not conclusive — a formal event study would be "
                "needed to attribute the shift to the measure."
            )
    else:
        n.policy_analysis.append(
            "No policy measures were retrieved for this period. Add entries to "
            "knowledge/policy_events.json, or enable web research, to populate this section."
        )

    for factor in (research.get("macro_factors") or [])[:6]:
        n.macro_context.append(
            f"{factor.get('factor')}: {factor.get('current_state')} "
            f"Housing-market impact ({factor.get('direction', 'neutral')}): {factor.get('housing_impact')}"
        )
    for theme in (research.get("themes") or []):
        if theme.get("key") in {"macro", "monetary", "supply"} and theme.get("summary"):
            n.macro_context.append(f"{theme['title']}: {theme['summary']}")
    if not n.macro_context:
        n.macro_context.append(
            "No macroeconomic context was retrieved for this run. The quantitative findings above "
            "stand on their own, but the reasons behind them cannot be evidenced without it."
        )

    # -- outlook ----------------------------------------------------------
    forecast = ts.get("forecast") or {}
    points = forecast.get("points") or []
    if points:
        last = points[-1]
        change = ((last["forecast"] / latest - 1) * 100) if latest else None
        n.outlook.append(
            f"Extrapolating the observed history with {forecast.get('method')} puts {label} at "
            f"{_fmt(last['forecast'])} by {last['date']}"
            + (f", {change:+.1f}% on the latest reading" if change is not None else "")
            + f", within a 95% band of {_fmt(last.get('lower'))} to {_fmt(last.get('upper'))}."
        )
        n.outlook.append(forecast.get("note", ""))
    else:
        n.outlook.append(
            "The sample is too short to support a statistical projection. A longer history is needed "
            "before a forward view can be put on a quantitative footing."
        )

    # -- recommendations --------------------------------------------------
    n.recommendations = _fallback_recommendations(label, yoy, cagr, regional, links, season, conc)

    # -- risks ------------------------------------------------------------
    if cagr is not None and cagr > 15:
        n.risks.append(
            f"Affordability erosion: at {cagr:+.1f}% compound annual growth, prices are very likely "
            "outrunning wages, which compresses the pool of qualifying buyers and eventually caps demand."
        )
    if _num(headline.get("volatility_pct")) is not None and _num(headline["volatility_pct"]) > 25:
        n.risks.append(
            f"Price instability: annualised volatility of {_num(headline['volatility_pct']):.1f}% is high, "
            "which raises collateral-valuation risk for mortgage lenders."
        )
    if "diverging" in str(regional.get("convergence", "")):
        n.risks.append(
            "Regional imbalance: the gap between leading and lagging regions is widening, concentrating "
            "construction capital and mortgage exposure in a few markets."
        )
    if conc.get("top_share_pct", 0) > 40:
        n.risks.append(
            f"Geographic concentration: {conc.get('top_group')} accounts for {conc.get('top_share_pct')}% "
            "of activity, so a local shock would transmit to the national aggregate."
        )
    n.risks.append(
        "Interest-rate sensitivity: subsidised and market mortgage rates are the main channel between "
        "monetary policy and housing demand, so a tightening cycle would show up first in transaction volumes."
    )
    n.risks.append(
        "Construction cost pass-through: imported materials and energy tariffs feed directly into new-build "
        "prices, so a currency depreciation or tariff step would raise primary-market prices independently of demand."
    )

    # -- limitations ------------------------------------------------------
    n.limitations.append(
        f"The quantitative findings rest entirely on the supplied dataset ({coverage.get('rows', 'n/a')} rows, "
        f"{start} to {end}). Any market segment not represented in that file is absent from this analysis."
    )
    n.limitations.append(
        "The narrative in this run was assembled from the computed statistics by the deterministic "
        "template writer, because no ANTHROPIC_API_KEY was configured. The numbers, tables and charts are "
        "unaffected; the interpretation is necessarily more mechanical than a written analysis would be."
    )
    if not research.get("web_used"):
        n.limitations.append(
            "Web research returned no sources for this run, so the policy and macro sections draw only "
            "on the local knowledge file."
        )
    # The ingestion and processing notes are printed in full under "Data quality
    # notes" in section 2. Repeating all of them verbatim here made the
    # limitations section twice as long as it needed to be, so it points there.
    if brief.get("notes"):
        n.limitations.append(
            f"A further {len(brief['notes'])} data-handling note(s) — records dropped, currency "
            "conversion, coverage thresholds and the columns excluded from the trend analysis — "
            "are listed in full under “Data quality notes” in section 2. Each one narrows what "
            "the figures above cover."
        )
    n.limitations.append(
        "Correlation and regression results describe association within this sample. They are not "
        "causal estimates and should not be used as elasticities for policy simulation."
    )
    return n


def _write_cross_section(n: Narrative, section: dict[str, Any]) -> None:
    """Prose for the price level and its regional spread, from computed numbers only."""
    overall = section.get("overall") or {}
    if not overall:
        return

    label = section.get("price_label", "advertised price")
    money = "$" if "USD" in str(section.get("currency_note", "")) else ""
    listings = overall.get("listings", 0)
    median, mean = _num(overall.get("median")), _num(overall.get("mean"))
    if median is None:
        return

    lead = (
        f"Across {listings:,} individual adverts the median {label} is {money}{median:,.0f}"
        + (
            f", or {money}{overall['median_per_sqm']:,.2f} per square metre"
            if overall.get("median_per_sqm") is not None
            else ""
        )
        + (
            f" on a median floor area of {overall['median_area_sqm']:,.0f} m²"
            if overall.get("median_area_sqm") is not None
            else ""
        )
        + "."
    )
    skew = _num(overall.get("skew_mean_over_median_pct"))
    if mean is not None and skew is not None:
        lead += (
            f" The mean of {money}{mean:,.0f} sits {skew:.0f}% above the median, so the "
            "distribution is right-skewed and the median is the figure that describes what a "
            "typical household faces."
        )
    n.executive_summary.append(lead)
    n.current_situation.insert(0, lead)

    p25, p75 = _num(overall.get("p25")), _num(overall.get("p75"))
    if p25 is not None and p75 is not None:
        n.current_situation.append(
            f"Half of all adverts fall between {money}{p25:,.0f} and {money}{p75:,.0f}; the "
            f"tenth of the market above {money}{_num(overall.get('p90')) or 0:,.0f} is what "
            "stretches the average."
        )
    n.key_findings.insert(
        0, f"The median {label} across {listings:,} adverts is {money}{median:,.0f}."
    )

    region = section.get("by_region") or {}
    top, bottom = region.get("most_expensive") or {}, region.get("cheapest") or {}
    field_name = region.get("group_field", "region")
    if top and bottom:
        ratio = _num(region.get("spread_ratio"))
        sentence = (
            f"{top.get(field_name)} is the most expensive {field_name} at a median of "
            f"{money}{_num(top.get('median')) or 0:,.0f}, against {bottom.get(field_name)} at "
            f"{money}{_num(bottom.get('median')) or 0:,.0f}"
            + (f" — a ratio of {ratio:.2f} to one." if ratio else ".")
        )
        n.regional_analysis.insert(0, sentence)
        n.executive_summary.append(sentence)
        # Key findings are scanned, not read, so this one is compressed rather
        # than repeated word for word from the summary above.
        n.key_findings.insert(
            1,
            f"{top.get(field_name)} is the most expensive {field_name} and "
            f"{bottom.get(field_name)} the cheapest"
            + (f", a gap of {ratio:.2f} times." if ratio else "."),
        )

        per_sqm_top, per_sqm_bottom = _num(top.get("median_per_sqm")), _num(bottom.get("median_per_sqm"))
        if per_sqm_top is not None and per_sqm_bottom is not None:
            n.regional_analysis.append(
                f"Measured per square metre the same ordering holds — {money}{per_sqm_top:,.2f} "
                f"against {money}{per_sqm_bottom:,.2f} — so the gap reflects the price of space "
                "itself rather than larger dwellings being advertised in the leading market."
                if per_sqm_top > per_sqm_bottom
                else
                f"Per square metre the ordering reverses ({money}{per_sqm_top:,.2f} against "
                f"{money}{per_sqm_bottom:,.2f}), which means the level gap is driven by dwelling "
                "size rather than by the price of space."
            )
        share = _num(top.get("share_of_listings_pct"))
        if share is not None and share > 50:
            n.regional_analysis.append(
                f"{top.get(field_name)} also supplies {share:.0f}% of all adverts in the sample, so "
                "the national median describes that market far more than it describes the country. "
                "Regional figures below should be read as separate markets, not as deviations from "
                "a meaningful national average."
            )

    dispersion = _num(region.get("dispersion_pct"))
    if dispersion is not None:
        n.regional_analysis.append(
            f"The coefficient of variation across regional medians is {dispersion:.1f}%, which is the "
            f"single-number measure of how unequal these markets are."
        )

    rooms = section.get("by_rooms") or []
    if len(rooms) >= 2:
        first, last = rooms[0], rooms[-1]
        first_sqm, last_sqm = _num(first.get("median_per_sqm")), _num(last.get("median_per_sqm"))
        if first_sqm is not None and last_sqm is not None:
            n.current_situation.append(
                f"Price per square metre falls from {money}{first_sqm:,.2f} for a "
                f"{first.get('size_band')} dwelling to {money}{last_sqm:,.2f} for a "
                f"{last.get('size_band')} one."
                + (
                    " That is the expected pattern — larger dwellings command a lower unit price — "
                    "and it is a useful check that the size and price fields are being read correctly."
                    if first_sqm > last_sqm
                    else " That inversion is unusual and suggests the size bands are picking up "
                    "location rather than size."
                )
            )
    # The cross-section's own caveats already reach the limitations section
    # through the brief's shared notes list; adding them here would print twice.


def _fallback_recommendations(
    label: str,
    yoy: float | None,
    cagr: float | None,
    regional: dict[str, Any],
    links: list[dict[str, Any]],
    season: dict[str, Any],
    conc: dict[str, Any],
) -> list[Recommendation]:
    out: list[Recommendation] = []

    if cagr is not None and cagr > 12:
        out.append(Recommendation(
            audience="Ministry of Construction and Housing / Ministry of Economy and Finance",
            action="Publish a quarterly affordability index pairing the price series with median regional wages, "
                   "and tie eligibility for subsidised mortgage products to that index rather than to a fixed price cap.",
            rationale=f"{label} has compounded at {cagr:+.1f}% a year. A fixed nominal cap loses its targeting "
                      "power every year that prices outrun wages.",
            priority="high", horizon="6-12 months",
        ))
        out.append(Recommendation(
            audience="Central Bank of Uzbekistan",
            action="Introduce or tighten macroprudential limits — a loan-to-value ceiling and a debt-service-to-income "
                   "cap — differentiated by region rather than applied uniformly.",
            rationale="Sustained double-digit price growth with mortgage expansion is the standard precondition for "
                      "collateral-valuation risk. Regional differentiation avoids over-tightening in slower markets.",
            priority="high", horizon="6-12 months",
        ))
    elif yoy is not None and yoy < 0:
        out.append(Recommendation(
            audience="Ministry of Construction and Housing",
            action="Phase the release of state-financed housing to avoid adding supply into a falling market, and "
                   "redirect near-term budget to completing stalled projects.",
            rationale=f"{label} is {yoy:.1f}% down year on year. Adding supply into weak demand deepens the correction "
                      "and raises developer default risk.",
            priority="high", horizon="immediate",
        ))

    if "diverging" in str(regional.get("convergence", "")):
        out.append(Recommendation(
            audience="Ministry of Economy and Finance / regional khokimiyats",
            action="Weight infrastructure and serviced-land allocation towards the lagging regions identified in the "
                   "regional table, and publish the allocation formula.",
            rationale=f"Regional dispersion is {regional.get('dispersion_pct')}% and {regional.get('convergence')}. "
                      "Untargeted national programmes will keep concentrating activity in the leading markets.",
            priority="medium", horizon="long term",
        ))

    if conc.get("top_share_pct", 0) > 40:
        out.append(Recommendation(
            audience="Bank supervision department, Central Bank of Uzbekistan",
            action=f"Set a supervisory concentration threshold for mortgage exposure to {conc.get('top_group')} and "
                   "require banks to stress-test a local price correction there.",
            rationale=f"{conc.get('top_group')} accounts for {conc.get('top_share_pct')}% of measured activity, so "
                      "system-wide mortgage books are exposed to one local market.",
            priority="medium", horizon="6-12 months",
        ))

    for link in links:
        if link.get("best_lag", 0) > 0:
            name = link.get("driver_label") or link.get("driver")
            out.append(Recommendation(
                audience="Central Bank of Uzbekistan / Ministry of Economy and Finance",
                action=f"Adopt {name} as a leading indicator in the housing-market monitoring dashboard, "
                       f"read with a {link['best_lag']}-period lead.",
                rationale=f"{name} correlates with {link.get('target_of', label)} at "
                          f"r = {link.get('best_lag_correlation')} when led by {link['best_lag']} period(s), "
                          "giving advance warning of turning points.",
                priority="medium", horizon="immediate",
            ))
            break

    if season.get("detected"):
        out.append(Recommendation(
            audience="State Statistics Committee",
            action="Publish a seasonally adjusted version of the housing price and transaction series alongside the raw data.",
            rationale=f"A seasonal pattern of strength {season.get('strength')} is present, peaking in "
                      f"{season.get('peak_period')}. Unadjusted period-on-period figures mislead readers.",
            priority="medium", horizon="6-12 months",
        ))

    out.append(Recommendation(
        audience="State Statistics Committee / cadastre agency",
        action="Publish transaction-level price data by region, dwelling type and floor area on a monthly cycle, "
               "with a stable identifier so repeat sales can be tracked.",
        rationale="Repeat-sales and hedonic indices cannot be constructed from the aggregated data currently "
                  "available, which limits every analysis of this market — including this one.",
        priority="high", horizon="long term",
    ))
    out.append(Recommendation(
        audience="Commercial bank lenders and developers",
        action="Reprice risk against the regional growth table rather than the national average, and size new "
               "projects to the demand signal in the lagging regions rather than the leading ones.",
        rationale="The dispersion between the fastest and slowest regions in this dataset is large enough that "
                  "a national average is not a usable planning assumption for any individual market.",
        priority="medium", horizon="immediate",
    ))
    return out


def _align_events_to_breaks(events: list[dict], breaks: list[dict], months: int = 9) -> list[str]:
    """Find policy dates that sit close to a detected structural break."""
    import pandas as pd

    matches = []
    for point in breaks:
        try:
            break_date = pd.Timestamp(point["date"])
        except Exception:
            continue
        for event in events:
            try:
                event_date = pd.Timestamp(str(event.get("date"))[:10])
            except Exception:
                continue
            gap = abs((break_date - event_date).days)
            if gap <= months * 31:
                matches.append(
                    f"the {point.get('direction')} of {_num(point.get('shift_pct')):+.1f}% detected at "
                    f"{point['date']} falls within {round(gap / 30)} month(s) of “{event.get('title')}” "
                    f"({event.get('date')})"
                )
                break
    return matches


# ---------------------------------------------------------------------------
def _yoy_phrase(metric: dict[str, Any]) -> str:
    """Year-on-year movement, in percentage points for rates and percent otherwise."""
    if metric.get("change_unit") == "pp":
        value = _num(metric.get("yoy_pp"))
        if value is None:
            return ""
        return f"{value:+.2f} percentage points against a year earlier"
    value = _num(metric.get("yoy_pct"))
    return "" if value is None else f"{value:+.1f}% year on year"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    magnitude = abs(number)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{number / threshold:,.2f}{suffix}"
    return f"{number:,.2f}" if magnitude < 100 else f"{number:,.0f}"


def _pct_of(ratio: Any) -> str:
    value = _num(ratio)
    return "an unknown share" if value is None else f"{value * 100:.0f}%"
