"""Build the qualitative half of the report: policy, macro and market context.

Search results and the local policy file go in; a structured, cited set of
findings comes out. Without an LLM key the findings are still produced — they
just present the evidence as organised, ranked source material instead of a
synthesised narrative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..llm import LLM, LLMUnavailable
from . import knowledge, websearch
from .websearch import SearchHit

LOGGER = logging.getLogger(__name__)

RESEARCH_SYSTEM = (
    "You are a senior housing-market economist covering Uzbekistan and Central Asia. "
    "You read primary sources carefully, separate fact from speculation, and never "
    "invent statistics. When a figure is not in the sources you were given, you say so "
    "rather than guessing. You always attribute claims to the source they came from."
)


@dataclass
class ThemeFinding:
    key: str
    title: str
    summary: str = ""
    points: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "points": self.points,
            "sources": self.sources,
        }


@dataclass
class PolicyEvent:
    date: str
    title: str
    category: str = "policy"
    summary: str = ""
    expected_impact: str = ""
    direction: str = "neutral"
    confidence: str = "medium"
    origin: str = "local record"
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class ResearchFindings:
    themes: list[ThemeFinding] = field(default_factory=list)
    policy_events: list[PolicyEvent] = field(default_factory=list)
    macro_factors: list[dict[str, Any]] = field(default_factory=list)
    sources: list[SearchHit] = field(default_factory=list)
    llm_used: bool = False
    web_used: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "themes": [t.to_dict() for t in self.themes],
            "policy_events": [e.to_dict() for e in self.policy_events],
            "macro_factors": self.macro_factors,
            "sources": [s.to_dict() for s in self.sources],
            "llm_used": self.llm_used,
            "web_used": self.web_used,
            "notes": self.notes,
        }

    def source_index(self) -> list[dict[str, str]]:
        """Numbered, de-duplicated source list for the report bibliography."""
        seen: dict[str, dict[str, str]] = {}
        for hit in self.sources:
            if hit.url not in seen:
                seen[hit.url] = {"title": hit.title, "url": hit.url, "domain": hit.domain}
        for event in self.policy_events:
            if event.source and event.source not in seen:
                seen[event.source] = {
                    "title": event.title,
                    "url": event.source,
                    "domain": websearch._domain(event.source),
                }
        return list(seen.values())


# ---------------------------------------------------------------------------
def gather(
    llm: LLM | None,
    *,
    period_start: str = "",
    period_end: str = "",
    regions: list[str] | None = None,
    data_highlights: str = "",
    use_web: bool = True,
    results_per_query: int = 6,
    pages_to_read: int = 3,
    policy_file: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ResearchFindings:
    """Run the research agenda and return structured, cited findings."""
    say = progress or (lambda msg: LOGGER.info(msg))
    findings = ResearchFindings()

    # 1. Local policy record --------------------------------------------------
    local = knowledge.load_policy_records(policy_file)
    windowed = knowledge.events_in_window(local, period_start, period_end)
    for record in windowed:
        findings.policy_events.append(
            PolicyEvent(
                date=str(record.get("date", ""))[:10],
                title=str(record.get("title", "Untitled measure")),
                category=str(record.get("category", "policy")),
                summary=str(record.get("summary", "")),
                expected_impact=str(record.get("expected_impact", "")),
                direction=str(record.get("direction", "neutral")),
                confidence=str(record.get("confidence", "medium")),
                origin="local policy record",
                source=str(record.get("source", "")),
            )
        )
    say(f"Loaded {len(findings.policy_events)} policy records from the local knowledge file.")

    # 2. Web research ---------------------------------------------------------
    agenda = knowledge.themes(period_start, period_end, regions)
    hits_by_theme: dict[str, list[SearchHit]] = {}

    if use_web:
        for theme in agenda:
            say(f"Researching: {theme.title}")
            hits = websearch.search_many(theme.queries, per_query=results_per_query)
            if hits:
                websearch.enrich(hits, limit=pages_to_read)
            hits_by_theme[theme.key] = hits
            findings.sources.extend(hits[: results_per_query + 2])
        findings.web_used = any(hits_by_theme.values())
        if not findings.web_used:
            findings.notes.append(
                "Web research returned no results — the machine may be offline or the "
                "search endpoint may be rate-limiting. The report relies on the supplied "
                "data and the local policy file only."
            )
    else:
        findings.notes.append("Web research was disabled for this run.")

    # De-duplicate the master source list.
    unique: dict[str, SearchHit] = {}
    for hit in findings.sources:
        if hit.url not in unique or hit.score > unique[hit.url].score:
            unique[hit.url] = hit
    findings.sources = sorted(unique.values(), key=lambda h: h.score, reverse=True)[:40]

    # 3. Synthesis ------------------------------------------------------------
    if llm is not None and llm.available and (hits_by_theme or findings.policy_events):
        try:
            say("Synthesising research findings with the language model...")
            _synthesise(llm, findings, agenda, hits_by_theme, data_highlights)
            findings.llm_used = True
        except LLMUnavailable as exc:
            findings.notes.append(f"LLM synthesis unavailable ({exc}); falling back to evidence digest.")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("research synthesis failed: %s", exc)
            findings.notes.append(f"LLM synthesis failed ({exc}); falling back to evidence digest.")

    if not findings.llm_used:
        _digest(findings, agenda, hits_by_theme)

    return findings


# ---------------------------------------------------------------------------
def _synthesise(
    llm: LLM,
    findings: ResearchFindings,
    agenda: list[knowledge.Theme],
    hits_by_theme: dict[str, list[SearchHit]],
    data_highlights: str,
) -> None:
    evidence_blocks = []
    for theme in agenda:
        hits = hits_by_theme.get(theme.key, [])[:6]
        if not hits:
            continue
        lines = [f"### THEME {theme.key}: {theme.title} (purpose: {theme.purpose})"]
        for i, hit in enumerate(hits, 1):
            body = (hit.body or hit.snippet)[:1800]
            lines.append(f"[{theme.key}-{i}] {hit.title} | {hit.url}\n{body}")
        evidence_blocks.append("\n".join(lines))

    known = "\n".join(
        f"- {e.date} | {e.title} ({e.confidence} confidence): {e.summary} Expected impact: {e.expected_impact}"
        for e in findings.policy_events
    ) or "(none on file)"

    prompt = f"""Analyse the Uzbek housing market using the source material below.

## What the analyst's own dataset shows
{data_highlights or "(no dataset summary provided)"}

## Policy events already on file
{known}

## Web source material
{chr(10).join(evidence_blocks) if evidence_blocks else "(no web sources retrieved)"}

## Your task
Produce a rigorous, cited synthesis. Rules:
- Use ONLY the material above for facts and figures. Never invent a statistic.
- If a theme has weak evidence, say the evidence is thin instead of padding it.
- Every point must carry the id(s) of the source it came from, e.g. "policy-2".
- Explain mechanisms, not just events: how does a measure reach prices, volumes or credit?
- Where the dataset and the sources disagree, say so explicitly.

Return JSON of exactly this shape:
{{
  "themes": [
    {{"key": "<theme key>", "summary": "<2-4 sentence synthesis>",
      "points": ["<specific finding with [source-id]>", "..."]}}
  ],
  "policy_events": [
    {{"date": "YYYY-MM-DD", "title": "...", "category": "policy|monetary|fiscal|regulatory|macro|demographic",
      "summary": "...", "expected_impact": "how it transmits to the housing market",
      "direction": "positive|negative|mixed|neutral", "confidence": "high|medium|low",
      "source_id": "<the source id it came from>"}}
  ],
  "macro_factors": [
    {{"factor": "...", "current_state": "...", "housing_impact": "...",
      "direction": "positive|negative|mixed|neutral", "source_id": "..."}}
  ]
}}

Include only policy events you actually found evidence for in the material above.
Do not repeat events already on file unless you have new detail to add."""

    result = llm.complete_json(prompt, system=RESEARCH_SYSTEM, max_tokens=8000)
    if not isinstance(result, dict):
        raise LLMUnavailable("unexpected synthesis shape")

    lookup = _source_lookup(agenda, hits_by_theme)
    titles = {theme.key: theme.title for theme in agenda}

    for item in result.get("themes", []) or []:
        key = str(item.get("key", ""))
        finding = ThemeFinding(
            key=key,
            title=titles.get(key, key.replace("_", " ").title()),
            summary=str(item.get("summary", "")).strip(),
            points=[str(p).strip() for p in (item.get("points") or []) if str(p).strip()],
        )
        finding.sources = [
            {"title": h.title, "url": h.url, "domain": h.domain}
            for h in hits_by_theme.get(key, [])[:5]
        ]
        if finding.summary or finding.points:
            findings.themes.append(finding)

    existing = {(e.date[:7], e.title.lower()[:40]) for e in findings.policy_events}
    for item in result.get("policy_events", []) or []:
        date = str(item.get("date", ""))[:10]
        title = str(item.get("title", "")).strip()
        if not date or not title:
            continue
        if (date[:7], title.lower()[:40]) in existing:
            continue
        hit = lookup.get(str(item.get("source_id", "")))
        findings.policy_events.append(
            PolicyEvent(
                date=date,
                title=title,
                category=str(item.get("category", "policy")),
                summary=str(item.get("summary", "")),
                expected_impact=str(item.get("expected_impact", "")),
                direction=str(item.get("direction", "neutral")),
                confidence=str(item.get("confidence", "medium")),
                origin="web research",
                source=hit.url if hit else "",
            )
        )
        existing.add((date[:7], title.lower()[:40]))

    for item in result.get("macro_factors", []) or []:
        hit = lookup.get(str(item.get("source_id", "")))
        findings.macro_factors.append(
            {
                "factor": str(item.get("factor", "")),
                "current_state": str(item.get("current_state", "")),
                "housing_impact": str(item.get("housing_impact", "")),
                "direction": str(item.get("direction", "neutral")),
                "source": hit.url if hit else "",
                "source_title": hit.title if hit else "",
            }
        )

    findings.policy_events.sort(key=lambda e: e.date)


def _source_lookup(
    agenda: list[knowledge.Theme], hits_by_theme: dict[str, list[SearchHit]]
) -> dict[str, SearchHit]:
    lookup: dict[str, SearchHit] = {}
    for theme in agenda:
        for i, hit in enumerate(hits_by_theme.get(theme.key, [])[:6], 1):
            lookup[f"{theme.key}-{i}"] = hit
    return lookup


# ---------------------------------------------------------------------------
def _digest(
    findings: ResearchFindings,
    agenda: list[knowledge.Theme],
    hits_by_theme: dict[str, list[SearchHit]],
) -> None:
    """No-LLM fallback: organise the retrieved evidence by theme, with citations."""
    for theme in agenda:
        hits = hits_by_theme.get(theme.key, [])[:5]
        if not hits:
            continue
        finding = ThemeFinding(
            key=theme.key,
            title=theme.title,
            summary=(
                f"{theme.purpose} The agent retrieved {len(hits)} sources for this theme. "
                "Written synthesis requires an ANTHROPIC_API_KEY; the ranked evidence is "
                "reproduced below so the findings can be verified directly."
            ),
            points=[f"{hit.snippet[:320]} [{hit.domain}]" for hit in hits if hit.snippet],
            sources=[{"title": h.title, "url": h.url, "domain": h.domain} for h in hits],
        )
        findings.themes.append(finding)

    if not findings.themes and not findings.policy_events:
        findings.notes.append(
            "No qualitative context could be assembled. Add entries to "
            "knowledge/policy_events.json or enable web research."
        )
