"""Decide which variables belong in a housing-market analysis, and drop the rest.

The profiler works out what every column *is* and the glossary works out what
every column *means*. Neither asks the next question: does this variable belong
in a report about house prices and rents at all?

A scraped marketplace feed carries advert view counts, seller ratings, photo
counts, map coordinates and internal record keys alongside the price. Left in,
they become candidate drivers, get correlated against the headline series, take
up chart slots, and eventually turn into a sentence about how "user ratings rose
alongside prices". None of that is housing-market analysis.

This module runs after the glossary — meaning is established before relevance is
judged, never the other way round — and removes the unrelated columns from the
analysis frame. Decisions are logged, not reported: the report describes the
market, not the columns that were never part of it.

The screen is deliberately conservative. A column carrying a recognised
housing or macroeconomic role is only removed on an unambiguous match (a record
key, an engagement counter, ingestion bookkeeping) or a data-quality failure
(empty, constant, or a per-row identifier). Anything else survives, and if the
screen would leave nothing behind it is abandoned entirely.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM, LLMUnavailable
from . import translate
from .profiler import ColumnProfile, Understanding

LOGGER = logging.getLogger(__name__)

# Roles that are, by definition, part of a housing-market or macroeconomic
# picture. A column with one of these survives anything short of a strong
# exclusion match or a data-quality failure.
CORE_ROLES = {
    "price",
    "price_per_sqm",
    "volume",
    "supply",
    "mortgage",
    "rate",
    "income",
    "inflation",
    "fx",
    "area",
    "population",
}

# Unambiguous exclusions: these are never a market measurement, whatever role
# the name pattern happened to assign them.
STRONG_EXCLUSIONS: list[tuple[str, str]] = [
    (r"(^|_)(id|uuid|guid|hash|pk|fk|key)(_|$)", "a record key with no analytical meaning"),
    (r"(url|link|href|slug|permalink|token|api_|endpoint)", "a link or technical token"),
    (
        r"(view|click|impression|favou?rite|bookmark|like|share|comment_count|call_count|"
        r"message_count|contact_count|response_count)",
        "an advert engagement counter, not a price or a quantity of housing",
    ),
    (
        r"(photo|image|video|thumbnail|attachment|media|gallery|avatar)",
        "advert media, not a market measurement",
    ),
    (
        r"(^|_)(lat|lon|lng|latitude|longitude|geo|coord|coords|zoom|radius|bbox)(_|$)",
        "a map coordinate",
    ),
    (
        r"(scrape|scraped|crawl|crawled|parse|parsed|import|export|etl|ingest|batch|"
        r"row_num|rownum|offset|page_no|page_number|checksum)",
        "ingestion bookkeeping from how the data was collected",
    ),
    (
        r"(^|_)(test|dummy|tmp|temp|unused|deprecated|reserved|placeholder)(_|$)",
        "a placeholder column",
    ),
]

# Softer exclusions: plausible in some datasets, so they only apply to a column
# that does not already carry a core housing or macroeconomic role.
WEAK_EXCLUSIONS: list[tuple[str, str]] = [
    (
        r"(rating|review|reputation|trust|stars)",
        "a seller or advert rating rather than a property measurement",
    ),
    (
        r"(phone|email|contact|username|user_|seller|agent_|broker|author|account|profile|shop)",
        "a detail about who placed the advert, not about the property",
    ),
    (
        r"^(is|has)_|(promoted|featured|boost|premium|vip|highlight|banner|paid_ad|sponsored)",
        "an advert placement flag sold by the platform",
    ),
    (
        r"(created_at|updated_at|modified|deleted|expire|expiry|status|state_code|version)",
        "record lifecycle bookkeeping",
    ),
]

# A column missing this much of its data cannot support a trend or an average.
MAX_MISSING_PCT = 95.0

# Above this many rows, a column with one distinct value per row is a key.
IDENTIFIER_MIN_ROWS = 20


@dataclass
class VariableDecision:
    """One screening verdict, kept for the run log and for tests."""

    name: str
    label: str
    role: str
    keep: bool
    reason: str
    source: str = "heuristic"  # heuristic | data quality | model review

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class RelevanceScreen:
    decisions: list[VariableDecision] = field(default_factory=list)
    applied: bool = False
    abandoned_reason: str = ""

    @property
    def kept(self) -> list[str]:
        return [d.name for d in self.decisions if d.keep]

    @property
    def dropped(self) -> list[str]:
        return [d.name for d in self.decisions if not d.keep]

    def decision_for(self, name: str) -> VariableDecision | None:
        for decision in self.decisions:
            if decision.name == name:
                return decision
        return None

    def summary(self) -> str:
        """A one-line account for the console and the run log."""
        if self.abandoned_reason:
            return f"Variable relevance screen not applied: {self.abandoned_reason}"
        if not self.dropped:
            return f"All {len(self.kept)} variable(s) are relevant to the housing market."
        shown = ", ".join(
            f"{d.label} ({d.reason})" for d in self.decisions if not d.keep
        )
        return (
            f"Kept {len(self.kept)} housing-market variable(s); "
            f"excluded {len(self.dropped)}: {shown}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "kept": self.kept,
            "dropped": [d.to_dict() for d in self.decisions if not d.keep],
        }


# ---------------------------------------------------------------------------
def screen(
    understanding: Understanding,
    glossary: list[dict[str, Any]] | None = None,
    llm: LLM | None = None,
) -> RelevanceScreen:
    """Judge every candidate variable, then remove the unrelated ones in place.

    ``understanding.tidy`` and the primary profile's ``metric_cols`` are filtered
    down to the relevant set, and matching entries are removed from ``glossary``
    so that nothing downstream — including the report writer — ever sees a
    variable that was screened out.
    """
    candidates = _candidates(understanding)
    if not candidates:
        return RelevanceScreen(abandoned_reason="the dataset carries no numeric variables")

    result = RelevanceScreen()
    for name in candidates:
        col = _find_column(understanding, name)
        role = understanding.role_for_metric(name)
        keep, reason, source = _verdict(name, role, col)
        result.decisions.append(
            VariableDecision(
                name=name,
                label=translate.humanize_label(name),
                role=role,
                keep=keep,
                reason=reason,
                source=source,
            )
        )

    if llm is not None and llm.available:
        try:
            _llm_review(result, glossary or [], llm)
        except LLMUnavailable as exc:
            LOGGER.info("relevance screen ran on heuristics alone: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("LLM relevance review failed: %s", exc)

    if not result.kept:
        result.abandoned_reason = (
            "every variable would have been excluded, which is more likely to be a "
            "screening error than a dataset with nothing in it"
        )
        for decision in result.decisions:
            decision.keep = True
            decision.reason = "kept: the screen was abandoned"
        LOGGER.warning("relevance screen abandoned: %s", result.abandoned_reason)
        return result

    _apply(understanding, result, glossary)
    result.applied = True
    LOGGER.info("%s", result.summary())
    return result


# ---------------------------------------------------------------------------
def _candidates(understanding: Understanding) -> list[str]:
    """The numeric variables the analysis would otherwise use."""
    names = list(understanding.metrics)
    if not names:
        names = list(understanding.primary_profile.metric_cols)
    return names


def _find_column(understanding: Understanding, name: str) -> ColumnProfile | None:
    profile = understanding.primary_profile
    col = profile.column(name)
    if col is not None:
        return col
    for other in understanding.profiles.values():
        col = other.column(name)
        if col is not None:
            return col
    return None


def _verdict(name: str, role: str, col: ColumnProfile | None) -> tuple[bool, str, str]:
    """Keep-or-drop for one variable, with the reason and what decided it."""
    lowered = name.lower()

    for pattern, reason in STRONG_EXCLUSIONS:
        if re.search(pattern, lowered):
            return False, reason, "heuristic"

    if role not in CORE_ROLES:
        for pattern, reason in WEAK_EXCLUSIONS:
            if re.search(pattern, lowered):
                return False, reason, "heuristic"

    if col is not None:
        quality = _data_quality_failure(col)
        if quality:
            return False, quality, "data quality"

    if role in CORE_ROLES:
        return True, f"measures {role.replace('_', ' ')}, central to a housing report", "heuristic"
    return True, "a numeric measure with no sign of being unrelated", "heuristic"


def _data_quality_failure(col: ColumnProfile) -> str:
    """Reasons a variable cannot support any statistic, whatever it measures."""
    if col.non_null == 0:
        return "no values at all"
    if col.missing_pct > MAX_MISSING_PCT:
        return f"{col.missing_pct:.0f}% missing, too sparse to measure anything"
    if col.unique <= 1:
        return "a single constant value, so it cannot explain any variation"
    if (
        col.non_null >= IDENTIFIER_MIN_ROWS
        and col.unique == col.non_null
        and "int" in col.dtype.lower()
    ):
        return "one distinct whole number per row, which is a key rather than a measurement"
    return ""


# ---------------------------------------------------------------------------
def _apply(
    understanding: Understanding,
    result: RelevanceScreen,
    glossary: list[dict[str, Any]] | None,
) -> None:
    """Remove the excluded variables from everything downstream."""
    dropped = set(result.dropped)
    if not dropped:
        return

    tidy = understanding.tidy
    if not tidy.empty and "metric" in tidy.columns:
        understanding.tidy = tidy[~tidy["metric"].isin(dropped)].reset_index(drop=True)

    for profile in understanding.profiles.values():
        profile.metric_cols = [c for c in profile.metric_cols if c not in dropped]

    if glossary is not None:
        survivors = [entry for entry in glossary if entry.get("name") not in dropped]
        glossary[:] = survivors


# ---------------------------------------------------------------------------
RELEVANCE_SYSTEM = (
    "You decide which variables in a dataset belong in an analysis of housing and rental "
    "prices in Uzbekistan, and which are unrelated to it. Relevant variables measure the "
    "property market or the economy around it: prices, rents, price per square metre, floor "
    "area, transaction and listing counts, completions, mortgage lending, interest rates, "
    "incomes, inflation, exchange rates, population. Unrelated variables describe the "
    "platform or the record rather than the market: identifiers, links, view and click "
    "counts, seller ratings, photo counts, coordinates, promotion flags and ingestion "
    "bookkeeping. You judge only from the evidence given, and when a variable could "
    "genuinely go either way you keep it — removing a real housing indicator is the more "
    "damaging mistake."
)


def _llm_review(result: RelevanceScreen, glossary: list[dict[str, Any]], llm: LLM) -> None:
    """Let the model correct the heuristic verdicts it disagrees with."""
    descriptions = {
        entry.get("name"): entry.get("description", "")
        for entry in glossary
        if isinstance(entry, dict)
    }
    payload = [
        {
            "name": d.name,
            "role": d.role,
            "meaning": str(descriptions.get(d.name, ""))[:300],
            "heuristic_verdict": "keep" if d.keep else "drop",
            "heuristic_reason": d.reason,
        }
        for d in result.decisions
    ]

    prompt = f"""A housing-market analyst is about to analyse a dataset. Below is every numeric
variable in it, what the profiler thinks it measures, and a provisional keep-or-drop verdict.

{_compact_json(payload)}

Correct the verdicts that are wrong. Drop a variable only if it is unrelated to housing and
rental market analysis — it describes the listing platform, the record or the data collection
rather than the property market or the economy around it. Keep anything that measures a
price, a rent, an area, a quantity of housing, credit, a rate, an income, a price level, an
exchange rate or a population, and keep anything genuinely ambiguous.

Return JSON of exactly this shape, listing ONLY the variables whose verdict should CHANGE:
{{"changes": [{{"name": "...", "keep": true, "reason": "<why, in one clause>"}}]}}"""

    review = llm.complete_json(prompt, system=RELEVANCE_SYSTEM, max_tokens=2000)
    if not isinstance(review, dict):
        return

    for change in review.get("changes", []) or []:
        if not isinstance(change, dict):
            continue
        decision = result.decision_for(str(change.get("name", "")))
        if decision is None or "keep" not in change:
            continue
        keep = bool(change["keep"])
        if keep == decision.keep:
            continue
        reason = str(change.get("reason", "")).strip() or "model review"
        LOGGER.info(
            "relevance review changed %s: %s -> %s (%s)",
            decision.name,
            "keep" if decision.keep else "drop",
            "keep" if keep else "drop",
            reason,
        )
        decision.keep = keep
        decision.reason = reason
        decision.source = "model review"


def _compact_json(obj: Any, limit: int = 6000) -> str:
    import json

    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    return text[:limit] + ("\n... (truncated)" if len(text) > limit else "")
