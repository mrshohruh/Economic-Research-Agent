"""Explain what each column in the dataset actually measures.

A column name is not a definition: "price1" and "price2" tell a reader nothing
about which is which, and a name alone cannot say whether two similarly named
columns are a unit conversion of each other, two genuinely different price
concepts, or unrelated. This module builds that explanation from what the
profiler already computed — role, unit, sample values, summary statistics —
and, when two or more columns share a name stem, from how their values
actually relate to each other in the data. The LLM, when available, is asked
only to turn that evidence into plain English, never to guess past it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from ..llm import LLM, LLMUnavailable
from . import translate
from .profiler import ColumnProfile, Understanding

LOGGER = logging.getLogger(__name__)

ROLE_DESCRIPTIONS: dict[str, str] = {
    "date": "The time period each row refers to.",
    "region": "A geographic breakdown (region, city or district).",
    "segment": "A category the data is broken down by, such as dwelling size or type.",
    "price": "A price level, in the local currency or USD depending on the source.",
    "price_per_sqm": "Price expressed per square metre of floor area.",
    "volume": "A count of transactions, deals or listings observed in the period.",
    "supply": "Newly built, commissioned or permitted housing units.",
    "mortgage": "Mortgage lending, as an amount or a count of loans.",
    "rate": "An interest rate or other percentage level.",
    "income": "Household income, wages or earnings.",
    "inflation": "A price-level index or inflation rate.",
    "fx": "An exchange rate between the local currency and USD.",
    "area": "A floor area measurement.",
    "population": "A population count.",
    "value": "A numeric measure that the column name alone did not identify further.",
    "category": "A qualitative attribute recorded for each row.",
    "unit": "A unit of measurement or currency label attached to another column.",
}

# Columns sharing this many characters of their lower-cased, digit-stripped
# name are treated as the same "family" worth comparing to one another.
_STEM_RE = re.compile(r"^(.*?)[\s_]*\d*$")


def build_glossary(understanding: Understanding, llm: LLM | None = None) -> list[dict[str, Any]]:
    """One entry per reportable column: name, label, role, unit and a plain-English description."""
    profile = understanding.primary_profile
    frame = understanding.dataset.tables.get(understanding.primary)
    columns = [c for c in profile.columns if c.role not in {"identifier", "unit"}]
    if not columns:
        return []

    entries: dict[str, dict[str, Any]] = {}
    for col in columns:
        entries[col.name] = {
            "name": col.name,
            "label": translate.humanize_label(col.name),
            "role": col.role,
            "unit": col.unit,
            "samples": [str(s) for s in col.samples[:5]],
            "description": _heuristic_description(col),
        }

    for members in _group_by_stem(columns).values():
        if len(members) > 1:
            _annotate_ambiguous_group(entries, members, frame)

    if llm is not None and llm.available:
        try:
            _llm_refine(entries, llm)
        except LLMUnavailable as exc:
            LOGGER.info("LLM glossary refinement skipped: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("LLM glossary refinement failed: %s", exc)

    return list(entries.values())


# ---------------------------------------------------------------------------
def _heuristic_description(col: ColumnProfile) -> str:
    base = ROLE_DESCRIPTIONS.get(col.role, "A recorded attribute of each row.")
    if col.is_metric and col.minimum is not None and col.maximum is not None:
        unit = f" {col.unit}" if col.unit and col.unit not in {"currency", "count", "units"} else ""
        base += (
            f" Observed values range from {col.minimum:,.2f} to {col.maximum:,.2f}"
            f" (average {col.mean:,.2f}){unit}."
        )
    elif col.samples:
        shown = ", ".join(str(s) for s in col.samples[:4])
        base += f" Example values: {shown}."
    return base


def _group_by_stem(columns: list[ColumnProfile]) -> dict[str, list[ColumnProfile]]:
    groups: dict[str, list[ColumnProfile]] = {}
    for col in columns:
        if not col.is_metric:
            continue
        stem = _STEM_RE.sub(r"\1", col.name.lower()).strip("_ ")
        groups.setdefault((stem, col.role), []).append(col)  # type: ignore[index]
    # Re-key by stem alone for the caller; role is only there to avoid grouping
    # an unrelated numeric column that happens to share a name prefix.
    return {f"{stem}:{role}": members for (stem, role), members in groups.items()}


def _annotate_ambiguous_group(
    entries: dict[str, dict[str, Any]], members: list[ColumnProfile], frame: pd.DataFrame | None
) -> None:
    names = [m.name for m in members]
    if frame is None or len(names) != 2 or any(n not in frame.columns for n in names):
        if len(names) > 1:
            for name in names:
                others = ", ".join(n for n in names if n != name)
                entries[name]["description"] += (
                    f" This column's name pattern is shared with {others}; the values could not be "
                    "compared directly, so treat them as independent measures unless the source "
                    "documentation says otherwise."
                )
        return

    a, b = names
    sa = pd.to_numeric(frame[a], errors="coerce")
    sb = pd.to_numeric(frame[b], errors="coerce")
    both = pd.concat([sa, sb], axis=1).dropna()
    if len(both) < 5:
        return

    corr = float(both[a].corr(both[b]))
    ratio = (both[a] / both[b]).replace([np.inf, -np.inf], np.nan).dropna()
    ratio_med = float(ratio.median()) if len(ratio) else None
    ratio_spread = float(ratio.std(ddof=0) / ratio.mean()) if len(ratio) and ratio.mean() else None

    if ratio_med and ratio_spread is not None and ratio_spread < 0.02 and corr > 0.98:
        note = (
            f"“{a}” and “{b}” move together almost perfectly (r = {corr:.3f}) at a near-constant ratio "
            f"of about {ratio_med:,.3g}, which is the signature of a unit conversion (e.g. one is the "
            "other expressed in a different currency or per a different base) rather than two "
            "independent indicators."
        )
    elif corr > 0.9:
        note = (
            f"“{a}” and “{b}” are highly correlated (r = {corr:.3f}) but not at a fixed ratio "
            f"(typical {a}/{b} ≈ {ratio_med:,.3g}), which fits two related but distinct price or value "
            "concepts — for example an asking price versus a transacted or discounted one — rather than "
            "a simple unit conversion. The dataset does not label which is which; this description "
            "reflects only what the numbers show."
        )
    else:
        note = (
            f"“{a}” and “{b}” share a name pattern but correlate only weakly (r = {corr:.3f}) and do not "
            "move at a fixed ratio, so they should be treated as measuring different things."
        )
    for name in names:
        entries[name]["description"] += " " + note


# ---------------------------------------------------------------------------
GLOSSARY_SYSTEM = (
    "You are a real estate analyst explaining a housing and rental dataset to a non-technical "
    "reader. You explain what each column measures using only the statistics and relationships "
    "given to you. You never invent a meaning the numbers do not support, and where two columns "
    "cannot be told apart from the data alone you say so plainly instead of guessing. You are "
    "particularly careful about the sale/rent distinction — a monthly rent and a purchase price "
    "are different quantities that a column name often fails to separate — and about whether a "
    "price is an advertised asking price or a transacted one."
)


def _llm_refine(entries: dict[str, dict[str, Any]], llm: LLM) -> None:
    payload = [
        {
            "name": e["name"],
            "role": e["role"],
            "unit": e["unit"],
            "samples": e["samples"],
            "current_description": e["description"],
        }
        for e in entries.values()
    ]
    if not payload:
        return

    prompt = f"""Here is the profiled schema of a housing-market dataset, one entry per column,
including a draft description already computed from the data (role, unit, sample values, and
for similarly-named columns, how their values statistically relate to each other):

{_compact_json(payload)}

Rewrite each "current_description" into one or two clear sentences a non-technical reader can
act on. Keep every specific number and every stated relationship between similarly-named
columns exactly as given — do not drop the disambiguation between columns that share a name
pattern. Do not introduce any fact not present in the input.

Return JSON of exactly this shape:
{{"descriptions": [{{"name": "<column name>", "description": "<rewritten description>"}}]}}"""

    result = llm.complete_json(prompt, system=GLOSSARY_SYSTEM, max_tokens=4000)
    if not isinstance(result, dict):
        return
    for item in result.get("descriptions", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        description = str(item.get("description", "")).strip()
        if name in entries and description:
            entries[name]["description"] = description


def _compact_json(obj: Any, limit: int = 8000) -> str:
    import json

    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    return text[:limit] + ("\n... (truncated)" if len(text) > limit else "")
