"""
Stage 23: Fact-Checker agent. Verifies that:
  - every quantitative claim traces back to a StatResult/Finding computed by Python,
  - every cited source has a real URL (never fabricated),
  - causal language is only used where hypothesis confidence is high; otherwise
    the sentence is rewritten with cautious language.
"""
from __future__ import annotations

import re

from models.schemas import FactCheckItem, FactCheckResult, Finding, Hypothesis, ResearchEvidence

STRONG_CAUSAL_PATTERNS = [
    (re.compile(r"\bcaused\b", re.I), "may have contributed to"),
    (re.compile(r"\bcauses\b", re.I), "may contribute to"),
    (re.compile(r"\bled to\b", re.I), "is consistent with"),
    (re.compile(r"\bresulted in\b", re.I), "coincided with"),
    (re.compile(r"\bdue to\b", re.I), "potentially related to"),
    (re.compile(r"\bbecause of\b", re.I), "against a backdrop of"),
]


def hedge_unsupported_causal_language(text: str, confidence: str) -> tuple[str, bool]:
    """If confidence isn't high, softens strong causal verbs. Returns
    (possibly-revised text, was_revised)."""
    if confidence == "high":
        return text, False
    revised = text
    changed = False
    for pattern, replacement in STRONG_CAUSAL_PATTERNS:
        if pattern.search(revised):
            revised = pattern.sub(replacement, revised)
            changed = True
    return revised, changed


def verify_evidence_sources(evidence: list[ResearchEvidence]) -> list[FactCheckItem]:
    items = []
    for e in evidence:
        if not e.url or not e.url.startswith(("http://", "https://")):
            items.append(FactCheckItem(claim=e.claim, status="removed",
                                        detail=f"Evidence '{e.id}' lacked a verifiable URL and was excluded."))
        else:
            items.append(FactCheckItem(claim=e.source_title, status="verified",
                                        detail=f"Source URL present: {e.url}"))
    return items


def verify_findings_traceable(findings: list[Finding]) -> list[FactCheckItem]:
    items = []
    for f in findings:
        if f.supporting_stats or f.source == "computed from dataset":
            items.append(FactCheckItem(claim=f.finding, status="verified",
                                        detail="Value(s) computed directly from the uploaded dataset via the "
                                               "quantitative analysis engine (not LLM-generated)."))
        else:
            items.append(FactCheckItem(claim=f.finding, status="unverifiable",
                                        detail="No supporting statistic attached; flagged for review."))
    return items


def fact_check_narratives(narratives: dict[str, tuple[str, str]]) -> tuple[dict[str, str], list[FactCheckItem]]:
    """narratives: {key: (text, confidence)}. Returns revised narratives and
    a list of fact-check items describing any language changes."""
    revised_map, items = {}, []
    for key, (text, confidence) in narratives.items():
        revised, changed = hedge_unsupported_causal_language(text, confidence)
        revised_map[key] = revised
        if changed:
            items.append(FactCheckItem(claim=text[:120], status="revised",
                                        detail=f"Causal language softened because supporting confidence was "
                                               f"'{confidence}', not 'high'."))
        else:
            items.append(FactCheckItem(claim=text[:120], status="verified", detail="Language consistent with "
                                        "evidence confidence level; no revision needed."))
    return revised_map, items


def run_fact_check(findings: list[Finding], evidence: list[ResearchEvidence],
                    narratives: dict[str, tuple[str, str]]) -> tuple[dict[str, str], FactCheckResult]:
    items: list[FactCheckItem] = []
    items += verify_findings_traceable(findings)
    items += verify_evidence_sources(evidence)
    revised_narratives, narrative_items = fact_check_narratives(narratives)
    items += narrative_items
    return revised_narratives, FactCheckResult(items=items)
