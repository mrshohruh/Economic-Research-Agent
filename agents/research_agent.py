"""
Stage 7-8: Research Agent. Executes the research plan's search queries,
collects structured evidence, builds the policy timeline, and links
evidence back to hypotheses for each finding.
"""
from __future__ import annotations

import logging

from models.schemas import Finding, Hypothesis, PolicyEvent, ResearchEvidence, ResearchPlan
from research.evidence import collect_evidence_for_query
from research.policy_tracker import extract_policy_events

logger = logging.getLogger(__name__)


def run_research(plan: ResearchPlan, research_current_news: bool = True,
                  research_historical_policy: bool = True) -> tuple[list[ResearchEvidence], list[PolicyEvent]]:
    all_evidence: list[ResearchEvidence] = []
    counter = 1
    if not research_current_news and not research_historical_policy:
        return [], []

    for q in plan.search_queries:
        batch = collect_evidence_for_query(q, claim_hint=f"Context relevant to: {q}",
                                            id_prefix="E", start_id=counter)
        counter += len(batch) + 1
        all_evidence.extend(batch)

    policy_events = extract_policy_events(all_evidence)
    if not all_evidence:
        logger.info("No web evidence collected (search unavailable or no results). "
                     "Report will explicitly note the evidence gap.")
    return all_evidence, policy_events


def _keyword_overlap(a: str, b: str) -> int:
    aw = set(w.lower() for w in a.split() if len(w) > 3)
    bw = set(w.lower() for w in b.split() if len(w) > 3)
    return len(aw & bw)


def link_evidence_to_hypotheses(hypotheses: list[Hypothesis], evidence: list[ResearchEvidence]) -> list[Hypothesis]:
    """Heuristic keyword linking (works without an LLM). If an LLM is
    available, agents/economist_agent.py refines confidence levels further."""
    for h in hypotheses:
        scored = sorted(evidence, key=lambda e: _keyword_overlap(h.label + " " + h.description,
                                                                   e.source_title + " " + e.snippet), reverse=True)
        supporting = [e.id for e in scored if _keyword_overlap(h.label + " " + h.description,
                                                                 e.source_title + " " + e.snippet) > 0][:3]
        h.supporting_evidence_ids = supporting
        if supporting:
            avg_quality = sum(next(e.source_quality for e in evidence if e.id == eid) for eid in supporting) / len(supporting)
            h.confidence = "high" if avg_quality >= 0.85 else "medium"
        else:
            h.confidence = "low"
    return hypotheses
