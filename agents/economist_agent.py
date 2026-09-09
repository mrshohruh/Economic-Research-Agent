"""
Stage 13 / 18-19: Economist Agent. Applies the appropriate economic
framework to interpret findings, evaluates competing explanations against
collected evidence, and writes the fact/correlation/explanation/causal-claim
language with appropriate hedging. Falls back to deterministic templated
prose when no LLM is configured, so the pipeline still produces a complete,
traceable narrative end-to-end.
"""
from __future__ import annotations

from agents.llm_interface import LLM
from models.schemas import Finding, Hypothesis, ResearchEvidence

FRAMEWORKS = {
    "inflation": ["demand-pull inflation", "cost-push inflation", "exchange-rate pass-through",
                  "monetary transmission", "supply shocks", "expectations", "imported inflation"],
    "labor": ["labor demand", "labor supply", "productivity", "unit labor costs", "wage-price dynamics"],
    "external": ["exchange rates", "terms of trade", "foreign demand", "commodity prices", "capital flows"],
    "fiscal": ["government expenditure", "taxation", "fiscal multipliers", "debt dynamics", "crowding out"],
}


def select_framework(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ("inflation", "price", "cpi")):
        return "inflation"
    if any(k in t for k in ("labor", "labour", "wage", "employment", "unemployment")):
        return "labor"
    if any(k in t for k in ("export", "import", "exchange rate", "trade", "current account", "capital flow")):
        return "external"
    if any(k in t for k in ("fiscal", "budget", "tax", "debt", "government spending")):
        return "fiscal"
    return "inflation"


def evaluate_hypotheses(finding: Finding, hypotheses: list[Hypothesis],
                         evidence: list[ResearchEvidence]) -> str:
    """Builds the internal competing-explanations comparison as markdown."""
    ev_by_id = {e.id: e for e in evidence}
    rows = ["| Hypothesis | Supporting evidence | Confidence |", "|---|---|---|"]
    for h in sorted(hypotheses, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.confidence]):
        support = "; ".join(ev_by_id[eid].institution or ev_by_id[eid].source_title
                             for eid in h.supporting_evidence_ids if eid in ev_by_id) or "No direct evidence found"
        rows.append(f"| {h.label} | {support} | {h.confidence.capitalize()} |")
    return "\n".join(rows)


def _template_finding_narrative(finding: Finding, hypotheses: list[Hypothesis],
                                 evidence: list[ResearchEvidence]) -> str:
    ev_by_id = {e.id: e for e in evidence}
    sentences = [f"The data indicate that {finding.finding.rstrip('.')}."]

    ranked = sorted(hypotheses, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.confidence])
    supported = [h for h in ranked if h.confidence in ("high", "medium") and h.supporting_evidence_ids]

    if supported:
        top = supported[0]
        cites = [ev_by_id[eid] for eid in top.supporting_evidence_ids if eid in ev_by_id]
        if cites:
            cite_txt = "; ".join(f"{c.institution or c.source_title} ({c.publication_date or 'n.d.'})" for c in cites[:2])
            sentences.append(
                f"One possible explanation consistent with the available evidence is {top.label.lower()}: "
                f"{top.description} This is suggested by reporting from {cite_txt}, "
                f"although the available evidence does not establish a definitive causal relationship."
            )
        else:
            sentences.append(f"A plausible contributing factor is {top.label.lower()}, though direct supporting "
                              f"evidence was not identified in this research pass.")
    else:
        sentences.append("Available evidence is insufficient to support a strong conclusion about the underlying "
                          "cause of this pattern; several competing explanations remain plausible.")

    others = [h for h in ranked if h != (supported[0] if supported else None)][:3]
    if others:
        other_labels = ", ".join(h.label.lower() for h in others)
        sentences.append(f"Other candidate explanations considered include {other_labels}, "
                          f"which cannot be ruled out on the basis of currently available evidence.")

    return " ".join(sentences)


def write_finding_narrative(finding: Finding, hypotheses: list[Hypothesis],
                             evidence: list[ResearchEvidence], topic: str) -> str:
    if LLM.enabled:
        framework = select_framework(topic)
        concepts = ", ".join(FRAMEWORKS[framework])
        ev_desc = "\n".join(
            f"- {e.institution or e.source_title}: {e.snippet[:220]}" for e in evidence
            if e.id in sum((h.supporting_evidence_ids for h in hypotheses), [])
        )[:3000]
        system = (
            "You are a professional economist writing a research report section. Use precise, cautious "
            "language. Distinguish FACT (directly from data), CORRELATION (co-movement), POSSIBLE EXPLANATION "
            "(hedged: 'may have contributed', 'is consistent with', 'suggests'), and CAUSAL CLAIM (only if very "
            "strongly evidenced). Never invent sources, dates, or numbers beyond what is given. Write 2-4 "
            f"sentences, using relevant concepts from: {concepts}."
        )
        user = (f"Finding: {finding.finding}\nMagnitude: {finding.magnitude}\nPeriod: {finding.period}\n"
                f"Candidate hypotheses: {[h.label for h in hypotheses]}\nEvidence:\n{ev_desc}")
        text = LLM.complete(system, user, max_tokens=500)
        if text:
            return text
    return _template_finding_narrative(finding, hypotheses, evidence)


def write_visual_analysis(title: str, description: str, key_numbers: str, topic: str) -> str:
    """Stage 12: analytical discussion for a table or figure, grounded in
    the real numbers computed by Python (passed in as `key_numbers`)."""
    if LLM.enabled:
        system = ("You are a professional economist writing the analytical commentary that accompanies a table "
                   "or figure in a research report. Reference the actual numbers given. Explain what changed, "
                   "the magnitude, timing, likely drivers, whether the movement is unusual, and note limitations. "
                   "Use cautious, non-causal language unless evidence is strong. 3-5 sentences.")
        user = f"Item: {title}\nWhat it shows: {description}\nComputed figures: {key_numbers}\nTopic: {topic}"
        text = LLM.complete(system, user, max_tokens=400)
        if text:
            return text
    return (f"{title} shows {description}. Based on the underlying data, {key_numbers} "
            f"The movement should be read alongside other findings in this report rather than in isolation, "
            f"since a single indicator rarely identifies a unique cause.")
