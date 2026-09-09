"""Builds a policy timeline from collected evidence. Only creates an entry
when the evidence snippet plausibly describes a policy action; dates are
taken from the evidence, never invented."""
from __future__ import annotations

import re

from agents.llm_interface import LLM
from models.schemas import PolicyEvent, ResearchEvidence

POLICY_KEYWORDS = [
    "policy rate", "interest rate", "central bank", "decree", "tariff", "subsidy",
    "tax", "import", "export", "regulation", "reform", "law", "quota", "duty",
    "exchange rate", "devalu", "monetary policy", "fiscal", "budget",
]


def extract_policy_events(evidence: list[ResearchEvidence]) -> list[PolicyEvent]:
    events: list[PolicyEvent] = []
    for e in evidence:
        text = f"{e.source_title} {e.snippet}".lower()
        if not any(k in text for k in POLICY_KEYWORDS):
            continue
        channel = _infer_channel(text)
        events.append(PolicyEvent(
            date=e.publication_date, institution=e.institution or "Unknown",
            policy=e.source_title, economic_channel=channel, evidence_id=e.id,
        ))
    return events


def _infer_channel(text: str) -> str:
    if any(k in text for k in ("policy rate", "interest rate", "monetary policy", "central bank")):
        return "Monetary transmission"
    if any(k in text for k in ("tariff", "subsidy", "tax", "duty", "energy")):
        return "Cost-push / direct price effect"
    if any(k in text for k in ("import", "export", "quota", "trade")):
        return "Supply / import-price channel"
    if any(k in text for k in ("exchange rate", "devalu")):
        return "Exchange-rate pass-through"
    if any(k in text for k in ("fiscal", "budget")):
        return "Fiscal transmission"
    return "Other / indirect channel"
