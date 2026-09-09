"""Formats the References section from collected ResearchEvidence. Only
real, collected sources are ever listed -- nothing is invented."""
from __future__ import annotations

from models.schemas import ResearchEvidence


def format_reference(e: ResearchEvidence) -> str:
    date = e.publication_date or "n.d."
    inst = f" ({e.institution})" if e.institution else ""
    return f"{e.source_title}{inst}. {date}. {e.url}"


def build_reference_list(evidence: list[ResearchEvidence]) -> list[str]:
    seen, refs = set(), []
    for e in sorted(evidence, key=lambda x: -x.source_quality):
        if e.url in seen:
            continue
        seen.add(e.url)
        refs.append(format_reference(e))
    return refs
