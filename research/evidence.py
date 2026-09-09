"""Converts raw search results into structured, traceable ResearchEvidence.
Never fabricates URLs, dates, or quotations -- only wraps what the search
provider actually returned."""
from __future__ import annotations

from research.search import SearchResult, search
from research.source_ranker import institution_of, source_quality
from models.schemas import ResearchEvidence


def collect_evidence_for_query(query: str, claim_hint: str, id_prefix: str,
                                start_id: int, max_results: int = 5) -> list[ResearchEvidence]:
    results = search(query, max_results=max_results)
    evidence: list[ResearchEvidence] = []
    for i, r in enumerate(results):
        q = source_quality(r.url)
        confidence = "high" if q >= 0.9 else ("medium" if q >= 0.7 else "low")
        evidence.append(ResearchEvidence(
            id=f"{id_prefix}{start_id + i}",
            claim=claim_hint,
            source_title=r.title or r.url,
            institution=institution_of(r.url),
            url=r.url,
            publication_date=r.published,
            snippet=r.snippet,
            relevance=0.6,
            source_quality=round(q, 2),
            confidence=confidence,
            query_used=query,
        ))
    return evidence
