"""
Pluggable web-search interface. Default provider is DuckDuckGo (no API key
required). The system is never hard-coded to one provider: swap
SEARCH_PROVIDER in .env or extend `_search_duckduckgo` with another backend.

If a search fails (no network, provider error, or SEARCH_PROVIDER=none) the
function returns an empty list and callers MUST record that evidence could
not be collected, rather than fabricating results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str | None = None


def search(query: str, max_results: int | None = None) -> list[SearchResult]:
    max_results = max_results or settings.MAX_SEARCH_RESULTS_PER_QUERY
    if settings.SEARCH_PROVIDER == "none":
        return []
    if settings.SEARCH_PROVIDER == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    logger.warning("Unknown SEARCH_PROVIDER=%s; no results returned.", settings.SEARCH_PROVIDER)
    return []


def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    try:
        from duckduckgo_search import DDGS
    except Exception as exc:
        logger.warning("duckduckgo_search not available: %s", exc)
        return []
    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", "").strip(),
                    url=r.get("href", "").strip(),
                    snippet=r.get("body", "").strip(),
                ))
    except Exception as exc:
        logger.warning("Web search failed for query %r: %s", query, exc)
        return []
    return [r for r in results if r.url]
