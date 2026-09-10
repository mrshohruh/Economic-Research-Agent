"""Web search and page reading for the policy / macro research step.

Uses DuckDuckGo through the ``ddgs`` package, which needs no API key. Every
function degrades to an empty result rather than raising, so the pipeline keeps
running when the machine is offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# The search backends log every engine attempt at INFO, including the ones that
# fail and get retried. That is our implementation detail, not the user's problem.
for _noisy in ("ddgs", "ddgs.ddgs", "primp", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "research"
CACHE_TTL_SECONDS = 60 * 60 * 24 * 3  # three days

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Domains whose housing / macro coverage of Uzbekistan is worth ranking up.
TRUSTED_DOMAINS = {
    "stat.uz": 1.0,
    "cbu.uz": 1.0,
    "lex.uz": 1.0,
    "gov.uz": 0.9,
    "president.uz": 0.9,
    "mineconomy.uz": 0.9,
    "imf.org": 0.9,
    "worldbank.org": 0.9,
    "adb.org": 0.85,
    "ebrd.com": 0.85,
    "un.org": 0.8,
    "oecd.org": 0.8,
    "gazeta.uz": 0.7,
    "kun.uz": 0.65,
    "spot.uz": 0.65,
    "review.uz": 0.6,
    "uza.uz": 0.6,
    "podrobno.uz": 0.55,
    "reuters.com": 0.7,
    "fitchratings.com": 0.7,
    "moodys.com": 0.7,
    "trend.az": 0.5,
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    query: str = ""
    domain: str = ""
    score: float = 0.0
    body: str = ""  # filled in only if the page was fetched

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["body"] = self.body[:400]
        return data

    @property
    def citation(self) -> str:
        return f"{self.title} — {self.domain}"


# ---------------------------------------------------------------------------
def search(query: str, max_results: int = 6, region: str = "wt-wt") -> list[SearchHit]:
    """Run one search query. Returns [] on any failure."""
    cached = _cache_get("search", query + region + str(max_results))
    if cached is not None:
        return [SearchHit(**hit) for hit in cached]

    hits: list[SearchHit] = []
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region=region, max_results=max_results))
    except Exception as exc:
        LOGGER.warning("search failed for %r: %s", query, exc)
        return []

    for item in raw:
        url = item.get("href") or item.get("url") or item.get("link") or ""
        title = _clean(item.get("title") or "")
        snippet = _clean(item.get("body") or item.get("snippet") or item.get("description") or "")
        if not url or not title:
            continue
        domain = _domain(url)
        hits.append(
            SearchHit(
                title=title,
                url=url,
                snippet=snippet,
                query=query,
                domain=domain,
                score=_score(domain, title, snippet),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    _cache_put("search", query + region + str(max_results), [asdict(h) for h in hits])
    return hits


def search_many(queries: list[str], per_query: int = 6, pause: float = 0.6) -> list[SearchHit]:
    """Run several queries and de-duplicate the combined results by URL."""
    seen: dict[str, SearchHit] = {}
    for i, query in enumerate(queries):
        for hit in search(query, max_results=per_query):
            existing = seen.get(hit.url)
            if existing is None or hit.score > existing.score:
                seen[hit.url] = hit
        if i < len(queries) - 1:
            time.sleep(pause)  # be polite to the endpoint
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)


def fetch(url: str, max_chars: int = 12000, timeout: int = 20) -> str:
    """Download a page and return its readable text, or '' on failure."""
    cached = _cache_get("page", url)
    if cached is not None:
        return str(cached)[:max_chars]

    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en,ru;q=0.8"},
            timeout=timeout,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()

        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = _clean(main.get_text(" "))
    except Exception as exc:
        LOGGER.info("could not fetch %s: %s", url, exc)
        return ""

    _cache_put("page", url, text[: max_chars * 2])
    return text[:max_chars]


def enrich(hits: list[SearchHit], limit: int = 3, max_chars: int = 8000) -> list[SearchHit]:
    """Download the top `limit` pages so the LLM sees real content, not snippets."""
    for hit in hits[:limit]:
        hit.body = fetch(hit.url, max_chars=max_chars)
    return hits


# ---------------------------------------------------------------------------
def _score(domain: str, title: str, snippet: str) -> float:
    score = 0.3
    for trusted, weight in TRUSTED_DOMAINS.items():
        if domain == trusted or domain.endswith("." + trusted):
            score = max(score, weight)
            break

    text = f"{title} {snippet}".lower()
    for keyword, bonus in (
        ("uzbekistan", 0.25), ("узбекистан", 0.25), ("o'zbekiston", 0.25),
        ("housing", 0.12), ("mortgage", 0.12), ("real estate", 0.1),
        ("жиль", 0.12), ("ипотек", 0.12), ("недвижим", 0.1),
        ("uy-joy", 0.12), ("ipoteka", 0.12),
        ("tashkent", 0.06), ("ташкент", 0.06),
    ):
        if keyword in text:
            score += bonus

    # Recent years mentioned in the title usually mean fresher analysis.
    for year, bonus in (("2026", 0.2), ("2025", 0.15), ("2024", 0.08)):
        if year in title:
            score += bonus
    return round(min(score, 2.0), 3)


def _domain(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url)
    if not match:
        return ""
    return match.group(1).lower().removeprefix("www.")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ---------------------------------------------------------------------------
# Tiny on-disk cache so repeated runs are fast and gentle on the endpoints.
# ---------------------------------------------------------------------------
def _cache_path(kind: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / kind / f"{digest}.json"


def _cache_get(kind: str, key: str) -> Any | None:
    path = _cache_path(kind, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - payload.get("_ts", 0) > CACHE_TTL_SECONDS:
            return None
        return payload.get("data")
    except Exception:
        return None


def _cache_put(kind: str, key: str, data: Any) -> None:
    path = _cache_path(kind, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"_ts": time.time(), "data": data}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("cache write failed: %s", exc)
