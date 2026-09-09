"""Ranks search results by source quality, prioritizing official/primary
sources (Central Bank, statistics agency, government, IMF/WB/ADB/EBRD,
major financial press) over lower-quality sources."""
from __future__ import annotations

from urllib.parse import urlparse

from config.settings import PRIORITY_SOURCES

TIER1 = {"cbu.uz", "stat.uz", "gov.uz", "mineconomy.uz", "lex.uz"}
TIER2 = {"imf.org", "worldbank.org", "adb.org", "ebrd.com"}
TIER3 = {"reuters.com", "bloomberg.com", "ft.com"}


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def source_quality(url: str) -> float:
    domain = domain_of(url)
    if any(domain.endswith(d) for d in TIER1):
        return 0.97
    if any(domain.endswith(d) for d in TIER2):
        return 0.92
    if any(domain.endswith(d) for d in TIER3):
        return 0.85
    if any(domain.endswith(d) for d in PRIORITY_SOURCES):
        return 0.8
    if domain.endswith(".gov") or domain.endswith(".int"):
        return 0.9
    return 0.55


def institution_of(url: str) -> str:
    domain = domain_of(url)
    mapping = {
        "cbu.uz": "Central Bank of Uzbekistan", "stat.uz": "Statistics Agency of Uzbekistan",
        "gov.uz": "Government of Uzbekistan", "mineconomy.uz": "Ministry of Economy and Finance",
        "lex.uz": "National Database of Legislation (LexUZ)",
        "imf.org": "International Monetary Fund", "worldbank.org": "World Bank",
        "adb.org": "Asian Development Bank", "ebrd.com": "EBRD",
        "reuters.com": "Reuters", "bloomberg.com": "Bloomberg", "ft.com": "Financial Times",
    }
    for suffix, name in mapping.items():
        if domain.endswith(suffix):
            return name
    return domain or "Unknown source"
