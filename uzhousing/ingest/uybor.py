"""Bounded collection of public Uybor.uz residential listings.

Uybor serves its listings from a public JSON API that its own website calls.
Unlike OLX it is reachable over plain HTTP, so no browser is needed. The same
rules apply as everywhere else in this package: paced requests, robots.txt is
honoured, access errors are fatal, and no login or access control is bypassed.
"""
from __future__ import annotations

import time

import requests

BASE = "https://uybor.uz"
API = "https://api.uybor.uz/api/v1/listings"

#: Residential categories only, so the figures stay comparable with the OLX
#: apartment and house categories. Land, offices, warehouses and whole
#: businesses are also advertised here and are not housing.
CATEGORY_PROPERTY = {7: "Kvartira", 8: "Hovli", 28: "Hovli"}

#: The API returns at most this many per request whatever is asked for.
MAX_LIMIT = 100

#: ``price`` means different things per listing, and reading it as a total
#: would be wrong by a factor of the floor area.
PRICE_TOTAL, PRICE_PER_SQM = "all", "sqm"

EMBED = "category,subCategory,region,city,district"

USD_CODES = {"usd"}
UZS_CODES = {"uzs"}


class UyborError(RuntimeError):
    pass


def _robots_blocked(session, path: str) -> bool:
    """True when uybor.uz's robots.txt disallows ``path`` for any crawler.

    The API lives on a separate host that publishes no robots.txt, which
    declares no restriction; the website's own rules are still checked so a
    future move of the API under uybor.uz cannot slip past them.
    """
    try:
        response = session.get(BASE + "/robots.txt", timeout=30)
    except requests.RequestException as exc:
        raise UyborError("Uybor robots.txt o'qilmadi") from exc
    if response.status_code != 200:
        return False
    applies = False
    for line in response.text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            applies = value in ("*", "uzhousingresearch")
        elif applies and key == "disallow" and value and path.startswith(value.rstrip("*")):
            return True
    return False


def fetch_page(session, *, page, limit=MAX_LIMIT):
    """Fetch one page. The API paginates by ``page`` and ignores ``offset``
    entirely, so asking by offset silently returns the first page again."""
    if not isinstance(page, int) or page < 1 or not 1 <= limit <= MAX_LIMIT:
        raise ValueError("Invalid Uybor pagination bounds")
    try:
        response = session.get(API, timeout=30, params={
            "mode": "search", "limit": limit, "page": page, "embed": EMBED},
            headers={"User-Agent": "UzHousingResearch/1.0", "Accept": "application/json"})
    except requests.RequestException as exc:
        raise UyborError("Uybor tarmoq so'rovi bajarilmadi") from exc
    if response.status_code in (401, 403, 429):
        raise UyborError(f"Uybor kirishni chekladi (HTTP {response.status_code}). "
                         "Jonli ma'lumot olinmadi.")
    if response.status_code != 200:
        raise UyborError(f"Uybor HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise UyborError("Uybor JSON javobi noto'g'ri") from exc
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise UyborError("Uybor JSON tuzilishi o'zgargan")
    return body


def housing_fields(listing: dict) -> dict | None:
    """Keep the housing fields of a residential listing, or ``None``.

    Seller identity, contact details, photos and promotion flags are dropped
    here rather than filtered later, so they never reach a snapshot.
    """
    category = (listing.get("category") or {}).get("id")
    property_type = CATEGORY_PROPERTY.get(category)
    if property_type is None or listing.get("operationType") not in ("sale", "rent"):
        return None
    name = lambda node: ((node or {}).get("name") or {}).get("uz") or None
    return {
        "id": listing.get("id"),
        "operationType": listing.get("operationType"),
        "property": property_type,
        "price": listing.get("price"),
        "priceCurrency": listing.get("priceCurrency"),
        "priceType": listing.get("priceType"),
        "pricePeriodUnit": listing.get("pricePeriodUnit"),
        "square": listing.get("square"),
        "room": listing.get("room"),
        "floor": listing.get("floor"),
        "floorTotal": listing.get("floorTotal"),
        "isNewBuilding": listing.get("isNewBuilding"),
        "region": name(listing.get("region")),
        "city": name(listing.get("city")),
        "district": name(listing.get("district")),
        "createdAt": listing.get("createdAt"),
    }


def collect(*, pages=5, session=None, pause=time.sleep, progress=print):
    """Return residential listings and a coverage record.

    ``pages`` is counted in requests of up to 100 listings, matching the API's
    own ceiling, so it is not the same unit as the OLX page count.
    """
    if not isinstance(pages, int) or not 1 <= pages <= 200:
        raise ValueError("Uybor pages must be between 1 and 200")
    owns = session is None
    session = session or requests.Session()
    try:
        if _robots_blocked(session, "/api/v1/listings"):
            raise UyborError("Uybor robots.txt ushbu manzilni yig'ishga ruxsat bermaydi")
        records, seen, total, stop = [], set(), None, "page_limit"
        for page in range(1, pages + 1):
            if page > 1:
                pause(2)
            body = fetch_page(session, page=page)
            total = body.get("total")
            results = body["results"]
            # Counted before anything is added: asking "are any of these new?"
            # after inserting them would always answer no.
            fresh = sum(1 for item in results if item.get("id") not in seen)
            for listing in results:
                identifier = listing.get("id")
                if identifier in seen:
                    continue
                seen.add(identifier)
                row = housing_fields(listing)
                if row is not None:
                    records.append(row)
            progress(f"Uybor: {len(records)} e'lon")
            if not fresh:
                stop = "repeated_page"
                break
            if len(results) < MAX_LIMIT:
                stop = "exhausted"
                break
        coverage = {"source": BASE, "requests": page, "listings": len(records),
                    "available": total, "stop": stop}
        return records, coverage
    finally:
        if owns:
            session.close()
