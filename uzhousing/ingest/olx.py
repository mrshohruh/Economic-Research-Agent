"""Bounded collection of public OLX property pages, with dated snapshots.

The website's embedded listing data is not a supported public developer API.
Access errors are fatal; no login, CAPTCHA or access-control bypass is attempted.
"""
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

from .olx_client import BASE, OLXClient, CollectionError, OfferUnavailable
from .olx_store import save_observations
CATEGORIES = {
    "rent_apartment": "/nedvizhimost/kvartiry/arenda-dolgosrochnaya/",
    "sale_apartment": "/nedvizhimost/kvartiry/prodazha/",
    "rent_house": "/nedvizhimost/doma/arenda-dolgosrochnaya/",
    "sale_house": "/nedvizhimost/doma/prodazha/",
}

def extract_offers(html: str) -> list[dict]:
    """Read JSON hydration objects; never execute JavaScript from the site."""
    found = {}
    def walk(value):
        if isinstance(value, dict):
            if value.get("id") is not None and isinstance(value.get("params"), list) and "location" in value:
                found[str(value["id"])] = value
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for script in BeautifulSoup(html, "html.parser").find_all("script"):
        source = script.string or script.get_text()
        candidates = [source.strip()]
        for match in re.finditer(r'JSON\.parse\(\s*("(?:\\.|[^"\\])*")\s*\)', source):
            try:
                candidates.append(json.loads(match.group(1)))
            except ValueError:
                pass
        assignment = re.search(r'(?:window\.)?__PRERENDERED_STATE__\s*=\s*(\{.*\})\s*;?\s*$', source, re.S)
        if assignment:
            candidates.append(assignment.group(1))
        encoded = re.search(r'__PRERENDERED_STATE__\s*=\s*("(?:\\.|[^"\\])*")', source)
        if encoded:
            try:
                candidates.append(json.loads(encoded.group(1)))
            except ValueError:
                pass
        for candidate in candidates:
            try:
                walk(json.loads(candidate))
            except (ValueError, TypeError):
                continue
    return list(found.values())


def collect(output_dir: Path, *, pages: int = 5, session=None, browser: bool = False,
            headless: bool = True, pause=time.sleep, progress=print,
            uybor_pages: int = 0) -> Path:
    """Collect the bounded category pages.

    ``browser=True`` routes every request through a real browser instead of a
    plain HTTP session, which is the only transport the site's CDN firewall
    currently serves. Nothing else about collection changes: the same robots.txt
    rules, pacing, schema checks and snapshot format apply either way.
    """
    if not 1 <= pages <= 100:
        raise ValueError("OLX pages must be between 1 and 100 per category")
    owned = None
    if session is None and browser:
        from .olx_browser import BrowserSession
        progress("Brauzer ishga tushirilmoqda...")
        session = owned = BrowserSession(headless=headless)
    client = OLXClient(session, pause=pause)
    try:
        return _collect(output_dir, pages=pages, client=client, progress=progress,
                        uybor_pages=uybor_pages, pause=pause)
    finally:
        client.close()
        if owned is not None:
            owned.close()


def discover_ids(html: str) -> list[str]:
    """Numeric IDs explicitly attached to listing cards; never decode URL slugs."""
    ids = []
    for card in BeautifulSoup(html, "html.parser").select('[data-cy="l-card"]'):
        value = str(card.get("data-id") or card.get("id") or "")
        if value.isascii() and value.isdecimal() and int(value) > 0:
            value = str(int(value))
            if value not in ids:
                ids.append(value)
    return ids


HOUSING_FIELDS = ("id", "url", "title", "params", "location", "category",
                  "created_time", "last_refresh_time")


def _housing_fields(offer, category):
    """Retain only housing fields, not seller contacts or photos."""
    return {key: offer.get(key) for key in HOUSING_FIELDS} | {"collection_category": category}


def _verified_category_id(client, card_ids, *, agreeing=2, sample=3):
    """Read the numeric category from listings the category page actually shows.

    The ID is never guessed: it is taken from real offers and only accepted when
    independent listings agree on it, so a mixed or promoted card cannot decide
    which category is collected. ``None`` means no ID was established and the
    caller must fall back to reading listing pages.
    """
    votes, unavailable, fetched = [], 0, {}
    for offer_id in card_ids[:sample]:
        try:
            offer = client.get_offer(offer_id)
        except OfferUnavailable:
            unavailable += 1
            continue
        fetched[offer_id] = offer
        category = offer.get("category")
        if isinstance(category, dict) and isinstance(category.get("id"), int):
            votes.append(category["id"])
    if len(votes) >= agreeing and len(set(votes)) == 1:
        return votes[0], unavailable, fetched
    return None, unavailable, fetched


def _collect(output_dir, *, pages, client, progress, uybor_pages=0, pause=time.sleep):
    stamp = datetime.now(timezone.utc)
    records, coverage = [], []
    # Check current robots rules, including wildcard paths used by OLX.
    robots = client.get("/robots.txt").text
    blocked = []
    applies = False
    for line in robots.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "user-agent":
            applies = value.strip().lower() in ("*", "uzhousingresearch")
        elif applies and key.strip().lower() == "disallow" and value.strip():
            blocked.append(value.strip())
    def check_path(path):
        if any(re.search("^" + re.escape(rule).replace(r"\*", ".*").replace(r"\$", "$"), path) for rule in blocked):
            raise CollectionError("OLX robots.txt ushbu sahifani avtomatik yig'ishga ruxsat bermaydi: " + path)
    client.check_path = check_path
    for category, path in CATEGORIES.items():
        seen = set()
        repeated = False
        unavailable = 0
        fetched = set()
        page = 1
        # The category page no longer embeds its listings, so every listing
        # would otherwise cost its own detail request. Where the category's own
        # ID can be verified from the listings it shows, the paged offers
        # endpoint returns the same records about forty at a time instead.
        progress(f"OLX: {category}, toifa aniqlanmoqda")
        first_html = client.get(path, params={"page": 1}).text
        category_id, unavailable, prefetched = _verified_category_id(client, discover_ids(first_html))
        if category_id is not None:
            strategy = f"category_api:{category_id}"
            for offer in client.iter_offers(category_id=category_id, max_pages=pages, limit=40):
                key = str(offer["id"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(_housing_fields(offer, category))
                progress(f"OLX: {category}, {len(seen)} e'lon")
            if not seen:
                raise CollectionError(
                    f"OLX toifasidan e'lonlar qaytmadi: {category} (category_id={category_id}). "
                    "Hisobot yaratilmagan.")
            coverage.append({"category": category, "pages_requested": pages, "listings": len(seen),
                             "unavailable": unavailable, "stop": client.last_stop, "strategy": strategy})
            continue
        strategy = "listing_pages"
        for page in range(1, pages + 1):
            url = BASE + path
            progress(f"OLX: {category}, {page}-sahifa")
            html = first_html if page == 1 else client.get(path, params={"page": page}).text
            offers = extract_offers(html)
            card_ids = discover_ids(html)
            embedded_ids = {str(offer["id"]) for offer in offers}
            for offer_id in card_ids:
                if offer_id in embedded_ids or offer_id in fetched:
                    continue
                fetched.add(offer_id)
                if offer_id in prefetched:  # already read while verifying the category
                    offers.append(prefetched[offer_id])
                    continue
                try:
                    offers.append(client.get_offer(offer_id))
                except OfferUnavailable:
                    unavailable += 1
            if not offers and card_ids and all(i in seen for i in card_ids):
                repeated = True
                break
            if not offers:
                raise CollectionError(f"OLX sahifasidan tuzilmali e'lonlar topilmadi: {url}, page={page}. Sahifa tuzilishi o'zgargan yoki kirish cheklangan bo'lishi mumkin. Hisobot yaratilmagan.")
            fresh = [offer for offer in offers if str(offer["id"]) not in seen]
            if not fresh:
                repeated = True
                break
            for offer in fresh:
                seen.add(str(offer["id"]))
                records.append(_housing_fields(offer, category))
        coverage.append({"category": category, "pages_requested": page, "listings": len(seen),
                         "unavailable": unavailable, "stop": "repeated_page" if repeated else "page_limit",
                         "strategy": strategy})
    # Fetch the official dated conversion rate; never silently use a stale default.
    fx = client.session.get("https://cbu.uz/uz/arkhiv-kursov-valyut/json/USD/", timeout=30)
    fx.raise_for_status()
    rate = fx.json()[0]
    if float(rate["Rate"]) <= 0:
        raise CollectionError("Markaziy bank valyuta kursi noto'g'ri")
    payload = {"source": BASE, "collected_at": stamp.isoformat(), "coverage": coverage,
               "fx": {"rate": float(rate["Rate"]), "date": rate["Date"], "source": "https://cbu.uz/uz/arkhiv-kursov-valyut/"}, "data": records}
    if uybor_pages:
        # A second marketplace, collected over plain HTTP; its listings are
        # kept in their own key so each source stays identifiable.
        from .uybor import collect as collect_uybor
        uybor_records, uybor_coverage = collect_uybor(
            pages=uybor_pages, pause=pause, progress=progress)
        payload["uybor"] = uybor_records
        payload["coverage"] = coverage + [uybor_coverage]
        payload["sources"] = [BASE, "https://uybor.uz"]
    folder = Path(output_dir) / "olx_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    save_observations(Path(output_dir) / "olx_history.sqlite", payload)
    return target
