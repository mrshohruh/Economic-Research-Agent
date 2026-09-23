"""Public OLX transport. Undocumented endpoints may change or deny access."""
from __future__ import annotations
import time
import requests

BASE = "https://www.olx.uz"

class CollectionError(RuntimeError):
    pass

class OfferUnavailable(CollectionError):
    """A discovered listing returned 404/410; this does not prove a sale."""

class PaginationLimit(CollectionError):
    """The marketplace refused a deeper page. Pages already read stay valid."""

#: OLX validates the offset and rejects anything beyond this with HTTP 400,
#: so roughly MAX_OFFSET + one page is all a single query will ever return.
#: This is a limit on one query, not on the category: narrower queries each get
#: their own window, which is what :mod:`uzhousing.ingest.olx_partition` uses.
MAX_OFFSET = 1000

#: The listing endpoint refuses a larger page with HTTP 400.
MAX_LIMIT = 50

#: How many listings one query can actually hand over.
REACHABLE = MAX_OFFSET + MAX_LIMIT

#: Filters a query may carry beside the category. Anything else is refused
#: rather than passed on, so a partition cannot widen what is collected.
ALLOWED_FILTERS = ("region_id", "city_id", "district_id",
                   "filter_float_price:from", "filter_float_price:to")

class OLXClient:
    def __init__(self, session=None, *, pause=time.sleep, check_path=None):
        self.session = session if session is not None else requests.Session()
        self.owns_session = session is None
        self.pause = pause
        self.check_path = check_path or (lambda path: None)
        self.requested = False
        self.last_stop = "page_limit"
        self.last_pages = 0

    def close(self):
        if self.owns_session:
            self.session.close()

    def get(self, path, *, params=None, detail=False, paged=False, absent_ok=False):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Only relative OLX paths are accepted")
        self.check_path(path)
        if self.requested:
            self.pause(2)
        self.requested = True
        try:
            response = self.session.get(BASE + path, params=params, timeout=30,
                headers={"User-Agent": "UzHousingResearch/1.0", "Accept": "application/json" if path.startswith("/api/") else "text/html,text/plain", "Accept-Language": "uz-UZ,uz;q=0.9,ru;q=0.8"})
        except requests.RequestException as exc:
            raise CollectionError("OLX tarmoq so'rovi bajarilmadi") from exc
        if response.status_code in (401, 403, 429):
            raise CollectionError(f"OLX kirishni chekladi (HTTP {response.status_code}). Jonli ma'lumot olinmadi. Keyinroq urinib ko'ring yoki OLX bilan ma'lumot olish bo'yicha bog'laning.")
        if detail and response.status_code in (404, 410):
            raise OfferUnavailable(f"OLX e'loni mavjud emas (HTTP {response.status_code})")
        if absent_ok and response.status_code == 404:
            # "This has no sub-list", which is an answer, not a failure. A
            # refusal is still a refusal: 401/403/429 were raised above.
            return None
        if paged and response.status_code == 400:
            raise PaginationLimit("OLX bu toifada chuqurroq sahifalashni qabul qilmadi")
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise CollectionError(f"OLX HTTP {response.status_code}: {path}") from exc
        return response

    @staticmethod
    def _json(response):
        try:
            value = response.json()
        except ValueError as exc:
            raise CollectionError("OLX JSON javobi noto'g'ri") from exc
        if not isinstance(value, dict):
            raise CollectionError("OLX JSON tuzilishi o'zgargan")
        return value

    def get_offer(self, offer_id):
        value = str(offer_id)
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise ValueError("A positive numeric listing ID is required")
        body = self._json(self.get(f"/api/v1/offers/{int(value)}/", detail=True))
        offer = body.get("data", body)
        if not isinstance(offer, dict) or str(offer.get("id")) != str(int(value)) or not isinstance(offer.get("params"), list) or not isinstance(offer.get("location"), dict):
            raise CollectionError("OLX e'lon identifikatori yoki tuzilishi mos emas")
        return offer

    @staticmethod
    def _filters(filters):
        """Validate a partition's filters; unknown keys are never sent."""
        if not filters:
            return {}
        unknown = set(filters) - set(ALLOWED_FILTERS)
        if unknown:
            raise ValueError(f"Unsupported OLX filters: {sorted(unknown)}")
        for key, value in filters.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"OLX filter {key} must be a non-negative integer")
        return dict(filters)

    def overflows(self, *, category_id, filters=None):
        """True when this query holds more listings than its window will serve.

        The site's search metadata would answer this as a single number, but
        robots.txt disallows ``*/api/v1/offers/metadata/``, so the question is
        put to the listing endpoint instead: ask for the last page the window
        reaches and see whether it is still full. A full page means listings
        remain out past the window and the query has to be split; a short or
        empty page means the whole query fits and can be read as it stands.

        This costs one request, the same as reading the count did, and it
        never needs a path the marketplace asks crawlers to leave alone.
        """
        try:
            last = self.get_offers(category_id=category_id, offset=MAX_OFFSET,
                                   limit=MAX_LIMIT, filters=filters)
        except PaginationLimit:
            # The window is narrower than MAX_OFFSET assumes. Nothing past it
            # is collectable however finely the query is split, so the walk
            # reads what it can and iter_offers reports the shortfall as
            # "depth_limit" rather than splitting blindly.
            return False
        return len(last) >= MAX_LIMIT

    def get_offers(self, *, category_id, offset=0, limit=MAX_LIMIT, filters=None):
        """Explicit verified category only; never make an unfiltered marketplace crawl."""
        if not isinstance(category_id, int) or isinstance(category_id, bool) or category_id <= 0:
            raise ValueError("A verified positive category_id is required")
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
            raise ValueError("Invalid pagination bounds")
        body = self._json(self.get("/api/v1/offers/", paged=True,
            params={"category_id": category_id, "offset": offset, "limit": limit}
                   | self._filters(filters)))
        offers = body.get("data")
        if not isinstance(offers, list) or any(not isinstance(o, dict) or o.get("id") is None or not isinstance(o.get("params"), list) or not isinstance(o.get("location"), dict) for o in offers):
            raise CollectionError("OLX ro'yxati tuzilishi o'zgargan")
        return offers

    def iter_offers(self, *, category_id, max_pages=None, limit=MAX_LIMIT, filters=None):
        """Read the category page by page.

        ``max_pages=None`` means every page the marketplace will serve: the
        walk ends on its own when a page repeats, comes back short or is
        refused, which is what ``MAX_OFFSET`` guarantees will happen.
        """
        if max_pages is not None and (not isinstance(max_pages, int)
                                      or isinstance(max_pages, bool) or max_pages < 1):
            raise ValueError("max_pages must be a positive integer, or None for every page")
        seen = set()
        self.last_stop = "page_limit"
        self.last_pages = 0
        page = -1
        while max_pages is None or page + 1 < max_pages:
            page += 1
            offset = page * limit
            if offset > MAX_OFFSET:
                # Asking anyway is a guaranteed HTTP 400; the pages already
                # collected are complete and usable.
                self.last_stop = "depth_limit"
                break
            try:
                offers = self.get_offers(category_id=category_id, offset=offset,
                                         limit=limit, filters=filters)
            except PaginationLimit:
                # The cap moved; stop here rather than lose the whole run.
                self.last_stop = "depth_limit"
                break
            self.last_pages = page + 1  # counted once the page really arrived
            added = 0
            for offer in offers:
                key = str(offer["id"])
                if key not in seen:
                    seen.add(key)
                    added += 1
                    yield offer
            if not added or len(offers) < limit:
                self.last_stop = "exhausted"
                break
