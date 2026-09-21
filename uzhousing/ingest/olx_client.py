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
#: so roughly MAX_OFFSET + one page is all a category will ever return.
MAX_OFFSET = 1000

class OLXClient:
    def __init__(self, session=None, *, pause=time.sleep, check_path=None):
        self.session = session if session is not None else requests.Session()
        self.owns_session = session is None
        self.pause = pause
        self.check_path = check_path or (lambda path: None)
        self.requested = False
        self.last_stop = "page_limit"

    def close(self):
        if self.owns_session:
            self.session.close()

    def get(self, path, *, params=None, detail=False, paged=False):
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

    def get_offers(self, *, category_id, offset=0, limit=40):
        """Explicit verified category only; never make an unfiltered marketplace crawl."""
        if not isinstance(category_id, int) or isinstance(category_id, bool) or category_id <= 0:
            raise ValueError("A verified positive category_id is required")
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 40:
            raise ValueError("Invalid pagination bounds")
        body = self._json(self.get("/api/v1/offers/", paged=True,
            params={"category_id": category_id, "offset": offset, "limit": limit}))
        offers = body.get("data")
        if not isinstance(offers, list) or any(not isinstance(o, dict) or o.get("id") is None or not isinstance(o.get("params"), list) or not isinstance(o.get("location"), dict) for o in offers):
            raise CollectionError("OLX ro'yxati tuzilishi o'zgargan")
        return offers

    def iter_offers(self, *, category_id, max_pages=5, limit=40):
        if not isinstance(max_pages, int) or not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        seen = set()
        self.last_stop = "page_limit"
        for page in range(max_pages):
            offset = page * limit
            if offset > MAX_OFFSET:
                # Asking anyway is a guaranteed HTTP 400; the pages already
                # collected are complete and usable.
                self.last_stop = "depth_limit"
                break
            try:
                offers = self.get_offers(category_id=category_id, offset=offset, limit=limit)
            except PaginationLimit:
                # The cap moved; stop here rather than lose the whole run.
                self.last_stop = "depth_limit"
                break
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
