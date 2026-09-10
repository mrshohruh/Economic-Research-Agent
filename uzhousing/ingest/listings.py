"""Flatten property-marketplace exports into one row per listing.

Scraped listing feeds (OLX and the portals built on the same API) arrive as
pages of adverts, each advert carrying its measured attributes inside a
``params`` array rather than as columns::

    {"data": [{"id": 1, "location": {...},
               "params": [{"key": "price", "value": {"value": 550, "currency": "UYE"}},
                          {"key": "total_area", "value": {"key": "64"}}, ...]},
              ...],
     "metadata": {...}}

A generic JSON flattener reads that file as one row *per page*, with every
advert buried in a single unhashable cell — no price column, no region column,
and therefore no housing statistics at all. This module recognises the shape and
lifts the adverts out into a real table: price, currency, region, city, floor
area, rooms and the derived price per square metre.

Everything here is arithmetic on the source file. No estimate, imputation or
model output is introduced: rows that cannot be read are dropped and counted,
never guessed at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# "у.е." (conventional units) is how Uzbek property is advertised in dollars.
# The marketplace reports it as the currency code UYE, at parity with the USD.
USD_CODES = {"UYE", "USD", "U.E.", "УЕ", "У.Е."}
UZS_CODES = {"UZS", "SUM", "СУМ", "СЎМ"}

# Fallback only. The real rate is passed in from settings so a run can be
# repeated against the rate that applied on the collection date.
DEFAULT_UZS_PER_USD = 12_650.0

# Sanity bounds. These reject data-entry noise (a 100,000 m² apartment, a rent
# of one dollar), not unusual-but-real listings, and every rejection is counted.
AREA_BOUNDS = (10.0, 1_000.0)
RENT_USD_BOUNDS = (10.0, 50_000.0)
SALE_USD_BOUNDS = (1_000.0, 5_000_000.0)

# params[].key -> the column it becomes. Anything not listed is ignored.
NUMERIC_PARAMS = {
    "total_area": "total_area",
    "total_living_area": "living_area",
    "kitchen_area": "kitchen_area",
    "number_of_rooms": "rooms",
    "floor": "floor",
    "total_floors": "total_floors",
    "ceiling_height": "ceiling_height",
}

LABEL_PARAMS = {
    "house_type": "building_type",
    "repairs": "condition",
    "furnished": "furnished",
    "layout": "layout",
    "wc": "bathroom",
    "comission": "agent_commission",
    "year_of_construction_rent": "year_built",
    "year_of_construction": "year_built",
}

# A rental feed names its own parameters. Sale feeds do not carry these.
RENT_MARKERS = ("year_of_construction_rent", "comission", "rent")

# The only columns that mean anything as a time series. Floor area, ceiling
# height and the rest are attributes of the dwellings that happen to be
# advertised: their month-to-month movement measures a change in the *mix* of
# adverts, not a change in the market, and reporting "ceiling height, -1.8% year
# on year" as an economic indicator would be meaningless.
TIME_SERIES_METRICS = ("price_usd", "price_per_sqm_usd")


@dataclass
class ListingTable:
    frame: pd.DataFrame
    listing_type: str = "unknown"
    uzs_per_usd: float = DEFAULT_UZS_PER_USD
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
def looks_like_listings(records: Iterable[Any]) -> bool:
    """True when these JSON records are pages of marketplace adverts."""
    for advert in _iter_adverts(records, limit=40):
        if isinstance(advert.get("params"), list) and (
            "location" in advert or "category" in advert
        ):
            return True
    return False


def flatten(
    records: Iterable[Any],
    uzs_per_usd: float = DEFAULT_UZS_PER_USD,
) -> ListingTable | None:
    """Turn listing pages into one tidy row per advert, or ``None`` if not that shape."""
    adverts = list(_iter_adverts(records))
    if not adverts:
        return None

    rows = [_advert_row(advert) for advert in adverts]
    rows = [row for row in rows if row]
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    notes: list[str] = [f"Read {len(frame):,} adverts out of the nested listing feed."]

    # One advert can be returned on several pages of the same crawl.
    if "listing_id" in frame.columns:
        before = len(frame)
        frame = frame.drop_duplicates(subset="listing_id", keep="first")
        if before > len(frame):
            notes.append(
                f"Removed {before - len(frame):,} duplicate advert(s) that the crawl "
                "returned on more than one page."
            )

    listing_type = _detect_listing_type(adverts, frame)
    frame, cleaning_notes = _derive_and_clean(frame, listing_type, uzs_per_usd)
    notes.extend(cleaning_notes)

    if frame.empty:
        return None
    return ListingTable(
        frame=frame.reset_index(drop=True),
        listing_type=listing_type,
        uzs_per_usd=uzs_per_usd,
        notes=notes,
    )


# ---------------------------------------------------------------------------
def _iter_adverts(records: Iterable[Any], limit: int | None = None) -> list[dict[str, Any]]:
    """Yield individual adverts whether they are wrapped in pages or not."""
    out: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        payload = record.get("data")
        if isinstance(payload, list):
            out.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(record.get("params"), list):
            out.append(record)
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def _advert_row(advert: dict[str, Any]) -> dict[str, Any] | None:
    location = advert.get("location") or {}
    row: dict[str, Any] = {
        "listing_id": advert.get("id"),
        "title": advert.get("title"),
        "url": advert.get("url"),
        "date": _clean_text(advert.get("created_time")),
        "region": _place_name(location.get("region")),
        "city": _place_name(location.get("city")),
        "seller": "business" if advert.get("business") else "private",
    }

    for param in advert.get("params") or []:
        if not isinstance(param, dict):
            continue
        key = str(param.get("key", ""))
        value = param.get("value")
        if not isinstance(value, dict):
            continue

        if key == "price":
            row["price"] = _to_float(value.get("value"))
            row["currency"] = str(value.get("currency") or "").strip().upper()
            row["negotiable"] = bool(value.get("negotiable"))
        elif key in NUMERIC_PARAMS:
            row[NUMERIC_PARAMS[key]] = _to_float(value.get("key", value.get("value")))
        elif key in LABEL_PARAMS:
            row[LABEL_PARAMS[key]] = _clean_text(value.get("label"))

    return row if row.get("price") is not None else None


def _place_name(place: Any) -> str | None:
    """Prefer the display name; fall back to the ASCII slug the feed also carries."""
    if not isinstance(place, dict):
        return None
    name = _clean_text(place.get("name"))
    if name:
        return name
    slug = _clean_text(place.get("normalized_name"))
    return slug.replace("-", " ").title() if slug else None


# ---------------------------------------------------------------------------
def _detect_listing_type(adverts: list[dict[str, Any]], frame: pd.DataFrame) -> str:
    """Rent or sale — the single most consequential fact about a price column.

    Reading a rental feed as a sales feed turns a $550 monthly rent into a
    $550 apartment, so this is decided from the feed's own parameter names
    first and only falls back to price magnitude.
    """
    keys = {
        str(param.get("key", "")).lower()
        for advert in adverts[:200]
        for param in advert.get("params") or []
        if isinstance(param, dict)
    }
    if any(marker in key for key in keys for marker in RENT_MARKERS):
        return "rent"

    text = " ".join(str(t).lower() for t in frame.get("title", pd.Series(dtype=str)).head(400))
    if re.search(r"\b(ijara|arenda|аренд|rent|kiraga|ijaraga)\b", text):
        return "rent"

    # A monthly rent and a purchase price are three orders of magnitude apart.
    usd = frame.loc[frame.get("currency", pd.Series(dtype=str)).isin(USD_CODES), "price"]
    if len(usd) >= 30 and float(usd.median()) < 5_000:
        return "rent"
    return "sale"


def _derive_and_clean(
    frame: pd.DataFrame, listing_type: str, uzs_per_usd: float
) -> tuple[pd.DataFrame, list[str]]:
    """Put every advert on one currency, then drop the physically impossible."""
    notes: list[str] = []
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()

    currency = frame.get("currency", pd.Series("", index=frame.index)).fillna("")
    price = pd.to_numeric(frame["price"], errors="coerce")

    is_usd = currency.isin(USD_CODES)
    is_uzs = currency.isin(UZS_CODES)
    frame["price_usd"] = np.where(
        is_usd, price, np.where(is_uzs, price / uzs_per_usd, np.nan)
    )
    frame["price_uzs"] = np.where(
        is_uzs, price, np.where(is_usd, price * uzs_per_usd, np.nan)
    )
    unknown = int((~is_usd & ~is_uzs & price.notna()).sum())
    if unknown:
        notes.append(
            f"{unknown:,} advert(s) quoted a currency other than som or у.е. and were "
            "excluded from the price statistics."
        )
    notes.append(
        f"Prices quoted in som were converted at {uzs_per_usd:,.0f} UZS/USD; "
        f"{int(is_usd.sum()):,} of {len(frame):,} adverts were already quoted in "
        f"у.е. (dollar-linked conventional units) and {int(is_uzs.sum()):,} in som."
    )

    area = pd.to_numeric(frame.get("total_area"), errors="coerce")
    bad_area = int(((area < AREA_BOUNDS[0]) | (area > AREA_BOUNDS[1])).sum())
    area = area.where((area >= AREA_BOUNDS[0]) & (area <= AREA_BOUNDS[1]))
    frame["total_area"] = area
    if bad_area:
        notes.append(
            f"{bad_area:,} advert(s) recorded a floor area outside "
            f"{AREA_BOUNDS[0]:.0f}–{AREA_BOUNDS[1]:.0f} m² and had that field discarded "
            "as a data-entry error."
        )

    low, high = RENT_USD_BOUNDS if listing_type == "rent" else SALE_USD_BOUNDS
    before = len(frame)
    frame = frame[frame["price_usd"].between(low, high)]
    if before > len(frame):
        notes.append(
            f"{before - len(frame):,} advert(s) fell outside the plausible "
            f"{'monthly rent' if listing_type == 'rent' else 'sale price'} range of "
            f"${low:,.0f}–${high:,.0f} and were excluded."
        )

    frame["price_per_sqm_usd"] = (frame["price_usd"] / frame["total_area"]).replace(
        [np.inf, -np.inf], np.nan
    )

    rooms = pd.to_numeric(frame.get("rooms"), errors="coerce")
    frame["rooms"] = rooms.where(rooms.between(1, 15))

    # The as-advertised price mixes som and у.е. in one column, so averaging it
    # is meaningless; price_uzs is price_usd times a constant, so keeping both
    # would put a perfect correlation into every driver table. Only the dollar
    # series survives into the analysis — the currency split is reported in the
    # notes above, and `currency` remains for anyone segmenting by it.
    frame = frame.drop(columns=[c for c in ("price", "price_uzs") if c in frame.columns])
    return frame, notes


# ---------------------------------------------------------------------------
def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, (list, dict)):
        return None
    try:
        out = float(str(value).replace(" ", "").replace(",", "."))
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _clean_text(value: Any) -> str | None:
    if value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    return re.sub(r"\s+", " ", text) or None
