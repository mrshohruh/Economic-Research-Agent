"""Collect a whole category by splitting it into queries the site will serve.

OLX refuses a paging offset beyond 1000, so any one query hands over at most
about a thousand listings however many pages are asked for. A category with
sixty thousand listings is therefore not reachable as a single query — but the
limit is on the query, not on the category: a narrower query gets its own full
window.

So a category is read by asking whether a query reaches past its own window —
which the listing endpoint answers, while the search metadata robots.txt asks
crawlers to leave alone would not — and splitting any query too large to page
through into smaller ones until each part fits. The splits follow the
marketplace's own structure — region, then city, then district, then price
bands — so every listing belongs to exactly one part and none is left out.
Parts are read in full and listings are de-duplicated, because a listing
sitting exactly on a price boundary appears in both neighbouring bands.
"""
from __future__ import annotations

from .olx_client import REACHABLE, CollectionError
from .olx_geo import cities, districts, regions

#: The widest price band, in so'm. The marketplace normalises every listing's
#: price into this one scale, so a band holds listings priced in any currency.
PRICE_FLOOR, PRICE_CEILING = 0, 10 ** 12

#: How far a price band may be halved before the remainder is accepted as
#: unreachable. Each level doubles the number of queries, and a band this
#: narrow holding over a thousand listings would mean identical asking prices.
MAX_PRICE_DEPTH = 16

PRICE_FROM, PRICE_TO = "filter_float_price:from", "filter_float_price:to"


def _price_children(filters):
    """Halve the current price band, or open one spanning every price."""
    low = filters.get(PRICE_FROM, PRICE_FLOOR)
    high = filters.get(PRICE_TO, PRICE_CEILING)
    if high - low < 2:
        return []
    # Asking prices are heavily skewed towards the bottom of the range, so the
    # band is halved geometrically: an arithmetic midpoint would leave almost
    # everything on one side and barely narrow the query at all. Splitting on
    # the ratio instead reaches a band that fits within a handful of steps,
    # whichever end of the market the listings are clustered at.
    middle = int((max(low, 1) * high) ** 0.5)
    if not low < middle < high:
        middle = (low + high) // 2
    # The endpoints are inclusive on both sides, so the bands are made to meet
    # without overlapping; a listing exactly on the seam is still de-duplicated.
    return [{PRICE_FROM: low, PRICE_TO: middle},
            {PRICE_FROM: middle + 1, PRICE_TO: high}]


def _children(client, cache, filters, price_depth):
    """The next, narrower set of queries covering exactly the same listings."""
    if "region_id" not in filters:
        return [{"region_id": r["id"]} for r in regions(client, cache)], "region"
    if "city_id" not in filters:
        found = cities(client, cache, filters["region_id"])
        if found:
            return [filters | {"city_id": c["id"]} for c in found], "city"
    if "district_id" not in filters:
        found = districts(client, cache, filters["city_id"]) if "city_id" in filters else []
        if found:
            return [filters | {"district_id": d["id"]} for d in found], "district"
    if price_depth < MAX_PRICE_DEPTH:
        return [filters | band for band in _price_children(filters)], "price"
    return [], "none"


def _describe(filters):
    if PRICE_FROM in filters:
        return f"narx {filters[PRICE_FROM]:,}-{filters[PRICE_TO]:,}"
    for key in ("district_id", "city_id", "region_id"):
        if key in filters:
            return f"{key.removesuffix('_id')} {filters[key]}"
    return "hammasi"


def iter_category(client, *, category_id, progress, report, cache, max_pages=None,
                  seen=None, filters=None, price_depth=0):
    """Yield every listing of a category, splitting queries that do not fit.

    ``report`` is filled in as the walk proceeds: it records how many listings
    were read, how many queries it took, and whichever parts stayed too large
    to page through in full. There is no site-wide total in it, because the
    only endpoint that publishes one is disallowed to crawlers. ``cache``
    holds the geo lists, which are read once and reused across categories.
    """
    filters = filters or {}
    top = seen is None
    seen = set() if top else seen
    report["queries"] = report.get("queries", 0) + 1
    if client.overflows(category_id=category_id, filters=filters):
        children, kind = _children(client, cache, filters, price_depth)
        if children:
            report["splits"] = report.get("splits", 0) + 1
            progress(f"OLX: {_describe(filters)}: {REACHABLE:,} e'londan ko'p, "
                     f"{len(children)} ta {kind} bo'yicha bo'linmoqda")
            for child in children:
                yield from iter_category(
                    client, category_id=category_id, progress=progress,
                    report=report, cache=cache, max_pages=max_pages, seen=seen,
                    filters=child,
                    price_depth=price_depth + 1 if kind == "price" else price_depth)
            report["collected"] = len(seen)
            return
        # Nothing left to split by: read the window this query does serve and
        # say plainly that listings stayed out of reach, even though the exact
        # number of them cannot be asked for.
        report.setdefault("unreachable", []).append(
            {"filters": dict(filters), "reachable": REACHABLE})
    for offer in client.iter_offers(category_id=category_id, max_pages=max_pages,
                                    filters=filters or None):
        key = str(offer["id"])
        if key in seen:
            continue
        seen.add(key)
        yield offer
        if len(seen) % 200 == 0:
            progress(f"OLX: {len(seen):,} e'lon yig'ildi")
    report["collected"] = len(seen)
