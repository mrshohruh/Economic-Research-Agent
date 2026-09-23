"""The marketplace's own region, city and district lists.

These are read from OLX's public geo-encoder rather than hard-coded, so a
renamed or newly added district cannot silently fall out of collection. They
exist to split one oversized query into several smaller ones; see
:mod:`uzhousing.ingest.olx_partition` for why that is necessary.
"""
from __future__ import annotations

from .olx_client import CollectionError


def _places(client, path, cache, key, *, absent_ok=False):
    """Read a geo list once per run and validate its shape."""
    if key in cache:
        return cache[key]
    response = client.get(path, absent_ok=absent_ok)
    if response is None:
        cache[key] = []
        return []
    body = client._json(response)
    data = body.get("data", body)
    if not isinstance(data, list):
        raise CollectionError(f"OLX geo ro'yxati tuzilishi o'zgargan: {path}")
    places = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise CollectionError(f"OLX geo ro'yxati tuzilishi o'zgargan: {path}")
        places.append({"id": item["id"], "name": item.get("name") or str(item["id"])})
    cache[key] = places
    return places


def regions(client, cache):
    return _places(client, "/api/v1/geo-encoder/regions/", cache, "regions")


def cities(client, cache, region_id):
    return _places(client, f"/api/v1/geo-encoder/regions/{int(region_id)}/cities/",
                   cache, f"cities:{int(region_id)}")


def districts(client, cache, city_id):
    """A city's districts, or an empty list where it has none.

    Only the larger cities are subdivided, so a 404 here is an ordinary answer
    and not a collection failure.
    """
    return _places(client, f"/api/v1/geo-encoder/cities/{int(city_id)}/districts/",
                   cache, f"districts:{int(city_id)}", absent_ok=True)
