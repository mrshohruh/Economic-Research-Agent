"""The category walk: split a query the site will not serve into ones it will."""
import pytest
from unittest.mock import Mock

from uzhousing.ingest import olx_partition
from uzhousing.ingest.olx import robots_checker
from uzhousing.ingest.olx_client import (MAX_LIMIT, MAX_OFFSET, OLXClient,
                                         CollectionError)
from uzhousing.ingest.olx_partition import PRICE_FROM, PRICE_TO, iter_category

#: The rules olx.uz publishes, verbatim in the parts that matter here.
OLX_ROBOTS = """User-agent: *
Disallow: */search/
Disallow: */api/v1/users/me/
Disallow: */api/v1/offers/*/page-views/
Disallow: */api/v1/offers/*/suggested/
Disallow: */api/v1/offers/metadata/
Allow: /
"""

REGIONS = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
CITIES = {1: [{"id": 11, "name": "A1"}, {"id": 12, "name": "A2"}], 2: []}
DISTRICTS = {11: [{"id": 111, "name": "A1a"}, {"id": 112, "name": "A1b"}], 12: []}


class FakeMarket:
    """A marketplace that pages the way OLX does.

    Every listing sits in one region, city, district and price, so a correct
    walk collects each exactly once; the pagination window is clamped exactly
    as the real endpoint clamps it.
    """

    def __init__(self, listings, reachable):
        self.listings = listings
        self.reachable = reachable
        self.probes = 0
        self.pages = 0

    def _match(self, filters):
        rows = self.listings
        for key in ("region_id", "city_id", "district_id"):
            if key in filters:
                rows = [r for r in rows if r[key] == filters[key]]
        if PRICE_FROM in filters:  # both ends inclusive, as the real filter is
            rows = [r for r in rows if filters[PRICE_FROM] <= r["price"] <= filters[PRICE_TO]]
        return rows

    def overflows(self, *, category_id, filters=None):
        # As the real probe does: is there anything past the window?
        self.probes += 1
        return len(self._match(filters or {})) > self.reachable

    def iter_offers(self, *, category_id, max_pages=None, filters=None, limit=50):
        self.pages += 1
        # Only the first `reachable` listings of any query are ever served.
        for row in self._match(filters or {})[:self.reachable]:
            yield row


def market(n=40, reachable=5):
    listings = []
    for i in range(n):
        region = REGIONS[i % 2]["id"]
        city = CITIES[region][i % 2]["id"] if CITIES[region] else None
        district = (DISTRICTS.get(city) or [{}])[i % 2].get("id") if city else None
        listings.append({"id": i, "region_id": region, "city_id": city,
                         "district_id": district, "price": (i % 8) * 1000 + 1})
    return FakeMarket(listings, reachable)


@pytest.fixture(autouse=True)
def _geo(monkeypatch):
    monkeypatch.setattr(olx_partition, "regions", lambda c, cache: REGIONS)
    monkeypatch.setattr(olx_partition, "cities", lambda c, cache, region_id: CITIES[region_id])
    monkeypatch.setattr(olx_partition, "districts",
                        lambda c, cache, city_id: DISTRICTS.get(city_id, []))


def walk(fake, monkeypatch, reachable=5, **kwargs):
    monkeypatch.setattr(olx_partition, "REACHABLE", reachable)
    report = {}
    got = list(iter_category(fake, category_id=13, progress=lambda _: None,
                             report=report, cache={}, **kwargs))
    return got, report


def test_a_query_that_fits_is_read_whole_without_splitting(monkeypatch):
    fake = market(n=4)
    got, report = walk(fake, monkeypatch)
    assert [o["id"] for o in got] == [0, 1, 2, 3]
    assert report["collected"] == 4
    assert report.get("splits") is None and fake.probes == 1


def test_every_listing_is_collected_exactly_once_across_the_splits(monkeypatch):
    fake = market(n=40)
    got, report = walk(fake, monkeypatch)
    ids = [o["id"] for o in got]
    assert sorted(ids) == list(range(40)) and len(ids) == len(set(ids))
    assert report["collected"] == 40
    assert report["splits"] >= 1 and "unreachable" not in report


def test_the_walk_reaches_more_than_one_query_ever_could(monkeypatch):
    # The point of the exercise: 40 listings behind a 5-per-query window.
    fake = market(n=40, reachable=5)
    got, _ = walk(fake, monkeypatch, reachable=5)
    assert len(got) == 40 > fake.reachable


def test_an_unsplittable_overflow_is_recorded_not_hidden(monkeypatch):
    # Every listing shares one region, city, district and price, so no split
    # can separate them and the shortfall has to be reported.
    rows = [{"id": i, "region_id": 1, "city_id": 11, "district_id": 111, "price": 500}
            for i in range(30)]
    fake = FakeMarket(rows, reachable=5)
    got, report = walk(fake, monkeypatch)
    assert len(got) == 5
    assert report["unreachable"] and report["unreachable"][0]["reachable"] == 5
    assert report["unreachable"][0]["filters"]["city_id"] == 11


def test_price_bands_meet_without_overlapping_or_leaving_a_gap():
    low, high = olx_partition._price_children({})
    assert low[PRICE_FROM] == olx_partition.PRICE_FLOOR
    assert high[PRICE_TO] == olx_partition.PRICE_CEILING
    assert low[PRICE_TO] + 1 == high[PRICE_FROM]
    assert olx_partition._price_children({PRICE_FROM: 5, PRICE_TO: 6}) == []


def _offers(n):
    return [{"id": i, "params": [], "location": {}} for i in range(n)]


def test_overflow_is_judged_from_the_last_page_the_window_serves():
    # A full last page means listings remain past the window; a short one
    # means the query fits and can be read as it stands.
    session = Mock()
    response = Mock(status_code=200, text="")
    session.get.return_value = response
    client = OLXClient(session, pause=lambda _: None)

    response.json.return_value = {"data": _offers(MAX_LIMIT)}
    assert client.overflows(category_id=13) is True
    assert session.get.call_args.kwargs["params"] == {
        "category_id": 13, "offset": MAX_OFFSET, "limit": MAX_LIMIT}

    response.json.return_value = {"data": _offers(3)}
    assert client.overflows(category_id=13) is False
    response.json.return_value = {"data": []}
    assert client.overflows(category_id=13) is False


def test_the_probe_never_asks_for_the_disallowed_metadata_endpoint():
    # The bug this guards: the search metadata answers "how many are there"
    # in one number, and robots.txt disallows it.
    session = Mock()
    response = Mock(status_code=200, text="")
    response.json.return_value = {"data": _offers(1)}
    session.get.return_value = response
    client = OLXClient(session, pause=lambda _: None,
                       check_path=robots_checker(OLX_ROBOTS))
    client.overflows(category_id=13, filters={"region_id": 5})
    assert "/metadata/" not in session.get.call_args.args[0]
    assert not hasattr(client, "count")


def test_the_published_rules_allow_every_path_the_walk_uses():
    check = robots_checker(OLX_ROBOTS)
    for path in ("/nedvizhimost/kvartiry/arenda-dolgosrochnaya/",
                 "/api/v1/offers/", "/api/v1/offers/123456/",
                 "/api/v1/geo-encoder/regions/",
                 "/api/v1/geo-encoder/regions/5/cities/",
                 "/api/v1/geo-encoder/cities/7/districts/"):
        check(path)  # must not raise
    for path in ("/api/v1/offers/metadata/search/",
                 "/api/v1/offers/123/page-views/", "/api/v1/users/me/"):
        with pytest.raises(CollectionError, match="robots"):
            check(path)


def test_only_known_filters_are_ever_sent():
    session = Mock()
    client = OLXClient(session, pause=lambda _: None)
    for bad in ({"user_id": 5}, {"region_id": "5"}, {"region_id": -1}):
        with pytest.raises(ValueError):
            client.overflows(category_id=13, filters=bad)
    assert session.get.call_count == 0
