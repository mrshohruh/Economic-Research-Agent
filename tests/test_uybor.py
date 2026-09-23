"""Uybor collection and pooling; every response here is a fixture."""
from unittest.mock import Mock

import pandas as pd
import pytest

from uzhousing.ingest.price_history import canonical_region
from uzhousing.ingest.uybor import UyborError, collect, fetch_page, housing_fields
from uzhousing.report.olx_bulletin import _uybor_rows, source_mix


def listing(**over):
    base = {"id": 1, "operationType": "sale", "price": 100_000, "priceCurrency": "usd",
            "priceType": "all", "pricePeriodUnit": None, "square": 100, "room": "3",
            "floor": 2, "floorTotal": 9, "isNewBuilding": False,
            "category": {"id": 7, "name": {"uz": "Kvartira"}},
            "region": {"name": {"uz": "Toshkent shahri"}},
            "district": {"name": {"uz": "Olmazor tumani"}},
            "city": None, "createdAt": "2026-09-21T00:00:00Z",
            "media": [{"url": "x"}], "userId": 5, "views": 10}
    base.update(over)
    return base


def reply(data=None, status=200, text=""):
    response = Mock(status_code=status, text=text)
    response.json.return_value = data
    return response


def test_only_residential_categories_are_kept():
    assert housing_fields(listing())["property"] == "Kvartira"
    assert housing_fields(listing(category={"id": 28, "name": {}}))["property"] == "Hovli"
    for other in (11, 12, 21, 18):  # land, office, warehouse, a whole business
        assert housing_fields(listing(category={"id": other, "name": {}})) is None


def test_seller_and_media_never_reach_a_record():
    row = housing_fields(listing())
    assert not {"media", "userId", "views", "seller", "phone"} & set(row)


def test_pagination_asks_by_page_because_offset_is_ignored():
    session = Mock()
    session.get.return_value = reply({"results": [], "total": 0})
    fetch_page(session, page=3)
    params = session.get.call_args.kwargs["params"]
    assert params["page"] == 3 and "offset" not in params
    for bad in (0, -1, "2"):
        with pytest.raises(ValueError):
            fetch_page(session, page=bad)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_refusal_is_fatal(status):
    session = Mock()
    session.get.return_value = reply(status=status)
    with pytest.raises(UyborError, match="chekladi"):
        fetch_page(session, page=1)


def test_broken_payload_is_reported():
    session = Mock()
    session.get.return_value = reply({"nope": 1})
    with pytest.raises(UyborError, match="tuzilishi"):
        fetch_page(session, page=1)


def test_collect_paginates_dedupes_and_stops(tmp_path):
    session = Mock()
    page1 = {"results": [listing(id=i) for i in range(100)], "total": 150}
    page2 = {"results": [listing(id=i) for i in range(95, 140)], "total": 150}
    session.get.side_effect = [reply(status=200, text="User-agent: *\nDisallow: /admin/"),
                               reply(page1), reply(page2)]
    rows, coverage = collect(pages=5, session=session, pause=lambda _: None,
                             progress=lambda _: None)
    assert len(rows) == 140  # 100 + 45 new, five repeats dropped
    assert coverage["stop"] == "exhausted" and coverage["available"] == 150


def test_robots_disallow_stops_collection():
    session = Mock()
    session.get.return_value = reply(status=200, text="User-agent: *\nDisallow: /api/")
    with pytest.raises(UyborError, match="robots"):
        collect(pages=1, session=session, pause=lambda _: None, progress=lambda _: None)


# -- mapping into the report's rows ----------------------------------------

def rows_of(*items, rate=12000.0):
    return _uybor_rows([housing_fields(i) for i in items], rate)


def test_per_sqm_price_is_multiplied_by_area_not_read_as_a_total():
    row = rows_of(listing(priceType="sqm", price=1500, square=80))[0]
    assert row["price"] == 120_000  # 1500 x 80, not 1500


def test_per_sotka_and_unpriceable_listings_are_dropped():
    assert rows_of(listing(priceType="sot", price=900)) == []
    # A per-m2 price with no area has no recoverable total.
    assert rows_of(listing(priceType="sqm", price=1500, square=None)) == []


def test_non_monthly_rent_is_excluded():
    assert rows_of(listing(operationType="rent", pricePeriodUnit="day")) == []
    assert len(rows_of(listing(operationType="rent", pricePeriodUnit="month"))) == 1


def test_currency_is_passed_through_for_the_shared_cleaner():
    # Converting here would be overwritten, and would bypass the shared screens.
    assert rows_of(listing(priceCurrency="uzs"))[0]["currency"] == "UZS"
    assert rows_of(listing(priceCurrency="usd"))[0]["price"] == 100_000
    assert rows_of(listing(priceCurrency="eur")) == []


def test_new_building_flag_maps_to_the_market_split():
    assert rows_of(listing(isNewBuilding=True))[0]["market"] == "Birlamchi"
    assert rows_of(listing(isNewBuilding=False))[0]["market"] == "Ikkilamchi"
    assert rows_of(listing(isNewBuilding=None))[0]["market"] == "Aniqlanmagan"


def test_pooling_needs_one_spelling_per_region():
    # OLX sends Russian, Uybor sends Uzbek: unmapped they would be two rows.
    assert canonical_region("Ташкентская область", "Ташкент") == canonical_region("Toshkent shahri")
    assert canonical_region("Toshkent viloyati") == canonical_region("Ташкентская область", "Чирчик")


def test_source_mix_reports_composition_only_when_pooled():
    frame = pd.DataFrame({"source": ["OLX.uz", "OLX.uz", "Uybor.uz"],
                          "kind": ["sale", "rent", "sale"],
                          "property": ["Kvartira"] * 3})
    mix = source_mix(frame)
    assert set(mix["Manba"]) == {"OLX.uz", "Uybor.uz"}
    assert mix["E'lonlar"].sum() == 3
    single = frame[frame["source"] == "OLX.uz"]
    assert source_mix(single).empty  # nothing pooled, nothing to disclose


def test_collect_without_a_page_limit_reads_until_the_api_runs_out():
    session = Mock()
    pages = [{"results": [listing(id=i) for i in range(n * 100, n * 100 + 100)],
              "total": 250} for n in range(2)]
    pages.append({"results": [listing(id=i) for i in range(200, 250)], "total": 250})
    session.get.side_effect = [reply(status=200, text="User-agent: *\nDisallow: /admin/"),
                               *(reply(p) for p in pages)]
    rows, coverage = collect(session=session, pause=lambda _: None, progress=lambda _: None)
    assert len(rows) == 250
    assert coverage["stop"] == "exhausted" and coverage["requests"] == 3


def test_page_count_must_be_positive_or_absent():
    for bad in (0, -1, 1001, "all"):
        with pytest.raises(ValueError):
            collect(pages=bad, session=Mock(), pause=lambda _: None, progress=lambda _: None)
