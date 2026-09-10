"""Tests for marketplace listing ingestion and the cross-sectional analysis.

Everything here runs offline: no API key, no web access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uzhousing import pipeline  # noqa: E402
from uzhousing.analysis import cross_section  # noqa: E402
from uzhousing.ingest import listings, loader, profiler  # noqa: E402

UZS_PER_USD = 12_500.0


def _advert(
    listing_id: int,
    region: str,
    city: str,
    price: float,
    currency: str = "UYE",
    area: float = 60.0,
    rooms: int = 2,
    created: str = "2026-08-01T10:00:00+05:00",
) -> dict:
    """One advert in the shape the marketplace API returns."""
    return {
        "id": listing_id,
        "url": f"https://example.invalid/{listing_id}",
        "title": "Ijaraga kvartira",
        "created_time": created,
        "business": False,
        "category": {"id": 1147, "type": "real_estate"},
        "location": {
            "city": {"id": 1, "name": city, "normalized_name": city.lower()},
            "region": {"id": 2, "name": region, "normalized_name": region.lower()},
        },
        "params": [
            {"key": "price", "name": "Цена", "type": "price",
             "value": {"value": price, "currency": currency, "negotiable": False}},
            {"key": "number_of_rooms", "type": "input", "value": {"key": str(rooms)}},
            {"key": "total_area", "type": "input", "value": {"key": str(area)}},
            {"key": "repairs", "type": "select", "value": {"key": "2", "label": "Евроремонт"}},
            {"key": "comission", "type": "select", "value": {"key": "no", "label": "Нет"}},
        ],
    }


def _pages(adverts: list[dict], per_page: int = 4) -> list[dict]:
    """Wrap adverts in the paged envelope the feed uses."""
    return [
        {"data": adverts[i : i + per_page], "metadata": {"total": len(adverts)}}
        for i in range(0, len(adverts), per_page)
    ]


@pytest.fixture
def feed() -> list[dict]:
    adverts = []
    listing_id = 1
    # Tashkent is deliberately the expensive, well-covered market; Jizzakh the cheap one.
    for _ in range(40):
        adverts.append(_advert(listing_id, "Ташкентская область", "Ташкент", 600))
        listing_id += 1
    for _ in range(40):
        adverts.append(_advert(listing_id, "Джизакская область", "Джизак", 200))
        listing_id += 1
    return _pages(adverts)


@pytest.fixture
def feed_file(tmp_path: Path, feed: list[dict]) -> Path:
    path = tmp_path / "listings.jsonl"
    path.write_text(
        "\n".join(json.dumps(page, ensure_ascii=False) for page in feed), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------
def test_detects_a_listing_feed(feed):
    assert listings.looks_like_listings(feed)


def test_ordinary_records_are_not_mistaken_for_listings():
    assert not listings.looks_like_listings([{"date": "2024-01", "price": 10}])


def test_flatten_produces_one_row_per_advert(feed):
    table = listings.flatten(feed, uzs_per_usd=UZS_PER_USD)
    assert table is not None
    assert len(table.frame) == 80
    for column in ("region", "city", "price_usd", "total_area", "rooms", "price_per_sqm_usd"):
        assert column in table.frame.columns


def test_rental_feed_is_recognised_as_rent(feed):
    table = listings.flatten(feed, uzs_per_usd=UZS_PER_USD)
    assert table.listing_type == "rent"


def test_som_prices_are_converted_to_dollars():
    adverts = [_advert(1, "Регион", "Город", 2_500_000, currency="UZS", area=50)]
    table = listings.flatten(_pages(adverts), uzs_per_usd=UZS_PER_USD)
    row = table.frame.iloc[0]
    assert row["price_usd"] == pytest.approx(200.0)
    assert row["price_per_sqm_usd"] == pytest.approx(4.0)


def test_mixed_currency_and_derived_columns_are_dropped():
    """Averaging an as-advertised column that mixes som and у.е. is meaningless."""
    table = listings.flatten(_pages([_advert(1, "R", "C", 500)]), uzs_per_usd=UZS_PER_USD)
    assert "price" not in table.frame.columns
    assert "price_uzs" not in table.frame.columns


def test_duplicate_adverts_are_removed():
    advert = _advert(7, "Регион", "Город", 500)
    table = listings.flatten(_pages([advert, advert]), uzs_per_usd=UZS_PER_USD)
    assert len(table.frame) == 1
    assert any("duplicate" in note for note in table.notes)


def test_impossible_floor_area_is_discarded_not_used():
    adverts = [
        _advert(1, "Регион", "Город", 500, area=100_000),
        _advert(2, "Регион", "Город", 500, area=60),
    ]
    table = listings.flatten(_pages(adverts), uzs_per_usd=UZS_PER_USD)
    # The advert survives; only its unusable area is dropped.
    assert len(table.frame) == 2
    assert table.frame["total_area"].isna().sum() == 1


def test_loader_routes_a_listing_file_through_the_flattener(feed_file):
    dataset = loader.load(feed_file, uzs_per_usd=UZS_PER_USD)
    assert dataset.listing_type == "rent"
    assert dataset.total_rows == 80
    assert "listings" in dataset.tables


# ---------------------------------------------------------------------------
# Cross-sectional statistics
# ---------------------------------------------------------------------------
@pytest.fixture
def section(feed) -> cross_section.CrossSection:
    table = listings.flatten(feed, uzs_per_usd=UZS_PER_USD)
    result = cross_section.analyse(
        table.frame, listing_type=table.listing_type, currency_note="Stated in USD."
    )
    assert result is not None
    return result


def test_overall_statistics_are_computed(section):
    assert section.overall["listings"] == 80
    assert section.overall["median"] == pytest.approx(400.0)
    assert section.price_label == "monthly asking rent"


def test_most_expensive_and_cheapest_regions_are_identified(section):
    region = section.by_region
    assert region.most_expensive["region"] == "Ташкентская область"
    assert region.cheapest["region"] == "Джизакская область"
    assert region.spread_ratio == pytest.approx(3.0)


def test_region_table_carries_the_columns_the_report_prints(section):
    columns = set(section.by_region.table.columns)
    assert {"region", "listings", "median", "mean", "median_per_sqm",
            "share_of_listings_pct", "vs_national_pct", "sample"} <= columns


def test_thin_groups_are_flagged_rather_than_ranked():
    adverts = [_advert(i, "Большой регион", "Город", 500) for i in range(1, 41)]
    adverts.append(_advert(99, "Крошечный регион", "Село", 9_000))
    table = listings.flatten(_pages(adverts), uzs_per_usd=UZS_PER_USD)
    result = cross_section.analyse(table.frame, listing_type=table.listing_type)

    region = result.by_region
    thin = region.table[region.table["region"] == "Крошечный регион"].iloc[0]
    assert thin["sample"].startswith("thin")
    # A single advert must not become the country's most expensive market.
    assert region.most_expensive["region"] == "Большой регион"


def test_rental_data_is_labelled_as_rent_not_house_prices(section):
    joined = " ".join(section.notes).lower()
    assert "rental market" in joined
    assert "not a purchase price" in joined


def test_cross_section_survives_without_a_region_column(feed):
    table = listings.flatten(feed, uzs_per_usd=UZS_PER_USD)
    frame = table.frame.drop(columns=["region", "city"])
    result = cross_section.analyse(frame, listing_type="rent")
    assert result is not None and result.by_region is None
    assert result.overall["listings"] == 80


def test_too_little_data_returns_nothing():
    frame = pd.DataFrame({"price_usd": [100.0, 200.0]})
    assert cross_section.analyse(frame) is None


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_produces_cross_sectional_figures(feed_file):
    dataset = loader.load(feed_file, uzs_per_usd=UZS_PER_USD)
    understanding = profiler.understand(dataset, None)
    analysis = pipeline.analyse(understanding)

    assert analysis.cross_section is not None
    assert analysis.cross_section.by_region is not None


def test_row_identifiers_are_not_treated_as_indicators(feed_file):
    """"listing_id" contains "listing", which the volume pattern would otherwise match."""
    dataset = loader.load(feed_file, uzs_per_usd=UZS_PER_USD)
    understanding = profiler.understand(dataset, None)
    assert understanding.primary_profile.role_of("listing_id") == "identifier"
    assert "listing_id" not in understanding.primary_profile.metric_cols
