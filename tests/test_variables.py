"""Regression checks for offline property-variable selection."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from uzhousing.analysis.variables import assess_variables
from uzhousing.analysis.cross_section import analyse
from uzhousing.ingest.profiler import ColumnProfile
from uzhousing.ingest.relevance import _verdict
from uzhousing.report.narrative import Narrative, _write_cross_section


def frame():
    return pd.DataFrame({
        "price_usd": np.arange(100, 200) * 1000,
        "total_area": np.arange(100, 200),
        "price_per_sqm_usd": np.repeat(1000, 100),
        "rooms": np.tile([2, 3], 50),
        "condition": ["basic"] * 50 + ["renovated"] * 50,
        "has_parking": [False] * 50 + [True] * 50,
        "listing_id": np.arange(100),
        "advert_views": np.arange(100) * 3,
        "mystery": np.arange(100) * 8,
        "floor": [2] * 100,
        "living_area": [np.nan] * 90 + list(range(10)),
    })


def test_semantics_precede_association_and_reasons_are_serializable():
    rows = assess_variables(frame(), "price_usd")
    result = {r["name"]: r for r in rows}
    assert result["total_area"]["selected"]
    assert result["total_area"]["association"] == 1.0
    assert result["condition"]["selected"]
    assert result["has_parking"]["selected"]
    for name in ("price_usd", "price_per_sqm_usd", "listing_id", "advert_views",
                 "mystery", "floor", "living_area"):
        assert not result[name]["selected"], name
    assert all(r["reason"] for r in rows)
    json.dumps(rows, allow_nan=False)


def test_unique_integer_prices_are_measurements():
    col = ColumnProfile(name="price_usd", dtype="int64", role="price",
                        non_null=100, unique=100)
    assert _verdict(col.name, col.role, col)[0]


def test_property_flags_and_macro_shares_survive():
    assert _verdict("has_parking", "value", None)[0]
    assert _verdict("mortgage_share", "mortgage", None)[0]
    assert not _verdict("advert_views", "value", None)[0]
    assert not _verdict("is_promoted", "value", None)[0]


def test_cross_section_and_offline_prose_include_associations():
    section = analyse(frame(), listing_type="sale")
    assert section.variable_assessment
    narrative = Narrative()
    _write_cross_section(narrative, section.to_dict())
    assert any("Spearman" in p for p in narrative.drivers)


def test_optional_columns_do_not_cancel_analysis():
    section = analyse(frame()[["price_usd", "condition"]])
    assert section is not None
    assert any(v["selected"] for v in section.variable_assessment)


def test_infinite_values_and_thin_categories_are_not_evidence():
    data = frame()
    data["total_area"] = np.inf
    data["condition"] = ["basic"] * 99 + ["renovated"]
    result = {v["name"]: v for v in assess_variables(data, "price_usd")}
    assert not result["total_area"]["selected"]
    assert not result["condition"]["selected"]
