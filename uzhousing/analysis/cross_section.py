"""Cross-sectional statistics over property microdata.

The time-series modules answer "how has the market moved?". This module answers
the prior question a housing report has to settle first: *what does the market
look like right now, and where is it expensive?* — mean and median price by
region, price per square metre, how the regions rank against each other, and how
price varies with dwelling size.

Every figure here is computed by pandas from the listing table. Nothing in this
module asks a language model for a number; the model's job is to interpret the
tables it produces, not to produce them.

Two design choices are deliberate and matter for the economics:

* **The median leads, not the mean.** Advertised property prices are strongly
  right-skewed — a handful of luxury listings drag the mean well above what a
  typical household faces. Both are reported, and the gap between them is itself
  a reported statistic.
* **Thin regions are labelled, not silently ranked.** A region represented by
  eleven adverts is not comparable to one represented by twenty thousand, so the
  sample size sits next to every figure and undersized groups are flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Below this many adverts a regional median is too noisy to rank on. The group
# is still reported — with the count visible — but marked as indicative.
MIN_GROUP_SAMPLE = 30

# Bands used to describe dwelling size in the size/price table.
ROOM_LABELS = {1: "1 room", 2: "2 rooms", 3: "3 rooms", 4: "4 rooms"}


@dataclass
class GroupStats:
    """Per-group price statistics, plus the frame the report prints."""

    field_name: str
    table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    most_expensive: dict[str, Any] = field(default_factory=dict)
    cheapest: dict[str, Any] = field(default_factory=dict)
    spread_ratio: float | None = None
    dispersion_pct: float | None = None
    reliable_groups: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_field": self.field_name,
            "most_expensive": self.most_expensive,
            "cheapest": self.cheapest,
            "spread_ratio": self.spread_ratio,
            "dispersion_pct": self.dispersion_pct,
            "reliable_groups": self.reliable_groups,
            "rows": self.table.to_dict(orient="records"),
        }


@dataclass
class CrossSection:
    """The complete cross-sectional picture of the market."""

    listing_type: str = "unknown"
    price_label: str = "price"
    currency_note: str = ""
    overall: dict[str, Any] = field(default_factory=dict)
    by_region: GroupStats | None = None
    by_city: GroupStats | None = None
    by_rooms: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    by_condition: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    # Kept for the distribution chart. Summary statistics alone cannot show
    # whether a market is one cluster or several.
    prices: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    prices_per_sqm: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.overall)

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_type": self.listing_type,
            "price_label": self.price_label,
            "currency_note": self.currency_note,
            "overall": self.overall,
            "by_region": self.by_region.to_dict() if self.by_region else {},
            "by_city": self.by_city.to_dict() if self.by_city else {},
            "by_rooms": self.by_rooms.to_dict(orient="records"),
            "by_condition": self.by_condition.to_dict(orient="records"),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
def analyse(
    frame: pd.DataFrame,
    listing_type: str = "unknown",
    price_col: str = "price_usd",
    area_col: str = "total_area",
    per_sqm_col: str = "price_per_sqm_usd",
    currency_note: str = "",
) -> CrossSection | None:
    """Compute the full cross-sectional picture, or ``None`` if the data cannot support it."""
    if frame is None or frame.empty or price_col not in frame.columns:
        return None
    prices = pd.to_numeric(frame[price_col], errors="coerce")
    if prices.notna().sum() < 20:
        return None

    result = CrossSection(
        listing_type=listing_type,
        price_label=_price_label(listing_type),
        currency_note=currency_note,
    )
    result.overall = _overall(frame, prices, area_col, per_sqm_col, listing_type)
    result.prices = prices.dropna()
    result.prices_per_sqm = pd.to_numeric(frame.get(per_sqm_col), errors="coerce").dropna()

    if "region" in frame.columns:
        result.by_region = _group_stats(frame, "region", price_col, per_sqm_col, area_col)
    if "city" in frame.columns:
        result.by_city = _group_stats(
            frame, "city", price_col, per_sqm_col, area_col, top_n=15
        )
    result.by_rooms = _by_rooms(frame, price_col, per_sqm_col, area_col)
    result.by_condition = _by_category(frame, "condition", price_col, per_sqm_col)

    result.notes = _quality_notes(result, frame, listing_type)
    return result


# ---------------------------------------------------------------------------
def _overall(
    frame: pd.DataFrame,
    prices: pd.Series,
    area_col: str,
    per_sqm_col: str,
    listing_type: str,
) -> dict[str, Any]:
    clean = prices.dropna()
    per_sqm = pd.to_numeric(frame.get(per_sqm_col), errors="coerce").dropna()
    area = pd.to_numeric(frame.get(area_col), errors="coerce").dropna()

    median = float(clean.median())
    mean = float(clean.mean())
    out: dict[str, Any] = {
        "listings": int(len(clean)),
        "median": round(median, 2),
        "mean": round(mean, 2),
        "p25": round(float(clean.quantile(0.25)), 2),
        "p75": round(float(clean.quantile(0.75)), 2),
        "p90": round(float(clean.quantile(0.90)), 2),
        "minimum": round(float(clean.min()), 2),
        "maximum": round(float(clean.max()), 2),
        # A mean well above the median is the signature of a right-skewed
        # market: a thin luxury tail pulling the average away from the typical.
        "skew_mean_over_median_pct": round((mean / median - 1) * 100, 1) if median else None,
        "listing_type": listing_type,
    }
    if len(per_sqm):
        out["median_per_sqm"] = round(float(per_sqm.median()), 2)
        out["mean_per_sqm"] = round(float(per_sqm.mean()), 2)
    if len(area):
        out["median_area_sqm"] = round(float(area.median()), 1)
    rooms = pd.to_numeric(frame.get("rooms"), errors="coerce").dropna()
    if len(rooms):
        out["median_rooms"] = round(float(rooms.median()), 1)
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if len(dates):
            out["posted_from"] = f"{dates.min():%Y-%m-%d}"
            out["posted_to"] = f"{dates.max():%Y-%m-%d}"
    return out


def _group_stats(
    frame: pd.DataFrame,
    field_name: str,
    price_col: str,
    per_sqm_col: str,
    area_col: str,
    top_n: int | None = None,
) -> GroupStats | None:
    """Rank groups by the median price a household actually faces."""
    work = frame[[c for c in (field_name, price_col, per_sqm_col, area_col) if c in frame.columns]].copy()
    work = work.dropna(subset=[field_name, price_col])
    if work.empty or work[field_name].nunique() < 2:
        return None

    grouped = work.groupby(field_name)
    table = pd.DataFrame(
        {
            "listings": grouped[price_col].size(),
            "median": grouped[price_col].median().round(2),
            "mean": grouped[price_col].mean().round(2),
            "p25": grouped[price_col].quantile(0.25).round(2),
            "p75": grouped[price_col].quantile(0.75).round(2),
        }
    )
    if per_sqm_col in work.columns:
        table["median_per_sqm"] = grouped[per_sqm_col].median().round(2)
    if area_col in work.columns:
        table["median_area_sqm"] = grouped[area_col].median().round(1)

    total = int(table["listings"].sum())
    table["share_of_listings_pct"] = (table["listings"] / total * 100).round(1)

    national_median = float(work[price_col].median())
    table["vs_national_pct"] = ((table["median"] / national_median - 1) * 100).round(1)
    table["sample"] = np.where(
        table["listings"] >= MIN_GROUP_SAMPLE, "adequate", "thin — indicative only"
    )

    table = table.sort_values("median", ascending=False)
    table.index.name = field_name
    table = table.reset_index()

    # Rankings are drawn only from groups with enough adverts to mean something.
    # A region represented by a single luxury advert must never be reported as
    # the country's most expensive market, so any adequately-sampled group
    # outranks every thin one; the full table is used only when none qualify.
    reliable = table[table["listings"] >= MIN_GROUP_SAMPLE]
    ranked = reliable if len(reliable) else table

    stats = GroupStats(field_name=field_name, reliable_groups=int(len(reliable)))
    top, bottom = ranked.iloc[0], ranked.iloc[-1]
    stats.most_expensive = _group_record(top, field_name)
    stats.cheapest = _group_record(bottom, field_name)

    # A spread needs two groups to span. With one, there is nothing to compare.
    if len(ranked) >= 2:
        if float(bottom["median"]):
            stats.spread_ratio = round(float(top["median"]) / float(bottom["median"]), 2)
        medians = ranked["median"].astype(float)
        if medians.mean():
            stats.dispersion_pct = round(float(medians.std(ddof=0) / medians.mean() * 100), 1)

    # When the table is truncated (cities, of which there are many), show the
    # best-evidenced groups rather than whichever thin group happens to top the
    # ranking — two adverts at a luxury address must not outrank the capital.
    stats.table = ranked.head(top_n) if top_n else table
    return stats


def _group_record(row: pd.Series, field_name: str) -> dict[str, Any]:
    keys = ("median", "mean", "listings", "median_per_sqm", "share_of_listings_pct", "vs_national_pct")
    record: dict[str, Any] = {field_name: str(row[field_name])}
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            record[key] = float(row[key])
    return record


def _by_rooms(
    frame: pd.DataFrame, price_col: str, per_sqm_col: str, area_col: str
) -> pd.DataFrame:
    """Price by dwelling size — the cheapest available check on price per m².

    Price per square metre normally *falls* as dwellings get larger. If it rises
    instead, the size bands are picking up location rather than size.
    """
    if "rooms" not in frame.columns:
        return pd.DataFrame()
    work = frame.dropna(subset=["rooms", price_col]).copy()
    if work.empty:
        return pd.DataFrame()

    rooms = pd.to_numeric(work["rooms"], errors="coerce")
    work["size_band"] = rooms.map(lambda r: ROOM_LABELS.get(int(r), "5+ rooms") if pd.notna(r) else None)
    grouped = work.groupby("size_band")

    table = pd.DataFrame(
        {
            "listings": grouped[price_col].size(),
            "median": grouped[price_col].median().round(2),
            "mean": grouped[price_col].mean().round(2),
        }
    )
    if per_sqm_col in work.columns:
        table["median_per_sqm"] = grouped[per_sqm_col].median().round(2)
    if area_col in work.columns:
        table["median_area_sqm"] = grouped[area_col].median().round(1)

    order = ["1 room", "2 rooms", "3 rooms", "4 rooms", "5+ rooms"]
    table = table.reindex([b for b in order if b in table.index])
    table.index.name = "size_band"
    return table.reset_index()


def _by_category(
    frame: pd.DataFrame, column: str, price_col: str, per_sqm_col: str, min_n: int = 25
) -> pd.DataFrame:
    """Median price by a qualitative attribute such as the state of repair."""
    if column not in frame.columns:
        return pd.DataFrame()
    work = frame.dropna(subset=[column, price_col])
    if work.empty or work[column].nunique() < 2:
        return pd.DataFrame()

    grouped = work.groupby(column)
    table = pd.DataFrame(
        {
            "listings": grouped[price_col].size(),
            "median": grouped[price_col].median().round(2),
        }
    )
    if per_sqm_col in work.columns:
        table["median_per_sqm"] = grouped[per_sqm_col].median().round(2)
    table = table[table["listings"] >= min_n].sort_values("median", ascending=False)
    if len(table) < 2:
        return pd.DataFrame()
    table.index.name = column
    return table.reset_index().head(10)


# ---------------------------------------------------------------------------
def _quality_notes(result: CrossSection, frame: pd.DataFrame, listing_type: str) -> list[str]:
    """Caveats a reader needs in order to use these numbers correctly."""
    notes = [
        "These are advertised asking prices from an open marketplace, not registered "
        "transaction prices. Asking prices lead agreed prices and exclude any negotiated "
        "discount, so levels here sit above what changes hands.",
    ]
    if listing_type == "rent":
        notes.append(
            "The dataset covers the rental market. Every price is a monthly rent, not a "
            "purchase price, so these figures speak to rental affordability and yields — "
            "they are not house prices and must not be read as such."
        )

    region = result.by_region
    if region and region.most_expensive:
        share = region.most_expensive.get("share_of_listings_pct")
        if share and share > 60:
            notes.append(
                f"Coverage is heavily concentrated: {region.most_expensive[region.field_name]} "
                f"accounts for {share:.0f}% of all adverts. The national median therefore "
                "describes that market far more than it describes the country."
            )
        thin = len(region.table) - region.reliable_groups
        if thin > 0:
            notes.append(
                f"{thin} of {len(region.table)} regions are represented by fewer than "
                f"{MIN_GROUP_SAMPLE} adverts. Their medians are shown for completeness but are "
                "too thin to rank confidently."
            )

    if "date" in frame.columns:
        notes.append(
            "The posting dates describe adverts that were live when the data was collected, "
            "not a complete history: listings that were filled or withdrawn earlier are absent. "
            "This is a snapshot of standing supply, so it supports a cross-sectional comparison "
            "but not a price index over time."
        )
    return notes


def _price_label(listing_type: str) -> str:
    return {
        "rent": "monthly asking rent",
        "sale": "asking price",
    }.get(listing_type, "advertised price")
