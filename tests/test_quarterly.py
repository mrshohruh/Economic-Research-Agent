"""Quarterly comparison tables, name folding, dated rates and the index.

Everything here is local: the archive is a fixture and the Central Bank is a
stub, so no test reaches the network.
"""
import sqlite3
from unittest.mock import Mock

import pandas as pd
import pytest

from uzhousing.ingest import fx_history
from uzhousing.ingest.price_history import (MIN_QUARTER_LISTINGS, build_series,
                                            canonical_district, canonical_region,
                                            comparison_table, fixed_weights,
                                            index_series, national, quarter_label,
                                            region_label, thin_groups)
from uzhousing.report import layout
from uzhousing.report.olx_bulletin import (Archive, Note, index_section,
                                          movement_bullets)


# ---------------------------------------------------------------------------
# folding the three sources' spellings onto one name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Ташкентская область", "Toshkent Viloyati"),          # OLX, Cyrillic
    ("Toshkent viloyati", "Toshkent Viloyati"),            # Uybor, Latin Uzbek
    ("Toshkent shahri", "Toshkent shahri"),                # the archive
    ("Каракалпакстан", "Qoraqalpogʻiston Respublikasi"),   # no "Республика"
    ("Qoraqalpog`iston Respublikasi", "Qoraqalpogʻiston Respublikasi"),
    ("Syrdar'inskaya oblast'", "Sirdaryo Viloyati"),       # transliterated
    ("Dzhizakskaya oblast'", "Jizzax Viloyati"),
    ("Farg`ona viloyati", "Farg'ona Viloyati"),            # a different apostrophe
    ("Джизакская область ", "Jizzax Viloyati"),            # trailing space
])
def test_every_spelling_of_a_region_folds_to_one_name(name, expected):
    assert canonical_region(name) == expected


def test_the_capital_is_told_from_its_region_by_the_city():
    # OLX files the city under the surrounding region and names the city.
    assert canonical_region("Ташкентская область", "Ташкент") == "Toshkent shahri"
    assert canonical_region("Ташкентская область", "Чирчик") == "Toshkent Viloyati"
    assert canonical_region("Ташкентская область") == "Toshkent Viloyati"


def test_an_unknown_region_keeps_its_own_spelling():
    # Merging it into a neighbour would attribute prices to the wrong place.
    assert canonical_region("Неизвестная область") == "Неизвестная область"
    assert canonical_region(None) == "Ko'rsatilmagan"


@pytest.mark.parametrize("name,expected", [
    ("Ташкент, Мирзо-Улугбекский район", "Mirzo Ulug'bek"),
    ("Мирзо-Улугбекский район", "Mirzo Ulug'bek"),
    ("Mirzo Ulug`bek tumani", "Mirzo Ulug'bek"),
    ("Яшнабадский район", "Yashnobod"),
    ("Yashnabad tumani", "Yashnobod"),
])
def test_district_spellings_fold_to_one_name(name, expected):
    assert canonical_district(name) == expected


def test_only_the_capitals_districts_are_districts():
    # A settlement in another region listed beside them would read as one.
    assert canonical_district("Чирчик") == "Ko'rsatilmagan"
    assert canonical_district("Самарканд, Центр") == "Ko'rsatilmagan"


def test_region_labels_shorten_without_merging_the_two_tashkents():
    assert region_label("Andijon Viloyati") == "Andijon"
    assert region_label("Toshkent Viloyati") == "Toshkent viloyati"
    assert region_label("Toshkent shahri") == "Toshkent shahri"


# ---------------------------------------------------------------------------
# the archive, aggregated by quarter and by district
# ---------------------------------------------------------------------------

def archive(tmp_path, name, rows):
    """A fixture file with the archive's real column names, districts included."""
    path = tmp_path / name
    db = sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE olx_house_price (month_and_year TEXT, state TEXT, '
                   'price1 REAL, currency TEXT, "Общая площадь:" REAL, '
                   '"Тип жилья:" TEXT, location_1 TEXT)')
        db.executemany("INSERT INTO olx_house_price VALUES (?,?,?,?,?,?,?)", rows)
        db.commit()
    finally:
        db.close()  # or Windows keeps the fixture locked
    return path


def listing(month, price, state="Toshkent shahri", market="Вторичный рынок",
            location="Ташкент, Чиланзарский район", area=100):
    return (month, state, price, "'у.е.'", area, market, location)


def test_quarters_and_districts_come_from_one_pass(tmp_path):
    archive(tmp_path, "a.db", [listing("01-2024", 100_000), listing("02-2024", 120_000),
                               listing("04-2024", 140_000)])
    tables = build_series(tmp_path)
    assert set(tables) == {"monthly", "quarterly", "quarterly_district"}
    quarterly = tables["quarterly"].set_index("quarter")
    assert quarterly.loc[pd.Period("2024Q1", "Q"), "listings"] == 2
    # Two months observed of three: the report says so rather than implying a full quarter.
    assert quarterly.loc[pd.Period("2024Q1", "Q"), "months"] == 2
    assert tables["quarterly_district"]["district"].unique().tolist() == ["Chilonzor"]


def test_regions_and_districts_are_aggregated_from_the_adverts_not_from_each_other(tmp_path):
    # Two districts with different medians: the region's median must come from
    # the adverts, not from the middle of the two district medians.
    rows = ([listing("01-2024", 100_000, location="Ташкент, Чиланзарский район")] * 3
            + [listing("01-2024", 400_000, location="Ташкент, Мирабадский район")])
    archive(tmp_path, "a.db", rows)
    tables = build_series(tmp_path)
    region = tables["quarterly"]["median_usd_sqm"].iloc[0]
    districts = tables["quarterly_district"]["median_usd_sqm"].tolist()
    assert region == 1000          # the median of 1000, 1000, 1000, 4000
    assert sorted(districts) == [1000, 4000]
    assert region != sum(districts) / 2


def test_a_file_without_districts_still_yields_the_other_tables(tmp_path):
    path = tmp_path / "a.db"
    db = sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE olx_house_price (month_and_year TEXT, state TEXT, '
                   'price1 REAL, currency TEXT, "Общая площадь:" REAL, "Тип жилья:" TEXT)')
        db.execute("INSERT INTO olx_house_price VALUES "
                   "('01-2024','Buxoro Viloyati',80000,'у.е.',80,'Новостройки')")
        db.commit()
    finally:
        db.close()
    tables = build_series(tmp_path)
    assert len(tables["quarterly"]) == 1
    assert tables["quarterly_district"].empty


def test_a_missing_required_column_is_reported_not_guessed(tmp_path):
    path = tmp_path / "a.db"
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE olx_house_price (month_and_year TEXT, price1 REAL)")
        db.commit()
    finally:
        db.close()
    with pytest.raises(ValueError, match="ustunlar topilmadi"):
        build_series(tmp_path)


def test_a_cache_written_before_quarters_existed_is_rebuilt(tmp_path):
    archive(tmp_path, "a.db", [listing("01-2024", 100_000)])
    cache = tmp_path / "cache.csv"
    build_series(tmp_path, cache=cache)
    cache.with_name("cache_quarterly.csv").unlink()
    # A partial cache must not be half-trusted: all three tables come from one read.
    assert not build_series(tmp_path, cache=cache)["quarterly"].empty


# ---------------------------------------------------------------------------
# the comparison table the reference's layout is built around
# ---------------------------------------------------------------------------

def quarterly(rows):
    return pd.DataFrame([{"quarter": pd.Period(q, "Q"), "region_uz": r,
                          "market": "Ikkilamchi", "listings": n, "months": 3,
                          "median_usd_sqm": v, "median_usd": v * 50}
                         for q, r, n, v in rows])


QUARTERS = [pd.Period(q, "Q") for q in ("2025Q1", "2025Q2", "2025Q3")]
RATES = {QUARTERS[0]: 12_000.0, QUARTERS[1]: 12_500.0, QUARTERS[2]: 13_000.0}


def test_two_levels_and_two_changes_priced_in_som():
    table = comparison_table(quarterly([
        ("2025Q1", "Buxoro Viloyati", 500, 1000),
        ("2025Q2", "Buxoro Viloyati", 500, 1100),
        ("2025Q3", "Buxoro Viloyati", 500, 1210)]), "region_uz", QUARTERS, RATES)
    assert list(table.columns) == ["Hudud", "2025-Ch2", "Δ Ch2/Ch1", "2025-Ch3", "Δ Ch3/Ch2"]
    # 1100 USD/m² at 12 500 so'm is 13.75 million so'm per m².
    assert table["2025-Ch2"].iloc[0] == pytest.approx(13.75)
    assert table["2025-Ch3"].iloc[0] == pytest.approx(15.73)
    assert table["Δ Ch3/Ch2"].iloc[0] == pytest.approx(14.4, abs=0.1)


def test_dollars_are_reported_when_a_quarter_has_no_dated_rate():
    # Pricing a quarter at another quarter's rate would invent a movement.
    table = comparison_table(quarterly([
        ("2025Q1", "Buxoro Viloyati", 500, 1000),
        ("2025Q2", "Buxoro Viloyati", 500, 1100),
        ("2025Q3", "Buxoro Viloyati", 500, 1210)]), "region_uz", QUARTERS,
        {QUARTERS[0]: 12_000.0})
    assert table["2025-Ch2"].iloc[0] == 1100


def test_a_thin_quarter_keeps_its_row_and_shows_no_median():
    thin = MIN_QUARTER_LISTINGS - 1
    table = comparison_table(quarterly([
        ("2025Q1", "Qoraqalpogʻiston Respublikasi", thin, 500),
        ("2025Q2", "Qoraqalpogʻiston Respublikasi", thin, 500),
        ("2025Q3", "Qoraqalpogʻiston Respublikasi", thin, 800),
        ("2025Q1", "Buxoro Viloyati", 500, 1000),
        ("2025Q2", "Buxoro Viloyati", 500, 1000),
        ("2025Q3", "Buxoro Viloyati", 500, 1000)]), "region_uz", QUARTERS, RATES)
    row = table.set_index("Hudud").loc["Qoraqalpogʻiston Respublikasi"]
    assert row.isna().all()           # present, but nothing is ranked
    assert "Buxoro Viloyati" in table["Hudud"].tolist()


def test_thin_groups_are_listed_with_their_counts():
    frame = quarterly([("2025Q3", "Navoiy Viloyati", 12, 500),
                       ("2025Q3", "Buxoro Viloyati", 500, 1000)])
    assert thin_groups(frame, "region_uz", QUARTERS[-1]) == {"Navoiy Viloyati": 12}


def test_rows_follow_the_reference_order():
    table = comparison_table(quarterly([
        (q, r, 500, 1000) for q in ("2025Q1", "2025Q2", "2025Q3")
        for r in ("Xorazm Viloyati", "Andijon Viloyati", "Toshkent shahri")]),
        "region_uz", QUARTERS, RATES)
    assert table["Hudud"].tolist() == ["Andijon Viloyati", "Toshkent shahri",
                                       "Xorazm Viloyati"]


def test_change_columns_are_named_so_the_renderer_colours_them():
    table = comparison_table(quarterly([
        (q, "Buxoro Viloyati", 500, 1000) for q in ("2025Q1", "2025Q2", "2025Q3")]),
        "region_uz", QUARTERS, RATES)
    assert [c for c in table.columns if c.startswith(layout.DELTA)]


def test_quarter_labels_match_the_reference():
    assert quarter_label(pd.Period("2025Q3", "Q")) == "2025-Ch3"


# ---------------------------------------------------------------------------
# commentary is read off the table it describes
# ---------------------------------------------------------------------------

def test_commentary_reads_the_table_rather_than_listing_it():
    table = pd.DataFrame({"Hudud": ["A", "B", "C"], "2025-Ch2": [10.0, 20.0, 30.0],
                          "Δ Ch2/Ch1": [1.0, 1.0, 1.0], "2025-Ch3": [9.0, 22.0, 27.0],
                          "Δ Ch3/Ch2": [-10.0, 10.0, -10.0]})
    items = movement_bullets(table, "mln so'm/m²")
    joined = " ".join(items)
    # Two or three findings, not one line per row of the table.
    assert 2 <= len(items) <= 3
    assert "2025-yil III choragida" in joined
    assert "2 ta hududda" in joined and "**A** (-10,0 foiz)" in joined
    # The spread is stated as a ratio, which is what the levels column means.
    assert "barobar" in joined and "**C**" in joined
    # Every figure is written the Uzbek way: comma decimal, the word "foiz".
    assert "%" not in joined


def test_commentary_says_so_when_nothing_is_comparable():
    table = pd.DataFrame({"Hudud": ["A"], "2025-Ch2": [float("nan")],
                          "Δ Ch2/Ch1": [float("nan")], "2025-Ch3": [float("nan")],
                          "Δ Ch3/Ch2": [float("nan")]})
    assert "hisoblanmadi" in " ".join(movement_bullets(table, "USD/m²"))


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------

def test_the_index_bases_on_the_first_observed_quarter():
    levels = index_series(quarterly([("2025Q1", "A", 100, 1000),
                                     ("2025Q2", "A", 100, 1100)]))
    assert levels["index"].tolist() == [100.0, 110.0]


def test_the_index_is_listing_weighted_not_a_plain_average():
    levels = index_series(quarterly([("2025Q1", "A", 90, 1000), ("2025Q1", "B", 10, 2000)]))
    assert levels["median_usd_sqm"].iloc[0] == pytest.approx(1100)  # not 1500


def test_the_index_holds_its_weights_fixed_against_a_shift_in_advert_volumes():
    # Prices do not move; only the share of adverts coming from the expensive
    # region does. A volume-weighted series would read that as a price rise.
    frame = quarterly([("2025Q1", "Toshkent shahri", 100, 2000),
                       ("2025Q1", "Buxoro Viloyati", 100, 500),
                       ("2025Q2", "Toshkent shahri", 900, 2000),
                       ("2025Q2", "Buxoro Viloyati", 100, 500)])
    levels = index_series(frame)
    assert levels.attrs["weighting"] == "fixed"
    assert levels["index"].tolist() == [100.0, 100.0]
    assert national(frame, "quarter")["median_usd_sqm"].iloc[-1] > 1250  # volume-weighted


def test_a_region_seen_in_only_one_quarter_is_not_in_the_fixed_basket():
    # A late arrival would otherwise enter the index as a price movement.
    frame = quarterly([("2025Q1", "Toshkent shahri", 100, 2000),
                       ("2025Q2", "Toshkent shahri", 100, 2000),
                       ("2025Q2", "Navoiy Viloyati", 100, 400)])
    weights = fixed_weights(frame)
    assert list(weights.index.get_level_values("region_uz")) == ["Toshkent shahri"]
    levels = index_series(frame)
    assert levels["index"].tolist() == [100.0, 100.0]
    # The basket covers half the adverts of the second quarter, and says so.
    assert levels["coverage"].tolist() == [100.0, 50.0]


def test_without_a_common_stratum_the_index_says_it_used_volume_weights():
    frame = quarterly([("2025Q1", "Toshkent shahri", 100, 2000),
                       ("2025Q2", "Navoiy Viloyati", 100, 400)])
    levels = index_series(frame)
    assert levels.attrs["weighting"] == "volume"


def test_a_partly_observed_quarter_is_flagged_not_hidden():
    frame = quarterly([("2025Q1", "A", 100, 1000), ("2025Q2", "A", 100, 1100)])
    frame.loc[frame["quarter"] == pd.Period("2025Q2", "Q"), "months"] = 1
    section = index_section(Archive(pd.DataFrame(), frame, pd.DataFrame()))
    text = " ".join([item for block in section.blocks
                     if hasattr(block, "items") for item in block.items]
                    + [line for block in section.blocks if isinstance(block, Note)
                       for line in block.paragraphs])
    assert "2025-yil II choragi (1 oy)" in text


def test_without_an_archive_the_index_section_says_so():
    section = index_section(None)
    assert not section.tables()
    assert "keltirilmadi" in " ".join(getattr(b, "text", "") for b in section.blocks)


# ---------------------------------------------------------------------------
# dated exchange rates
# ---------------------------------------------------------------------------

def rate_response(date, rate="12000.5"):
    response = Mock(status_code=200)
    response.json.return_value = [{"Ccy": "USD", "Rate": rate, "Date": date}]
    return response


def test_a_rate_dated_anything_but_the_day_asked_for_is_refused(tmp_path):
    # The endpoint answers today's rate for a date it does not recognise.
    session = Mock()
    session.get.return_value = rate_response("19.09.2026")
    rates = fx_history.quarter_end_rates([pd.Period("2025Q3", "Q")],
                                         cache=tmp_path / "r.csv", session=session,
                                         progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    assert rates == {}


def test_a_quarter_ending_on_a_holiday_uses_the_last_rate_in_force(tmp_path):
    # 2025-03-31 was a public holiday; the rate in force is the previous one.
    session = Mock()
    # An unpublished day answers with today's rate, whichever day was asked for.
    session.get.side_effect = [rate_response("19.09.2026"),   # 31.03 not published
                               rate_response("19.09.2026"),   # 30.03 not published
                               rate_response("29.03.2025", "12913.19")]
    rates = fx_history.quarter_end_rates([pd.Period("2025Q1", "Q")],
                                         cache=tmp_path / "r.csv", session=session,
                                         progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    assert rates == {pd.Period("2025Q1", "Q"): (12913.19, "2025-03-29")}


def test_a_quarter_that_has_not_ended_is_not_requested(tmp_path):
    session = Mock()
    rates = fx_history.quarter_end_rates([pd.Period("2026Q4", "Q")],
                                         cache=tmp_path / "r.csv", session=session,
                                         progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    assert rates == {} and session.get.call_count == 0


def test_rates_are_cached_and_the_second_run_asks_for_nothing(tmp_path):
    cache = tmp_path / "r.csv"
    session = Mock()
    session.get.return_value = rate_response("30.09.2025", "12067.76")
    first = fx_history.quarter_end_rates([pd.Period("2025Q3", "Q")], cache=cache,
                                         session=session, progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    quiet = Mock()
    again = fx_history.quarter_end_rates([pd.Period("2025Q3", "Q")], cache=cache,
                                         session=quiet, progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    assert first == again and quiet.get.call_count == 0


def test_a_failed_fetch_returns_what_is_known_rather_than_raising(tmp_path):
    session = Mock()
    session.get.side_effect = RuntimeError("network down")
    rates = fx_history.quarter_end_rates([pd.Period("2025Q3", "Q")],
                                         cache=tmp_path / "r.csv", session=session,
                                         progress=lambda _: None,
                                         today=pd.Timestamp("2026-09-21"))
    assert rates == {}


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

def test_a_change_is_coloured_by_its_direction():
    assert layout.fmt_change(2.5)[0] == "+2.5%"
    assert layout.fmt_change(2.5)[1] == layout.PALETTE["rise"]
    assert layout.fmt_change(-2.5)[1] == layout.PALETTE["fall"]
    assert layout.fmt_change(float("nan"))[0] == "—"


def test_a_suppressed_number_is_a_dash_not_a_zero():
    assert layout.fmt(float("nan")) == "—"
    assert layout.fmt(1234.5) == "1 234.50"


def test_bold_markup_survives_into_word():
    assert layout.split_bold("a **b** c") == [("a ", False), ("b", True), (" c", False)]
