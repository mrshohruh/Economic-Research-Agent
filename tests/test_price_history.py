"""Historical archive aggregation; every archive here is a local fixture."""
import sqlite3

import pandas as pd
import pytest

from uzhousing.ingest.price_history import (
    REGION_UZ_FROM_RU, build_monthly, gap_months, live_month, national)
from uzhousing.report.olx_bulletin import (Bullets, Note, Text, annual_history,
                                            history_section)


def bullet_text(section):
    """Every line of commentary in a section, whatever block it came from."""
    lines = []
    for block in section.blocks:
        if isinstance(block, Bullets):
            lines += list(block.items)
        elif isinstance(block, Text):
            lines.append(block.text)
    return lines


def section_text(section):
    """Commentary plus the source and note callouts under the tables."""
    return bullet_text(section) + [line for block in section.blocks
                                   if isinstance(block, Note)
                                   for line in block.paragraphs]


def archive(tmp_path, name, rows):
    """Write a fixture file with the archive's real column names."""
    path = tmp_path / name
    db = sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE olx_house_price (month_and_year TEXT, state TEXT, '
                   'price1 REAL, currency TEXT, "Общая площадь:" REAL, "Тип жилья:" TEXT)')
        db.executemany("INSERT INTO olx_house_price VALUES (?,?,?,?,?,?)", rows)
        db.commit()
    finally:
        db.close()  # or Windows keeps the fixture locked
    return path


def test_prices_are_screened_and_som_excluded(tmp_path):
    archive(tmp_path, "a.db", [
        ("01-2024", "Toshkent shahri", 100_000, "'у.е.'", 100, "Вторичный рынок"),
        ("01-2024", "Toshkent shahri", 120_000, None, 100, "Новостройки"),
        # Som-quoted: excluded, because the archive carries no dated rate.
        ("01-2024", "Toshkent shahri", 900_000_000, "'сум'", 100, "Вторичный рынок"),
        ("01-2024", "Toshkent shahri", 500, "'у.е.'", 100, "Вторичный рынок"),   # under floor
        ("01-2024", "Toshkent shahri", 100_000, "'у.е.'", 2, "Вторичный рынок"), # area too small
    ])
    series = build_monthly(tmp_path)
    assert series["listings"].sum() == 2
    assert set(series["market"]) == {"Ikkilamchi", "Birlamchi"}
    assert series.loc[series["market"] == "Ikkilamchi", "median_usd_sqm"].iloc[0] == 1000


def test_unknown_market_is_kept_as_unknown(tmp_path):
    archive(tmp_path, "a.db", [("01-2024", "Buxoro Viloyati", 80_000, "'у.е.'", 80, "nan")])
    series = build_monthly(tmp_path)
    assert series["market"].tolist() == ["Aniqlanmagan"]


def test_cache_is_reused_and_rebuilt_on_demand(tmp_path):
    path = archive(tmp_path, "a.db", [("01-2024", "Navoiy Viloyati", 80_000, "'у.е.'", 80, "Новостройки")])
    cache = tmp_path / "cache.csv"
    first = build_monthly(tmp_path, cache=cache)
    assert cache.exists()
    path.unlink()  # cache alone must satisfy the next call
    again = build_monthly(tmp_path, cache=cache)
    assert again["median_usd_sqm"].tolist() == first["median_usd_sqm"].tolist()
    assert isinstance(again["month"].iloc[0], pd.Period)
    with pytest.raises(FileNotFoundError):
        build_monthly(tmp_path, cache=cache, rebuild=True)


def live_frame(rows):
    return pd.DataFrame([{"kind": "sale", "property": "Kvartira", "market": m,
                          "region": r, "city": c, "sqm_usd": s, "price_usd": s * 50}
                         for r, c, m, s in rows])


def test_live_month_separates_tashkent_city_from_its_region():
    frame = live_frame([
        ("Ташкентская область", "Ташкент", "Ikkilamchi", 1500),
        ("Ташкентская область", "Чирчик", "Ikkilamchi", 900),
        ("Самаркандская область", "Самарканд", "Birlamchi", 800),
    ])
    out = live_month(frame, "2026-09-21T00:00:00+00:00")
    mapped = dict(zip(out["region_uz"], out["median_usd_sqm"]))
    assert mapped["Toshkent shahri"] == 1500
    assert mapped["Toshkent Viloyati"] == 900
    assert mapped["Samarqand Viloyati"] == 800
    assert out["month"].iloc[0] == pd.Period("2026-09", freq="M")


def test_live_month_ignores_rent_and_houses():
    frame = live_frame([("Бухарская область", "Бухара", "Ikkilamchi", 700)])
    frame.loc[len(frame)] = {"kind": "rent", "property": "Kvartira", "market": "Ikkilamchi",
                             "region": "Бухарская область", "city": "Бухара",
                             "sqm_usd": 9, "price_usd": 500}
    frame.loc[len(frame)] = {"kind": "sale", "property": "Hovli", "market": "Ikkilamchi",
                             "region": "Бухарская область", "city": "Бухара",
                             "sqm_usd": 300, "price_usd": 90000}
    out = live_month(frame, "2026-09-21T00:00:00+00:00")
    assert out["listings"].sum() == 1
    assert out["median_usd_sqm"].iloc[0] == 700


def test_unmapped_region_is_named_not_dropped():
    out = live_month(live_frame([("Неизвестная область", "X", "Ikkilamchi", 500)]),
                     "2026-09-21T00:00:00+00:00")
    assert out["region_uz"].tolist() == ["Ko'rsatilmagan"]


def test_every_mapped_region_is_distinct():
    assert len(set(REGION_UZ_FROM_RU.values())) == len(REGION_UZ_FROM_RU)


def series_of(points):
    return pd.DataFrame([{"month": pd.Period(m, freq="M"), "region_uz": "R", "market": "Ikkilamchi",
                          "listings": n, "median_usd_sqm": v, "median_usd": v * 50}
                         for m, n, v in points])


def test_national_median_is_listing_weighted():
    frame = pd.DataFrame([
        {"month": pd.Period("2024-01", freq="M"), "region_uz": "A", "market": "x",
         "listings": 90, "median_usd_sqm": 1000, "median_usd": 1},
        {"month": pd.Period("2024-01", freq="M"), "region_uz": "B", "market": "x",
         "listings": 10, "median_usd_sqm": 2000, "median_usd": 1}])
    out = national(frame)
    assert out["median_usd_sqm"].iloc[0] == 1100  # not the unweighted 1500
    assert out["listings"].iloc[0] == 100


def test_gap_months_finds_only_interior_holes():
    assert gap_months(series_of([("2024-01", 5, 100), ("2024-04", 5, 100)])) == [
        pd.Period("2024-02", freq="M"), pd.Period("2024-03", freq="M")]
    assert gap_months(series_of([("2024-01", 5, 100), ("2024-02", 5, 100)])) == []


def test_annual_change_leaves_the_base_year_blank():
    table = annual_history(series_of([("2023-01", 10, 1000), ("2024-01", 10, 1100)]))
    assert table["Mediana, USD/m²"].tolist() == [1000, 1100]
    assert pd.isna(table["Δ oldingi yilga"].iloc[0])
    assert table["Δ oldingi yilga"].iloc[1] == pytest.approx(10.0)


def test_history_section_compares_across_the_most_recent_gap():
    # Two gaps: the comparison must use the later one, not the first.
    series = series_of([("2022-01", 500, 900), ("2022-04", 500, 950),
                        ("2025-09", 500, 1200), ("2026-09", 400, 1400)])
    section = history_section(series, {})
    joined = " ".join(bullet_text(section))
    assert "2025-09" in joined and "1 200" in joined and "1 400" in joined
    # Uzbek prose writes the decimal with a comma and the word "foiz".
    assert "+16,7 foiz" in joined
    assert section.tables() and not section.tables()[0].empty


def test_history_section_warns_when_the_live_sample_is_tiny():
    series = series_of([("2025-09", 100_000, 1200), ("2026-09", 50, 1400)])
    said = section_text(history_section(series, {}))
    assert any("50 ta e'londan" in text and "100 000 ta e'londan" in text
               and "tarkib o'zgarishi" in text for text in said)


def test_history_section_falls_back_without_an_archive():
    section = history_section(None, {})
    assert not section.tables()
    assert "indeksi hisoblanmadi" in " ".join(
        block.text for block in section.blocks if isinstance(block, Text))
