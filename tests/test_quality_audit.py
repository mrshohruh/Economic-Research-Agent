"""The data-quality audit, the Uzbek writing conventions and the editorial pass.

Everything here is local: the audit and the editor read frames and blocks that
the tests build, and the official-statistics client is given a stub session so
no test reaches the network.
"""
import pandas as pd
import pytest

from uzhousing.analysis import quality
from uzhousing.report import editorial, style
from uzhousing.report.olx_bulletin import Bullets, Note, Section, Tbl, Text
from uzhousing.research import official


def listings(rows):
    """A cleaned cross-section in the bulletin's own shape."""
    frame = pd.DataFrame(rows)
    defaults = {"kind": "sale", "property": "Kvartira", "region": "Toshkent shahri",
                "district": "Chilonzor", "rooms": 2.0, "total_area": 50.0,
                "price_usd": 50_000.0, "source": "OLX.uz"}
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
        frame[column] = frame[column].fillna(value) if column != "rooms" else frame[column]
    frame["sqm_usd"] = frame["price_usd"] / frame["total_area"]
    frame["sqm_uzs"] = frame["sqm_usd"] * 12_000
    return frame


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def test_every_screen_reports_itself_even_when_it_finds_nothing():
    # A screen that found nothing is evidence that the check ran.
    result = quality.audit(listings([{"price_usd": 40_000.0 + 1_000 * i,
                                      "total_area": 50.0 + i} for i in range(5)]))
    assert {"place", "cross_source", "rooms", "classification", "outlier", "repeat"} \
        <= {finding.code for finding in result.findings}
    assert not result.flagged


def test_a_rent_posted_as_a_sale_is_not_priced():
    rows = [{} for _ in range(4)]
    rows[0]["price_usd"] = 400.0          # 8 USD/m²: a monthly rent, not a sale
    result = quality.audit(listings(rows))
    finding = next(f for f in result.findings if f.code == "classification")
    assert finding.affected == 1
    assert result.frame["sqm_usd"].isna().sum() == 1
    # The row itself stays: it is still an advert, it just carries no price.
    assert len(result.frame) == 4


def test_the_same_flat_on_two_sites_is_counted_once():
    rows = [{"source": "OLX.uz"}, {"source": "Uybor.uz"}, {"source": "OLX.uz",
                                                           "price_usd": 80_000.0}]
    result = quality.audit(listings(rows))
    assert next(f for f in result.findings if f.code == "cross_source").affected == 1
    assert len(result.frame) == 2


def test_repeats_inside_one_site_are_reported_but_kept():
    # Identical adverts under different ids are ordinary on a classifieds site;
    # dropping them would shrink every median's sample without evidence.
    result = quality.audit(listings([{}, {}, {}]))
    assert next(f for f in result.findings if f.code == "repeat").affected == 2
    assert len(result.frame) == 3


def test_an_impossible_room_count_is_cleared_not_the_row():
    rows = [{"rooms": 40.0}, {"rooms": 3.0}]
    result = quality.audit(listings(rows))
    assert next(f for f in result.findings if f.code == "rooms").affected == 1
    assert result.frame["rooms"].isna().sum() == 1
    assert result.frame["price_usd"].notna().all()


def test_an_unlisted_region_is_kept_out_of_the_regional_tables():
    rows = [{"region": "Туркестанская область"}, {"region": "Toshkent shahri"}]
    result = quality.audit(listings(rows))
    finding = next(f for f in result.findings if f.code == "place")
    assert finding.affected == 1 and "Туркестанская область" in finding.detail
    assert result.frame["place_ok"].tolist() == [False, True]


def test_yields_outside_the_plausible_band_are_held_back():
    table = pd.DataFrame({"Hudud": ["A", "B", "C"], "Yalpi ko'rsatkich, %":
                          [8.0, 0.4, 61.0]})
    kept, rejected = quality.screen_yields(table, "Yalpi ko'rsatkich, %")
    assert kept["Hudud"].tolist() == ["A"]
    assert rejected["Hudud"].tolist() == ["B", "C"]


# ---------------------------------------------------------------------------
# Uzbek writing conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [(20.3712, "20,37"), (1025.0, "1 025"), (20.0, "20,00"),
                                            (12500.5, "12 500,50"), (None, "—")])
def test_numbers_are_written_the_uzbek_way(value, expected):
    assert style.num(value) == expected


def test_a_rate_of_change_carries_its_sign_but_not_a_percent_sign():
    # "foiz" is written outside the figure so a sentence can inflect it.
    assert style.pct(3.44) == "+3,4"
    assert style.pct(-3.44) == "-3,4"


@pytest.mark.parametrize("suffix,expected", [
    ("i", "2026-yil II choragi"), ("ida", "2026-yil II choragida"),
    ("idan", "2026-yil II choragidan")])
def test_quarters_are_named_in_roman_numerals(suffix, expected):
    assert style.quarter_phrase("2026-Ch2", suffix) == expected
    assert style.quarter_phrase(pd.Period("2026Q2", "Q"), suffix) == expected


def test_a_period_phrase_is_recognised_as_one_fact():
    # The narrative masker protects the whole phrase, so the model cannot pair
    # a year with a quarter the evidence never named.
    assert style.PERIOD_PATTERN.fullmatch("2026-yil II choragida")
    assert style.LOOSE_QUARTER.search("III chorakda narx oshdi")


# ---------------------------------------------------------------------------
# the editorial pass
# ---------------------------------------------------------------------------

def report(*paragraphs):
    return [Section("BO'LIM", [Bullets(list(paragraphs))])]


def codes(content):
    return {issue.code for issue in editorial.review(content)}


def test_a_worn_phrase_is_reported_with_advice():
    issues = editorial.review(report("Narxlar darajalar bo'yicha o'zgardi."))
    assert [issue.code for issue in issues][0] == "phrase"
    assert "darajalar bo'yicha" in issues[0].detail


def test_a_clause_repeated_in_two_paragraphs_is_reported_once_per_paragraph():
    shared = "bozor bir jinsli emas va farq katta"
    issues = editorial.review(report(f"Toshkentda {shared} bo'ldi.",
                                     f"Buxoroda {shared} bo'ldi."))
    assert [issue.code for issue in issues].count("repeat") == 1


def test_two_paragraphs_opening_the_same_way_are_reported():
    assert "opening" in codes(report("Umumiy holda narx oshdi, chunki taklif kam.",
                                     "Umumiy holda narx tushdi, chunki taklif ko'p."))


def test_a_paragraph_that_only_restates_a_number_is_reported():
    assert "flat" in codes(report("Mediana 20,37 mln so'm/m² ni tashkil etdi."))
    # The same figure with an argument attached is not.
    assert "flat" not in codes(report(
        "Mediana 20,37 mln so'm/m², ya'ni o'rta hududdan ikki barobar yuqori."))


def test_a_section_with_nothing_to_show_is_reported_and_not_rendered():
    empty = Section("BO'SH", [Tbl("Jadval", pd.DataFrame()), Text("")])
    assert "blank" in codes([empty])
    assert not editorial.has_content(empty)
    assert editorial.has_content(Section("BOR", [Note("Manba", ["matn"])]))


def test_the_revision_list_groups_issues_by_the_block_they_sit_in():
    content = report("Narxlar darajalar bo'yicha o'zgardi.", "Mediana 20,37 edi.")
    grouped = editorial.revisable(editorial.review(content))
    assert list(grouped) == [(0, 0)]
    assert editorial.summary(editorial.review(content))["issues"] >= 2


# ---------------------------------------------------------------------------
# official statistics
# ---------------------------------------------------------------------------

class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class StubSession:
    """Answers every World Bank series with one dated observation."""
    def __init__(self, values=None, fail=False):
        self.values = values or {}
        self.fail = fail
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        code = url.split("indicator/")[1].split("?")[0]
        rows = [{"date": str(year), "value": value}
                for year, value in sorted(self.values.get(code, {2025: 1.0}).items())]
        return StubResponse([{"page": 1}, rows])


def test_indicators_arrive_dated_and_cited(tmp_path):
    macro = official.gather(cache=tmp_path / "c.csv", session=StubSession(),
                            progress=lambda _: None, today="2026-09-23")
    assert len(macro.observations) == len(official.SERIES)
    assert all(item.period.endswith("-yil") and item.source for item in macro.observations)
    # The indicators no classifieds site carries are named with their publisher.
    assert macro.missing and all(item["source"] for item in macro.missing)


def test_a_second_run_reads_the_cache_instead_of_the_network(tmp_path):
    cache = tmp_path / "c.csv"
    official.gather(cache=cache, session=StubSession(), progress=lambda _: None,
                    today="2026-09-23")
    quiet = StubSession()
    macro = official.gather(cache=cache, session=quiet, progress=lambda _: None,
                            today="2026-09-24")
    assert quiet.calls == 0 and macro.observations


def test_a_failed_fetch_reports_what_is_known_rather_than_raising(tmp_path):
    macro = official.gather(cache=tmp_path / "c.csv", session=StubSession(fail=True),
                            progress=lambda _: None, today="2026-09-23")
    assert not macro.fetched
    assert all(item.source for item in macro.observations)


def test_the_deflator_spans_the_quarters_and_names_what_it_assumed():
    cpi = {2024: 10.0, 2025: 8.0}
    quarters = [pd.Period(q, "Q") for q in ("2024Q1", "2025Q1", "2026Q1")]
    level = official.price_level(cpi, quarters)
    assert level[quarters[0]] == pytest.approx(1.0)
    assert level[quarters[1]] > level[quarters[0]]
    # The current year is published late; carrying the last rate forward is
    # allowed, but the report has to be able to say which year that was.
    assert official.carried_years(cpi, quarters) == [2026]


def test_no_deflator_is_offered_before_the_first_published_year():
    # Nothing can be assumed backwards, so the adjustment is refused instead.
    assert official.price_level({2025: 8.0}, [pd.Period("2020Q1", "Q"),
                                              pd.Period("2025Q1", "Q")]) == {}


# ---------------------------------------------------------------------------
# plain language
# ---------------------------------------------------------------------------

def test_a_unit_is_spelled_out_in_prose_and_says_what_it_is_measured_per():
    assert style.amount(20.37, "mln so'm/m\u00b2") == "20,37 million so'm"
    assert style.amount(118.1, "ming so'm/m\u00b2/oy") == "118,1 ming so'm"
    assert style.per("mln so'm/m\u00b2") == "har bir kvadrat metr uchun"
    assert style.per("ming so'm/m\u00b2/oy") == "har bir kvadrat metr uchun oyiga"


def test_table_notation_inside_a_sentence_is_an_editorial_fault():
    assert "notation" in codes(report("Mediana 20,37 mln so'm/m\u00b2 ni tashkil etdi."))
    assert "notation" in codes(report("O'zgarish +3,4% ni tashkil etdi."))
    # The same finding written in words is not.
    assert "notation" not in codes(report(
        "Har bir kvadrat metr uchun 20,37 million so'm so'ralmoqda, ya'ni "
        "o'rtadagi hududdan ikki barobar ko'p."))


def test_an_undecoded_reference_is_reported_as_its_own_fault():
    issues = editorial.review(report("Narx [[F46]] choragida oshdi."))
    assert "token" in {issue.code for issue in issues}
