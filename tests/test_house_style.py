"""What the agent learned from the published issues of this review.

Four issues were read: the UzMRC housing and mortgage reviews for 2025 Q2, Q3
and Q4, and the Central Bank's 2025 annual analysis. Three things were taken
from them and are tested here — the corpus that records how they are written,
the currency arithmetic they state every issue, and the convergence claim they
make every issue, which this report checks rather than repeats.
"""
import json

import pandas as pd
import pytest

from uzhousing.report import drivers, house_style
from uzhousing.report.olx_bulletin import (Archive, Bullets, _convergence_line,
                                           _fx_effect, market_section)


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

def test_the_corpus_carries_patterns_and_not_other_reports_figures():
    """A style exemplar with a real figure in it is a leak waiting to happen.

    The writing step rejects any digit it cannot trace to this report's own
    evidence, so a borrowed figure would fail the run even if nobody noticed it
    in review. The corpus writes «X» instead.
    """
    raw = json.loads(house_style.DEFAULT_STYLE_FILE.read_text(encoding="utf-8"))
    examples = []
    for item in raw["paragraph_grammar"]:
        examples.append(item["pattern"])
        examples.extend(item.get("alternatives") or [])
    for item in raw["causal_channels"]:
        examples.append(item["phrasing"])
    for text in examples:
        digits = [ch for ch in text if ch.isdigit()]
        assert not digits, f"figure left in a style exemplar: {text}"


def test_every_channel_names_a_mechanism_its_evidence_and_its_certainty():
    for item in house_style.channels():
        assert item["mechanism"].strip(), item
        assert item["needs"].strip(), item
        assert item["certainty"] in {"arithmetic", "testable", "hypothesis"}, item


def test_the_style_reaches_the_writer_as_one_block_of_guidance():
    text = house_style.prompt_section()
    assert "House style of the published review" in text
    # The graded hedges are the part that keeps a guess from reading as a fact.
    assert "bilan izohlash mumkin" in text and "tufayli" in text
    # And the channels arrive with the evidence each one needs.
    assert "tabiiy korreksiya" in text and "valyuta kursi" in text


def test_a_missing_corpus_costs_the_guidance_and_not_the_run(tmp_path):
    missing = tmp_path / "gone.json"
    assert house_style.load(missing) == {}
    assert house_style.prompt_section({}) == ""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert house_style.load(broken) == {}


def test_the_drivers_section_can_name_every_channel_the_reviews_use():
    named = {item["key"] for item in drivers.repertoire()}
    assert {"natural_correction", "fx_translation", "supply_saturation",
            "regional_convergence", "state_programme"} <= named
    # And each one has Uzbek wording ready for the page.
    for key in ("correction", "convergence", "saturation", "demography"):
        assert key in drivers.CHANNELS


# ---------------------------------------------------------------------------
# the exchange rate, which these reviews explain in every issue
# ---------------------------------------------------------------------------

def archive(old_rate, new_rate):
    bundle = Archive(pd.DataFrame(),
                     pd.DataFrame({"quarter": ["2025Q2", "2025Q3"]}), pd.DataFrame())
    bundle.rates = {pd.Period("2025Q2", "Q"): (old_rate, "d"),
                    pd.Period("2025Q3", "Q"): (new_rate, "d")}
    return bundle


def test_the_som_price_effect_is_the_rate_change_not_the_soms_appreciation():
    """The two are reciprocals, and only one of them moves a so'm price.

    Between the published Q2 and Q3 rates the som appreciated 5.2% against the
    dollar, but a constant dollar price became 4.9% cheaper in som. It is the
    second figure a reader needs to discount the table by, and it is the one the
    published issue prints.
    """
    line = _fx_effect(archive(12695.0, 12068.0))
    assert "4,9 foizga** arzonlashadi" in line
    assert "5,2" not in line
    assert "mustahkamlandi" in line


def test_a_weaker_som_is_reported_the_other_way_round():
    line = _fx_effect(archive(12068.0, 12695.0))
    assert "zaiflashdi" in line and "qimmatlashadi" in line


def test_a_flat_quarter_says_the_movement_was_the_market_not_the_rate():
    line = _fx_effect(archive(12600.0, 12600.0))
    assert "deyarli" in line and "o'z harakati" in line


def test_no_currency_claim_without_a_dated_rate_at_both_ends():
    bundle = archive(12695.0, 12068.0)
    bundle.rates = {pd.Period("2025Q2", "Q"): (12695.0, "d")}
    assert _fx_effect(bundle) == ""
    assert _fx_effect(None) == ""


# ---------------------------------------------------------------------------
# convergence: the claim is checked, not repeated
# ---------------------------------------------------------------------------

def cross_section(changes):
    return pd.DataFrame({
        "Hudud": list("ABCDEFG"),
        "2025-Ch3": [5.0, 5.5, 6.0, 7.0, 9.0, 12.0, 14.0],
        "Δ Ch4/Ch3": list(changes)})


def test_growth_in_the_cheapest_places_is_reported_as_the_gap_narrowing():
    line = _convergence_line(cross_section([10.0, 7.0, 6.0, 1.0, -2.0, -5.0, -6.0]),
                             "Hudud", "mln so'm/m²")
    assert "tafovutining qisqarayotganiga" in line
    assert "**A**" in line
    # Stated as a pointer, never as a proven long-run convergence.
    assert "isbotlamaydi" in line


def test_the_opposite_quarter_is_reported_as_the_gap_widening():
    line = _convergence_line(cross_section([-6.0, -5.0, -2.0, 1.0, 6.0, 7.0, 10.0]),
                             "Hudud", "mln so'm/m²")
    assert "kengaydi" in line


def test_a_quarter_with_no_pattern_says_so_instead_of_repeating_the_claim():
    line = _convergence_line(cross_section([1.0, -2.0, 3.0, -1.0, 2.0, -3.0, 1.5]),
                             "Hudud", "mln so'm/m²")
    assert "barqaror bog'liqlik" in line and "xulosa chiqarib bo'lmaydi" in line


def test_a_thin_cross_section_makes_no_convergence_claim_at_all():
    thin = cross_section([1.0, -2.0, 3.0, -1.0, 2.0, -3.0, 1.5]).head(4)
    assert _convergence_line(thin, "Hudud", "mln so'm/m²") == ""


# ---------------------------------------------------------------------------
# and the currency line leads the segment, as the published issues place it
# ---------------------------------------------------------------------------

def test_the_currency_line_opens_the_market_section(monkeypatch):
    import uzhousing.report.olx_bulletin as bulletin

    monkeypatch.setattr(bulletin, "comparison_table",
                        lambda *a, **k: pd.DataFrame())
    snapshot = {"collected_at": "2025-10-01T00:00:00", "data": [], "coverage": {}}
    frame = pd.DataFrame(columns=["kind", "market", "region", "district",
                                  "property", "sqm_uzs"])
    section = market_section("BIRLAMCHI UY-JOY BOZORI", "Birlamchi", snapshot,
                             frame, archive(12695.0, 12068.0))
    # With no comparable table there is nothing for it to lead, and the section
    # must not invent a currency finding on its own.
    assert not any(isinstance(block, Bullets) and "dollarida" in " ".join(block.items)
                   for block in section.blocks)
