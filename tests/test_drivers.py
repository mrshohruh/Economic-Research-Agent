"""The half of the review that says why the quarter moved.

Two separate promises are tested here. The first is negative: a movement is
never described as the shape of a table — "the indicator in the middle of the
row of regional changes was +0.5 per cent" tells a reader nothing they can
picture, so no sentence the report builds may read that way. The second is
positive: policy and news evidence reaches the page, ranked, attributed and
flagged by how far the evidence actually goes.
"""
import pandas as pd
import pytest

from uzhousing.report import drivers, editorial, style
from uzhousing.report.olx_bulletin import (Bullets, Note, Section, Tbl, TITLES,
                                           _band, _segment_summary,
                                           drivers_section, movement_bullets)
from uzhousing.research.context import PolicyEvent, ResearchFindings, ThemeFinding


# ---------------------------------------------------------------------------
# a movement is described by places and sizes, never by the shape of the table
# ---------------------------------------------------------------------------

#: Wording that describes the distribution instead of the market. Every one of
#: these names no region, no price and nothing a household could act on.
ABSTRACTIONS = ("o'rtada turgan", "qatorining o'rtasi", "o'zgarishlar qatori",
                "markaziy qiymat", "median o'zgarish")


def comparison(changes):
    """A comparison table shaped like the one the bulletin builds."""
    names = [chr(ord("A") + i) for i in range(len(changes))]
    return pd.DataFrame({
        "Hudud": names,
        "2025-Ch2": [10.0 + i for i in range(len(changes))],
        "Δ Ch2/Ch1": [1.0] * len(changes),
        "2025-Ch3": [10.0 + i for i in range(len(changes))],
        "Δ Ch3/Ch2": list(changes)})


@pytest.mark.parametrize("changes", [
    (-4.0, -3.0, -2.0, -1.0, -8.0),   # a broad fall
    (4.0, 3.0, 2.0, 1.0, 8.0),        # a broad rise
    (-6.0, 5.0, -1.0, 2.0, -3.0),     # a quarter that split
])
def test_a_movement_is_told_through_places_not_through_the_distribution(changes):
    joined = " ".join(movement_bullets(comparison(changes), "mln so'm/m²"))
    for phrase in ABSTRACTIONS:
        assert phrase not in joined.lower(), phrase
    # The regions that carry the finding are named, with their own figures.
    assert "**" in joined and "foiz" in joined
    assert "%" not in joined


def test_the_bulk_of_the_moves_is_given_as_a_range_a_reader_can_picture():
    band = _band(pd.Series([-1.0, -2.0, -3.0, -4.0, -12.0]))
    assert "dan" in band and "foizgacha" in band
    # A set that moved by one size is stated as that size, not as a false range.
    assert _band(pd.Series([3.0, 3.0, 3.0])) == "3,0 foiz atrofida"


def test_the_summary_line_names_the_region_that_moved_most():
    table = comparison((-2.0, -1.0, -3.0, -9.0, -1.5))
    line = _segment_summary(table, "mln so'm/m²", "Birlamchi uy-joy")
    assert "**D**" in line and "-9,0" in line
    for phrase in ABSTRACTIONS:
        assert phrase not in line.lower()


def test_the_editor_rejects_the_abstraction_if_the_writer_reintroduces_it():
    section = Section("TEST", [Bullets([
        "Hududiy o'zgarishlar qatorining o'rtasidagi ko'rsatkich +0,5 foiz bo'ldi."])])
    assert "phrase" in {issue.code for issue in editorial.review([section])}


# ---------------------------------------------------------------------------
# policy and news reach the page, attributed and weighted
# ---------------------------------------------------------------------------

def findings():
    found = ResearchFindings()
    found.price_drivers = [
        {"driver": "Imtiyozli ipoteka ajratmalarining qisqarishi", "channel": "credit",
         "evidence": "Byudjet resurslari asosidagi ajratmalar chorak davomida sekinlashdi.",
         "effect": "down", "segment": "primary", "strength": "documented",
         "source": "https://cbu.uz/x", "source_title": "MB choraklik sharhi"},
        {"driver": "Yangi uylarning ko'plab topshirilishi", "channel": "supply",
         "evidence": "Poytaxtda topshirilgan kvartiralar soni oshdi.",
         "effect": "down", "segment": "primary", "strength": "speculative",
         "source": "https://stat.uz/y", "source_title": "Statistika agentligi"},
    ]
    found.policy_events = [PolicyEvent(
        date="2025-03-27", title="PF-63-son Farmon", category="policy",
        summary="Yashil ta'mir dasturi joriy etildi.",
        expected_impact="Ta'mirlash kreditlari ikkilamchi bozor talabini qo'llab-quvvatlaydi.",
        direction="positive", confidence="high", source="https://lex.uz/z")]
    found.themes = [ThemeFinding(key="policy", title="Policy",
                                 summary="Davlat dasturlari birlamchi bozorga qaratilgan.")]
    return found


def test_each_driver_names_its_channel_its_direction_and_how_far_the_evidence_goes():
    section = drivers_section(findings())
    assert section.title == TITLES["drivers"]
    items = section.blocks[0].items
    assert "ipoteka va kredit shartlari orqali" in items[0]
    assert "narxni pastga tortadi" in items[0]
    # A documented measure and a guess must not read alike.
    assert "hujjat bilan qayd etilgan" in items[0]
    assert "tasdiqlanmagan" in items[1]
    assert "MB choraklik sharhi" in items[0]


def test_the_measures_in_force_are_tabled_with_the_date_written_out():
    table = [block for block in drivers_section(findings()).blocks
             if isinstance(block, Tbl)][0].frame
    assert list(table["Sana"]) == ["2025-yil 27-mart"]
    assert table.iloc[0]["Yo'nalishi"] == "narxni oshiruvchi"
    assert table.iloc[0]["Ishonchlilik"] == "yuqori"


def test_every_source_behind_an_explanation_is_listed_for_checking():
    note = [block for block in drivers_section(findings()).blocks
            if isinstance(block, Note)][0]
    listed = " ".join(note.paragraphs)
    assert "https://cbu.uz/x" in listed and "https://stat.uz/y" in listed
    assert "https://lex.uz/z" in listed
    # And the section says plainly that these are read, not measured.
    assert "o'lchangan emas" in listed


def test_a_quarter_with_no_research_says_so_rather_than_inventing_a_reason():
    section = drivers_section(None)
    assert isinstance(section.blocks[0], Note)
    assert "izohlanmadi" in section.blocks[0].paragraphs[0]

    empty = drivers_section(ResearchFindings())
    text = " ".join(empty.blocks[0].paragraphs)
    assert "topilmadi" in text
    assert not [block for block in empty.blocks if isinstance(block, Bullets)]


def test_the_explanation_section_passes_the_editorial_read():
    assert editorial.review([drivers_section(findings())]) == []


# ---------------------------------------------------------------------------
# a policy date survives the narrative round trip as one fact
# ---------------------------------------------------------------------------

def test_a_written_out_date_is_masked_as_a_single_reference():
    from uzhousing.report.olx_narrative import PROSE_MASK
    sentence = f"Farmon {style.date_phrase('2025-03-27')} imzolandi."
    assert PROSE_MASK.findall(sentence) == ["2025-yil 27-martda"]
