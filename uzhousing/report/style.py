"""How the bulletin writes numbers, shares and periods in Uzbek prose.

Uzbek publications separate decimals with a comma and thousands with a space,
and name a quarter in Roman numerals: ``20,37 mln so'm``, ``3,4 foiz``,
``2026-yil II chorak``. The tables keep the compact machine forms — ``20.37``,
``+3.4%``, ``Δ Ch2/Ch1`` — because a table cell is read as a value, not as a
sentence. This module owns the prose forms so the two never drift apart, and so
a figure is written the same way in every section.
"""
from __future__ import annotations

import re

import pandas as pd

#: Roman numerals, as a quarterly review names its quarters.
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

_QUARTER = re.compile(r"(\d{4})[-\s]*(?:Ch|CH|ch|Q)(\d)")


def num(value, decimals: int = 2) -> str:
    """A number as Uzbek prose writes it: ``20,37``, ``1 025``, ``12 500,50``.

    A whole number of a thousand or more keeps no decimals, as the tables print
    it: ``1 025,00 ta e'lon`` is noise, not precision.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return "—"
    if float(value).is_integer() and abs(value) >= 1000:
        decimals = 0
    text = f"{float(value):,.{decimals}f}"
    whole, _, fraction = text.partition(".")
    whole = whole.replace(",", " ")
    return f"{whole},{fraction}" if fraction else whole


def count(value) -> str:
    """A whole count: ``1 025``."""
    return f"{int(value):,}".replace(",", " ")


def pct(value, decimals: int = 1, sign: bool = True) -> str:
    """A rate of change without the percent sign: ``+3,4``, ``-1,2``, ``0,0``.

    The word ``foiz`` is written outside the figure, so a sentence can inflect
    it — ``3,4 foizga oshdi``, ``3,4 foizni tashkil etdi`` — instead of being
    forced around a ``%`` sign.
    """
    if value is None or pd.isna(value):
        return "—"
    text = num(abs(value), decimals)
    if value < 0:
        return f"-{text}"
    return f"+{text}" if sign else text


def quarter_phrase(label, suffix: str = "ida") -> str:
    """A quarter as a reviewer writes it: ``2026-Ch2`` to ``2026-yil II choragida``.

    ``suffix`` is the case ending the sentence needs: ``i`` bare, ``ida`` in,
    ``iga`` to, ``idan`` from, ``igacha`` until.
    """
    if isinstance(label, pd.Period):
        year, quarter = label.year, label.quarter
    else:
        found = _QUARTER.search(str(label).strip())
        if not found:
            return str(label)
        year, quarter = int(found.group(1)), int(found.group(2))
    return f"{year}-yil {ROMAN.get(quarter, quarter)} chorag{suffix}"


def year_phrase(value, suffix: str = "da") -> str:
    """``2024`` to ``2024-yilda``."""
    return f"{int(value)}-yil{suffix}"


#: How a unit is *spoken* in a sentence, as against printed in a table cell.
#: A table column may be headed "Mediana, mln so'm/m²", because a column head is
#: read as a label. The same notation inside a sentence makes the reader stop
#: and decode it, so prose spells the unit out and says what it is measured per.
UNITS = {
    "mln so'm/m²": ("million so'm", "har bir kvadrat metr uchun", 2),
    "ming so'm/m²/oy": ("ming so'm", "har bir kvadrat metr uchun oyiga", 1),
    "USD/m²": ("AQSH dollari", "har bir kvadrat metr uchun", 0),
    "mln so'm": ("million so'm", "", 2),
}

#: What the report calls the figure it reports most often. A median is not an
#: average, and the difference matters, so the word is glossed where it is
#: first used rather than left as jargon.
MEDIAN_SALE = "so'ralayotgan o'rta narx"
MEDIAN_RENT = "so'ralayotgan o'rta ijara haqi"
MEDIAN_GLOSS = ("o'rta narx — e'lonlarning yarmi undan arzon, yarmi qimmat "
                "bo'lgan chegara")


def amount(value, unit: str) -> str:
    """A figure with its unit spelled out: ``20,37 million so'm``."""
    word, _, decimals = UNITS.get(unit, (unit, "", 2))
    return f"{num(value, decimals)} {word}".strip()


def per(unit: str) -> str:
    """What the figure is measured per: ``har bir kvadrat metr uchun``."""
    return UNITS.get(unit, ("", "", 2))[1]


def metric_name(unit: str) -> str:
    """The name of the price this unit carries, in words a reader knows."""
    return MEDIAN_RENT if "oy" in unit else MEDIAN_SALE


#: Month names as Uzbek dates are written, so a policy date reads as a date
#: rather than as three separate figures the model has to keep in order.
MONTHS = ("yanvar", "fevral", "mart", "aprel", "may", "iyun",
          "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr")


def date_phrase(value, suffix: str = "da") -> str:
    """An ISO date as a sentence writes it: ``2025-03-27`` to ``2025-yil 27-martda``."""
    text = str(value)[:10]
    try:
        year, month, day = (int(part) for part in text.split("-"))
        name = MONTHS[month - 1]
    except (ValueError, IndexError):
        return text
    return f"{year}-yil {day}-{name}{suffix}"


#: A written-out date, protected by the narrative masker as one fact for the
#: same reason a quarter is: its parts must not be recombined.
DATE_PATTERN = re.compile(r"\d{4}-yil\s+\d{1,2}-(?:" + "|".join(MONTHS) + r")\w*")

#: A period written the prose way, for the narrative masker to protect as one
#: fact: the year is a figure and the Roman numeral beside it must not drift.
PERIOD_PATTERN = re.compile(r"\d{4}-yil\s+(?:IV|III|II|I)\s+chora[gqk]\w*")

#: A Roman quarter written outside a protected period phrase — a numeral the
#: model chose for itself, which is a period claim the evidence never made.
LOOSE_QUARTER = re.compile(r"\b(?:IV|III|II|I)\s+chora[gqk]", re.IGNORECASE)
