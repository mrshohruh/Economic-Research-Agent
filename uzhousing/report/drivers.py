"""Why the quarter moved: the policy and news half of the bulletin.

The tables say what happened to asking prices. They cannot say why, because an
advert carries no reason. A reader of a quarterly review asks the question
anyway, and a review that refuses to answer it sends them somewhere else.

So the report answers it from a second body of evidence: decrees and government
programmes, Central Bank decisions, construction and mortgage statistics, and
the reporting around them, retrieved for the quarter the data covers and read by
the model in :mod:`uzhousing.research.context`. What comes back is ranked by how
much of the movement each force plausibly accounts for and flagged by how well
the evidence supports it, and this module prints it that way — a documented
measure and a reasonable guess never read alike on the page.

Nothing here computes a price. The section stands beside the tables and is
explicitly an interpretation of them; where the evidence does not reach, the
section says so instead of filling the gap.
"""
from __future__ import annotations

import pandas as pd

from . import house_style, style

#: How a force reaches a price. Naming the channel is what separates an
#: explanation from a coincidence: a measure that never touches credit,
#: supply, income or the exchange rate has no route to the asking price.
#:
#: The list follows the repertoire the published reviews actually use, read off
#: four issues and recorded in ``knowledge/report_style.json``. Those analysts
#: return to the same handful of mechanisms quarter after quarter, which is a
#: feature rather than a rut: a reader who met "tabiiy korreksiya" last quarter
#: knows what is being claimed this quarter.
CHANNELS = {
    "credit": "ipoteka va kredit shartlari orqali",
    "supply": "yangi uy-joy taklifining kengayishi orqali",
    "currency": "valyuta kursi orqali",
    "income": "aholi daromadi va xarid qobiliyati orqali",
    "seasonal": "mavsumiy talab orqali",
    "policy": "davlat ipoteka dasturlari orqali",
    "sentiment": "bozor ishtirokchilarining kutilmalari orqali",
    "correction": "avvalgi yillardagi yuqori o'sishning tabiiy korreksiyasi orqali",
    "convergence": "hududlararo narx tafovutining qisqarishi orqali",
    "saturation": "talabning vaqtinchalik to'yinishi orqali",
    "demography": "demografik o'sish va yangi oilalar soni orqali",
}

#: Which way the force pushes the asking price.
EFFECTS = {
    "up": "narxni yuqoriga suradi",
    "down": "narxni pastga tortadi",
    "mixed": "narxga ikki tomonlama ta'sir qiladi",
    "none": "narxga sezilarli ta'sir ko'rsatmaydi",
}

#: How far the evidence goes. This is the flag that keeps the section honest:
#: a mechanism that is merely plausible is printed as merely plausible.
STRENGTH = {
    "documented": "manbada hujjat bilan qayd etilgan",
    "likely": "mexanizm ma'lum, ammo bu chorak uchun dalil bilvosita",
    "speculative": "faraz sifatida keltirildi, tasdiqlanmagan",
}

DIRECTIONS = {
    "positive": "narxni oshiruvchi",
    "negative": "narxni pasaytiruvchi",
    "mixed": "ikki tomonlama",
    "neutral": "neytral",
}

CONFIDENCE = {"high": "yuqori", "medium": "o'rtacha", "low": "past"}

CATEGORIES = {
    "policy": "Davlat siyosati",
    "monetary": "Pul-kredit siyosati",
    "fiscal": "Byudjet siyosati",
    "regulatory": "Tartibga solish",
    "macro": "Makroiqtisodiyot",
    "demographic": "Demografiya",
}

#: Beyond this the section stops being an explanation and becomes a list.
MAX_DRIVERS = 5
MAX_EVENTS = 8


def _clean(text) -> str:
    return " ".join(str(text or "").split())


def _sentence(text) -> str:
    """A fragment closed off once, whether or not the source closed it."""
    cleaned = _clean(text).rstrip(".;, ")
    return f"{cleaned}." if cleaned else ""


def _driver_bullet(item: dict) -> str:
    """One force, written so the reader can check the reasoning themselves."""
    name = _clean(item.get("driver"))
    channel = CHANNELS.get(str(item.get("channel", "")).lower(), "")
    effect = EFFECTS.get(str(item.get("effect", "")).lower(), "")
    strength = STRENGTH.get(str(item.get("strength", "")).lower(),
                            STRENGTH["speculative"])
    evidence = _clean(item.get("evidence"))
    source = _clean(item.get("source_title")) or _clean(item.get("source"))
    opening = f"**{name}**"
    if channel and effect:
        opening += f" — {channel} {effect}"
    elif effect:
        opening += f" — {effect}"
    parts = [opening + "."]
    if evidence:
        parts.append(evidence if evidence.endswith(("!", "?")) else _sentence(evidence))
    tail = f"Dalil darajasi: {strength}"
    if source:
        tail += f"; manba — {source}"
    parts.append(tail + ".")
    return " ".join(parts)


def bullets(findings) -> list:
    """The ranked explanation, strongest force first."""
    items = [_driver_bullet(item)
             for item in (findings.price_drivers or [])[:MAX_DRIVERS]
             if _clean(item.get("driver"))]
    if items:
        return items
    # No ranked drivers came back, but a policy record still explains part of
    # the quarter; it is printed as background rather than as an attribution.
    return [f"**{_clean(event.title)}** ({style.date_phrase(event.date)}). "
            f"{_sentence(event.summary)} Kutilayotgan ta'sir: "
            f"{_sentence(event.expected_impact) or 'baholanmadi.'} Ushbu chora "
            f"quyidagi jadvalda qayd etilgan, biroq uning aynan shu chorakdagi "
            f"narx harakatiga qo'shgan hissasi o'lchanmadi."
            for event in (findings.policy_events or [])[-3:]]


def events_table(findings) -> pd.DataFrame:
    """The measures in force over the reported window, newest last."""
    rows = []
    for event in (findings.policy_events or [])[-MAX_EVENTS:]:
        rows.append({
            "Sana": style.date_phrase(event.date, ""),
            "Chora": _clean(event.title),
            "Turi": CATEGORIES.get(str(event.category).lower(), _clean(event.category)),
            "Uy-joy narxiga kutilayotgan ta'siri":
                _clean(event.expected_impact) or _clean(event.summary),
            "Yo'nalishi": DIRECTIONS.get(str(event.direction).lower(),
                                         _clean(event.direction)),
            "Ishonchlilik": CONFIDENCE.get(str(event.confidence).lower(),
                                           _clean(event.confidence)),
        })
    return pd.DataFrame(rows)


def repertoire() -> list:
    """The mechanisms this review recognises, for the run log and the prompt.

    Kept here rather than in the corpus file alone so a caller can see what the
    section is able to say without parsing the corpus itself.
    """
    return [{"key": item.get("key"), "name": item.get("name_uz"),
             "certainty": item.get("certainty")} for item in house_style.channels()]


def theme_bullets(findings, keys=("policy", "mortgage", "monetary", "supply")) -> list:
    """The standing background behind the quarter, one paragraph per theme."""
    found = {theme.key: theme for theme in (findings.themes or [])}
    items = []
    for key in keys:
        theme = found.get(key)
        summary = _clean(getattr(theme, "summary", ""))
        if theme is None or not summary:
            continue
        items.append(summary)
    return items


def sources_note(findings, limit: int = 10) -> list:
    """Where every claim in this section came from, so a reader can check it."""
    listed = findings.source_index()[:limit]
    if not listed:
        return []
    return ["Ushbu bo'limdagi baholashlar quyidagi tashqi manbalarga tayanadi: "
            + "; ".join(f"{_clean(item['title'])} ({item['url']})" for item in listed)
            + "."]


def caveat() -> list:
    """What this section is, and what it is not.

    The bulletin's own figures are measured; these explanations are read off
    other people's reporting. Keeping the two apart on the page is the whole
    point of putting them in different sections.
    """
    return [
        "Ushbu bo'lim e'lonlar ma'lumotlaridan emas, tashqi manbalardan — "
        "normativ hujjatlar, Markaziy bank va statistika e'lonlari hamda ommaviy "
        "axborot vositalari xabarlaridan olingan. E'lon narxlari sababni "
        "ko'rsatmaydi, shuning uchun bu yerdagi bog'liqliklar o'lchangan emas, "
        "izohlangan bog'liqliklardir.",
        "Har bir omil yonida dalil darajasi ko'rsatilgan: hujjat bilan "
        "tasdiqlangan chora bilan ehtimoliy izoh bir xil vaznga ega emas.",
    ]
