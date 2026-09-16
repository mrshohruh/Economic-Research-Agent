"""Translate Russian and Uzbek text in the dataset into English automatically.

Uzbek statistical and marketplace data routinely mixes three languages in the
same file: column headers in Uzbek ("narx", "hudud"), category values in
Russian ("евроремонт", "трёхкомнатная"), and English. A report written for an
English-speaking reader should not surface any of the source language, so this
module is the single place that turns a raw token or value into English.

Two tiers, cheapest first:

1. A curated dictionary covers the vocabulary that actually recurs in Uzbek
   housing/macro data (regions, room counts, condition, building type, common
   economic terms). This needs no network call and is used everywhere,
   including places with no LLM configured.
2. Whatever the dictionary cannot resolve — a free-text phrase, an unfamiliar
   category value — is batched into one LLM call per report run, when an LLM
   is available. Nothing is invented: values the LLM cannot translate either
   are returned unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from ..llm import LLM, LLMUnavailable

LOGGER = logging.getLogger(__name__)

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Uzbek Latin script uses these letters/digraphs in words that never occur in
# English text, so their presence is a reliable "this needs translating" flag
# even when there is no dictionary hit.
_UZ_HINT_RE = re.compile(r"(o'|g'|q|x)\w*(li|siz|dagi|ning|lar|ligi)\b", re.IGNORECASE)

# Regions and major cities of Uzbekistan: Uzbek Latin and Russian Cyrillic
# forms both map to the standard English spelling. Place names are kept, not
# invented — this is a spelling standardisation, not a translation.
REGION_MAP: dict[str, str] = {
    "toshkent shahri": "Tashkent City", "toshkent shahar": "Tashkent City",
    "toshkent viloyati": "Tashkent Region", "toshkent": "Tashkent",
    "ташкент город": "Tashkent City", "ташкент": "Tashkent",
    "ташкентская область": "Tashkent Region",
    "samarqand viloyati": "Samarkand Region", "samarqand": "Samarkand",
    "самарканд": "Samarkand", "самаркандская область": "Samarkand Region",
    "buxoro viloyati": "Bukhara Region", "buxoro": "Bukhara",
    "бухара": "Bukhara", "бухарская область": "Bukhara Region",
    "andijon viloyati": "Andijan Region", "andijon": "Andijan",
    "андижан": "Andijan", "андижанская область": "Andijan Region",
    "farg'ona viloyati": "Fergana Region", "farg'ona": "Fergana", "fargona": "Fergana",
    "фергана": "Fergana", "ферганская область": "Fergana Region",
    "namangan viloyati": "Namangan Region", "namangan": "Namangan",
    "наманган": "Namangan", "наманганская область": "Namangan Region",
    "xorazm viloyati": "Khorezm Region", "xorazm": "Khorezm",
    "хорезм": "Khorezm", "хорезмская область": "Khorezm Region",
    "qashqadaryo viloyati": "Kashkadarya Region", "qashqadaryo": "Kashkadarya",
    "кашкадарья": "Kashkadarya", "кашкадарьинская область": "Kashkadarya Region",
    "surxondaryo viloyati": "Surkhandarya Region", "surxondaryo": "Surkhandarya",
    "сурхандарья": "Surkhandarya", "сурхандарьинская область": "Surkhandarya Region",
    "jizzax viloyati": "Jizzakh Region", "jizzax": "Jizzakh",
    "джизак": "Jizzakh", "джизакская область": "Jizzakh Region",
    "navoiy viloyati": "Navoi Region", "navoiy": "Navoi",
    "навои": "Navoi", "навоийская область": "Navoi Region",
    "sirdaryo viloyati": "Syrdarya Region", "sirdaryo": "Syrdarya",
    "сырдарья": "Syrdarya", "сырдарьинская область": "Syrdarya Region",
    "qoraqalpog'iston respublikasi": "Republic of Karakalpakstan",
    "qoraqalpogiston": "Karakalpakstan",
    "каракалпакстан": "Karakalpakstan",
    "andijan": "Andijan", "fergana": "Fergana",
}

# Recurring housing/economics vocabulary, Uzbek and Russian -> English.
# Longer phrases are listed before the single words they contain so a phrase
# match wins over a token-by-token one.
PHRASE_MAP: dict[str, str] = {
    # room counts
    "bir xonali": "1-room", "ikki xonali": "2-room", "uch xonali": "3-room",
    "to'rt xonali": "4-room", "tort xonali": "4-room", "besh xonali": "5-room",
    "bir xona": "1 room", "ikki xona": "2 rooms", "uch xona": "3 rooms",
    "однокомнатная": "1-room", "двухкомнатная": "2-room", "трёхкомнатная": "3-room",
    "трехкомнатная": "3-room", "четырёхкомнатная": "4-room", "четырехкомнатная": "4-room",
    "пятикомнатная": "5-room",
    # condition / repair
    "yangi qurilgan": "newly built", "ta'mirlangan": "renovated", "tamirlangan": "renovated",
    "evro ta'mir": "European-style renovation", "evroremont": "European-style renovation",
    "o'rta ta'mir": "average condition", "yaxshi ta'mir": "good condition",
    "требует ремонта": "needs repair", "косметический ремонт": "cosmetic repair",
    "евроремонт": "European-style renovation", "хорошее состояние": "good condition",
    "среднее состояние": "average condition", "черновая отделка": "shell condition",
    # yes/no, furnished
    "mebel bilan": "furnished", "mebelsiz": "unfurnished",
    "с мебелью": "furnished", "без мебели": "unfurnished",
    "jismoniy shaxs": "private individual", "yuridik shaxs": "business",
    "частное лицо": "private individual", "юридическое лицо": "business",
    # market segment
    "birlamchi bozor": "primary market", "ikkilamchi bozor": "secondary market",
    "первичный рынок": "primary market", "вторичный рынок": "secondary market",
}

WORD_MAP: dict[str, str] = {
    # time
    "sana": "date", "oy": "month", "chorak": "quarter", "yil": "year", "yillik": "annual",
    "kunlik": "daily", "haftalik": "weekly", "oylik": "monthly", "choraklik": "quarterly",
    "дата": "date", "период": "period", "месяц": "month", "квартал": "quarter", "год": "year",
    "годовой": "annual", "ежедневный": "daily", "еженедельный": "weekly", "ежемесячный": "monthly",
    "квартальный": "quarterly",
    # geography
    "hudud": "region", "viloyat": "region", "shahar": "city", "tuman": "district", "rayon": "district",
    "область": "region", "регион": "region", "город": "city", "район": "district",
    # segmentation
    "turi": "type", "sinf": "class", "toifasi": "category", "segment": "segment",
    "тип": "type", "класс": "class", "категория": "category", "сегмент": "segment",
    "xona": "room", "xonali": "room", "комната": "room", "комнат": "rooms",
    # prices / money
    "narx": "price", "narxi": "price", "qiymat": "value", "qiymati": "value",
    "цена": "price", "стоимость": "price", "значение": "value",
    "ijara": "rent", "ijarasi": "rent", "аренда": "rent",
    "valyuta": "currency", "kurs": "rate", "kursi": "rate",
    "валюта": "currency", "курс": "rate",
    # activity
    "bitim": "deal", "sotuv": "sale", "sotuvi": "sale", "savdo": "trade", "tranzaksiya": "transaction",
    "сделка": "deal", "продажа": "sale", "торговля": "trade", "транзакция": "transaction",
    "soni": "count", "miqdori": "quantity", "количество": "count",
    # supply / construction
    "qurilish": "construction", "foydalanishga": "commissioning", "quril": "construction",
    "строительство": "construction", "ввод": "commissioning",
    # finance
    "stavka": "rate", "stavkasi": "rate", "foiz": "percent", "foizi": "percent",
    "ставка": "rate", "процент": "percent",
    "ipoteka": "mortgage", "kredit": "credit", "krediti": "loan",
    "ипотека": "mortgage", "кредит": "credit",
    "maosh": "wage", "daromad": "income", "daromadi": "income",
    "зарплата": "wage", "доход": "income",
    "inflyatsiya": "inflation", "инфляция": "inflation", "дефлятор": "deflator",
    # physical
    "maydon": "area", "maydoni": "area", "площадь": "area",
    "aholi": "population", "aholisi": "population", "население": "population",
    # misc qualifiers
    "o'rtacha": "average", "urtacha": "average", "umumiy": "total", "yalpi": "gross",
    "средний": "average", "средняя": "average", "общий": "total", "валовой": "gross",
    "o'sish": "growth", "o'sishi": "growth", "kamayish": "decline", "kamayishi": "decline",
    "рост": "growth", "снижение": "decline", "спад": "decline",
    "yangi": "new", "eski": "old", "yaxshi": "good", "yomon": "poor", "o'rta": "average",
    "новый": "new", "старый": "old", "хороший": "good", "плохой": "poor",
    "ha": "yes", "yo'q": "no", "yoq": "no", "да": "yes", "нет": "no",
    "erkak": "male", "ayol": "female", "мужской": "male", "женский": "female",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _lookup_phrase(text: str) -> str | None:
    norm = _norm(text)
    if norm in REGION_MAP:
        return REGION_MAP[norm]
    if norm in PHRASE_MAP:
        return PHRASE_MAP[norm]
    return None


def is_translatable(text: Any) -> bool:
    """True when ``text`` looks like Russian or Uzbek rather than English."""
    if not isinstance(text, str) or not text.strip():
        return False
    if CYRILLIC_RE.search(text):
        return True
    norm = _norm(text)
    if norm in REGION_MAP or norm in PHRASE_MAP:
        return True
    tokens = re.findall(r"[a-z']+", norm)
    if any(tok in WORD_MAP for tok in tokens):
        return True
    return bool(_UZ_HINT_RE.search(norm))


def translate_text(text: str) -> tuple[str, bool]:
    """Best-effort dictionary translation. Returns (text, fully_resolved)."""
    if not isinstance(text, str) or not text.strip():
        return text, True

    whole = _lookup_phrase(text)
    if whole is not None:
        return whole, True

    if not is_translatable(text):
        return text, True

    parts = re.split(r"(\s+)", text)
    out_parts = []
    all_resolved = True
    for part in parts:
        if part.isspace() or not part:
            out_parts.append(part)
            continue
        stripped = part.strip(".,;:()[]\"'")
        key = stripped.lower()
        translated = WORD_MAP.get(key)
        if translated is not None:
            if stripped[:1].isupper():
                translated = translated.capitalize()
            out_parts.append(part.replace(stripped, translated))
        else:
            out_parts.append(part)
            if re.search(r"[a-zA-Z]", stripped):
                all_resolved = False
            elif CYRILLIC_RE.search(stripped):
                all_resolved = False
    return "".join(out_parts), all_resolved


def humanize_label(name: str) -> str:
    """Column/metric name -> a readable English label for display.

    Underscores become spaces, known abbreviations are cased conventionally,
    and Russian/Uzbek tokens are translated via the dictionary above.
    """
    text = str(name).replace("_", " ").strip()
    text = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", text)
    translated, _ = translate_text(text)
    fixes = {
        "pct": "%", "avg": "average", "med": "median", "sqm": "m²", "m2": "m²",
        "usd": "USD", "uzs": "UZS", "gdp": "GDP", "cpi": "CPI", "yoy": "YoY", "fx": "FX",
        "bn": "bn", "mn": "mn",
    }
    words = []
    for word in translated.split():
        lowered = word.lower()
        words.append(fixes.get(lowered, word))
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else str(name)


# ---------------------------------------------------------------------------
# LLM fallback for whatever the dictionary can't resolve
# ---------------------------------------------------------------------------
TRANSLATE_SYSTEM = (
    "You translate short labels and category values from a Uzbekistan housing-market "
    "dataset (Uzbek or Russian) into concise English. You translate meaning, not just "
    "transliterate; you never explain or add commentary — only the translation itself."
)


def _llm_translate_batch(values: list[str], llm: LLM) -> dict[str, str]:
    if not values:
        return {}
    numbered = {str(i): v for i, v in enumerate(values)}
    prompt = (
        "Translate each of these dataset values/labels into short, natural English. "
        "Keep proper nouns (place names) as standard English spellings. "
        "Return JSON exactly as {\"translations\": {\"<index>\": \"<english>\", ...}} "
        "with one entry per index below, no omissions.\n\n"
        + "\n".join(f"{i}: {v}" for i, v in numbered.items())
    )
    try:
        result = llm.complete_json(prompt, system=TRANSLATE_SYSTEM, max_tokens=2000)
    except LLMUnavailable as exc:
        LOGGER.info("LLM translation skipped: %s", exc)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("LLM translation failed: %s", exc)
        return {}

    translations = result.get("translations") if isinstance(result, dict) else None
    if not isinstance(translations, dict):
        return {}

    out: dict[str, str] = {}
    for i, original in numbered.items():
        english = translations.get(i) or translations.get(str(i))
        if isinstance(english, str) and english.strip():
            out[original] = english.strip()
    return out


def translate_labels(values: Iterable[Any], llm: LLM | None = None) -> dict[str, str]:
    """Translate a batch of category values, dictionary first then LLM for the rest."""
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        translated, resolved = translate_text(value)
        mapping[value] = translated
        if not resolved and is_translatable(value):
            unresolved.append(value)

    if unresolved and llm is not None and llm.available:
        llm_map = _llm_translate_batch(unresolved, llm)
        mapping.update(llm_map)

    return mapping
