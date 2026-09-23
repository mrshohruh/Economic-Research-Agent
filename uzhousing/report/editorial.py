"""The editorial pass: read the finished prose the way a desk editor would.

A report assembled from many sections drifts in a small number of predictable
ways. The same clause appears in four sections, every section opens with the
same sentence shape, a paragraph restates a table cell and stops, a translated
turn of phrase survives into Uzbek, or a block ends up empty. None of that is a
numerical error, so nothing upstream catches it.

This module reads the assembled blocks and reports those faults with the block
they sit in, so the writing step can revise exactly those paragraphs and the
run log can record what was left. It changes nothing on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Turns of phrase to keep out of the report: literal translations of English
#: analytical formulas, and the filler the template used to fall back on. Each
#: is matched case-insensitively against the prose.
BANNED = {
    "darajalar bo'yicha": "ko'rsatkich nomini yozing (mediana taklif narxi, ijara narxi, indeks)",
    "o'sish kengroq tarqaldi": "o'sish qamrovini raqam bilan ayting",
    "alohida o'qilishi lozim": "nega farq qilishini bir jumlada ayting",
    "bevosita tenglashtirilmaydi": "nima uchun taqqoslab bo'lmasligini ayting",
    "bevosita taqqoslanmaydi": "nima uchun taqqoslab bo'lmasligini ayting",
    "joriy kesim": "qaysi sana va qaysi manbalar nazarda tutilganini yozing",
    "darajasini tashkil etdi": "ko'rsatkich nomi bilan yozing",
    "bo'lib qolmoqda": "holatni raqam bilan bayon qiling",
    "e'tiborga olish lozim": "shartni jumla ichida ayting",
    # A distribution described as a distribution. "The indicator in the middle
    # of the row of regional changes was +0.5 per cent" names no place, no
    # price and nothing a reader can act on; it is the shape of the table, not
    # a finding about the market.
    "o'rtada turgan": "hududni nomi bilan ayting yoki o'zgarishlar oralig'ini bering",
    "o'rtasidagi ko'rsatkich": "qaysi hudud va qaysi narx nazarda tutilganini yozing",
    "qatorining o'rtasi": "hududni nomi bilan ayting yoki oraliq bering",
    "o'zgarishlar qatori": "necha hududda qanday o'zgargani va eng kattasi kim ekanini yozing",
    "median o'zgarish": "eng katta siljishni nomi va raqami bilan ayting",
    "markaziy qiymat": "ko'rsatkichni nomi va hududi bilan yozing",
    "taqsimotning o'rtasi": "hududni nomlang yoki o'zgarishlar oralig'ini bering",
}

#: Words that turn a restated number into an argument. A paragraph carrying
#: figures but none of these is describing a table row, not reading it.
ANALYTICAL = (
    "chunki", "shu sababli", "shu bois", "ya'ni", "ammo", "lekin", "biroq",
    "aksincha", "nisbatan", "taqqoslaganda", "shuningdek", "bundan", "bu esa",
    "demak", "shu bilan birga", "ehtimol", "mumkin", "ko'rsatadi", "bildiradi",
    "farq", "sabab", "izohlamaydi", "aniqlamaydi", "tarkib", "ulush", "barobar",
    "sur'at", "tendentsiya", "ta'sir",
)

#: Notation that belongs in a table cell or an axis label, never in a sentence.
#: A reader meeting "mln so'm/m²" mid-sentence stops to decode it; the same
#: figure written "har bir kvadrat metr uchun 20,37 million so'm" simply reads.
NOTATION = (
    ("so'm/m²", "birlikni so'z bilan yozing: har bir kvadrat metr uchun ... so'm"),
    ("USD/m²", "birlikni so'z bilan yozing: har bir kvadrat metr uchun ... dollar"),
    ("m²/oy", "oyiga bir kvadrat metr uchun ... deb yozing"),
    ("%", "foiz so'zini yozing"),
    ("Δ", "o'zgarishni so'z bilan ayting"),
)

#: A reference that failed to decode. It must never reach the page, so the
#: editor reports it as a fault of its own rather than as odd prose.
TOKEN_RESIDUE = re.compile(r"\[\[F?\d*\]?\]?")

#: A paragraph longer than this reads as two paragraphs run together.
MAX_CHARS = 700

#: Words in a repeated run before it counts as a stock phrase.
SHINGLE = 6

_WORD = re.compile(r"[\w'’ʻ]+", re.UNICODE)
_SENTENCE = re.compile(r"[.!?]+\s+")


@dataclass
class Issue:
    code: str
    section: str
    where: tuple
    detail: str
    text: str = ""

    def line(self) -> str:
        return f"{self.section}: {self.detail}"


def _words(text: str) -> list:
    return [word.lower() for word in _WORD.findall(str(text))]


def _prose(content) -> list:
    """Every rewritable paragraph, with the block it belongs to."""
    from .olx_bulletin import Bullets, Text
    found = []
    for si, section in enumerate(content):
        for bi, block in enumerate(section.blocks):
            if isinstance(block, Bullets):
                for index, item in enumerate(block.items):
                    found.append((section.title, (si, bi, index), str(item)))
            elif isinstance(block, Text):
                found.append((section.title, (si, bi, 0), str(block.text)))
    return found


def review(content) -> list:
    """Every editorial fault in the assembled report, newest concern first."""
    issues, seen_shingles, openings = [], {}, {}
    for title, where, text in _prose(content):
        stripped = text.strip()
        if not stripped:
            issues.append(Issue("empty", title, where, "bo'sh xatboshi"))
            continue
        lowered = stripped.lower()
        for phrase, advice in BANNED.items():
            if phrase in lowered:
                issues.append(Issue("phrase", title, where,
                                    f"qolipga aylangan ibora: «{phrase}» — {advice}",
                                    stripped))
        if len(stripped) > MAX_CHARS:
            issues.append(Issue("long", title, where,
                                f"xatboshi {len(stripped)} belgi, {MAX_CHARS} dan uzun",
                                stripped))
        words = _words(stripped)
        opening = " ".join(words[:3])
        if opening:
            first = openings.setdefault(opening, (title, where))
            if first[1] != where:
                issues.append(Issue("opening", title, where,
                                    f"xatboshi «{opening}» bilan boshlangan boshqa "
                                    f"xatboshi ham bor ({first[0]})", stripped))
        for start in range(max(len(words) - SHINGLE + 1, 0)):
            run = " ".join(words[start:start + SHINGLE])
            first = seen_shingles.setdefault(run, (title, where))
            if first[1] != where:
                issues.append(Issue("repeat", title, where,
                                    f"takrorlangan ibora: «{run}» ({first[0]})",
                                    stripped))
                break
        for mark, advice in NOTATION:
            if mark in stripped:
                issues.append(Issue("notation", title, where,
                                    f"jumla ichida jadval belgisi: «{mark}» — {advice}",
                                    stripped))
        if TOKEN_RESIDUE.search(stripped):
            issues.append(Issue("token", title, where,
                                "matnda ochilmagan havola qoldi", stripped))
        if _restates_numbers(stripped):
            issues.append(Issue("flat", title, where,
                                "xatboshi raqamni takrorlaydi, uning ma'nosini "
                                "izohlamaydi", stripped))
    for si, section in enumerate(content):
        if not has_content(section):
            issues.append(Issue("blank", section.title, (si, -1, 0),
                                "bo'limda ko'rsatiladigan blok yo'q"))
    return issues


def _restates_numbers(text: str) -> bool:
    """A paragraph with figures, no argument and nothing to draw from them."""
    if not re.search(r"\d", text):
        return False
    lowered = text.lower()
    if any(word in lowered for word in ANALYTICAL):
        return False
    sentences = [part for part in _SENTENCE.split(text) if part.strip()]
    return len(sentences) <= 2


def has_content(section) -> bool:
    """Whether a section has anything a reader would see on the page."""
    from .olx_bulletin import Bullets, Chart, Note, Tbl, Text
    for block in section.blocks:
        if isinstance(block, Tbl):
            if block.frame is not None and not block.frame.empty:
                return True
        elif isinstance(block, Bullets):
            if any(str(item).strip() for item in block.items):
                return True
        elif isinstance(block, Text):
            if str(block.text).strip():
                return True
        elif isinstance(block, (Note, Chart)):
            return True
    return False


def revisable(issues) -> dict:
    """Issues a rewrite could fix, grouped by the block they sit in."""
    grouped = {}
    for issue in issues:
        if issue.code in ("phrase", "repeat", "opening", "long", "flat", "notation"):
            grouped.setdefault(issue.where[:2], []).append(issue)
    return grouped


def summary(issues) -> dict:
    """What the run log records about the editorial pass."""
    counts = {}
    for issue in issues:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return {"issues": len(issues), "by_code": counts,
            "detail": [issue.line() for issue in issues[:40]]}
