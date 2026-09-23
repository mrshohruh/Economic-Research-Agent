"""Uzbek housing bulletin, laid out like the supplied quarterly market review.

The reference report is a structure and design reference only. Every figure
here is computed from observed adverts: the dated OLX/Uybor snapshot for the
current cross-section, and the local OLX archive for the quarterly history. No
number is carried over from the reference, and the sections it contains that
these sources cannot support are named as unavailable rather than filled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..analysis import quality
from ..ingest.listings import _advert_row, _derive_and_clean, _place_name
from ..ingest.price_history import (CANONICAL_REGIONS, MIN_QUARTER_LISTINGS,
                                    SCOPE_NOTE, build_series, canonical_district,
                                    canonical_region, comparison_table, gap_months,
                                    index_series, live_month, national,
                                    quarter_label, region_label, thin_groups)
from ..research import official
from . import drivers, editorial, layout, style
from .style import num, pct, quarter_phrase

OLX_SOURCE, UYBOR_SOURCE = "OLX.uz", "Uybor.uz"

#: The units the tables are printed in. Prose asks :mod:`style` to spell them
#: out; only table headings and axis labels use the compact form.
SALE_UNIT = "mln so'm/m²"
RENT_UNIT = "ming so'm/m²/oy"

#: Section titles, in one place so the contents page and the sections agree.
TITLES = {
    "summary": "QISQACHA XULOSA",
    "quality": "MA'LUMOTLAR SIFATI TEKSHIRUVI",
    "primary": "BIRLAMCHI UY-JOY BOZORI",
    "secondary": "IKKILAMCHI UY-JOY BOZORI",
    "unknown": "BOZOR TURI KO'RSATILMAGAN SOTUV E'LONLARI",
    "index": "UY-JOY NARXLARI INDEKSI",
    "rent": "IJARA BOZORI",
    "rent_districts": "TOSHKENT SHAHRI TUMANLARIDA IJARA",
    "sale_districts": "TOSHKENT SHAHRI TUMANLARIDA SOTUV",
    "yield": "IJARA RENTABELLIGI (YILLIK)",
    "drivers": "NARX O'ZGARISHLARINING SABABLARI: SIYOSAT VA BOZOR OMILLARI",
    "macro": "MAKROIQTISODIY SHAROIT, QURILISH VA IPOTEKA",
    "mix": "MANBALAR TARKIBI",
    "method": "METODOLOGIYA VA MANBALAR",
    "appendix": "ILOVA: BATAFSIL JADVALLAR",
}

#: Expanded on the contents page, as the reference expands its own.
ABBREVIATIONS = [
    ("OLX", "olx.uz e'lonlar platformasi"),
    ("Uybor", "uybor.uz e'lonlar platformasi"),
    ("MB", "O'zbekiston Respublikasi Markaziy banki"),
    ("USD/UZS", "Markaziy bankning rasmiy dollar kursi"),
    ("m²", "kvadrat metr"),
    ("Ch1-Ch4", "yilning birinchi-to'rtinchi choragi"),
    ("O'rta (median) narx", "e'lonlarning yarmi undan arzon, yarmi qimmat bo'lgan "
     "narx; o'rtacha arifmetik qiymat emas, shuning uchun bir nechta juda qimmat "
     "e'lon uni siljita olmaydi"),
    ("y.e.", "shartli birlik; ushbu hisobotda AQSH dollariga teng deb olindi"),
]


# ---------------------------------------------------------------------------
# section model
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A titled run of blocks. The renderer decides how each block looks."""
    title: str
    blocks: list = field(default_factory=list)
    #: Tables a later section needs, so the summary reads the figures the
    #: segment sections printed rather than recomputing its own.
    stats: dict = field(default_factory=dict)

    def tables(self):
        return [block.frame for block in self.blocks if isinstance(block, Tbl)]


@dataclass
class Text:
    text: str


@dataclass
class Bullets:
    items: list


@dataclass
class Tbl:
    """A table. ``appendix`` sends the detail to the back of the report.

    A section reads better with one chart that carries its finding than with a
    fourteen-row table the reader is left to scan, but the table itself is
    still evidence and still belongs in the document. Flagged tables are moved
    into the appendix, numbered, and pointed at from the note under the chart
    they belong to.
    """
    caption: str
    frame: pd.DataFrame
    empty_note: str = "Ushbu bo'lim uchun yetarli, aniq tasniflangan kuzatuvlar mavjud emas."
    appendix: bool = False


@dataclass
class Chart:
    caption: str
    draw: object


@dataclass
class Note:
    label: str
    paragraphs: list
    tone: str = "source"


@dataclass
class Archive:
    """The archived series, already aggregated, with the rates that price it."""
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    district: pd.DataFrame
    rates: dict = field(default_factory=dict)

    @property
    def quarters(self) -> list:
        """The three most recent quarters: two levels and two changes."""
        found = sorted(self.quarterly["quarter"].unique()) if not self.quarterly.empty else []
        return [pd.Period(q, freq="Q") for q in found[-3:]]

    @property
    def rate_values(self) -> dict:
        return {quarter: rate for quarter, (rate, _) in self.rates.items()}

    @property
    def in_som(self) -> bool:
        """Whether every quarter shown has a dated rate to price it with."""
        return bool(self.quarters) and all(self.rate_values.get(q) for q in self.quarters)

    @property
    def unit(self) -> str:
        return SALE_UNIT if self.in_som else "USD/m²"


# ---------------------------------------------------------------------------
# normalising the live snapshot
# ---------------------------------------------------------------------------

def _uybor_rows(listings, rate):
    """Uybor listings in the bulletin's own row shape.

    ``price`` is per square metre on some listings and per sotka on others, so
    it is only read as a total where the listing says it is one.
    """
    rows = []
    for item in listings or []:
        price, area = item.get("price"), item.get("square")
        try:
            price = float(price)
            area = float(area) if area not in (None, "") else float("nan")
        except (TypeError, ValueError):
            continue
        kind = "rent" if item.get("operationType") == "rent" else "sale"
        if item.get("priceType") == "sqm":
            if not area or area != area:
                continue  # a per-m² price without an area has no total
            price *= area
        elif item.get("priceType") not in (None, "all"):
            continue  # per-sotka and other bases are not a dwelling price
        if kind == "rent" and item.get("pricePeriodUnit") not in (None, "month"):
            continue  # the bulletin reports monthly rent only
        currency = {"usd": "USD", "uzs": "UZS"}.get(
            str(item.get("priceCurrency") or "").lower())
        if currency is None:
            continue  # an unknown currency cannot be put on one scale
        rooms = item.get("room")
        try:
            rooms = float(str(rooms).split("+")[0])
        except (TypeError, ValueError):
            rooms = float("nan")
        rows.append({
            # price/currency, not price_usd: the shared cleaner converts and
            # screens every source the same way, and would overwrite it anyway.
            "listing_id": f"uybor:{item.get('id')}", "kind": kind,
            "property": item.get("property") or "Kvartira",
            "price": price, "currency": currency, "date": item.get("createdAt"),
            "total_area": area, "rooms": rooms,
            "region": item.get("region"), "city": item.get("city"),
            "district": item.get("district"),
            "market": "Birlamchi" if item.get("isNewBuilding") is True
                      else "Ikkilamchi" if item.get("isNewBuilding") is False else "Aniqlanmagan",
            "source": UYBOR_SOURCE})
    return rows


def source_mix(frame):
    """Listings behind the pooled medians, per source and segment.

    The medians in this report pool every source, so the composition is
    reported here: a change in the mix moves a pooled median on its own.
    """
    if "source" not in frame or frame["source"].nunique() < 2:
        return pd.DataFrame()
    table = (frame.groupby(["source", "kind", "property"], observed=True)
             .size().reset_index(name="E'lonlar"))
    table["kind"] = table["kind"].map({"sale": "Sotuv", "rent": "Ijara"})
    return table.rename(columns={"source": "Manba", "kind": "Turi", "property": "Uy turi"})


def normalise(snapshot: dict) -> pd.DataFrame:
    rows = []
    for ad in snapshot["data"]:
        row = _advert_row(ad)
        category = ad.get("collection_category", "")
        if row is None or category not in ("rent_apartment", "sale_apartment", "rent_house", "sale_house"):
            continue
        row["kind"] = "rent" if category.startswith("rent") else "sale"
        row["property"] = "Kvartira" if category.endswith("apartment") else "Hovli"
        row["district"] = _place_name((ad.get("location") or {}).get("district"))
        row["market"] = "Aniqlanmagan"
        for param in ad.get("params", []):
            if param.get("key") in ("market", "market_type"):
                value = param.get("value") or {}
                label = str(value.get("label", "")).lower()
                key = str(value.get("key", "")).lower()
                if key == "primary" or "первич" in label or "birlamchi" in label:
                    row["market"] = "Birlamchi"
                elif key == "secondary" or "вторич" in label or "ikkilamchi" in label:
                    row["market"] = "Ikkilamchi"
        row["source"] = OLX_SOURCE
        rows.append(row)
    rows += _uybor_rows(snapshot.get("uybor"), snapshot["fx"]["rate"])
    if not rows:
        raise ValueError("Tahlil uchun uy-joy e'lonlari mavjud emas")
    frame = pd.DataFrame(rows).drop_duplicates(["listing_id", "kind", "property"])
    # Missing area must remain missing, never prevent valid total-price analysis.
    if "total_area" not in frame:
        frame["total_area"] = float("nan")
    if "rooms" not in frame:
        frame["rooms"] = float("nan")
    collected = len(frame)
    parts = [_derive_and_clean(group, kind, snapshot["fx"]["rate"])[0]
             for kind, group in frame.groupby("kind")]
    frame = pd.concat(parts, ignore_index=True)
    # How many records the price, currency and area screens rejected, so the
    # data-quality audit can account for every record between the snapshot and
    # the tables rather than starting from what survived.
    screened = collected - len(frame)
    if frame.empty:
        raise ValueError("Tekshiruvdan o'tgan narxlar mavjud emas")
    frame["sqm_uzs"] = frame["price_usd"] * snapshot["fx"]["rate"] / frame["total_area"]
    frame["sqm_usd"] = frame["price_usd"] / frame["total_area"]
    # One spelling per region and district, or OLX's Russian and Uybor's Uzbek
    # names would split the same place into two rows of every pooled table.
    frame["region"] = [canonical_region(r, c) for r, c in zip(frame["region"], frame.get("city"))]
    frame["district"] = [canonical_district(d) for d in frame["district"]]
    for col in ("region", "city", "district"):
        frame[col] = frame[col].fillna("Ko'rsatilmagan")
    if "source" not in frame:
        frame["source"] = OLX_SOURCE
    frame.attrs["screened"] = screened
    return frame


def regional_table(frame, kind, market=None, district=False):
    selected = frame[frame.kind == kind]
    if market:
        selected = selected[selected.market == market]
    keys = ["district", "property"] if district else ["region", "property"]
    if district:
        selected = selected[selected.district != "Ko'rsatilmagan"]
    else:
        # A place name the region list does not recognise cannot be ranked
        # against the regions; the audit reports how many listings that is.
        selected = selected[selected.region.isin(CANONICAL_REGIONS)]
    metric = "sqm_uzs"
    selected = selected.dropna(subset=[metric])
    table = selected.groupby(keys, dropna=False)[metric].agg(["count", "median"]).reset_index()
    table["median"] /= 1_000 if kind == "rent" else 1_000_000
    # Thin groups have counts, but no prominently ranked estimate.
    table.loc[table["count"] < 15, "median"] = float("nan")
    table["median"] = table["median"].round(2)
    if not district:
        table["region"] = [region_label(value) for value in table["region"]]
    return table.rename(columns={
        "region": "Hudud", "district": "Tuman", "property": "Uy turi", "count": "E'lonlar",
        "median": "Mediana, ming so'm/m²/oy" if kind == "rent" else "Mediana, mln so'm/m²"})


# ---------------------------------------------------------------------------
# commentary computed from the tables it describes
# ---------------------------------------------------------------------------

def _thousands(value) -> str:
    """A count with space-separated thousands, as the tables print them."""
    return style.count(value)


def _band(values) -> str:
    """Where the bulk of a set of changes sat, written so a reader can picture it.

    The middle of a column of changes is a statistic about a table: "the middle
    of the row of regional changes was +0.5 per cent" names no place, no price
    and nothing a household could act on. The quarter to three-quarter band,
    said as a range, does — it is the size most of the moves actually were.
    """
    magnitudes = values.abs().dropna()
    if magnitudes.empty:
        return ""
    low, high = float(magnitudes.quantile(0.25)), float(magnitudes.quantile(0.75))
    if round(low, 1) == round(high, 1):
        return f"{num(low, 1)} foiz atrofida"
    return f"{num(low, 1)} dan {num(high, 1)} foizgacha"


def movement_bullets(table: pd.DataFrame, unit: str, label="Hudud",
                     segment="uy-joy", scope="hududlar") -> list:
    """Two or three findings read off the comparison table, not a tour of it.

    The table prints every region; repeating it in prose adds nothing. What the
    commentary adds is what the rows mean together: whether the move is shared
    or carried by a few places, whether it continued or turned against the
    previous quarter, and how far apart the dearest and the cheapest market
    now sit. The reader has the rest in the table.
    """
    if table.empty or len(table.columns) < 3:
        return []
    change = table.columns[-1]
    level = table.columns[-2]
    rows = table.dropna(subset=[change])
    if rows.empty:
        return ["Taqqoslanadigan kuzatuv yetarli bo'lmagani uchun bu segmentda "
                "chorakma-chorak o'zgarish hisoblanmadi."]
    falls = rows[rows[change] < 0].sort_values(change)
    rises = rows[rows[change] > 0].sort_values(change, ascending=False)
    noun = str(label).lower()
    metric = f"{style.per(unit)} {style.metric_name(unit)}"

    def named(frame, count=3):
        return ", ".join(f"**{row[label]}** ({pct(row[change])} foiz)"
                         for _, row in frame.head(count).iterrows())

    items = [_movement_opening(rows, falls, rises, change, level, noun, segment,
                               scope, metric, named)]
    turn = _turning_point(table, change, label, noun)
    if turn:
        items.append(turn)
    # Whether the move sat where prices were lowest: the published reviews'
    # standing claim, checked against this quarter rather than repeated.
    converging = _convergence_line(table, label, unit)
    if converging:
        items.append(converging)
    spread = _spread_line(table, level, label, unit, noun)
    if spread:
        items.append(spread)
    return items


def _movement_opening(rows, falls, rises, change, level, noun, segment, scope,
                      metric, named) -> str:
    """The finding itself: direction, how widely it is shared, and by whom.

    The sentence shape follows the data rather than a fixed template, so a
    broad decline, a narrow one and a mixed quarter do not all read alike.
    """
    period = quarter_phrase(level)
    total = len(rows)
    if len(falls) >= max(2, 0.7 * total):
        band = _band(falls[change])
        spread = f" Qolgan {noun}larda arzonlashish {band} bo'ldi." if band else ""
        return (f"{period} {segment} bozorida narxlar asosan pasaydi: "
                f"taqqoslangan {total} ta {noun}dan {len(falls)} tasida "
                f"{metric} tushdi. Eng sezilarli pasayish {named(falls)} da "
                f"qayd etildi.{spread} Arzonlashish bir nechta {noun} bilan "
                f"cheklanmagani uchun uni {scope}ga xos umumiy siljish deb "
                f"o'qish mumkin.")
    if len(rises) >= max(2, 0.7 * total):
        band = _band(rises[change])
        spread = f" Boshqa {noun}larda o'sish {band} yetdi." if band else ""
        return (f"{period} {segment} bozorida narxlar ko'tarildi: {total} ta "
                f"{noun}dan {len(rises)} tasida {metric} oshdi. Eng tez "
                f"qimmatlashish {named(rises)} da bo'ldi.{spread} O'sish "
                f"{scope}ning aksariyatini qamragani uchun bu ayrim "
                f"joylardagi tebranish emas.")
    return (f"{period} {segment} bozori ikkiga bo'lindi: {len(falls)} ta "
            f"{noun}da {metric} pasaydi, {len(rises)} tasida esa ko'tarildi. "
            f"Eng ko'p qimmatlashgani — {named(rises, 2)}, eng ko'p "
            f"arzonlashgani — {named(falls, 2)}. Qarama-qarshi yo'nalishdagi "
            f"bu siljishlar bir-birini so'ndiradi, shuning uchun xaridor uchun "
            f"ahamiyatlisi {scope} bo'yicha umumiy raqam emas, o'zi "
            f"qidirayotgan {noun}dagi narx.")


def _fx_effect(archive) -> str:
    """How much of the quarter's so'm movement is the exchange rate.

    Uzbek listings are posted in dollars, so a som price can fall in a quarter
    when nothing about the dwelling or the market changed — the som simply
    bought more dollars. The published reviews state this every issue, and they
    are right to: without it a reader reads a currency move as a price move.

    This is arithmetic, not a hypothesis, so it is written as fact. It is also
    the one explanation the report can make entirely from its own inputs: the
    dated official rate at each end of the comparison.
    """
    if archive is None or not archive.in_som or len(archive.quarters) < 2:
        return ""
    rates = archive.rate_values
    now, before = archive.quarters[-1], archive.quarters[-2]
    new_rate, old_rate = rates.get(now), rates.get(before)
    if not new_rate or not old_rate:
        return ""
    # The figure quoted is the one that matters for reading the table: what a
    # constant dollar price does to its own so'm equivalent. That is the change
    # in the rate itself, not the som's appreciation against the dollar — the
    # two are reciprocals and quoting one while claiming the other would put a
    # wrong number on a correct sentence.
    passthrough = (new_rate / old_rate - 1) * 100
    if abs(passthrough) < 0.1:
        return ("Chorak davomida so'mning dollarga nisbatan rasmiy kursi deyarli "
                "o'zgarmadi, shuning uchun jadvaldagi so'mdagi harakat kurs "
                "ta'siri emas, taklif narxlarining o'z harakati hisoblanadi.")
    verb = "mustahkamlandi" if passthrough < 0 else "zaiflashdi"
    effect = "arzonlashadi" if passthrough < 0 else "qimmatlashadi"
    return (f"Yurtimizda uy-joy e'lonlari odatda AQSH dollarida joylashtiriladi, "
            f"jadvaldagi narxlar esa chorak oxiridagi rasmiy kurs bo'yicha so'mga "
            f"o'tkazilgan. {quarter_phrase(quarter_label(now))} milliy valyuta "
            f"dollarga nisbatan oldingi chorakka qaraganda {verb}: bir dollarning "
            f"kursi **{pct(passthrough)} foizga** o'zgardi. Shu sababli dollardagi "
            f"narx umuman o'zgarmagan taqdirda ham so'mdagi narx "
            f"**{num(abs(passthrough), 1)} foizga** {effect} — quyidagi "
            f"o'zgarishlarning shunchasi bozorning emas, kursning harakatidir.")


def _convergence_line(table, label, unit) -> str:
    """Whether the quarter's growth sat in the places that were cheapest.

    The published reviews return to this claim every issue — that prices are
    rising where they were lowest, so the gap with the capital is closing. It is
    a claim about the report's own cross-section, which means it can be checked
    rather than asserted: rank the regions by price and by change and see
    whether the two run against each other.

    Saying "this quarter the pattern did not hold" is a finding too, and a more
    useful one than repeating the claim on a quarter that contradicts it.
    """
    change, level = table.columns[-1], table.columns[-2]
    rows = table.dropna(subset=[change, level])
    if len(rows) < 6:
        return ""
    correlation = rows[level].corr(rows[change], method="spearman")
    if pd.isna(correlation):
        return ""
    noun = str(label).lower()
    cheap = rows.nsmallest(3, level)
    dear = rows.nlargest(3, level)
    if correlation <= -0.4:
        names = ", ".join(f"**{row[label]}**" for _, row in cheap.iterrows())
        return (f"O'sish narxi past bo'lgan {noun}larda to'plangan: eng arzon uchta "
                f"{noun} ({names}) o'rtacha {pct(cheap[change].mean())} foiz "
                f"o'zgargan bo'lsa, eng qimmat uchtasi {pct(dear[change].mean())} "
                f"foiz. Bu {noun}lar orasidagi narx tafovutining qisqarayotganiga "
                "ishora qiladi — ammo bir chorakdagi taqsimot uzoq muddatli "
                "yaqinlashuvni isbotlamaydi.")
    if correlation >= 0.4:
        names = ", ".join(f"**{row[label]}**" for _, row in dear.iterrows())
        return (f"Bu chorakda o'sish aksincha qimmat {noun}larda to'plandi "
                f"({names}), arzonlarida esa narx {pct(cheap[change].mean())} foiz "
                f"o'zgargan. Ya'ni {noun}lar orasidagi tafovut qisqarmadi, aksincha "
                "kengaydi.")
    return (f"Narx darajasi bilan chorakdagi o'zgarish o'rtasida barqaror bog'liqlik "
            f"kuzatilmadi: arzon {noun}lar ham, qimmatlari ham har ikki yo'nalishda "
            "harakatlandi. Shu sababli bu chorak uchun tafovutning qisqarishi yoki "
            "kengayishi haqida xulosa chiqarib bo'lmaydi.")


def _turning_point(table, change, label, noun) -> str:
    """Whether this quarter continued the previous one or turned against it.

    The comparison table carries two change columns, so the report can say
    something the levels alone cannot: whether the direction held.
    """
    changes = [column for column in table.columns if str(column).startswith(layout.DELTA)]
    if len(changes) < 2:
        return ""
    before, now = changes[-2], changes[-1]
    rows = table.dropna(subset=[before, now])
    if len(rows) < 3:
        return ""
    turned = rows[(rows[before] > 0) & (rows[now] < 0) |
                  (rows[before] < 0) & (rows[now] > 0)]
    share = len(turned) / len(rows)
    if share >= 0.5:
        names = ", ".join(f"**{row[label]}**" for _, row in turned.head(3).iterrows())
        return (f"O'zgarish yo'nalishi barqaror emas: taqqoslanadigan {len(rows)} ta "
                f"{noun}dan {len(turned)} tasida oldingi chorakdagi harakat teskarisiga "
                f"aylandi ({names}). Ketma-ket ikki chorakdagi bunday almashinuv "
                f"barqaror tendentsiyadan ko'ra e'lonlar tarkibining o'zgarishiga "
                f"ko'proq o'xshaydi va uni faqat taklif ma'lumotlari bilan "
                f"ajratib bo'lmaydi.")
    held = rows[(rows[before] > 0) & (rows[now] > 0) |
                (rows[before] < 0) & (rows[now] < 0)]
    if len(held) >= 0.6 * len(rows):
        direction = "o'sish" if (held[now] > 0).mean() > 0.5 else "pasayish"
        pace = float(held[now].abs().mean() - held[before].abs().mean())
        speed = "tezlashdi" if pace > 0 else "sekinlashdi"
        was, is_now = num(abs(held[before]).mean(), 1), num(abs(held[now]).mean(), 1)
        # Region and district tables sit in the same section, so the two
        # sentences are built differently rather than repeating one shape.
        if noun == "tuman":
            return (f"Poytaxtning {len(held)} ta tumanida {direction} ikkinchi "
                    f"chorak ketma-ket kuzatildi, sur'at esa {was} foizdan "
                    f"{is_now} foizga {speed}. Yo'nalishning saqlanib qolishi buni "
                    "bir martalik tebranish emas, davom etayotgan harakat sifatida "
                    "o'qishga asos beradi.")
        return (f"{direction.capitalize()} {len(held)} ta {noun}da ikki chorak "
                f"ketma-ket takrorlandi ({was} foizdan {is_now} foizga {speed}). "
                "Bir yo'nalishdagi ketma-ket harakat bitta chorakdagi o'zgarishga "
                "qaraganda ishonchliroq belgi hisoblanadi.")
    return ""


def _spread_line(table, level, label, unit, noun) -> str:
    """How far apart the ends of the market sit, and whether that is unusual."""
    levels = table.dropna(subset=[level]).sort_values(level)
    if len(levels) < 3:
        return ""
    cheap, dear = levels.iloc[0], levels.iloc[-1]
    if not cheap[level]:
        return ""
    ratio = dear[level] / cheap[level]
    # The mid-priced place is named, not described. "The region in the middle"
    # is a position in a sorted column; a reader wants to know which one it is.
    middle = levels.iloc[len(levels) // 2]
    basis = style.per(unit)
    if noun == "tuman":
        return (f"Poytaxtning o'zida narxlar {num(ratio, 1)} barobar farq qiladi: "
                f"{basis} **{dear[label]}**da {style.amount(dear[level], unit)} "
                f"so'ralmoqda, **{cheap[label]}**da esa "
                f"{style.amount(cheap[level], unit)}; **{middle[label]}** kabi "
                f"o'rta narxli tumanlarda {style.amount(middle[level], unit)} "
                f"atrofida. Xaridor uchun tuman tanlash uy turini tanlashdan kam "
                "ta'sir qilmaydi.")
    return (f"Hududlar orasidagi tafovut {num(ratio, 1)} barobar: {basis} "
            f"**{dear[label]}**da {style.amount(dear[level], unit)} so'ralsa, "
            f"**{cheap[label]}**da {style.amount(cheap[level], unit)} so'ralmoqda; "
            f"**{middle[label]}** singari o'rta narxli hududlarda "
            f"{style.amount(middle[level], unit)} atrofida. Shu sababli "
            "respublika bo'yicha bitta umumiy raqam ko'pchilik hududdagi "
            "haqiqiy holatni ko'rsatmaydi.")


def _thin_note(quarterly, group, quarter, market, label) -> list:
    """Groups held back for a thin sample, as a line for the source callout.

    It belongs under the table with the rest of the method rather than among
    the findings: it explains a dash in a column, it is not a market movement.
    """
    thin = thin_groups(quarterly, group, quarter, market=market)
    if not thin:
        return []
    listed = ", ".join(f"{region_label(name) if group == 'region_uz' else name} "
                       f"({count} e'lon)" for name, count in sorted(thin.items()))
    return [f"Kuzatuvi {MIN_QUARTER_LISTINGS} tadan kam bo'lgani uchun quyidagi "
            f"{label} medianasi ko'rsatilmadi va jadvalda chiziqcha bilan "
            f"berildi: {listed}."]


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

def _history_chart(series):
    """Draw the monthly series, leaving unobserved months as a visible break."""
    monthly = national(series)
    months = pd.PeriodIndex(monthly["month"], freq="M")
    full = pd.period_range(months.min(), months.max(), freq="M")
    values = monthly.set_index(months)["median_usd_sqm"].reindex(full)

    def draw(ax):
        ax.plot([p.to_timestamp() for p in full], values.values,
                color=layout.HEX["primary"], linewidth=1.8, marker="o", markersize=2.5)
        for start, end in _gap_spans(full, values):
            ax.axvspan(start.to_timestamp(), end.to_timestamp(),
                       color=layout.HEX["fall"], alpha=0.12, zorder=0)
            ax.annotate("Kuzatuv yo'q", xy=(start.to_timestamp(), ax.get_ylim()[1]),
                        xytext=(4, -12), textcoords="offset points",
                        fontsize=7.5, color=layout.HEX["fall"])
        ax.set_ylabel("Mediana, USD/m²")
        ax.set_title("Kvartira sotuvi: oylik mediana (USD/m²)")
        ax.grid(alpha=0.4, axis="y")
    return draw


def _index_chart(levels):
    """The index line, with every point labelled as the reference labels its own."""
    labels = [quarter_label(q) for q in levels["quarter"]]
    values = levels["index"].tolist()

    def draw(ax):
        ax.plot(labels, values, color=layout.HEX["primary"], linewidth=1.8,
                marker="o", markersize=4, markerfacecolor="white",
                markeredgecolor=layout.HEX["primary"])
        for x, y in zip(labels, values):
            ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=layout.HEX["deep"])
        ax.set_ylim(min(values) - 12, max(values) + 12)
        ax.set_ylabel("Indeks")
        ax.set_title(f"Uy-joy taklif narxlari indeksi ({labels[0]} = 100)")
        ax.grid(alpha=0.4, axis="y")
        ax.tick_params(axis="x", rotation=45)
    return draw


def _yield_chart(table):
    column = table.columns[-1]
    rows = table.dropna(subset=[column]).sort_values(column, ascending=True).tail(14)
    names = [" / ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in row)
             for row in rows[[c for c in ("Hudud", "Uy turi", "Xonalar")
                              if c in rows]].values]

    def draw(ax):
        bars = ax.barh(names, rows[column], color=layout.HEX["band"], height=0.68)
        layout.label_bars(ax, bars, rows[column].tolist(), decimals=2, horizontal=True)
        ax.set_xlabel(column)
        ax.set_xlim(0, rows[column].max() * 1.18)
        ax.set_title("Yalpi ijara rentabelligi ko'rsatkichi")
        ax.grid(alpha=0.4, axis="x")
    return draw


def _change_chart(table, label, title, unit):
    """Each place's quarter-on-quarter change, coloured by direction.

    A reader sees the breadth of a move here in one glance — how many bars sit
    on each side of zero — which is the finding the section is about. The
    levels behind it stay in the appendix table.
    """
    change, level = table.columns[-1], table.columns[-2]
    rows = table.dropna(subset=[change]).sort_values(change)

    def draw(ax):
        colours = [layout.HEX["rise"] if value > 0 else layout.HEX["fall"]
                   for value in rows[change]]
        bars = ax.barh(rows[label].tolist(), rows[change], color=colours, height=0.68)
        layout.label_bars(ax, bars, rows[change].tolist(), decimals=1, horizontal=True)
        ax.axvline(0, color=layout.HEX["muted"], linewidth=0.9)
        span = max(rows[change].abs().max(), 1) * 1.35
        ax.set_xlim(-span, span)
        ax.set_xlabel(f"{quarter_phrase(level, 'idagi')} o'zgarish, foiz")
        ax.set_title(title)
        ax.grid(alpha=0.4, axis="x")
    return draw


def _level_chart(table, label, title, unit, top=12):
    """The dearest places of one property type, as bars with their values."""
    value = table.columns[-1]
    rows = (table[table["Uy turi"] == "Kvartira"] if "Uy turi" in table else table)
    rows = rows.dropna(subset=[value]).sort_values(value).tail(top)

    def draw(ax):
        bars = ax.barh(rows[label].tolist(), rows[value], color=layout.HEX["band"],
                       height=0.68)
        layout.label_bars(ax, bars, rows[value].tolist(), decimals=2, horizontal=True)
        ax.set_xlim(0, rows[value].max() * 1.2)
        ax.set_xlabel(unit)
        ax.set_title(title)
        ax.grid(alpha=0.4, axis="x")
    return draw


def som_index(levels):
    """The so'm price series rebased to the first quarter, or None.

    Consumer-price inflation is measured in so'm, so only a so'm-denominated
    price series can be deflated by it. The headline index is built on
    dollar-linked asking prices; deflating that by Uzbek CPI would subtract
    domestic inflation from a price that never carried it.
    """
    if "som_sqm" not in levels or levels["som_sqm"].isna().any():
        return None
    base = levels["som_sqm"].iloc[0]
    if not base:
        return None
    return (levels["som_sqm"] / base * 100).round(1)


def _real_index_chart(levels, deflator, nominal_som):
    """The so'm price series against the same series deflated by consumer prices.

    Nominal prices rise in an economy whose prices all rise. Drawing the two
    lines together answers the question the nominal series cannot: whether
    housing outpaced inflation or merely kept up with it.
    """
    labels = [quarter_label(q) for q in levels["quarter"]]
    nominal = nominal_som.tolist()
    real = [value / deflator[quarter] for value, quarter
            in zip(nominal_som, levels["quarter"])]

    def draw(ax):
        ax.plot(labels, nominal, color=layout.HEX["primary"], linewidth=1.8,
                marker="o", markersize=4, markerfacecolor="white",
                markeredgecolor=layout.HEX["primary"], label="So'mdagi nominal narx")
        ax.plot(labels, real, color=layout.HEX["fall"], linewidth=1.6,
                linestyle="--", marker="s", markersize=3.4,
                label="Inflyatsiyaga tuzatilgan")
        ax.axhline(100, color=layout.HEX["muted"], linewidth=0.8)
        ax.set_ylabel("Indeks")
        ax.set_title(f"So'mdagi taklif narxlari: nominal va real ({labels[0]} = 100)")
        ax.grid(alpha=0.4, axis="y")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7.5)
    return draw


def _gap_spans(full, values):
    """Contiguous runs of unobserved months, as (start, end) periods."""
    spans, start = [], None
    for period in full:
        if pd.isna(values.get(period)):
            start = start or period
        elif start is not None:
            spans.append((start, period))
            start = None
    return spans


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def market_section(title, market, snapshot, frame, archive) -> Section:
    """One market segment: its quarterly history, then today's cross-section."""
    section = Section(title)
    segment = f"{title.split()[0].lower()} uy-joy"
    if archive and archive.quarters and len(archive.quarters) > 1:
        quarters, rates, unit = archive.quarters, archive.rate_values, archive.unit
        method = []
        regions = comparison_table(archive.quarterly, "region_uz", quarters, rates,
                                   market=market)
        if not regions.empty:
            regions["Hudud"] = [region_label(value) for value in regions["Hudud"]]
            section.stats["regions"] = regions
            # The currency line leads: it says how much of everything below is
            # the exchange rate rather than the market, so it belongs before
            # the movement rather than as a footnote after it.
            movement = movement_bullets(regions, unit, segment=segment)
            currency = _fx_effect(archive)
            section.blocks += [
                Bullets(([currency] if currency else []) + movement),
                Chart(f"{segment.capitalize()} bozori: hududlar bo'yicha o'zgarish",
                      _change_chart(regions, "Hudud",
                                    f"{segment.capitalize()}: chorakma-chorak "
                                    "o'zgarish", unit)),
                Tbl(f"Hududlar bo'yicha {segment} narxlari ({unit})", regions,
                    appendix=True)]
            method += _thin_note(archive.quarterly, "region_uz", quarters[-1],
                                 market, "hududlar")
        districts = comparison_table(archive.district, "district", quarters, rates,
                                     market=market, label="Tuman")
        if not districts.empty:
            section.stats["districts"] = districts
            section.blocks += [
                Bullets(movement_bullets(districts, unit, label="Tuman",
                                         segment=segment,
                                         scope="Toshkent shahri tumanlari")),
                Tbl(f"Toshkent shahrida {segment} narxlari ({unit})", districts,
                    appendix=True)]
            method += _thin_note(archive.district, "district", quarters[-1],
                                 market, "tumanlar")
        if section.blocks:
            section.blocks.append(Note("Manba", [
                "Arxivlangan OLX e'lonlaridan ushbu dastur tomonidan hisoblangan. "
                "Faqat kvartira sotuvi e'lonlari; narxlar chorak oxiridagi Markaziy "
                "bank kursi bo'yicha so'mga o'tkazilgan."
                if archive.in_som else
                "Arxivlangan OLX e'lonlaridan hisoblangan. Chorak oxiri uchun rasmiy "
                "kurs olinmagani sababli narxlar dollarda keltirildi.",
                f"Kuzatuvi {MIN_QUARTER_LISTINGS} tadan kam guruhlar medianasi "
                "ko'rsatilmaydi."] + method))
    else:
        section.blocks.append(Text(
            "Chorakma-chorak taqqoslash uchun arxiv berilmadi, shuning uchun bu bo'limda "
            f"{snapshot['collected_at'][:10]} sanasidagi takliflar tahlil qilindi. "
            "Arxiv bilan ishga tushirish: run.py --olx --olx-archive <papka>."))
    historical = bool(section.blocks) and isinstance(section.blocks[0], Bullets)
    snapshot_table = regional_table(frame, "sale", market)
    section.stats["snapshot"] = snapshot_table
    observed = snapshot["collected_at"][:10]
    section.blocks += [
        Text(_today_line(snapshot_table, segment, observed, historical)),
        Tbl(f"{observed} holatiga {segment} takliflari, hududlar bo'yicha "
            "(mln so'm/m²)", snapshot_table, appendix=True),
        Note("Eslatma", ["Bozor turi faqat e'londagi aniq belgi asosida ajratildi; "
                         "yangi ta'mir yoki qurilish yili birlamchi bozor belgisi "
                         "sifatida olinmadi."], tone="note")]
    return section


def _today_line(table, segment, observed, historical) -> str:
    """What the snapshot adds to the quarterly picture, in one sentence."""
    rows = _ranked(table, "Kvartira")
    if rows.empty:
        return (f"{observed} sanasidagi takliflar orasida {segment} bozori bo'yicha "
                "mediana chiqarish uchun yetarli kuzatuv to'planmadi.")
    value = rows.columns[-1]
    listed = int(table[table.columns[2]].sum()) if len(table.columns) > 2 else len(rows)
    top = rows.iloc[0]
    # One region with a median is a single observation, not a range: saying
    # "from 19,33 to 19,33" would dress a thin sample as a distribution.
    spread = (f"kvartira uchun so'ralayotgan o'rta narx hududga qarab "
              f"{style.amount(rows[value].min(), SALE_UNIT)} bilan "
              f"{style.amount(rows[value].max(), SALE_UNIT)} orasida, eng "
              f"yuqorisi {top[table.columns[0]]}da"
              if len(rows) > 1 else
              f"narx hisoblashga yetarli e'lon faqat {top[table.columns[0]]}da "
              f"to'plandi — {style.amount(top[value], SALE_UNIT)}")
    return (f"{observed} sanasida to'plangan {style.count(listed)} ta {segment} "
            f"taklifida har bir kvadrat metr uchun {spread}. Bu bir kunlik "
            "e'lonlar kesimi bo'lib, chorak davomidagi holatni ko'rsatmaydi."
            + ("" if not historical else " Tarixiy qator faqat arxivlangan OLX "
               "kvartira sotuvidan, bu jadval esa OLX va Uybor takliflaridan "
               "tuzilgani uchun ikki qismning sathi bir xil o'lchovda emas."))


def index_section(archive, levels=None, macro=None) -> Section:
    section = Section(TITLES["index"])
    if not archive or archive.quarterly.empty:
        section.blocks.append(Text(
            "Uy-joy narxlari indeksi arxiv qatoridan hisoblanadi. Arxiv berilmagani "
            "uchun indeks ushbu hisobotda keltirilmadi."))
        return section
    if levels is None:
        levels = index_series(archive.quarterly, archive.rate_values)
    if levels.empty or len(levels) < 2:
        section.blocks.append(Text("Indeks uchun yetarli chorak kuzatilmadi."))
        return section
    section.stats["levels"] = levels
    first, last = levels.iloc[0], levels.iloc[-1]
    change = (last["index"] / 100 - 1) * 100
    table = pd.DataFrame({
        "Chorak": [quarter_label(q) for q in levels["quarter"]],
        "Oylar": levels["months"].astype(int),
        "E'lonlar": levels["listings"].astype(int),
        "Indeks": levels["index"],
        f"{layout.DELTA} oldingi chorakka":
            (levels["index"].pct_change() * 100).round(1)})
    if archive.in_som:
        table.insert(2, "Mediana, mln so'm/m²", levels["som_sqm"].round(2))
    else:
        table.insert(2, "Mediana, USD/m²", levels["median_usd_sqm"].round(0))
    if "coverage" in levels:
        table.insert(3, "Qamrov, %", levels["coverage"])
    deflator = official.price_level(macro.cpi if macro else {}, levels["quarter"])
    nominal_som = som_index(levels)
    real = deflator and nominal_som is not None
    section.blocks += [
        Bullets([
            f"Taklif narxlari indeksi {quarter_phrase(quarter_label(last['quarter']))} "
            f"**{num(last['index'], 0)}** ni tashkil etdi: bazaviy "
            f"{quarter_phrase(quarter_label(first['quarter']), 'iga')} nisbatan "
            f"**{pct(change)} foiz**. Indeks qat'iy vaznlarda qurilgani uchun bu "
            "o'zgarish e'lonlar tarkibining siljishi emas, narxlarning o'zgarishi."]
            + (_real_growth(levels, deflator, macro, nominal_som) if real else [])
            + _momentum(levels) + _currency_note(levels, archive)),
        Chart("1-rasm. Taklif narxlari indeksi",
              _real_index_chart(levels, deflator, nominal_som) if real
              else _index_chart(levels)),
        Tbl("Choraklar bo'yicha indeks va mediana", table),
        Note("Eslatma", [
            "Indeks qat'iy vaznlarda qurilgan: har bir hudud va bozor segmentining "
            "vazni butun davr bo'yicha e'lonlar ulushiga teng va choraklar orasida "
            "o'zgarmaydi, shuning uchun bir chorakda e'lonlar oqimi o'zgarishi "
            "indeksni siljitmaydi. Qat'iy savat barcha choraklarda kuzatilgan "
            "qatlamlardan tuzilgani uchun uning bozorni qamrash ulushi jadvalda "
            "alohida ustunda berilgan."
            if levels.attrs.get("weighting") == "fixed" else
            "Barcha choraklarda kuzatilgan umumiy qatlam topilmagani uchun indeks "
            "har chorakning o'z e'lonlar taqsimotida vaznlandi; bunda tarkib "
            "o'zgarishi natijaga ta'sir qilishi mumkin.",
            "Ko'rsatkich taklif narxlaridan qurilgan, aholi soniga vaznlanmagan va "
            "bitim narxlariga asoslanmagan. Rasmiy uy-joy narxlari indeksi boshqa "
            "usulda hisoblanadi, shuning uchun ikki qator bir-birining o'rnini "
            "bosmaydi.", SCOPE_NOTE] + _partial_quarters(levels), tone="note")]
    return section


def _real_growth(levels, deflator, macro, nominal_som) -> list:
    """Whether so'm prices outran consumer prices or merely kept up with them.

    Both sides of the comparison are in so'm: the price series priced at the
    dated official rate, and the consumer-price level the statistics publish.
    """
    if not deflator or macro is None or nominal_som is None:
        return []
    first, last = levels.iloc[0], levels.iloc[-1]
    inflation = (deflator[last["quarter"]] / deflator[first["quarter"]] - 1) * 100
    nominal = nominal_som.iloc[-1] - 100
    real = (nominal_som.iloc[-1] / 100 / deflator[last["quarter"]] - 1) * 100
    verdict = ("uy-joy narxlari umumiy narxlar o'sishidan tez oshgan"
               if real > 1 else
               "uy-joy narxlari umumiy narxlar o'sishidan orqada qolgan"
               if real < -1 else
               "uy-joy narxlari umumiy narxlar o'sishi bilan deyarli bir xil sur'atda "
               "harakatlangan")
    carried = official.carried_years(macro.cpi, levels["quarter"])
    assumed = ("" if not carried else
               " Rasmiy inflyatsiya hali e'lon qilinmagan "
               + ", ".join(style.year_phrase(year, "") for year in carried)
               + " uchun oxirgi ma'lum yillik sur'at qo'llanildi.")
    return [f"So'mdagi narxlar shu davrda **{pct(nominal)} foizga** ko'tarilgan, "
            f"iste'mol narxlari esa **{pct(inflation)} foizga**; real hisobda "
            f"**{pct(real)} foiz** qoladi — {verdict}. Hisob yillik inflyatsiyani "
            f"choraklarga teng taqsimlashga asoslangan taxmin.{assumed}"]


def _momentum(levels) -> list:
    """How the last year of the index moved, and where it turned.

    The index table already prints every quarter; what the commentary adds is
    the shape of the recent path and the quarters that broke it.
    """
    steps = levels.assign(change=levels["index"].pct_change() * 100).dropna(subset=["change"])
    if len(steps) < 2:
        return []
    recent = steps.tail(4)
    last = recent.iloc[-1]
    earlier = recent["change"].iloc[:-1].mean()
    # A tenth of a point either way is not a change of pace; calling it one
    # would put a finding where the data has none.
    pace = ("deyarli o'zgarmadi" if abs(last["change"] - earlier) < 0.3
            else "tezlashdi" if last["change"] > earlier else "sekinlashdi")
    falls = steps[steps["change"] < 0]
    items = [f"O'sish sur'ati {pace}: {quarter_phrase(quarter_label(last['quarter']))} "
             f"indeks oldingi chorakka nisbatan **{pct(last['change'])} foiz**, "
             f"undan avvalgi uch chorakda esa o'rtacha {pct(earlier)} foiz o'zgargan. "
             + (f"Butun qator davomida pasayish faqat {len(falls)} chorakda "
                f"({quarter_phrase(quarter_label(falls.sort_values('change').iloc[0]['quarter']))} "
                f"eng chuquri, {pct(falls['change'].min())} foiz) kuzatilgan, "
                "shuning uchun qatorni o'sish yo'nalishidagi harakat deb o'qish mumkin."
                if not falls.empty else
                "Qator davomida birorta chorakda pasayish qayd etilmagan.")]
    return items


def _currency_note(levels, archive) -> list:
    """Separate the dollar movement from the exchange-rate movement.

    The index is built on dollar prices, so a reader comparing it with the
    so'm levels in the same table sees two different numbers for the same
    period. The difference is the exchange rate, and saying so is cheaper than
    letting it be read as a second price movement.
    """
    if not archive or not archive.in_som or "som_sqm" not in levels:
        return []
    priced = levels.dropna(subset=["som_sqm"])
    if len(priced) < 2:
        return []
    first, last = priced.iloc[0], priced.iloc[-1]
    dollars = (last["median_usd_sqm"] / first["median_usd_sqm"] - 1) * 100
    som = (last["som_sqm"] / first["som_sqm"] - 1) * 100
    return [f"{quarter_phrase(quarter_label(first['quarter']), 'idan')} "
            f"{quarter_phrase(quarter_label(last['quarter']), 'igacha')} median narx "
            f"dollarda **{pct(dollars)} foiz**, so'mda esa **{pct(som)} foiz** "
            "o'zgargan. Ikki raqam orasidagi farq rasmiy kurs o'zgarishi bo'lib, "
            "uni ikkinchi narx harakati sifatida o'qish xato bo'ladi."]


def _partial_quarters(levels) -> list:
    """Name the quarters the archive observed for fewer than three months.

    A quarter standing on one month is not comparable with a full one, and the
    difference is invisible in the index unless it is said.
    """
    partial = [(quarter_label(q), int(m)) for q, m in
               zip(levels["quarter"], levels["months"]) if 0 < m < 3]
    if not partial:
        return []
    listed = ", ".join(f"{quarter_phrase(label, 'i')} ({months} oy)"
                       for label, months in partial)
    return [f"Arxiv quyidagi choraklarni to'liq kuzatmagan: {listed}. Bu choraklar "
            "uch oylik choraklar bilan bir xil asosda taqqoslanmaydi."]


def history_section(series, snapshot):
    """The monthly trend and what the breaks in it do and do not allow."""
    if series is None or getattr(series, "empty", True):
        return Section("TARIXIY TAQQOSLASH", [Text(
            "Ushbu hisobot bir yig'ish sanasidagi kesimdir. E'lon joylashtirilgan sana "
            "oldingi davrdagi narx kuzatuvi hisoblanmaydi. Shu sababli joriy e'lonlardan "
            "choraklik o'sish yoki tarixiy uy-joy indeksi hisoblanmadi."), Text(
            "Har bir muvaffaqiyatli yig'ish alohida sanali faylda saqlanadi. To'liq "
            "choraklik tahlil uchun davrlar bo'yicha taqqoslanadigan qamrov va yig'ish "
            "jadvali kerak.")])
    monthly = national(series)
    first, last = monthly["month"].iloc[0], monthly["month"].iloc[-1]
    gaps = gap_months(series)
    section = Section("TARIXIY TAQQOSLASH: OYLIK QATOR")
    annual = annual_history(series)
    items, method = _annual_bullets(annual), _partial_years(annual)
    if gaps:
        spans = _gap_spans(pd.period_range(first, last, freq="M"),
                           monthly.set_index(pd.PeriodIndex(monthly["month"], freq="M"))["median_usd_sqm"]
                           .reindex(pd.period_range(first, last, freq="M")))
        listed = ", ".join(f"{a}–{(b - 1)}" for a, b in spans)
        method.append(
            f"Qatorda {len(gaps)} oy uchun kuzatuv mavjud emas ({listed}). Bu oylar "
            "uchun ma'lumot to'ldirilmadi va grafikda chiziq uzilgan holda "
            "ko'rsatilgan; uzilishdan keyingi taqqoslash ikki nuqta orasidagi "
            "o'zgarish bo'lib, uzluksiz tendentsiya emas.")
        # The comparison spans the most recent break, which is the one standing
        # between the archive and today, not an older one inside the archive.
        gap_start, resumed = spans[-1]
        before = monthly[monthly["month"] < gap_start]
        after = monthly[monthly["month"] >= resumed]
        if not before.empty and not after.empty:
            previous, current = before.iloc[-1], after.iloc[-1]
            items.append(
                f"Oxirgi kuzatuvda ({current['month']}) kvartiraning bir kvadrat "
                f"metri uchun o'rta narx "
                f"**{_thousands(round(current['median_usd_sqm']))} dollar**, "
                f"uzilishdan oldingi oxirgi oyga ({previous['month']}, "
                f"{_thousands(round(previous['median_usd_sqm']))} dollar) nisbatan "
                f"**{pct((current['median_usd_sqm'] / previous['median_usd_sqm'] - 1) * 100)} "
                "foiz**. Oraliqdagi oylar kuzatilmagani uchun bu farq qachon va "
                "qanday sur'atda to'plangani ma'lum emas — u bir necha yilga "
                "tarqalgan harakatning yig'indisi bo'lishi mumkin.")
            if current["listings"] < previous["listings"] / 10:
                method.append(
                    f"Joriy oy {_thousands(current['listings'])} ta e'londan, "
                    f"taqqoslanayotgan tarixiy oy esa "
                    f"{_thousands(previous['listings'])} ta e'londan hisoblangan. "
                    "Kichikroq tanlanma hududlar va uy turlari bo'yicha boshqacha "
                    "tarkibga ega bo'lishi mumkin, shuning uchun farqning bir qismi "
                    "narx harakati emas, tarkib o'zgarishi bo'lishi mumkin.")
    section.blocks += [
        Bullets(items),
        Chart("2-rasm. Kvartira sotuvi: oylik median narx", _history_chart(series)),
        Tbl("Yillar bo'yicha median narx va o'zgarish", annual, appendix=True),
        Note("Manba", [
            "Tarixiy qiymatlar arxivdagi e'lonlardan, oxirgi qiymat esa shu "
            "yig'ishdan olingan. Ikkalasi ham bir xil usulda tozalangan: y.e. USDga "
            "teng, so'mdagi e'lonlar chiqarilgan (tarixiy kurs arxivda yo'q), narx "
            "1 000–5 000 000 USD, maydon 10–1 000 m²."] + method)]
    return section


def _partial_years(annual) -> list:
    """Years the archive observed for fewer than twelve months.

    The yearly change reads as a full-year movement unless the years that are
    not full are named, and the first and last year of a series rarely are.
    """
    if annual is None or annual.empty or "Oylar" not in annual:
        return []
    partial = [f"{int(row['Yil'])} ({int(row['Oylar'])} oy)"
               for _, row in annual.iterrows() if 0 < row["Oylar"] < 12]
    if not partial:
        return []
    return ["Yillik jadvalda quyidagi yillar to'liq kuzatilmagan: "
            + ", ".join(partial) + ". Ular to'liq yillar bilan bir xil asosda "
            "taqqoslanmaydi."]


def _annual_bullets(annual) -> list:
    """The long view the yearly table carries: level, pace and where it turned."""
    if annual is None or annual.empty or len(annual) < 2:
        return []
    value, change = annual.columns[-2], annual.columns[-1]
    rows = annual.dropna(subset=[value])
    first, last = rows.iloc[0], rows.iloc[-1]
    total = (last[value] / first[value] - 1) * 100
    years = int(last["Yil"]) - int(first["Yil"])
    annualised = ((last[value] / first[value]) ** (1 / years) - 1) * 100 if years else total
    verb = "ko'tarildi" if total > 0 else "tushdi"
    items = [f"Kvartiraning bir kvadrat metri uchun so'ralgan o'rta narx "
             f"{style.year_phrase(first['Yil'], 'dagi')} {_thousands(first[value])} "
             f"dollardan {style.year_phrase(last['Yil'], 'da')} "
             f"**{_thousands(last[value])} dollarga** {verb}: butun davr uchun "
             f"**{pct(total)} foiz**, har yili o'rtacha {pct(annualised)} foizdan."]
    paced = rows.dropna(subset=[change])
    if len(paced) > 1:
        fastest = paced.loc[paced[change].idxmax()]
        slowest = paced.loc[paced[change].idxmin()]
        spread = fastest[change] - slowest[change]
        items.append(
            f"O'sish yillar bo'yicha juda notekis taqsimlangan: "
            f"{style.year_phrase(fastest['Yil'])} **{pct(fastest[change])} foiz**, "
            f"{style.year_phrase(slowest['Yil'])} esa **{pct(slowest[change])} foiz** — "
            f"ikki yil orasidagi farq {num(spread, 1)} foiz punkt. Bunday tarqoqlikda "
            "o'rtacha yillik sur'at alohida yillardagi holatni deyarli tushuntirmaydi; "
            "qator nima uchun shunday harakatlangani e'lon ma'lumotlaridan aniqlanmaydi.")
    return items


def annual_history(series):
    """Listing-weighted yearly median USD/m2, with the change on the year before."""
    monthly = national(series)
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly.assign(year=[p.year for p in monthly["month"]],
                             _w=monthly["median_usd_sqm"] * monthly["listings"])
    rows = monthly.groupby("year", observed=True).agg(
        months=("month", "nunique"), listings=("listings", "sum"), _w=("_w", "sum")).reset_index()
    rows["median"] = (rows["_w"] / rows["listings"]).round(0)
    rows["change"] = (rows["median"].pct_change() * 100).round(1)
    # Left numeric so the renderer prints an em dash for the base year, as it
    # does for every other suppressed figure.
    table = rows[["year", "months", "listings", "median", "change"]].copy()
    return table.rename(columns={"year": "Yil", "months": "Oylar", "listings": "E'lonlar",
                                 "median": "Mediana, USD/m²",
                                 "change": f"{layout.DELTA} oldingi yilga"})


def yield_table(frame):
    """Gross rental-yield proxy on matched region, property and room strata.

    The stratum is keyed on the canonical region rather than the city: the
    sources spell city names in different alphabets and Uybor often sends
    none, which would split one stratum into several and compare a region
    against itself. The capital is already a region of its own.
    """
    groups = (frame.dropna(subset=["rooms", "sqm_usd"])
              .groupby(["region", "property", "rooms", "kind"])["sqm_usd"]
              .agg(["count", "median"]))
    rows = []
    for key, row in groups.iterrows():
        if key[-1] != "rent" or row["count"] < 15:
            continue
        sale_key = key[:-1] + ("sale",)
        if sale_key not in groups.index or groups.loc[sale_key, "count"] < 15:
            continue
        rows.append({"Hudud": region_label(key[0]), "Uy turi": key[1],
                     "Xonalar": int(key[2]),
                     "Yalpi ko'rsatkich, %": round(1200 * row["median"]
                                                   / groups.loc[sale_key, "median"], 2)})
    return pd.DataFrame(rows)


def _ranked(table, prop):
    """Rows of one property type that carry a median, dearest first."""
    if table is None or table.empty or "Uy turi" not in table:
        return pd.DataFrame()
    value = table.columns[-1]
    rows = table[table["Uy turi"] == prop].dropna(subset=[value])
    return rows.sort_values(value, ascending=False)


def _levels_sentence(table, unit, label="Hudud", prop="Kvartira", lead="", count=3):
    """What the spread across places means, not a walk down the column.

    A cross-section table carries no change column, so the finding it supports
    is how far apart the ends of the market sit and how much of the market the
    dearest places account for — not a list of every row in order.
    """
    rows = _ranked(table, prop)
    if rows.empty:
        return ""
    value = rows.columns[-1]
    noun = str(label).lower()
    dear, cheap = rows.iloc[0], rows.iloc[-1]
    middle = rows[value].median()
    lead = lead.rstrip() + " " if lead else ""
    if len(rows) < 3 or not cheap[value]:
        return (f"{lead}**{dear[label]}** {style.amount(dear[value], unit)} bilan "
                f"oldinda; narx hisoblashga yetarli kuzatuvga ega {noun} soni "
                f"({len(rows)} ta) kam bo'lgani uchun tarqoqlik baholanmadi.")
    ratio = dear[value] / cheap[value]
    above = int((rows[value] > middle).sum())
    # The closing clause changes with the shape of the distribution, so two
    # sections built from the same helper do not end with the same sentence.
    if ratio >= 2.5:
        takeaway = (f"Bunday tafovutda respublika bo'yicha bitta o'rtacha "
                    f"ko'rsatkich {noun}larning ko'pchiligi uchun ma'noga ega emas.")
    elif above <= len(rows) / 3:
        takeaway = (f"Taqsimot yuqoriga qarab cho'zilgan: bir nechta qimmat "
                    f"{noun} umumiy manzarani o'ziga tortadi.")
    else:
        takeaway = (f"Narxlar {noun}lar orasida nisbatan tekis taqsimlangan, "
                    "shuning uchun chekka qiymatlar emas, o'rta qism bozorni "
                    "tavsiflaydi.")
    return (f"{lead}{style.per(unit)} **{dear[label]}**da "
            f"{style.amount(dear[value], unit)}, **{cheap[label]}**da esa "
            f"{style.amount(cheap[value], unit)} so'ralmoqda — farq "
            f"{num(ratio, 1)} barobar. {len(rows)} ta {noun}dan {above} tasi "
            f"{style.amount(middle, unit)}lik o'rta chegaradan yuqorida. {takeaway}")


def _segment_summary(table, unit, segment):
    """One line on a segment's quarter, for the executive summary."""
    if table is None or table.empty or len(table.columns) < 3:
        return ""
    change, level, label = table.columns[-1], table.columns[-2], table.columns[0]
    rows = table.dropna(subset=[change])
    levels = table.dropna(subset=[level]).sort_values(level)
    if rows.empty or len(levels) < 2:
        return ""
    dear = levels.iloc[-1]
    falling = int((rows[change] < 0).sum())
    # The single largest move, named. A summary that reports the middle of a
    # column of changes tells the reader nothing they can picture or act on.
    moved = rows.reindex(rows[change].abs().sort_values(ascending=False).index)
    lead_row = moved.iloc[0]
    if falling >= 0.7 * len(rows):
        shape, meaning = "narxlar keng ko'lamda pasaydi", ("harakat butun segmentga "
                                                           "tegishli")
    elif falling <= 0.3 * len(rows):
        shape, meaning = "narxlar keng ko'lamda oshdi", "harakat butun segmentga tegishli"
    else:
        shape, meaning = ("narxlar har xil yo'nalishda harakatlandi",
                          "segment bo'yicha yagona xulosa chiqarib bo'lmaydi")
    direction = "pasaydi" if lead_row[change] < 0 else "oshdi"
    return (f"{segment} bozorida {quarter_phrase(level)} {shape}: "
            f"{len(rows)} ta hududdan **{falling} tasida** so'ralayotgan o'rta narx "
            f"pasaydi — ya'ni {meaning}. Eng kuchli siljish "
            f"**{lead_row[label]}**da: narx bir chorakda **{pct(lead_row[change])} "
            f"foizga** {direction}. Narxi eng baland hudud **{dear[label]}**: "
            f"{style.per(unit)} {style.amount(dear[level], unit)}.")


def summary_section(snapshot, frame, archive, segments, levels, rent, yields,
                    macro=None) -> Section:
    """The two or three findings the report stands on, before how it was built.

    Not a précis of every section: a reader who stops here should leave with
    what moved, how broadly, and what it means for affordability — the rest of
    the document is where the detail belongs.
    """
    section = Section(TITLES["summary"])
    if snapshot.get("synthetic"):
        section.blocks.append(Note("SINOV NAMUNASI", [
            "Barcha raqamlar sun'iy ma'lumotlardan olingan; haqiqiy bozor natijalari emas."
        ], tone="note"))
    unit = archive.unit if archive else SALE_UNIT
    items = []
    if levels is not None and not levels.empty and len(levels) > 1:
        first, last = levels.iloc[0], levels.iloc[-1]
        step = (last["index"] / levels["index"].iloc[-2] - 1) * 100
        deflator = official.price_level(macro.cpi if macro else {}, levels["quarter"])
        nominal_som = som_index(levels)
        real = ""
        if deflator and nominal_som is not None:
            # In so'm on both sides: a dollar-linked price carries no domestic
            # inflation, so deflating it by the Uzbek consumer-price index
            # would remove something it never contained.
            real_change = (nominal_som.iloc[-1] / 100 / deflator[last["quarter"]] - 1) * 100
            real = (f" So'mdagi narxlardan iste'mol narxlari o'sishi chegirilsa, "
                    f"real o'zgarish **{pct(real_change)} foiz**ni tashkil etadi.")
        items.append(
            f"{quarter_phrase(quarter_label(last['quarter']))} taklif narxlari indeksi "
            f"**{num(last['index'], 0)}**: oldingi chorakka nisbatan "
            f"**{pct(step)} foiz**, bazaviy "
            f"{quarter_phrase(quarter_label(first['quarter']), 'iga')} nisbatan "
            f"**{pct((last['index'] / 100 - 1) * 100)} foiz**.{real}")
    for segment, table in segments.items():
        line = _segment_summary(table, unit, segment)
        if line:
            items.append(line)
    flats = _ranked(rent, "Kvartira")
    if not flats.empty and yields is not None and not yields.empty:
        column, value = yields.columns[-1], flats.columns[-1]
        items.append(
            f"Ijara bozorida so'ralayotgan o'rta haq **{top_label(flats)}**da eng baland "
            f"(har bir kvadrat metr uchun oyiga "
            f"{style.amount(flats.iloc[0][value], RENT_UNIT)}), taqqoslanadigan "
            f"qatlamlarda yalpi rentabellik esa **{num(yields[column].min())}–"
            f"{num(yields[column].max())} foiz** oralig'ida. Bu ko'rsatkich sotuv "
            "narxining ijara daromadiga nisbatan qanchalik yuqori turganini o'lchaydi: "
            "past rentabellik uy-joyning ijara daromadidan ko'ra boshqa sabablarga "
            "ko'ra qimmatlashayotganiga ishora qiladi, biroq sababni bu ma'lumotlar "
            "aniqlamaydi.")
    elif not flats.empty:
        value = flats.columns[-1]
        items.append(
            f"Ijara bozorida so'ralayotgan o'rta haq **{top_label(flats)}**da eng baland: "
            f"har bir kvadrat metr uchun oyiga "
            f"{style.amount(flats.iloc[0][value], RENT_UNIT)}. Ijara oylik, sotuv esa "
            "bir martalik to'lov bo'lgani uchun ikki ko'rsatkich faqat rentabellik "
            "orqali bog'lanadi.")
    section.blocks.append(Bullets(items))
    # Both sources, or the total would be smaller than the rows it explains.
    collected = len(snapshot["data"]) + len(snapshot.get("uybor") or [])
    coverage = [
        "Hisobotdagi narxlar o'rta (median) qiymat sifatida berilgan: "
        f"{style.MEDIAN_GLOSS}. Sotuv narxi har bir kvadrat metr uchun million "
        "so'mda, ijara haqi esa oyiga har bir kvadrat metr uchun ming so'mda "
        "keltiriladi.",
        f"Kuzatuv sanasi: {snapshot['collected_at'][:10]}. Ikkala manbadan yig'ilgan "
        f"{_thousands(collected)} ta yozuvdan {_thousands(len(frame))} ta "
        "takrorlanmagan, narxi yaroqli e'lon tahlilga kiritildi.",
        "Kvartira va hovli takliflari alohida ko'rsatiladi; ular bir xil bozor emas."]
    if archive and archive.quarters:
        coverage.insert(1, "Chorakma-chorak jadvallar "
                        f"{quarter_phrase(quarter_label(archive.quarters[0]), 'idan')} "
                        f"{quarter_phrase(quarter_label(archive.quarters[-1]), 'igacha')} "
                        "bo'lgan davrni qamrab oladi.")
    else:
        coverage.insert(1, "Arxiv berilmagani uchun chorakma-chorak taqqoslash va "
                        "uy-joy narxlari indeksi hisoblanmadi.")
    section.blocks.append(Note("Qamrov", coverage))
    return section


def top_label(rows) -> str:
    """The name in the first column of the first row of a ranked table."""
    return str(rows.iloc[0][rows.columns[0]])


def sections(snapshot, frame, history=None, archive=None, audit=None, macro=None,
             research=None) -> list:
    """Every section of the bulletin, in the order the reference sets them out.

    The audit runs first and the rest of the report is built from the frame it
    returns, so a suspect record is never behind a headline figure. Detailed
    tables are flagged for the appendix as they are built and moved there once
    every section exists.
    """
    archive = archive or (Archive(history, pd.DataFrame(), pd.DataFrame())
                          if history is not None and not getattr(history, "empty", True)
                          else None)
    if audit is None:
        audit = quality.audit(frame, screened=int(frame.attrs.get("screened", 0)))
    frame = audit.frame
    result = [quality_section(audit, frame)]

    segments = {}
    for title, market, name in ((TITLES["primary"], "Birlamchi", "Birlamchi uy-joy"),
                                (TITLES["secondary"], "Ikkilamchi", "Ikkilamchi uy-joy")):
        section = market_section(title, market, snapshot, frame, archive)
        segments[name] = section.stats.get("regions")
        result.append(section)

    unknown = regional_table(frame, "sale", "Aniqlanmagan")
    result.append(Section(TITLES["unknown"], [
        Bullets(_unknown_bullets(unknown, frame)),
        Tbl("Turi noma'lum sotuv takliflari (mln so'm/m²)", unknown, appendix=True)]))

    levels = (index_series(archive.quarterly, archive.rate_values)
              if archive is not None and not archive.quarterly.empty else pd.DataFrame())
    result.append(index_section(archive, levels, macro))
    result.append(history_section(history, snapshot))

    rent = regional_table(frame, "rent")
    rent_districts = regional_table(frame, "rent", district=True)
    rent_blocks = [Bullets([
        _levels_sentence(rent, RENT_UNIT,
                         lead="Kvartira ijarasining hududiy tarqoqligi katta:"),
        _levels_sentence(rent_districts, RENT_UNIT, label="Tuman",
                         lead="Toshkent shahri ichida ham bir xil emas:")])]
    if not rent.empty:
        rent_blocks.append(Chart(
            "Hududlar bo'yicha kvartira ijarasi",
            _level_chart(rent, "Hudud", "Kvartira ijarasi: median narx",
                         "ming so'm/m²/oy")))
    rent_blocks += [
        Tbl("Hududlar bo'yicha ijara narxlari (ming so'm/m²/oy)", rent, appendix=True),
        Tbl("Toshkent shahri tumanlarida ijara narxlari (ming so'm/m²/oy)",
            rent_districts, appendix=True),
        Note("Eslatma", [
            "Faqat uzoq muddatli ijara toifalari olindi; kunlik ijara kiritilmadi. "
            "Arxivda ijara e'lonlari bo'lmagani uchun ijara bo'yicha chorakma-chorak "
            "dinamika hisoblanmadi.",
            "Tuman kesimi faqat Toshkent shahri uchun beriladi: boshqa hududlarda "
            "e'lonlar tuman nomini izchil ko'rsatmaydi."], tone="note")]
    result.append(Section(TITLES["rent"], rent_blocks))

    sale_districts = regional_table(frame, "sale", district=True)
    district_blocks = [Bullets([
        _levels_sentence(sale_districts, SALE_UNIT, label="Tuman",
                         lead="Poytaxt ichidagi farq hududlararo farqdan kam emas:")])]
    if not sale_districts.empty:
        district_blocks.append(Chart(
            "Toshkent tumanlarida kvartira sotuvi",
            _level_chart(sale_districts, "Tuman",
                         "Toshkent tumanlari: kvartira sotuvining median narxi",
                         "mln so'm/m²")))
    district_blocks += [
        Tbl("Toshkent shahri tumanlarida sotuv takliflari (mln so'm/m²)",
            sale_districts, appendix=True),
        Note("Eslatma", [
            "Ushbu jadval birlamchi, ikkilamchi va turi noma'lum takliflarni "
            "birlashtiradi, shuning uchun uning sathi segment jadvallaridan farq "
            "qiladi."], tone="note")]
    result.append(Section(TITLES["sale_districts"], district_blocks))

    yields, rejected = quality.screen_yields(yield_table(frame),
                                             "Yalpi ko'rsatkich, %")
    yield_blocks = [Bullets(_yield_bullets(yields, rejected))]
    if yields is not None and not yields.empty:
        yield_blocks.append(Chart("3-rasm. Yalpi ijara rentabelligi", _yield_chart(yields)))
    yield_blocks += [
        Tbl("Taqqoslanadigan qatlamlar bo'yicha yalpi ko'rsatkich", yields,
            appendix=True),
        Note("Eslatma", [
            "Yalpi ko'rsatkich = 12 × median oylik ijara (USD/m²) / median sotuv narxi "
            "(USD/m²) × 100. Bir xil hudud, uy turi va xonalar soni solishtiriladi; "
            "har ikki guruhda kamida 15 ta kuzatuv talab qilinadi va "
            f"{quality.YIELD_BAND[0]:.0f}–{quality.YIELD_BAND[1]:.0f} foiz oralig'idan "
            "chiqqan qatlamlar mos kelmaydigan taqqoslash sifatida olib tashlanadi.",
            "Shahar nomi manbalar o'rtasida izchil emas, shuning uchun qatlam hudud "
            "darajasida olindi. Bu aynan bir mulkning daromadliligi emas; bo'sh turish, "
            "soliq va ta'mirlash xarajatlari chegirilmagan."], tone="note")]
    result.append(Section(TITLES["yield"], yield_blocks))

    result.append(macro_section(macro, levels))
    # The explanation sits after the measured sections and before the method,
    # so a reader meets the movement first and the reasons for it second.
    result.append(drivers_section(research))

    mix = source_mix(frame)
    if not mix.empty:
        result.append(Section(TITLES["mix"], [
            Bullets(_mix_bullets(mix)),
            Tbl("Manbalar bo'yicha e'lonlar soni", mix)]))

    result.insert(0, summary_section(snapshot, frame, archive, segments, levels,
                                     rent, yields, macro))
    result.append(method_section(snapshot, archive, audit, macro))
    return _with_appendix(result)


def quality_section(audit, frame) -> Section:
    """What the screens found, before any figure built on them is read."""
    flagged = audit.flagged
    checks = len([item for item in audit.findings if item.code != "screened"])
    opening = (f"Tahlilga kirgan {style.count(len(frame))} ta e'lon quyidagi "
               f"{checks} ta tekshiruvdan o'tkazildi: hudud nomining ro'yxatga "
               "mosligi, manbalar o'rtasidagi va manba ichidagi takrorlanish, "
               "xonalar soni, sotuv va ijara tasnifi hamda m² narxining "
               "taqsimotdagi o'rni. Guruh kattaligi alohida qoida bilan boshqariladi: "
               "kuzatuvi 15 tadan kam guruhda mediana chiqarilmaydi.")
    if not flagged:
        opening += (" Hech bir tekshiruv shubhali yozuv aniqlamadi, shuning uchun "
                    "quyidagi jadvallar to'liq tanlanmaga tayanadi.")
    else:
        worst = max(flagged, key=lambda item: item.affected)
        opening += (f" Eng ko'p yozuv «{worst.check.lower()}» tekshiruvida ajratildi "
                    f"({style.count(worst.affected)} ta). Ajratilgan yozuvlar "
                    "medianalarga kiritilmadi, lekin e'lonlar sanog'ida qoldi, "
                    "shuning uchun jadvaldagi e'lonlar soni va mediana bir xil "
                    "yozuvlar to'plamiga tayanmasligi mumkin.")
    blocks = [Bullets([opening] + audit.lines()[:3]),
              Tbl("O'tkazilgan tekshiruvlar va natijalar", audit.table())]
    if audit.serious:
        blocks.append(Note("Diqqat", [
            "Quyidagi tekshiruvlar tanlanmaning sezilarli qismiga tegdi va natijalarni "
            "o'qishda hisobga olinishi kerak: "
            + "; ".join(item.check.lower() for item in audit.serious) + "."],
            tone="note"))
    return Section(TITLES["quality"], blocks)


def drivers_section(research) -> Section:
    """What moved the quarter, from sources outside the advert data.

    The section is written even when the research step came back empty: a
    reader who is told the question was not answered can go and answer it, a
    reader shown nothing assumes it was never asked.
    """
    if research is None:
        return Section(TITLES["drivers"], [Note("Manba", [
            "Siyosat va yangiliklar tahlili ushbu ishga ulanmagan holda "
            "bajarildi, shuning uchun narx harakatining sabablari izohlanmadi. "
            "Tahlil WEB_RESEARCH sozlamasi yoqilganda va tarmoq ulanishi "
            "mavjud bo'lganda hisobotga qo'shiladi."], tone="note")])

    blocks = []
    items = drivers.bullets(research)
    background = drivers.theme_bullets(research)
    if not items and not background:
        blocks.append(Note("Manba", [
            "Chorak bo'yicha siyosat va bozor yangiliklari izlandi, biroq "
            "narx harakatini izohlaydigan ishonchli manba topilmadi. Sabablar "
            "izohsiz qoldirildi: dalilsiz izoh jadvaldagi raqamga qiymat "
            "qo'shmaydi."], tone="note"))
        return Section(TITLES["drivers"], blocks)
    if items:
        blocks.append(Bullets(items))
    table = drivers.events_table(research)
    if not table.empty:
        blocks.append(Tbl("Hisobot davrida kuchda bo'lgan chora-tadbirlar", table))
    if background:
        blocks.append(Bullets(background))
    notes = drivers.caveat() + drivers.sources_note(research)
    blocks.append(Note("Eslatma", notes, tone="note"))
    return Section(TITLES["drivers"], blocks)


def macro_section(macro, levels) -> Section:
    """The variables no classifieds site carries, from official statistics."""
    if macro is None or (not macro.observations and not macro.missing):
        return Section(TITLES["macro"], [Note("Manba", [
            "Makroiqtisodiy ko'rsatkichlar olinmadi: tarmoq ulanishi mavjud "
            "bo'lmagani uchun rasmiy manbalarga murojaat qilinmadi."], tone="note")])
    items = []
    inflation = macro.get("Iste'mol narxlari inflyatsiyasi")
    income = macro.get("Aholi jon boshiga YaMD (Atlas usuli)")
    credit = macro.get("Banklarning o'rtacha kredit stavkasi")
    if inflation:
        items.append(
            f"Uy-joy narxlari umumiy narx darajasi bilan birga harakatlanadi: "
            f"{inflation.period} iste'mol narxlari **{num(inflation.value, 1)} foizga** "
            "oshgan. Yuqoridagi nominal o'sish shu fonda o'qilishi kerak — undan "
            "pastdagi har qanday o'sish real hisobda pasayishni bildiradi.")
    if income and credit:
        items.append(
            f"To'lovga qobiliyat tomonidan qaraganda, aholi jon boshiga yillik daromad "
            f"{income.period} **{style.count(income.value)} AQSH dollari**, banklarning "
            f"o'rtacha kredit stavkasi esa {credit.period} **{num(credit.value, 1)} "
            "foiz**. Bunday stavkada ipoteka to'lovi daromadning katta qismini "
            "egallaydi, shuning uchun narxlarning o'sishi qarz hisobiga emas, "
            "jamg'arma va o'tkazmalar hisobiga moliyalashtirilayotgan bo'lishi mumkin; "
            "bu ma'lumotlar bunday taxminni tasdiqlay olmaydi.")
    if macro.missing:
        names = ", ".join(item["name"].lower() for item in macro.missing[:4])
        items.append(
            f"Quyidagi ko'rsatkichlar ushbu hisobotda hisoblanmadi: {names}. Ular "
            "e'lonlar platformalarida yo'q va faqat rasmiy manbalarda — Markaziy bank "
            "va Statistika agentligi nashrlarida — e'lon qilinadi; manzillari quyidagi "
            "eslatmada keltirilgan.")
    blocks = [Bullets(items or ["Rasmiy ko'rsatkichlar ro'yxati bo'sh."])]
    table = macro.table()
    if not table.empty:
        blocks.append(Tbl("Rasmiy manbalardagi makroiqtisodiy ko'rsatkichlar", table))
    if macro.missing:
        blocks.append(Note("Manba", [
            "Hisobotga kirmagan ko'rsatkichlar va ularni nashr etuvchi manbalar: "
            + "; ".join(f"{item['name']} — {item['source']}"
                        + (f" ({item['url']})" if item.get("url") else "")
                        for item in macro.missing[:7]) + ".",
            "Ushbu qiymatlar knowledge/official_indicators.json fayliga kiritilsa, "
            "keyingi hisobotda o'z manbasi bilan birga jadvalga qo'shiladi."],
            tone="note"))
    return Section(TITLES["macro"], blocks)


def _unknown_bullets(unknown, frame) -> list:
    """How large the unclassified group is, and whether it biases the split."""
    sales = frame[frame["kind"] == "sale"]
    share = (sales["market"] == "Aniqlanmagan").mean() * 100 if len(sales) else 0
    lead = (f"Sotuv e'lonlarining **{num(share, 1)} foizida** bozor turi "
            "ko'rsatilmagan. Bu guruh birlamchi yoki ikkilamchi bozorga taxmin bilan "
            "qo'shilmadi: shunday qilinsa, ikki segmentning medianasi o'lchangan "
            "narxdan ko'ra tasniflash qoidasiga bog'liq bo'lib qolardi."
            + (" Guruhning kattaligini hisobga olsak, segment medianalari butun "
               "sotuv bozorini emas, uning turi aniq ko'rsatilgan qismini tavsiflaydi."
               if share > 25 else ""))
    return [lead, _levels_sentence(unknown, SALE_UNIT,
                                   lead="Guruh ichida narxlar keng tarqalgan:")]


def _mix_bullets(mix) -> list:
    """What the source composition does to a pooled median."""
    total = int(mix["E'lonlar"].sum())
    by_source = mix.groupby("Manba")["E'lonlar"].sum().sort_values(ascending=False)
    largest = by_source.index[0]
    return [
        f"Medianalar barcha manbalar bo'yicha birlashtirilgan: jami {style.count(total)} "
        f"ta e'londan **{num(100 * by_source.iloc[0] / total, 1)} foizi** "
        f"{largest} ulushiga to'g'ri keladi.",
        "Manbalar bir xil e'lonlar to'plamiga ega emas, shuning uchun ikki yig'ish "
        "orasidagi farqning bir qismi narx harakati emas, manbalar nisbatining "
        "o'zgarishi bo'lishi mumkin. Shu sababli chorakma-chorak indeks qat'iy "
        "vaznlarda hisoblanadi, joriy jadvallar esa bitta sanaga tegishli.",
        "Bir e'lon ikki saytda joylashtirilgan bo'lsa, u narx, maydon, xona soni va "
        "joylashuvi bo'yicha aniqlanib, bir marta hisobga olinadi; umumiy "
        "identifikator bo'lmagani uchun barcha takrorlanishlar aniqlanmasligi mumkin."]


#: Printed once at the head of the appendix rather than under every table.
APPENDIX_NOTE = (
    "Jadvallardagi guruhlar uylarning joylashuvi, maydoni va holati bo'yicha bir xil "
    "emas, shuning uchun ikki guruh orasidagi farq sof narx farqi emas. Kuzatuvi "
    "yetarli bo'lmagan guruhlarda mediana o'rniga chiziqcha turadi.")


def _number_figures(result) -> None:
    """Number every figure in the order the reader meets it.

    A caption that names its own number in one section and not in another is a
    typographic accident the reader notices before the finding.
    """
    number = 0
    for section in result:
        for block in section.blocks:
            if isinstance(block, Chart):
                number += 1
                caption = str(block.caption)
                if "-rasm." in caption:
                    caption = caption.split("-rasm.", 1)[1].strip()
                block.caption = f"{number}-rasm. {caption}"


def _with_appendix(result) -> list:
    """Move the flagged detail tables to a numbered appendix at the back.

    The chart above stays where the argument is made; the table follows it in
    the appendix with a number, and the section's own note says where it went.
    """
    _number_figures(result)
    appendix, index = [], 0
    for section in result:
        moved = []
        for block in list(section.blocks):
            if isinstance(block, Tbl) and block.appendix:
                if block.frame is None or block.frame.empty:
                    section.blocks.remove(block)
                    continue
                index += 1
                block.caption = f"A{index}-jadval. {block.caption}"
                section.blocks.remove(block)
                appendix.append(block)
                moved.append(f"A{index}")
        if moved:
            line = ("Batafsil jadval ilovada: " if len(moved) == 1
                    else "Batafsil jadvallar ilovada: ") + ", ".join(moved) + "."
            notes = [block for block in section.blocks if isinstance(block, Note)]
            if notes:
                notes[-1].paragraphs = list(notes[-1].paragraphs) + [line]
            else:
                section.blocks.append(Note("Eslatma", [line], tone="note"))
    if appendix:
        result.append(Section(TITLES["appendix"],
                              [Note("Eslatma", [APPENDIX_NOTE], tone="note")] + appendix))
    return result


def _yield_bullets(yields, rejected=None) -> list:
    """What the yields say about pricing, not a walk along the column."""
    if yields is None or yields.empty:
        return ["Bir vaqtning o'zida yetarli ijara va sotuv kuzatuviga ega qatlam "
                "topilmagani uchun yalpi rentabellik hisoblanmadi."]
    column = yields.columns[-1]
    rows = yields.sort_values(column, ascending=False)
    payback = 100 / rows[column].median()

    def named(frame):
        return ", ".join(
            f"**{row['Hudud']}** ({row['Uy turi'].lower()}, {int(row['Xonalar'])} xona — "
            f"{num(row[column])} foiz)" for _, row in frame.iterrows())

    items = [
        f"Taqqoslanadigan {len(rows)} ta qatlamda yalpi rentabellik "
        f"**{num(rows[column].min())}** dan **{num(rows[column].max())} foizgacha**, "
        f"medianasi {num(rows[column].median())} foiz. Bu sotib olingan uy-joy o'z "
        f"narxini ijara hisobiga taxminan {num(payback, 0)} yilda qoplashini "
        "bildiradi — xarajatlar chegirilmagan holda.",
        f"Yuqori chekkada {named(rows.head(2))}, quyi chekkada esa "
        f"{named(rows.tail(2).iloc[::-1])} turibdi. Qatlamlar orasidagi bunday farq "
        "sotuv narxi bilan ijara narxi hamma joyda bir xil nisbatda "
        "shakllanmasligini ko'rsatadi; farqning sababini e'lon ma'lumotlari "
        "aniqlab bermaydi."]
    if rejected is not None and not rejected.empty:
        items.append(
            f"{len(rejected)} ta qatlam hisobdan chiqarildi: ularning ko'rsatkichi "
            f"{quality.YIELD_BAND[0]:.0f}–{quality.YIELD_BAND[1]:.0f} foiz oralig'idan "
            "tashqarida bo'lib, bu ijara va sotuv e'lonlari bir xil turdagi uy-joyga "
            "tegishli emasligidan darak beradi.")
    return items


#: Every caveat that governs the whole report, in one box. A caveat repeated in
#: each section stops being read; one box the reader can return to, and section
#: notes that carry only what is specific to their own table, is read once.
CAVEATS = [
    "Barcha ko'rsatkichlar e'lon (so'ralgan) narxlaridan hisoblangan. Bitim narxi "
    "odatda savdolashuvdan keyin shakllanadi, shuning uchun bu yerdagi darajalar "
    "bozor narxining yuqori chegarasi sifatida o'qilishi to'g'riroq bo'ladi.",
    "Chorakma-chorak dinamika arxivlangan OLX kvartira sotuvi e'lonlaridan, sanaga "
    "oid jadvallar esa OLX va Uybor takliflaridan tuzilgan. Ikki qism bir xil "
    "qamrovga ega emas, shuning uchun ularning sathi emas, yo'nalishi taqqoslanadi.",
    "Ijara ko'rsatkichlari oylik to'lov, sotuv ko'rsatkichlari esa bir martalik narx. "
    "Ular faqat rentabellik orqali bog'lanadi; jadvallarni yonma-yon qo'yib "
    "taqqoslash noto'g'ri xulosaga olib keladi.",
    "Guruh kuzatuvi 15 tadan (chorak jadvallarida "
    f"{MIN_QUARTER_LISTINGS} tadan) kam bo'lsa, mediana ko'rsatilmaydi va jadvalda "
    "chiziqcha bilan beriladi. Bunday guruh haqida hech qanday xulosa chiqarilmaydi.",
    "Hisobot tavsifiy: u nima o'zgarganini o'lchaydi, nima uchun o'zgarganini emas. "
    "Sabab-oqibat bog'liqligi e'lon ma'lumotlaridan aniqlanmaydi va matndagi "
    "izohlar faqat tekshirilishi kerak bo'lgan taxminlardir.",
]


def method_section(snapshot, archive, audit=None, macro=None) -> Section:
    rate = snapshot["fx"]
    items = [
        "Manbalar: " + ", ".join(snapshot.get("sources")
                                 or [snapshot.get("source") or "https://www.olx.uz"])
        + f". Yig'ish vaqti (UTC): {snapshot['collected_at']}.",
        f"Yig'ish sanasidagi hisob-kitoblar uchun Markaziy bankning rasmiy kursi "
        f"olindi: 1 USD = {num(rate['rate'])} so'm, kurs sanasi {rate['date']}. "
        f"Manba: {rate['source']}.",
        "y.e. (shartli birlik) USDga teng deb qabul qilindi. Boshqa valyutalar "
        "chiqarildi. Sotuv uchun 1 000–5 000 000 USD, ijara uchun 10–50 000 USD "
        "oralig'idagi narxlar olindi. Maydon 10–1 000 m² bo'lmasa, m² narxi "
        "hisoblanmadi.",
        "Bir yig'ish ichida takrorlangan e'lon identifikatorlari chiqarildi. Turli "
        "identifikator bilan qayta joylangan ayni uylar qolishi mumkin.",
        "Hudud va tuman nomlari manbalar o'rtasida bitta yozuvga keltirildi; "
        "snapshotda asl yozuv saqlanadi.",
        "Hisob-kitoblar dastur orqali bajarildi; matn o'zbek tilidagi shablon asosida "
        "yozildi. Sabab-oqibat xulosalari yoki bozor prognozi ishlab chiqilmadi."]
    if archive and archive.rates:
        dated = ", ".join(f"{quarter_label(q)}: {num(rate)} so'm ({date})"
                          for q, (rate, date) in sorted(archive.rates.items())
                          if q in archive.quarters)
        items.append(
            "Chorak jadvallari uchun chorak oxiridagi rasmiy kurs ishlatildi — "
            f"{dated}. Chorakning oxirgi kuni dam olish yoki bayram bo'lsa, o'sha kunda "
            "amalda bo'lgan oxirgi e'lon qilingan kurs olindi.")
    elif archive:
        items.append("Chorak oxiri uchun rasmiy kurs olinmadi, shuning uchun tarixiy "
                     "jadvallar dollarda keltirildi; so'mdagi qiymat taxmin qilinmadi.")
    items.append(
        "Namuna sifatida berilgan choraklik sharh faqat tuzilma va ko'rinish uchun "
        "asos bo'ldi. Undan birorta ham raqam ko'chirilmadi.")
    # Say why each category stopped: a run that read everything on offer is a
    # different claim from one cut short by a page limit.
    reasons = {"exhausted": "sayt boshqa e'lon bermaguncha to'liq o'qildi",
               "repeated_page": "sahifalar takrorlangunicha o'qildi",
               "depth_limit": "sayt chuqurroq sahifalarni bermadi",
               "page_limit": "sahifa chegarasi qo'yildi",
               "partitioned": "toifa hudud va narx bo'yicha bo'lib to'liq o'qildi"}
    unknown_reason = "yig'ish cheklangan"

    def line(item):
        name = item.get("category") or item.get("source")
        scale = (f"{item['queries']} so'rov" if item.get("queries")
                 else f"{item.get('pages_requested') or item.get('requests')} sahifa")
        # Where the marketplace states its own total, print the share collected
        # rather than leaving the reader to assume the sample is everything.
        share = ""
        if item.get("available"):
            share = f" ({item['listings']} / saytdagi {item['available']})"
        return (f"{name}: {scale}, {item['listings']} e'lon{share}; "
                f"{reasons.get(item.get('stop'), unknown_reason)}.")

    coverage = [line(item) for item in snapshot["coverage"]]
    blocks = [Note("Metodologik ogohlantirishlar", CAVEATS, tone="note"), Bullets(items)]
    if audit is not None and audit.flagged:
        blocks.append(Note("Ma'lumotlar sifati", audit.lines()))
    if macro is not None and macro.sources:
        blocks.append(Note("Tashqi manbalar", [
            "Makroiqtisodiy ko'rsatkichlar: " + "; ".join(macro.sources) + ". "
            "Ular e'lonlardan emas, rasmiy statistikadan olingan va o'z davri bilan "
            "birga keltirilgan."]))
    if coverage:
        blocks.append(Note("Qamrov", coverage))
    return Section(TITLES["method"], blocks)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _render_charts(content, assets: Path) -> dict:
    """Draw every figure once; the PDF and the Word file embed the same image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    with plt.rc_context(layout.figure_style()):
        for si, section in enumerate(content):
            for bi, block in enumerate(section.blocks):
                if not isinstance(block, Chart):
                    continue
                fig, ax = plt.subplots(figsize=(8.4, 4.0))
                block.draw(ax)
                fig.tight_layout()
                path = assets / f"{si:02d}_{bi:02d}.png"
                fig.savefig(path, dpi=170)
                plt.close(fig)
                paths[(si, bi)] = path
    return paths


def _pdf_story(content, styles, charts, contents_title="MUNDARIJA"):
    from reportlab.platypus import (Image, KeepTogether, NextPageTemplate, PageBreak,
                                    Spacer)
    story = [NextPageTemplate("body"), Spacer(1, 2), PageBreak(),
             layout.heading(contents_title, styles, toc=False),
             layout.table_of_contents(styles), Spacer(1, 16),
             layout.heading("MAXSUS QISQARTMALAR", styles, level=2, toc=False),
             layout.abbreviations_table(ABBREVIATIONS, styles)]
    for si, section in enumerate(content):
        # A section with nothing to show would still take a page break and a
        # heading, and print as a blank page.
        if not editorial.has_content(section):
            continue
        story += [PageBreak(), layout.heading(section.title, styles)]
        for bi, block in enumerate(section.blocks):
            if isinstance(block, Text):
                story.append(layout.body(block.text, styles))
            elif isinstance(block, Bullets):
                story += layout.bullets(block.items, styles) + [Spacer(1, 4)]
            elif isinstance(block, Note):
                story += [Spacer(1, 4),
                          layout.callout(block.label, block.paragraphs, styles,
                                         tone=block.tone),
                          Spacer(1, 6)]
            elif isinstance(block, Chart):
                path = charts.get((si, bi))
                if path:
                    # The caption travels with its figure: a caption alone at a
                    # page foot, or a figure opening a page with no caption, is
                    # the pagination fault this avoids.
                    story += [Spacer(1, 6),
                              KeepTogether([
                                  Image(str(path), width=layout.CONTENT_WIDTH,
                                        height=layout.CONTENT_WIDTH * 4.0 / 8.4),
                                  layout.body(block.caption, styles, "caption")]),
                              Spacer(1, 6)]
            elif isinstance(block, Tbl):
                if block.frame is None or block.frame.empty:
                    story += [layout.body(block.caption, styles, "caption"),
                              layout.body(block.empty_note, styles)]
                    continue
                table = layout.data_table(block.frame, styles)
                head = [layout.body(block.caption, styles, "caption")]
                # A short table keeps its caption; a long one would not fit on
                # any page together with it, so it is allowed to break.
                story.append(KeepTogether(head + [table]) if len(block.frame) <= 12
                             else head[0])
                if len(block.frame) > 12:
                    story.append(table)
                story.append(Spacer(1, 8))
    return story


def _write_pdf(path, content, snapshot, charts, title, period):
    styles = layout.stylesheet()
    doc = layout.BulletinDoc(
        path,
        cover={"title_lines": layout.wrap_title(title),
               "period": period,
               "strapline": "E'LONLAR MA'LUMOTLARI ASOSIDAGI TAHLILIY SHARH",
               "footnote": ["Manba: olx.uz va uybor.uz ochiq e'lonlari; "
                            "Markaziy bank kursi.",
                            "Barcha raqamlar e'lon (so'ralgan) narxlari asosida "
                            "hisoblangan; bitim narxlari emas."]},
        footer="SINOV: sun'iy ma'lumotlar" if snapshot.get("synthetic")
               else "O'zbekiston uy-joy va ijara bozori sharhi")
    doc.multiBuild(_pdf_story(content, styles, charts))
    return path


def _write_docx(path, content, charts, title, period, observed):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    for name in ("Heading 1", "Heading 2", "Title"):
        document.styles[name].font.color.rgb = RGBColor.from_string("1C7C7A")
    heading = document.add_heading(title, 0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in (period, "E'lonlar ma'lumotlari asosidagi tahliliy sharh", observed):
        paragraph = document.add_paragraph(line)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()
    content = [section for section in content if editorial.has_content(section)]
    document.add_heading("Mundarija", 1)
    for section in content:
        document.add_paragraph(section.title, style="List Bullet")
    document.add_heading("Maxsus qisqartmalar", 2)
    for short, long in ABBREVIATIONS:
        layout.docx_runs(document.add_paragraph(), [(short, True), (f" - {long}", False)])
    for si, section in enumerate(content):
        document.add_page_break()
        document.add_heading(section.title, 1)
        for bi, block in enumerate(section.blocks):
            if isinstance(block, Text):
                document.add_paragraph(block.text)
            elif isinstance(block, Bullets):
                for item in block.items:
                    layout.docx_runs(document.add_paragraph(style="List Bullet"),
                                     layout.split_bold(item))
            elif isinstance(block, Note):
                layout.docx_callout(document, block.label, block.paragraphs)
            elif isinstance(block, Chart):
                # Not ``path``: that name is this function's output file.
                figure = charts.get((si, bi))
                if figure:
                    document.add_picture(str(figure), width=Inches(6.3))
                    caption = document.add_paragraph(block.caption)
                    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif isinstance(block, Tbl):
                document.add_heading(block.caption, 2)
                if block.frame is None or block.frame.empty:
                    document.add_paragraph(block.empty_note)
                    continue
                layout.docx_table(document, block.frame)
    document.save(path)
    return path


def _period_label(snapshot, archive) -> str:
    """What the cover calls the period: the quarters covered, or the snapshot date."""
    observed = snapshot["collected_at"][:10]
    if archive and archive.quarters:
        return (f"{quarter_label(archive.quarters[0])} - "
                f"{quarter_label(archive.quarters[-1])}")
    return f"KESIM: {observed}"


def write(snapshot_path: Path, output_dir: Path, title="", history=None, archive=None,
          llm=None, macro=None, research=None, progress=lambda _: None):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    frame = normalise(snapshot)
    audit = quality.audit(frame, screened=int(frame.attrs.get("screened", 0)))
    if audit.flagged:
        # A Windows console is not always on a Unicode code page, and a
        # progress line is not worth a crash: the report keeps the typography.
        progress("Ma'lumotlar sifati tekshiruvi: "
                 + ", ".join(f"{item.check.lower()} - {item.affected}"
                             .replace("²", "2") for item in audit.flagged))
    if archive is not None and history is None:
        history = archive.monthly
    content = sections(snapshot, frame, history, archive, audit=audit, macro=macro,
                       research=research)
    narrative_meta = {"generated_by": "template", "model": None}
    if llm is not None:
        from .olx_narrative import enrich
        narrative_meta = enrich(content, llm, progress=progress)
    review = editorial.summary(editorial.review(content))
    if review["issues"]:
        progress(f"Tahririy tekshiruv: {review['issues']} ta izoh qoldi.")
    stamp = Path(snapshot_path).stem
    folder = Path(output_dir) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    assets = Path(output_dir) / "tables" / stamp
    assets.mkdir(parents=True, exist_ok=True)
    for si, section in enumerate(content):
        for bi, block in enumerate(section.blocks):
            if isinstance(block, Tbl) and block.frame is not None and not block.frame.empty:
                block.frame.to_csv(assets / f"{si:02d}_{bi:02d}.csv", index=False,
                                   encoding="utf-8-sig")
    charts = _render_charts(content, assets)
    title = title or "O'zbekiston uy-joy va ijara bozori sharhi"
    observed = snapshot["collected_at"][:10]
    period = _period_label(snapshot, archive)
    docx_path = folder / f"olx_uz_{stamp}.docx"
    pdf_path = docx_path.with_suffix(".pdf")
    _write_docx(docx_path, content, charts, title, period, observed)
    _write_pdf(pdf_path, content, snapshot, charts, title, period)
    log = Path(output_dir) / "runs" / f"olx_{stamp}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "snapshot": str(snapshot_path), "rows": len(frame),
        "coverage": snapshot["coverage"], "pdf": str(pdf_path), "docx": str(docx_path),
        "language": "uz", "period": period, "narrative": narrative_meta,
        "quality": [vars(item) for item in audit.findings],
        "editorial": review,
        "macro": [item.row() for item in (macro.observations if macro else [])],
        "research": research.to_dict() if research is not None else None,
        "scope": "synthetic_demo" if snapshot.get("synthetic")
                 else "bounded_current_snapshot",
        "quarters": [str(q) for q in (archive.quarters if archive else [])],
        "quarter_rates": {str(q): rate for q, (rate, _) in
                          (archive.rates.items() if archive else [])},
        "sections": [section.title for section in content],
        "figures": [str(p) for p in charts.values()],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return docx_path, pdf_path, log


def combined_history(archive, output_dir, snapshot_path, *, rebuild=False, progress=print):
    """The archived tables, with this snapshot appended to the monthly series.

    Returns ``None`` when no archive was given, which leaves the bulletin's
    standing "no historical comparison" wording in place.
    """
    if not archive:
        return None
    from ..ingest.fx_history import CACHE_NAME, quarter_end_rates
    cache = Path(output_dir) / "price_history_monthly.csv"
    tables = build_series(archive, cache=cache, rebuild=rebuild, progress=progress)
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    monthly = tables["monthly"]
    current = live_month(normalise(snapshot), snapshot["collected_at"])
    if not current.empty:
        # A re-run on the same month replaces that month rather than doubling it.
        monthly = monthly[monthly["month"] != current["month"].iloc[0]]
        monthly = pd.concat([monthly, current], ignore_index=True).sort_values(
            ["month", "region_uz", "market"], ignore_index=True)
    bundle = Archive(monthly, tables["quarterly"], tables["quarterly_district"])
    # Every quarter the index prints needs its own dated rate, not only the
    # three the comparison tables show. They are cached after the first run.
    bundle.rates = quarter_end_rates(sorted(tables["quarterly"]["quarter"].unique()),
                                     cache=Path(output_dir) / CACHE_NAME,
                                     progress=progress)
    return bundle


def _movement_brief(archive) -> str:
    """What the research step is being asked to explain, in one short brief.

    The searches are the same every quarter, but the question is not: a quarter
    in which the capital fell and the valley rose needs different reading from
    one that moved together. Handing the model the movement keeps the retrieved
    material pointed at this quarter instead of at the market in general.
    """
    if archive is None or archive.quarterly.empty:
        return ""
    quarterly = archive.quarterly
    quarters = archive.quarters
    if len(quarters) < 2:
        return ""
    now, before = quarters[-1], quarters[-2]
    lines = []
    for market in sorted(quarterly["market"].dropna().unique()):
        rows = quarterly[(quarterly["market"] == market)]
        mask = pd.PeriodIndex(rows["quarter"], freq="Q")
        latest, prior = rows[mask == now], rows[mask == before]
        if latest.empty or prior.empty:
            continue
        pair = latest.merge(prior, on="region_uz", suffixes=("_now", "_before"))
        pair = pair.dropna(subset=["median_usd_sqm_now", "median_usd_sqm_before"])
        if pair.empty:
            continue
        move = (pair["median_usd_sqm_now"] / pair["median_usd_sqm_before"] - 1) * 100
        fell = int((move < 0).sum())
        worst = pair.loc[move.idxmin(), "region_uz"]
        best = pair.loc[move.idxmax(), "region_uz"]
        lines.append(
            f"- {market} market, {now} vs {before}: asking prices fell in {fell} of "
            f"{len(pair)} regions (largest fall {move.min():.1f}% in {worst}, "
            f"largest rise {move.max():.1f}% in {best}).")
    if not lines:
        return ""
    return ("Asking prices from classified adverts, own computation:\n"
            + "\n".join(lines)
            + "\nExplain what over this quarter would move asking prices this way.")


def _research(settings, archive, llm, progress):
    """Policy, monetary and market news for the quarter the data covers.

    Never fatal: an offline machine, a rate-limited search endpoint or a failed
    synthesis costs the report its explanation section, not the report.
    """
    if not settings.web_research:
        progress("Siyosat va yangiliklar tahlili o'chirilgan (WEB_RESEARCH).")
        return None
    from ..research import context
    quarters = archive.quarters if archive else []
    start = str(quarters[0].start_time.date()) if quarters else ""
    end = str(quarters[-1].end_time.date()) if quarters else ""
    progress("Uy-joy bozoriga oid siyosat va yangiliklar o'rganilmoqda...")
    try:
        return context.gather(
            llm, period_start=start, period_end=end,
            data_highlights=_movement_brief(archive),
            use_web=True,
            results_per_query=settings.search_results_per_query,
            pages_to_read=settings.pages_to_read,
            policy_file=settings.policy_file,
            language="uz", progress=progress)
    except Exception as exc:  # pragma: no cover - network and vendor failures
        progress(f"Siyosat tahlili bajarilmadi: {exc}")
        return None


def run_olx(settings, *, pages=None, progress=print, title="", snapshot_path=None,
            browser=False, headless=True, archive=None, rebuild_history=False,
            uybor_pages=0):
    from ..ingest.olx import collect
    from ..pipeline import RunResult
    import time
    start = time.monotonic()
    from ..llm import LLM, LLMUnavailable
    llm = LLM(settings.llm_api_key, settings.model)
    if not llm.available:
        # llm.status carries the real cause — a missing key, but equally an
        # uninstalled client or a rejected credential. Naming only the key sent
        # the reader to a correctly-filled .env when the module was the problem.
        raise LLMUnavailable(
            f"OLX hisoboti uchun model kerak, lekin u ishga tushmadi: {llm.status}. "
            f"Kalit {settings.llm_key_name} sifatida .env faylida bo'lishi kerak."
        )
    progress(f"Model tahlili: {settings.model}")
    snapshot_path = Path(snapshot_path) if snapshot_path else collect(
        settings.output_dir, pages=pages, progress=progress,
        browser=browser, headless=headless, uybor_pages=uybor_pages)
    bundle = combined_history(archive, settings.output_dir, snapshot_path,
                              rebuild=rebuild_history, progress=progress)
    progress("Rasmiy makroiqtisodiy ko'rsatkichlar olinmoqda...")
    macro = official.gather(cache=Path(settings.output_dir) / official.CACHE_NAME,
                            progress=progress)
    research = _research(settings, bundle, llm, progress)
    progress("O'zbek tilidagi PDF va Word hisobotlari tayyorlanmoqda...")
    docx, pdf, log = write(snapshot_path, settings.output_dir, title, archive=bundle,
                           llm=llm, macro=macro, research=research, progress=progress)
    return RunResult(report_path=docx, pdf_path=pdf, run_log=log,
                     duration_seconds=time.monotonic() - start)
