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

from ..ingest.listings import _advert_row, _derive_and_clean, _place_name
from ..ingest.price_history import (MIN_QUARTER_LISTINGS, SCOPE_NOTE, build_series,
                                    canonical_district, canonical_region,
                                    comparison_table, gap_months, index_series,
                                    live_month, national, quarter_label,
                                    region_label, thin_groups)
from . import layout

OLX_SOURCE, UYBOR_SOURCE = "OLX.uz", "Uybor.uz"

#: Section titles, in one place so the contents page and the sections agree.
TITLES = {
    "summary": "QISQACHA XULOSA",
    "primary": "BIRLAMCHI UY-JOY BOZORI",
    "secondary": "IKKILAMCHI UY-JOY BOZORI",
    "unknown": "BOZOR TURI KO'RSATILMAGAN SOTUV E'LONLARI",
    "index": "UY-JOY NARXLARI INDEKSI",
    "rent": "IJARA BOZORI",
    "rent_districts": "TOSHKENT SHAHRI TUMANLARIDA IJARA",
    "sale_districts": "TOSHKENT SHAHRI TUMANLARIDA SOTUV",
    "yield": "IJARA RENTABELLIGI (YILLIK)",
    "unavailable": "UY-JOY QURILISHI VA IPOTEKA BOZORI",
    "mix": "MANBALAR TARKIBI",
    "method": "METODOLOGIYA VA MANBALAR",
}

#: Expanded on the contents page, as the reference expands its own.
ABBREVIATIONS = [
    ("OLX", "olx.uz e'lonlar platformasi"),
    ("Uybor", "uybor.uz e'lonlar platformasi"),
    ("MB", "O'zbekiston Respublikasi Markaziy banki"),
    ("USD/UZS", "Markaziy bankning rasmiy dollar kursi"),
    ("m²", "kvadrat metr"),
    ("Ch1-Ch4", "yilning birinchi-to'rtinchi choragi"),
    ("Mediana", "taqsimotning o'rta qiymati (o'rtacha emas)"),
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
    caption: str
    frame: pd.DataFrame
    empty_note: str = "Ushbu bo'lim uchun yetarli, aniq tasniflangan kuzatuvlar mavjud emas."


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
        return "mln so'm/m²" if self.in_som else "USD/m²"


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
    parts = [_derive_and_clean(group, kind, snapshot["fx"]["rate"])[0]
             for kind, group in frame.groupby("kind")]
    frame = pd.concat(parts, ignore_index=True)
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
    return frame


def regional_table(frame, kind, market=None, district=False):
    selected = frame[frame.kind == kind]
    if market:
        selected = selected[selected.market == market]
    keys = ["district", "property"] if district else ["region", "property"]
    if district:
        selected = selected[selected.district != "Ko'rsatilmagan"]
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

def movement_bullets(table: pd.DataFrame, unit: str, label="Hudud") -> list:
    """The reference's commentary pattern, every figure read off the table."""
    if table.empty or len(table.columns) < 3:
        return []
    change = table.columns[-1]
    level = table.columns[-2]
    rows = table.dropna(subset=[change])
    if rows.empty:
        return ["Taqqoslash uchun yetarli kuzatuvga ega hudud topilmadi."]
    falls = rows[rows[change] < 0].sort_values(change)
    rises = rows[rows[change] > 0].sort_values(change, ascending=False)

    def named(frame, count=3):
        return ", ".join(f"**{row[label]}** ({row[change]:+.1f}%)"
                         for _, row in frame.head(count).iterrows())

    noun = str(label).lower()
    items = [f"{level} davrida taqqoslanadigan {len(rows)} ta {noun}dan "
             f"**{len(falls)} tasida narx pasaydi**, {len(rises)} tasida oshdi."]
    if not falls.empty:
        items.append(f"Eng sezilarli pasayish: {named(falls)}.")
    if not rises.empty:
        items.append(f"Narx o'sishi qayd etilgan {noun}lar: {named(rises)}.")
    levels = table.dropna(subset=[level]).sort_values(level)
    if len(levels) > 1:
        cheap, dear = levels.iloc[0], levels.iloc[-1]
        items.append(
            f"Eng arzon: **{cheap[label]}** ({cheap[level]:,.2f} {unit}); "
            f"eng qimmat: **{dear[label]}** ({dear[level]:,.2f} {unit}).")
    return items


def _thin_note(quarterly, group, quarter, market, label) -> list:
    thin = thin_groups(quarterly, group, quarter, market=market)
    if not thin:
        return []
    listed = ", ".join(f"{region_label(name) if group == 'region_uz' else name} "
                       f"({count} e'lon)" for name, count in sorted(thin.items()))
    return [f"Kuzatuvi {MIN_QUARTER_LISTINGS} tadan kam bo'lgani uchun mediana "
            f"ko'rsatilmagan {label}: {listed}. Ular jadvalda chiziqcha bilan qoldirildi."]


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
    if archive and archive.quarters and len(archive.quarters) > 1:
        quarters, rates, unit = archive.quarters, archive.rate_values, archive.unit
        regions = comparison_table(archive.quarterly, "region_uz", quarters, rates,
                                   market=market)
        if not regions.empty:
            regions["Hudud"] = [region_label(value) for value in regions["Hudud"]]
            section.blocks += [
                Bullets(movement_bullets(regions, unit)
                        + _thin_note(archive.quarterly, "region_uz", quarters[-1],
                                     market, "hududlar")),
                Tbl(f"Hududlar bo'yicha {title.split()[0].lower()} uy-joy narxlari "
                    f"({unit})", regions)]
        districts = comparison_table(archive.district, "district", quarters, rates,
                                     market=market, label="Tuman")
        if not districts.empty:
            section.blocks += [
                Bullets(movement_bullets(districts, unit, label="Tuman")
                        + _thin_note(archive.district, "district", quarters[-1],
                                     market, "tumanlar")),
                Tbl(f"Toshkent shahrida {title.split()[0].lower()} uy-joy narxlari "
                    f"({unit})", districts)]
        if section.blocks:
            section.blocks.append(Note("Manba", [
                "Arxivlangan OLX e'lonlaridan ushbu dastur tomonidan hisoblangan. "
                "Faqat kvartira sotuvi e'lonlari; narxlar chorak oxiridagi Markaziy "
                "bank kursi bo'yicha so'mga o'tkazilgan."
                if archive.in_som else
                "Arxivlangan OLX e'lonlaridan hisoblangan. Chorak oxiri uchun rasmiy "
                "kurs olinmagani sababli narxlar dollarda keltirildi.",
                "Bular e'lon (so'ralgan) narxlari, bitim narxlari emas. "
                f"Kuzatuvi {MIN_QUARTER_LISTINGS} tadan kam guruhlar medianasi "
                "ko'rsatilmaydi."]))
    else:
        section.blocks.append(Text(
            "Chorakma-chorak taqqoslash uchun arxiv berilmadi, shuning uchun bu bo'limda "
            "faqat joriy yig'ish kesimi keltirildi. Arxiv bilan ishga tushirish: "
            "run.py --olx --olx-archive <papka>."))
    snapshot_table = regional_table(frame, "sale", market)
    section.blocks += [
        Text("Quyidagi jadval — shu yig'ishdagi joriy takliflar kesimi. "
             "U yuqoridagi tarixiy qator bilan bir xil manba emas: joriy kesim "
             "OLX va Uybor e'lonlarini birlashtiradi, tarixiy qator esa faqat "
             "arxivlangan OLX kvartira sotuvi e'lonlaridan iborat."
             if section.blocks and isinstance(section.blocks[0], Bullets)
             else "Joriy yig'ishdagi takliflar kesimi."),
        Tbl("Joriy kesim: hududlar bo'yicha taklif narxlari (mln so'm/m²)",
            snapshot_table),
        Note("Eslatma", ["Bozor turi faqat e'londagi aniq belgi asosida ajratildi; "
                         "yangi ta'mir yoki qurilish yili birlamchi bozor belgisi "
                         "sifatida olinmadi. Taklif narxlari yakuniy sotuv narxidan "
                         "farq qilishi mumkin."], tone="note")]
    return section


def index_section(archive) -> Section:
    section = Section(TITLES["index"])
    if not archive or archive.quarterly.empty:
        section.blocks.append(Text(
            "Uy-joy narxlari indeksi arxiv qatoridan hisoblanadi. Arxiv berilmagani "
            "uchun indeks ushbu hisobotda keltirilmadi."))
        return section
    levels = index_series(archive.quarterly, archive.rate_values)
    if levels.empty or len(levels) < 2:
        section.blocks.append(Text("Indeks uchun yetarli chorak kuzatilmadi."))
        return section
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
        table.insert(2, "Daraja, mln so'm/m²", levels["som_sqm"].round(2))
    else:
        table.insert(2, "Daraja, USD/m²", levels["median_usd_sqm"].round(0))
    section.blocks += [
        Bullets([
            f"Indeks {quarter_label(first['quarter'])} choragini 100 deb oladi. "
            f"{quarter_label(last['quarter'])} holatiga indeks **{last['index']:.0f}** "
            f"bo'lib, bazaviy chorakka nisbatan **{change:+.1f}%** o'zgarishni bildiradi.",
            f"Qator {len(levels)} ta chorakni va jami "
            f"{int(levels['listings'].sum()):,} ta e'lonni qamrab oladi.".replace(",", " "),
            "Indeks taklif narxlaridan qurilgan va e'lonlar soniga qarab "
            "vaznlangan. U aholi soniga vaznlanmagan va bitim narxlariga "
            "asoslanmagan, shuning uchun rasmiy uy-joy narxlari indeksi bilan "
            "bevosita taqqoslanmaydi."]
            + _currency_note(levels, archive) + _partial_quarters(levels)),
        Chart("1-rasm. Uy-joy taklif narxlari indeksi", _index_chart(levels)),
        Tbl("Choraklar bo'yicha indeks va daraja", table),
        Note("Eslatma", [
            "Respublika darajasi hududiy medianalarning e'lonlar soniga ko'ra "
            "vaznlangan o'rtachasi; medianalar yig'indisidan mediana chiqmaydi, "
            "shuning uchun bu ko'rsatkich respublika medianasi emas.",
            SCOPE_NOTE], tone="note")]
    return section


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
    return [f"Dollar hisobidagi daraja {quarter_label(first['quarter'])} dan "
            f"{quarter_label(last['quarter'])} gacha **{dollars:+.1f}%**, so'mdagi "
            f"daraja esa **{som:+.1f}%** o'zgargan. Farq shu davrdagi rasmiy kurs "
            "o'zgarishidan kelib chiqadi va ikkinchi narx harakati emas."]


def _partial_quarters(levels) -> list:
    """Name the quarters the archive observed for fewer than three months.

    A quarter standing on one month is not comparable with a full one, and the
    difference is invisible in the index unless it is said.
    """
    partial = [(quarter_label(q), int(m)) for q, m in
               zip(levels["quarter"], levels["months"]) if 0 < m < 3]
    if not partial:
        return []
    listed = ", ".join(f"{label} ({months} oy)" for label, months in partial)
    return [f"**Eslatma:** arxiv quyidagi choraklarni to'liq kuzatmagan: {listed}. "
            "Bu choraklar uch oylik choraklar bilan bir xil asosda taqqoslanmaydi."]


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
    items = [
        f"Qator {first} dan {last} gacha bo'lgan {monthly['month'].nunique()} oylik "
        f"kuzatuvni qamrab oladi; jami {int(monthly['listings'].sum()):,} ta e'lon. "
        "Har oy uchun mediana alohida hisoblangan.".replace(",", " ")]
    section = Section("TARIXIY TAQQOSLASH: OYLIK QATOR")
    if gaps:
        spans = _gap_spans(pd.period_range(first, last, freq="M"),
                           monthly.set_index(pd.PeriodIndex(monthly["month"], freq="M"))["median_usd_sqm"]
                           .reindex(pd.period_range(first, last, freq="M")))
        listed = ", ".join(f"{a}–{(b - 1)}" for a, b in spans)
        items.append(
            f"**DIQQAT:** {len(gaps)} oy uchun kuzatuv mavjud emas ({listed}). Bu oylar "
            "uchun ma'lumot to'ldirilmadi va grafikda chiziq uzilgan holda ko'rsatilgan. "
            "Uzilishdan keyingi taqqoslash ikki nuqta orasidagi o'zgarish bo'lib, "
            "uzluksiz tendentsiya emas.")
        # The comparison spans the most recent break, which is the one standing
        # between the archive and today, not an older one inside the archive.
        gap_start, resumed = spans[-1]
        before = monthly[monthly["month"] < gap_start]
        after = monthly[monthly["month"] >= resumed]
        if not before.empty and not after.empty:
            previous, current = before.iloc[-1], after.iloc[-1]
            items.append(
                f"Uzilishdan oldingi oxirgi kuzatuv ({previous['month']}): "
                f"{previous['median_usd_sqm']:,.0f} USD/m² "
                f"({int(previous['listings']):,} e'lon). Joriy kuzatuv "
                f"({current['month']}): {current['median_usd_sqm']:,.0f} USD/m² "
                f"({int(current['listings']):,} e'lon). Ikki nuqta orasidagi farq: "
                f"{(current['median_usd_sqm'] / previous['median_usd_sqm'] - 1) * 100:+.1f}%. "
                "Bu oraliqdagi oylar kuzatilmagani uchun farqning qachon yuz bergani "
                "aniqlanmaydi.")
            if current["listings"] < previous["listings"] / 10:
                items.append(
                    f"**OGOHLANTIRISH:** joriy oy atigi {int(current['listings']):,} ta "
                    f"e'londan hisoblangan, tarixiy oy esa {int(previous['listings']):,} "
                    "ta e'londan. Bunday kichik tanlanma hududlar va uy turlari bo'yicha "
                    "boshqacha tarkibga ega bo'lishi mumkin, shuning uchun farqning bir "
                    "qismi bozor narxi emas, tanlanma tarkibi o'zgarishidan kelib "
                    "chiqishi mumkin. Taqqoslashni mustahkamlash uchun yig'ishni ko'proq "
                    "sahifa bilan takrorlang.")
    section.blocks += [
        Bullets(items),
        Chart("2-rasm. Kvartira sotuvi: oylik mediana", _history_chart(series)),
        Tbl("Yillar bo'yicha mediana va o'zgarish", annual_history(series)),
        Note("Manba", [
            "Tarixiy qiymatlar arxivdagi e'lonlardan, joriy qiymat esa shu yig'ishdan "
            "olingan. Ikkalasi ham bir xil usulda tozalangan: y.e. USDga teng, so'mdagi "
            "e'lonlar chiqarilgan (tarixiy kurs arxivda yo'q), narx 1 000–5 000 000 USD, "
            "maydon 10–1 000 m².", SCOPE_NOTE])]
    return section


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


def sections(snapshot, frame, history=None, archive=None) -> list:
    """Every section of the bulletin, in the order the reference sets them out."""
    observed = snapshot["collected_at"][:10]
    archive = archive or (Archive(history, pd.DataFrame(), pd.DataFrame())
                          if history is not None and not getattr(history, "empty", True)
                          else None)
    result = []

    summary = Section(TITLES["summary"])
    if snapshot.get("synthetic"):
        summary.blocks.append(Note("SINOV NAMUNASI", [
            "Barcha raqamlar sun'iy ma'lumotlardan olingan; haqiqiy bozor natijalari emas."
        ], tone="note"))
    # Both sources, or the total would be smaller than the rows it explains.
    collected = len(snapshot["data"]) + len(snapshot.get("uybor") or [])
    scope = [
        f"Kuzatuv sanasi: **{observed}**. Ikkala manbadan yig'ilgan {collected:,} ta "
        f"yozuvdan **{len(frame):,} ta** takrorlanmagan, narxi yaroqli e'lon tahlil "
        "qilindi.".replace(",", " "),
        "Sotuv narxlari mln so'm/m², oylik ijara narxlari ming so'm/m² hisobida "
        "keltirilgan. Kvartira va hovlilar alohida ko'rsatiladi.",
        "E'lon narxi yakuniy bitim narxi emas. Sahifalar soni cheklangan tanlanma "
        "butun bozorni ifodalamaydi."]
    if archive and archive.quarters:
        scope.insert(1,
            "Hisobot ikki qismdan iborat: **chorakma-chorak dinamika** arxivlangan "
            "OLX kvartira sotuvi e'lonlaridan, **joriy kesim** esa shu yig'ishdagi "
            "OLX va Uybor takliflaridan hisoblangan.")
        scope.insert(2, f"Chorakma-chorak jadvallar **{quarter_label(archive.quarters[0])} – "
                        f"{quarter_label(archive.quarters[-1])}** davrini qamrab oladi.")
    else:
        scope.insert(1,
            "Hisobot bir yig'ish sanasidagi kesimdir: arxiv berilmagani uchun "
            "chorakma-chorak taqqoslash va uy-joy narxlari indeksi hisoblanmadi.")
    summary.blocks.append(Bullets(scope))
    result.append(summary)

    result.append(market_section(TITLES["primary"], "Birlamchi", snapshot, frame, archive))
    result.append(market_section(TITLES["secondary"], "Ikkilamchi", snapshot, frame, archive))

    unknown = Section(TITLES["unknown"], [
        Bullets(["Bozor turi e'londa ko'rsatilmagan takliflar alohida keltirildi. "
                 "Ular birlamchi yoki ikkilamchi bozorga taxmin bilan qo'shilmadi."]),
        Tbl("Turi noma'lum sotuv takliflari (mln so'm/m²)",
            regional_table(frame, "sale", "Aniqlanmagan"))])
    result.append(unknown)

    result.append(index_section(archive))
    result.append(history_section(history, snapshot))

    rent = regional_table(frame, "rent")
    result.append(Section(TITLES["rent"], [
        Bullets(["Faqat uzoq muddatli ijara toifalaridagi e'lonlar olindi; kunlik "
                 "ijara toifalari kiritilmadi.",
                 "Arxivda ijara e'lonlari yo'q, shuning uchun ijara bo'yicha "
                 "chorakma-chorak taqqoslash berilmaydi. Quyidagilar joriy kesim."]),
        Tbl("Hududlar bo'yicha ijara narxlari (ming so'm/m²/oy)", rent),
        Tbl("Toshkent shahri tumanlarida ijara narxlari (ming so'm/m²/oy)",
            regional_table(frame, "rent", district=True)),
        Note("Eslatma", ["Tuman nomi manbada ko'rsatilgan kuzatuvlar. Toshkent "
                         "shahridan tashqari joylar tuman kesimida keltirilmaydi."],
             tone="note")]))

    result.append(Section(TITLES["sale_districts"], [
        Bullets(["Birlamchi, ikkilamchi va turi noma'lum sotuv takliflari birgalikda; "
                 "uy turi alohida."]),
        Tbl("Toshkent shahri tumanlarida sotuv takliflari (mln so'm/m²)",
            regional_table(frame, "sale", district=True))]))

    yields = yield_table(frame)
    yield_blocks = [Bullets([
        "Taxminiy yalpi ko'rsatkich = 12 × median oylik ijara (USD/m²) / median sotuv "
        "narxi (USD/m²) × 100.",
        "Bir xil hudud, uy turi va xonalar soni solishtiriladi; har ikki guruhda "
        "kamida 15 ta kuzatuv talab qilinadi. Shahar nomi manbalar o'rtasida "
        "izchil emas, shuning uchun qatlam hudud darajasida olindi.",
        "Bu aynan bir mulkning daromadliligi emas; bo'sh turish, soliq va ta'mirlash "
        "xarajatlari chegirilmagan."])]
    if not yields.empty:
        yield_blocks.append(Chart("3-rasm. Yalpi ijara rentabelligi", _yield_chart(yields)))
    yield_blocks.append(Tbl("Taqqoslanadigan qatlamlar bo'yicha yalpi ko'rsatkich", yields))
    result.append(Section(TITLES["yield"], yield_blocks))

    result.append(Section(TITLES["unavailable"], [
        Note("Ma'lumot mavjud emas", [
            "Foydalanishga topshirilgan uy-joylar hajmi, ajratilgan ipoteka kreditlari, "
            "o'rtacha foiz stavkalari va bank mahsulotlari shartlari e'lonlar "
            "platformalarida mavjud emas.",
            "Bu ko'rsatkichlar uchun Markaziy bank, Qurilish va uy-joy kommunal "
            "xo'jaligi vazirligi va statistika idorasining rasmiy ma'lumotlari zarur; "
            "ular ushbu hisobotda hisoblanmadi va namuna hisobotdan ko'chirilmadi."],
            tone="note")]))

    mix = source_mix(frame)
    if not mix.empty:
        result.append(Section(TITLES["mix"], [
            Bullets([
                "Joriy kesimdagi medianalar barcha manbalar bo'yicha birlashtirilgan. "
                "Quyidagi jadval har bir medianaga qaysi manbadan nechta e'lon "
                "kirganini ko'rsatadi.",
                "**DIQQAT:** manbalar bir xil e'lonlar to'plami emas. Birlashtirilgan "
                "medianaga manbalar nisbati ham ta'sir qiladi, shuning uchun yig'ishlar "
                "orasidagi farq narx o'zgarishi emas, tarkib o'zgarishi bo'lishi mumkin.",
                "Bir e'lon ikki saytda joylangan bo'lsa, u ikki marta hisoblanishi "
                "mumkin: manbalar o'rtasida takrorlanishni aniqlaydigan umumiy "
                "identifikator yo'q."]),
            Tbl("Manbalar bo'yicha e'lonlar soni", mix)]))

    result.append(method_section(snapshot, archive))
    return result


def method_section(snapshot, archive) -> Section:
    rate = snapshot["fx"]
    items = [
        "Manbalar: " + ", ".join(snapshot.get("sources")
                                 or [snapshot.get("source") or "https://www.olx.uz"])
        + f". Yig'ish vaqti (UTC): {snapshot['collected_at']}.",
        f"Joriy kesim uchun Markaziy bank kursi: 1 USD = {rate['rate']:,.2f} so'm; "
        f"kurs sanasi: {rate['date']}. Manba: {rate['source']}.".replace(",", " "),
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
        dated = ", ".join(f"{quarter_label(q)}: {rate:,.2f} so'm ({date})".replace(",", " ")
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
    coverage = [f"{item.get('category') or item.get('source')}: "
                f"{item.get('pages_requested') or item.get('requests')} sahifa, "
                f"{item['listings']} e'lon; yig'ish cheklangan."
                for item in snapshot["coverage"]]
    blocks = [Bullets(items)]
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


def _table_note(frame: pd.DataFrame):
    """One line describing the spread, for a table with no change columns."""
    if frame.empty or len(frame.columns) < 2:
        return None
    column = frame.columns[-1]
    if str(column).startswith(layout.DELTA) or not pd.api.types.is_numeric_dtype(frame[column]):
        return None
    valid = frame.dropna(subset=[column])
    if len(valid) < 2:
        return None
    return (f"Kamida 15 ta kuzatuvli {len(valid)} ta guruh mavjud. Guruhlar bo'yicha "
            f"{str(column).lower()}: {valid[column].min():,.2f} dan "
            f"{valid[column].max():,.2f} gacha. Farqlar uylarning joylashuvi, maydoni "
            "va holati tarkibiga ham bog'liq; ular vaqt bo'yicha narx o'zgarishi emas."
            ).replace(",", " ")


def _pdf_story(content, styles, charts, contents_title="MUNDARIJA"):
    from reportlab.platypus import Image, NextPageTemplate, PageBreak, Spacer
    story = [NextPageTemplate("body"), Spacer(1, 2), PageBreak(),
             layout.heading(contents_title, styles, toc=False),
             layout.table_of_contents(styles), Spacer(1, 16),
             layout.heading("MAXSUS QISQARTMALAR", styles, level=2, toc=False),
             layout.abbreviations_table(ABBREVIATIONS, styles)]
    for si, section in enumerate(content):
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
                    story += [Spacer(1, 6),
                              Image(str(path), width=layout.CONTENT_WIDTH,
                                    height=layout.CONTENT_WIDTH * 4.0 / 8.4),
                              layout.body(block.caption, styles, "caption"),
                              Spacer(1, 6)]
            elif isinstance(block, Tbl):
                story.append(layout.body(block.caption, styles, "caption"))
                if block.frame is None or block.frame.empty:
                    story.append(layout.body(block.empty_note, styles))
                    continue
                story.append(layout.data_table(block.frame, styles))
                note = _table_note(block.frame)
                if note:
                    story += [Spacer(1, 4), layout.body(note, styles, "source")]
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
                note = _table_note(block.frame)
                if note:
                    document.add_paragraph(note)
    document.save(path)
    return path


def _period_label(snapshot, archive) -> str:
    """What the cover calls the period: the quarters covered, or the snapshot date."""
    observed = snapshot["collected_at"][:10]
    if archive and archive.quarters:
        return (f"{quarter_label(archive.quarters[0])} - "
                f"{quarter_label(archive.quarters[-1])}")
    return f"KESIM: {observed}"


def write(snapshot_path: Path, output_dir: Path, title="", history=None, archive=None, llm=None):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    frame = normalise(snapshot)
    if archive is not None and history is None:
        history = archive.monthly
    content = sections(snapshot, frame, history, archive)
    narrative_meta = {"generated_by": "template", "model": None}
    if llm is not None:
        from .olx_narrative import enrich
        narrative_meta = enrich(content, llm)
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


def run_olx(settings, *, pages=5, progress=print, title="", snapshot_path=None,
            browser=False, headless=True, archive=None, rebuild_history=False,
            uybor_pages=0):
    from ..ingest.olx import collect
    from ..pipeline import RunResult
    import time
    start = time.monotonic()
    from ..llm import LLM, LLMUnavailable
    llm = LLM(settings.anthropic_api_key, settings.model)
    if not llm.available:
        raise LLMUnavailable("OLX hisoboti uchun ANTHROPIC_API_KEY talab qilinadi.")
    progress(f"Claude tahlili: {settings.model}")
    snapshot_path = Path(snapshot_path) if snapshot_path else collect(
        settings.output_dir, pages=pages, progress=progress,
        browser=browser, headless=headless, uybor_pages=uybor_pages)
    bundle = combined_history(archive, settings.output_dir, snapshot_path,
                              rebuild=rebuild_history, progress=progress)
    progress("O'zbek tilidagi PDF va Word hisobotlari tayyorlanmoqda...")
    docx, pdf, log = write(snapshot_path, settings.output_dir, title, archive=bundle, llm=llm)
    return RunResult(report_path=docx, pdf_path=pdf, run_log=log,
                     duration_seconds=time.monotonic() - start)
