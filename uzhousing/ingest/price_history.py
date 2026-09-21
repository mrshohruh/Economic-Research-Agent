"""Monthly asking-price history, aggregated from archived OLX listing databases.

The archive holds millions of individual adverts across many files, which is far
more than the report needs and more than fits comfortably in memory. Each file
covers its own months, so a median taken per file is exact rather than an
average of averages, and the result is cached as a small monthly table.

What this produces is a series of **median asking prices for apartment sales**:
monthly for the trend chart, and quarterly by region and by Tashkent district
for the comparison tables. It is not a transaction index and not a rent series:
the archive holds no rentals (see :data:`SCOPE_NOTE`).

Region and district medians are aggregated separately from the adverts rather
than one from the other, because a median cannot be recovered from a set of
medians. One pass over the archive produces every table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

#: Columns read from each archive file. Everything else is ignored.
COLUMNS = {
    "month_and_year": "month",
    "state": "region_uz",
    "price1": "price",
    "currency": "currency",
    "Общая площадь:": "area",
    "Тип жилья:": "market_ru",
    # "Ташкент, Мирзо-Улугбекский район" for the capital, a plain settlement
    # name elsewhere. Only the capital's districts are reported.
    "location_1": "location",
}

#: Columns an archive file may not carry. A file without them still yields
#: every table the column does not feed, rather than failing the whole run.
OPTIONAL = {"location_1"}

#: у.е. is a dollar-linked unit quoted at parity with the USD, as in the live
#: collector. Som-quoted adverts are excluded rather than converted, because a
#: 2022 som price needs a 2022 rate and the archive carries no dated rate.
USD_CODES = {"у.е.", "уе", "y.e.", "u.e.", "usd", "$"}
UZS_CODES = {"сум", "so'm", "som", "uzs"}

#: Matches the live bulletin's screens, so the two are directly comparable.
MIN_PRICE_USD, MAX_PRICE_USD = 1_000, 5_000_000
MIN_AREA, MAX_AREA = 10, 1_000

MARKET_UZ = {
    "Вторичный рынок": "Ikkilamchi",
    "Новостройки": "Birlamchi",
    "От застройщика": "Birlamchi",
}

#: The archive names regions in Latin Uzbek; the live feed names them in
#: Russian. Tashkent is the case that matters: the archive separates the city
#: from the surrounding region, while the live feed files both under the region
#: and distinguishes them by city, so the city name decides.
REGION_UZ_FROM_RU = {
    "Ташкентская область": "Toshkent Viloyati",
    "Самаркандская область": "Samarqand Viloyati",
    "Бухарская область": "Buxoro Viloyati",
    "Навоийская область": "Navoiy Viloyati",
    "Ферганская область": "Farg'ona Viloyati",
    "Кашкадарьинская область": "Qashqadaryo Viloyati",
    "Хорезмская область": "Xorazm Viloyati",
    "Республика Каракалпакстан": "Qoraqalpogʻiston Respublikasi",
    "Сурхандарьинская область": "Surxondaryo Viloyati",
    "Сырдарьинская область": "Sirdaryo Viloyati",
    "Джизакская область": "Jizzax Viloyati",
    "Андижанская область": "Andijon Viloyati",
    "Наманганская область": "Namangan Viloyati",
}
TASHKENT_CITY_RU = "Ташкент"
TASHKENT_CITY_UZ = "Toshkent shahri"


#: Apostrophes arrive as six different characters across the three sources
#: ('  `  ´  ʻ  ʼ  ’), and two of them are letters as far as
#: ``str.isalnum`` is concerned, so they are removed by name.
APOSTROPHES = "'`´ʻʼ‘’"

#: Words naming the administrative unit rather than the place.
_UNIT_WORDS = {
    "city": ("shahri", "shahar", "город", "шахри"),
    "region": ("viloyati", "viloyat", "oblast", "область", "области", "skaya", "ская"),
    "republic": ("respublikasi", "respublika", "республика"),
}

#: Every spelling the three sources use, folded to one key. OLX writes Russian
#: in Cyrillic, Uybor writes Latin Uzbek and sometimes a Latin transliteration
#: of the Russian, and the archive writes Latin Uzbek. All three must land on
#: one row or the same region is counted twice in every pooled table.
REGION_KEY = {
    "andijon": "Andijon Viloyati", "андижан": "Andijon Viloyati",
    "andizhan": "Andijon Viloyati",
    "buxoro": "Buxoro Viloyati", "бухар": "Buxoro Viloyati",
    "bukhar": "Buxoro Viloyati", "buxor": "Buxoro Viloyati",
    "jizzax": "Jizzax Viloyati", "джизак": "Jizzax Viloyati",
    "dzhizak": "Jizzax Viloyati",
    "navoiy": "Navoiy Viloyati", "навоий": "Navoiy Viloyati",
    "навои": "Navoiy Viloyati", "navoi": "Navoiy Viloyati",
    "namangan": "Namangan Viloyati", "наманган": "Namangan Viloyati",
    "samarqand": "Samarqand Viloyati", "самарканд": "Samarqand Viloyati",
    "samarkand": "Samarqand Viloyati",
    "sirdaryo": "Sirdaryo Viloyati", "сырдарьин": "Sirdaryo Viloyati",
    "syrdarin": "Sirdaryo Viloyati", "sirdaryo​": "Sirdaryo Viloyati",
    "surxondaryo": "Surxondaryo Viloyati",
    "сурхандарьин": "Surxondaryo Viloyati",
    "surkhandarin": "Surxondaryo Viloyati",
    "fargona": "Farg'ona Viloyati", "ферган": "Farg'ona Viloyati",
    "fergan": "Farg'ona Viloyati",
    "xorazm": "Xorazm Viloyati", "хорезм": "Xorazm Viloyati",
    "khorezm": "Xorazm Viloyati",
    "qashqadaryo": "Qashqadaryo Viloyati",
    "кашкадарьин": "Qashqadaryo Viloyati",
    "kashkadarin": "Qashqadaryo Viloyati",
    "qoraqalpogiston": "Qoraqalpogʻiston Respublikasi",
    "каракалпакстан": "Qoraqalpogʻiston Respublikasi",
    "karakalpakstan": "Qoraqalpogʻiston Respublikasi",
    "qoraqalpoq": "Qoraqalpogʻiston Respublikasi",
    "toshkent": "Toshkent Viloyati", "ташкент": "Toshkent Viloyati",
    "tashkent": "Toshkent Viloyati",
}

#: Kept for callers that still map Russian region names directly.
REGION_UZ_FROM_RU = {
    "Ташкентская область": "Toshkent Viloyati",
    "Самаркандская область": "Samarqand Viloyati",
    "Бухарская область": "Buxoro Viloyati",
    "Навоийская область": "Navoiy Viloyati",
    "Ферганская область": "Farg'ona Viloyati",
    "Кашкадарьинская область": "Qashqadaryo Viloyati",
    "Хорезмская область": "Xorazm Viloyati",
    "Республика Каракалпакстан": "Qoraqalpogʻiston Respublikasi",
    "Сурхандарьинская область": "Surxondaryo Viloyati",
    "Сырдарьинская область": "Sirdaryo Viloyati",
    "Джизакская область": "Jizzax Viloyati",
    "Андижанская область": "Andijon Viloyati",
    "Наманганская область": "Namangan Viloyati",
}
TASHKENT_CITY_RU = "Ташкент"
TASHKENT_CITY_UZ = "Toshkent shahri"

#: Tashkent's districts, in the Latin Uzbek spelling the report uses. OLX names
#: them in Russian, Uybor in a mix of Russian and Latin Uzbek, and the archive
#: in Russian, so all three are folded onto one key or the same district becomes
#: several rows of every table.
DISTRICT_UZ = {
    "bektemir": "Bektemir",
    "chilonzor": "Chilonzor", "chilanzar": "Chilonzor",
    "mirobod": "Mirobod", "mirabad": "Mirobod",
    "mirzoulugbek": "Mirzo Ulug'bek", "mirzo-ulugbek": "Mirzo Ulug'bek",
    "olmazor": "Olmazor", "almazar": "Olmazor",
    "sergeli": "Sergeli", "sergeliy": "Sergeli",
    "shayxontohur": "Shayxontohur", "shaykhantakhur": "Shayxontohur",
    "shayhontohur": "Shayxontohur",
    "uchtepa": "Uchtepa", "uchtepin": "Uchtepa",
    "yakkasaroy": "Yakkasaroy", "yakkasaray": "Yakkasaroy",
    "yashnobod": "Yashnobod", "yashnabad": "Yashnobod",
    "yunusobod": "Yunusobod", "yunusabad": "Yunusobod",
    "yangihayot": "Yangihayot", "yangihayet": "Yangihayot",
}

#: Cyrillic district stems, transliterated onto the keys above.
_DISTRICT_RU = {
    "бектемир": "bektemir", "чиланзар": "chilonzor", "мирабад": "mirobod",
    "мирзоулугбек": "mirzoulugbek", "алмазар": "olmazor", "сергелий": "sergeli",
    "шайхантахур": "shayxontohur", "шайхантаур": "shayxontohur",
    "учтепин": "uchtepa", "учтепа": "uchtepa", "яккасарай": "yakkasaroy",
    "яшнабад": "yashnobod", "юнусабад": "yunusobod", "янгихаёт": "yangihayot",
    "янгихает": "yangihayot",
}

#: Words that mark the administrative unit rather than name it.
_DISTRICT_NOISE = ("район", "райони", "tumani", "tuman", "туман", "ский", "skiy", "cкий")

UNKNOWN = "Ko'rsatilmagan"


def _fold(text: str) -> str:
    """Lowercase, drop the unit word and every separator, so spellings meet."""
    lowered = " ".join(str(text or "").lower().split())
    for character in APOSTROPHES:
        lowered = lowered.replace(character, "")
    for noise in _DISTRICT_NOISE:
        lowered = lowered.replace(noise, " ")
    return "".join(ch for ch in lowered if ch.isalnum())


def canonical_district(name, region=None) -> str:
    """One Latin Uzbek spelling per Tashkent district, whatever arrived.

    Anything that is not one of the capital's districts stays unknown: the
    report's district tables are Tashkent's, and a settlement name from another
    region listed beside them would read as one.
    """
    text = " ".join(str(name or "").split())
    if not text or text.lower() in ("none", "nan", UNKNOWN.lower()):
        return UNKNOWN
    # The archive writes "Ташкент, <district>"; anything else has no district.
    if "," in text:
        head, _, tail = text.partition(",")
        if _fold(head) not in ("ташкент", "toshkent", "toshkentshahri"):
            return UNKNOWN
        text = tail
    folded = _fold(text)
    folded = _DISTRICT_RU.get(folded, folded)
    return DISTRICT_UZ.get(folded, UNKNOWN)


def canonical_region(region, city=None) -> str:
    """One spelling per region, whichever source and language it arrived in.

    The three sources disagree on alphabet, on transliteration and on which of
    six apostrophe characters to use, so the name is folded to a bare stem
    before it is looked up. Tashkent is the case that matters: the city and the
    surrounding region fold to the same stem, so the unit word — or the city
    name beside it — decides which of the two it is.

    An unrecognised name keeps its source spelling rather than being merged
    into a neighbour or dropped.
    """
    text = " ".join(str(region or "").split())
    if not text or text.lower() in ("none", "nan", UNKNOWN.lower()):
        return UNKNOWN
    folded, unit = _fold_region(text)
    canonical = REGION_KEY.get(folded)
    if canonical is None:
        return text
    if canonical == "Toshkent Viloyati":
        # OLX files the capital under the surrounding region and tells them
        # apart by the city, so the city name outranks the region's own word.
        city_folded, _ = _fold_region(city)
        if REGION_KEY.get(city_folded) == "Toshkent Viloyati" or unit == "city":
            return TASHKENT_CITY_UZ
    return canonical


def _fold_region(text: str):
    """A bare stem plus which administrative unit the name mentioned."""
    lowered = " ".join(str(text or "").lower().split())
    for character in APOSTROPHES:
        lowered = lowered.replace(character, "")
    unit = ""
    for name, words in _UNIT_WORDS.items():
        for word in words:
            if word in lowered:
                unit = unit or name
                lowered = lowered.replace(word, " ")
    return "".join(ch for ch in lowered if ch.isalnum()), unit


#: Every name the pooled tables know. Anything else is reported as it arrived,
#: and the archive series treats it as unplaced rather than inventing a row.
CANONICAL_REGIONS = set(REGION_KEY.values()) | {TASHKENT_CITY_UZ}


#: How a region is printed in a table: short enough for a five-column table to
#: fit beside its commentary, and the spelling the reference report uses. The
#: two Tashkents must stay apart, so the region keeps its word.
REGION_LABEL = {"Toshkent Viloyati": "Toshkent viloyati",
                "Toshkent shahri": "Toshkent shahri",
                "Qoraqalpogʻiston Respublikasi": "Qoraqalpog'iston"}

#: Table row order. Alphabetical by the Uzbek name, but with the capital and
#: its region kept together where a reader expects them.
REGION_ORDER = ["Andijon Viloyati", "Buxoro Viloyati", "Jizzax Viloyati",
                "Navoiy Viloyati", "Namangan Viloyati", "Samarqand Viloyati",
                "Sirdaryo Viloyati", "Surxondaryo Viloyati", "Toshkent Viloyati",
                "Toshkent shahri", "Farg'ona Viloyati", "Xorazm Viloyati",
                "Qashqadaryo Viloyati", "Qoraqalpogʻiston Respublikasi"]


def region_label(name) -> str:
    text = str(name)
    if text in REGION_LABEL:
        return REGION_LABEL[text]
    return text[:-len(" Viloyati")] if text.endswith(" Viloyati") else text


SCOPE_NOTE = (
    "Tarixiy qator faqat kvartira sotuvi e'lonlariga tegishli. Arxivda ijara "
    "e'lonlari yo'q, shuning uchun ijara bo'yicha tarixiy taqqoslash berilmaydi. "
    "Bular e'lon (so'ralgan) narxlari, bitim narxlari emas."
)


def _clean_text(series: pd.Series) -> pd.Series:
    """Archive text arrives quoted in some files and as the string 'nan' in others."""
    return (series.astype("string").str.strip().str.strip("'\"").str.strip()
            .replace({"nan": pd.NA, "none": pd.NA, "None": pd.NA, "": pd.NA}))


def _read_file(path: Path) -> pd.DataFrame:
    # Closed explicitly: sqlite3's context manager ends the transaction but
    # leaves the handle open, which keeps the archive file locked on Windows.
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        present = {row[1] for row in db.execute("PRAGMA table_info(olx_house_price)")}
        missing = set(COLUMNS) - present - OPTIONAL
        if missing:
            raise ValueError(f"{path.name}: ustunlar topilmadi: {sorted(missing)}")
        wanted = [name for name in COLUMNS if name in present]
        quoted = ", ".join(f'"{name}"' for name in wanted)
        frame = pd.read_sql(f"SELECT {quoted} FROM olx_house_price", db)
    finally:
        db.close()
    frame = frame.rename(columns=COLUMNS)
    for name in OPTIONAL:
        frame[COLUMNS[name]] = frame.get(COLUMNS[name], pd.NA)

    currency = _clean_text(frame["currency"]).str.lower()
    # A missing currency is not a missing price: whole files carry no currency
    # at all, and their price scale is unambiguously dollar-linked. Som-scale
    # values are excluded below by the price screen rather than assumed away.
    usd = currency.isin(USD_CODES) | currency.isna()
    frame = frame[usd & ~currency.isin(UZS_CODES)].copy()

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["area"] = pd.to_numeric(frame["area"], errors="coerce")
    frame = frame[frame["price"].between(MIN_PRICE_USD, MAX_PRICE_USD)
                  & frame["area"].between(MIN_AREA, MAX_AREA)]

    frame["market"] = _clean_text(frame["market_ru"]).map(MARKET_UZ).fillna("Aniqlanmagan")
    frame["region_uz"] = _clean_text(frame["region_uz"]).fillna(UNKNOWN)
    frame["district"] = [canonical_district(value)
                         for value in _clean_text(frame["location"]).fillna("")]
    # Computed here rather than read from the archive's own column, so the
    # historical and live figures are produced the same way.
    frame["usd_sqm"] = frame["price"] / frame["area"]
    frame["month"] = pd.to_datetime(_clean_text(frame["month"]), format="%m-%Y",
                                    errors="coerce").dt.to_period("M")
    return frame.dropna(subset=["month"])


def _aggregate(frame: pd.DataFrame, keys) -> pd.DataFrame:
    """Median asking price per group, taken over the adverts themselves."""
    grouped = frame.groupby(list(keys), observed=True)
    out = grouped.agg(listings=("usd_sqm", "size"),
                      months=("month", "nunique"),
                      median_usd_sqm=("usd_sqm", "median"),
                      median_usd=("price", "median")).reset_index()
    # A monthly row always covers one month; only a quarter can be partial.
    return out.drop(columns="months") if "month" in keys else out


def _tables(frame: pd.DataFrame) -> dict:
    """Every table this module publishes, from one read of one archive file.

    Regions and districts are aggregated from the adverts separately. Deriving
    one from the other would be a median of medians, which is not a median.
    """
    frame = frame.assign(quarter=frame["month"].dt.asfreq("Q"))
    capital = frame[frame["district"] != UNKNOWN]
    return {
        "monthly": _aggregate(frame, ["month", "region_uz", "market"]),
        "quarterly": _aggregate(frame, ["quarter", "region_uz", "market"]),
        "quarterly_district": _aggregate(capital, ["quarter", "district", "market"]),
    }


#: The key columns of each published table, used to merge parts and spot a
#: cache written before one of them existed.
TABLE_KEYS = {
    "monthly": ["month", "region_uz", "market"],
    "quarterly": ["quarter", "region_uz", "market"],
    "quarterly_district": ["quarter", "district", "market"],
}
PERIOD_FREQ = {"month": "M", "quarter": "Q"}


def archive_files(folder) -> list[Path]:
    return sorted(Path(folder).glob("*.db"))


def _cache_paths(cache) -> dict:
    """Where each table is cached, named after the monthly cache the caller gave."""
    cache = Path(cache)
    return {"monthly": cache,
            "quarterly": cache.with_name(f"{cache.stem}_quarterly{cache.suffix}"),
            "quarterly_district": cache.with_name(
                f"{cache.stem}_quarterly_district{cache.suffix}")}


def _read_cached(paths: dict) -> dict | None:
    """Every cached table, or None if any is missing or predates a column.

    A partial cache is treated as no cache: the tables are cut from one pass
    over the archive and must describe the same read.
    """
    tables = {}
    for name, path in paths.items():
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        keys = TABLE_KEYS[name]
        if not set(keys) <= set(frame.columns):
            return None  # written before districts or quarters existed
        period = keys[0]
        frame[period] = pd.PeriodIndex(frame[period], freq=PERIOD_FREQ[period])
        tables[name] = frame
    return tables


def _combine(parts: list[pd.DataFrame], keys: list[str], label: str, progress) -> pd.DataFrame:
    """Stack the per-file aggregates, refusing to average medians together.

    Each archive file holds whole months and whole quarters, so a group belongs
    to exactly one file and stacking is exact. If that ever stops being true the
    overlap is reported rather than silently resolved into a median of medians.
    """
    series = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=keys + ["listings", "median_usd_sqm", "median_usd"])
    duplicated = series.duplicated(keys, keep=False)
    if duplicated.any():
        periods = sorted({str(value) for value in series.loc[duplicated, keys[0]]})
        progress(f"DIQQAT: {label} jadvalida {', '.join(periods)} davri bir nechta "
                 "arxiv faylida uchradi; eng katta tanlanmali fayl olindi, "
                 "medianalar qo'shilmadi.")
    return (series.sort_values("listings")
            .drop_duplicates(keys, keep="last")
            .sort_values(keys, ignore_index=True))


def build_series(folder, *, cache=None, rebuild: bool = False, progress=print) -> dict:
    """The monthly, quarterly and Tashkent-district tables.

    One pass over the archive produces all three, because each is a different
    grouping of the same adverts and re-reading gigabytes per table is waste.
    """
    paths = _cache_paths(cache) if cache else None
    if paths and not rebuild:
        cached = _read_cached(paths)
        if cached is not None:
            return cached
    files = archive_files(folder)
    if not files:
        raise FileNotFoundError(f"Arxivda .db fayllar topilmadi: {folder}")
    parts = {name: [] for name in TABLE_KEYS}
    for index, path in enumerate(files, 1):
        progress(f"Tarixiy arxiv o'qilmoqda ({index}/{len(files)}): {path.name}")
        for name, table in _tables(_read_file(path)).items():
            parts[name].append(table)
    tables = {name: _combine(parts[name], TABLE_KEYS[name], name, progress)
              for name in TABLE_KEYS}
    if paths:
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            period = TABLE_KEYS[name][0]
            frame = tables[name]
            frame.assign(**{period: frame[period].astype(str)}).to_csv(path, index=False)
    return tables


def build_monthly(folder, *, cache=None, rebuild: bool = False, progress=print) -> pd.DataFrame:
    """The monthly series alone. See :func:`build_series`."""
    return build_series(folder, cache=cache, rebuild=rebuild, progress=progress)["monthly"]


def live_month(frame: pd.DataFrame, collected_at) -> pd.DataFrame:
    """Aggregate the live snapshot into one month of the same series.

    ``frame`` is the bulletin's normalised table. Only apartment sales are used,
    because those are the only listings the archive can be compared against.
    """
    sales = frame[(frame["kind"] == "sale") & (frame["property"] == "Kvartira")].copy()
    sales = sales.dropna(subset=["sqm_usd"])
    if sales.empty:
        return pd.DataFrame(columns=["month", "region_uz", "market", "listings",
                                     "median_usd_sqm", "median_usd"])
    month = pd.Period(pd.Timestamp(collected_at), freq="M")
    city = sales.get("city", pd.Series("", index=sales.index)).fillna("")
    region = sales["region"].fillna("")
    sales["region_uz"] = [
        TASHKENT_CITY_UZ if str(c).strip() == TASHKENT_CITY_RU
        else REGION_UZ_FROM_RU.get(str(r).strip(), "Ko'rsatilmagan")
        for r, c in zip(region, city)]
    grouped = sales.groupby(["region_uz", "market"], observed=True).agg(
        listings=("sqm_usd", "size"), median_usd_sqm=("sqm_usd", "median"),
        median_usd=("price_usd", "median")).reset_index()
    grouped.insert(0, "month", month)
    return grouped


def national(series: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Listing-weighted national figure per period, across regions and markets.

    This is a weighted mean of regional medians, not a national median: the
    cached tables hold medians, and a median cannot be recovered from medians.
    The report says so wherever this number appears.
    """
    if series.empty:
        return series
    weighted = series.assign(_w=series["median_usd_sqm"] * series["listings"])
    out = weighted.groupby(period, observed=True).agg(
        listings=("listings", "sum"), _w=("_w", "sum")).reset_index()
    out["median_usd_sqm"] = out["_w"] / out["listings"]
    return out.drop(columns="_w").sort_values(period, ignore_index=True)


#: A median is not shown for a group with fewer observations than this; the
#: count still is, so a thin group is visible rather than absent.
MIN_LISTINGS = 15

#: The archive's quarterly tables are a different scale entirely — most regions
#: bring tens of thousands of adverts a quarter — so the live snapshot's floor
#: would let a twenty-advert region be ranked beside them, and a quarter-on-
#: quarter move of a few listings would read as a market movement. A region
#: below this floor keeps its row and shows a dash.
MIN_QUARTER_LISTINGS = 100

#: Marks a change column, so the renderer knows to colour it by direction.
DELTA = "Δ"


#: Quarters are labelled as the reference report labels them: 2025-Ch3.
def quarter_label(quarter) -> str:
    quarter = pd.Period(quarter, freq="Q")
    return f"{quarter.year}-Ch{quarter.quarter}"


def in_som(usd_per_sqm, rate):
    """USD/m² at a dated rate, expressed in millions of so'm per m²."""
    if rate is None or pd.isna(usd_per_sqm):
        return float("nan")
    return usd_per_sqm * rate / 1_000_000


def comparison_table(quarterly: pd.DataFrame, group: str, quarters, rates,
                     *, market=None, label="Hudud",
                     minimum: int = MIN_QUARTER_LISTINGS) -> pd.DataFrame:
    """The reference's table shape: two dated levels and the change into each.

    ``quarters`` are the quarters to show, oldest first; the oldest supplies the
    first change column and is not printed as a level of its own. Values are
    millions of so'm per m² where a dated rate exists for every quarter shown,
    and dollars per m² where one does not — never one standing in for the other.

    A group observed too thinly to carry a median keeps its row and shows a
    dash, so the reader can tell "not enough adverts" from "not in the data".
    """
    quarters = [pd.Period(q, freq="Q") for q in quarters]
    if len(quarters) < 2 or quarterly.empty:
        return pd.DataFrame()
    selected = quarterly[quarterly["quarter"].isin(quarters)]
    if market:
        selected = selected[selected["market"] == market]
    if selected.empty:
        return pd.DataFrame()
    som = all(rates.get(q) for q in quarters)
    values = {}
    for quarter in quarters:
        rows = selected[selected["quarter"] == quarter]
        rate = rates.get(quarter)
        values[quarter] = {
            name: (in_som(median, rate) if som else median)
            for name, count, median in zip(rows[group], rows["listings"],
                                           rows["median_usd_sqm"])
            if count >= minimum}
    present = {name for name in selected[group].unique() if name != UNKNOWN}
    # The reference lists its regions in a fixed order; anything the order does
    # not know about follows it alphabetically rather than being dropped.
    known = [name for name in REGION_ORDER if name in present]
    names = known + sorted(present - set(known))
    table = pd.DataFrame({label: names})
    previous = None
    for index, quarter in enumerate(quarters):
        column = [values[quarter].get(name, float("nan")) for name in names]
        if index:  # the oldest quarter only supplies the first comparison
            change = [(new / old - 1) * 100 if old and new
                      and not pd.isna(old) and not pd.isna(new) else float("nan")
                      for new, old in zip(column, previous)]
            short, before = quarter_label(quarter)[5:], quarter_label(quarters[index - 1])[5:]
            table[quarter_label(quarter)] = [round(v, 2) for v in column]
            table[f"{DELTA} {short}/{before}"] = [round(v, 1) for v in change]
        previous = column
    return table


def thin_groups(quarterly: pd.DataFrame, group: str, quarter, *, market=None,
                minimum: int = MIN_QUARTER_LISTINGS) -> dict:
    """Groups whose sample in ``quarter`` is too small to carry a median."""
    quarter = pd.Period(quarter, freq="Q")
    rows = quarterly[quarterly["quarter"] == quarter]
    if market:
        rows = rows[rows["market"] == market]
    return {name: int(count) for name, count in zip(rows[group], rows["listings"])
            if count < minimum and name != UNKNOWN}


def index_series(quarterly: pd.DataFrame, rates=None) -> pd.DataFrame:
    """An asking-price index, the first observed quarter set to 100.

    Built from listings, not transactions, and weighted by listing count rather
    than by population, so it is not comparable with an official house-price
    index however similar the shape.
    """
    if quarterly.empty:
        return pd.DataFrame()
    levels = national(quarterly, "quarter")
    base = levels["median_usd_sqm"].iloc[0]
    levels = levels.assign(index=(levels["median_usd_sqm"] / base * 100).round(1))
    # How much of each quarter was actually observed: the archive's first file
    # skips months, and a one-month quarter is not a quarter.
    observed = quarterly.groupby("quarter", observed=True)["months"].max()
    levels["months"] = [int(observed.get(q, 0)) for q in levels["quarter"]]
    if rates:
        levels["som_sqm"] = [in_som(v, rates.get(q))
                             for v, q in zip(levels["median_usd_sqm"], levels["quarter"])]
    return levels


def gap_months(series: pd.DataFrame) -> list:
    """Months with no observation between the first and last, as Periods."""
    if series.empty:
        return []
    months = pd.PeriodIndex(sorted(series["month"].unique()), freq="M")
    full = pd.period_range(months.min(), months.max(), freq="M")
    return [p for p in full if p not in set(months)]
