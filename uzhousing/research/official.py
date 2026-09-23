"""Macro indicators the listing sites cannot supply, from official statistics.

Mortgage lending, construction volumes, household income, population and
interest rates are not on a classifieds site, and writing "not available" in
their place tells the reader nothing they did not already know. What the
report can do is fetch the official series, date it and cite it.

Two sources feed this module. The World Bank's open API carries Uzbekistan's
national accounts, prices, population and credit indicators in a stable,
machine-readable, dated form, and is used for everything it covers. Anything
it does not carry — quarterly mortgage disbursements, dwellings commissioned,
transaction registrations — is listed in ``knowledge/official_indicators.json``
with the Uzbek institution that publishes it, so the report names the source
instead of naming a gap. Filling a value in that file puts it in the table
with its own citation.

Every fetch degrades to what is already cached: an offline run reports the
indicators it has and says which are missing, and never guesses a figure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

WORLD_BANK = ("https://api.worldbank.org/v2/country/UZB/indicator/{code}"
              "?format=json&per_page=80")
WORLD_BANK_SOURCE = "https://data.worldbank.org/country/UZ"

#: Where the cached values are written, next to the other caches.
CACHE_NAME = "official_indicators.csv"

#: How long a cached value is used before the series is fetched again. The
#: underlying series are annual, so a week is generous.
CACHE_DAYS = 7

#: The curated pointers to Uzbek official publications.
KNOWLEDGE = Path(__file__).resolve().parents[2] / "knowledge" / "official_indicators.json"


@dataclass(frozen=True)
class Series:
    code: str
    name: str
    unit: str
    decimals: int = 1


#: The World Bank series the report reads, named as the bulletin names them.
SERIES = (
    Series("FP.CPI.TOTL.ZG", "Iste'mol narxlari inflyatsiyasi", "foiz, yillik", 1),
    Series("FR.INR.LEND", "Banklarning o'rtacha kredit stavkasi", "foiz", 1),
    Series("FS.AST.PRVT.GD.ZS", "Xususiy sektorga ichki kredit", "YaIMga nisbatan, foiz", 1),
    Series("NE.GDI.FTOT.ZS", "Asosiy kapitalga yalpi investitsiyalar", "YaIMga nisbatan, foiz", 1),
    Series("NY.GNP.PCAP.CD", "Aholi jon boshiga YaMD (Atlas usuli)", "AQSH dollari", 0),
    Series("SP.POP.TOTL", "Aholi soni", "kishi", 0),
    Series("SP.URB.TOTL.IN.ZS", "Shahar aholisi ulushi", "foiz", 1),
)


@dataclass
class Observation:
    name: str
    value: float
    unit: str
    period: str
    source: str
    url: str = ""
    decimals: int = 1

    def row(self) -> dict:
        return {"Ko'rsatkich": self.name, "Qiymat": round(self.value, self.decimals),
                "O'lchov": self.unit, "Davr": self.period, "Manba": self.source}


@dataclass
class Macro:
    """What was retrieved, what is still missing and where it is published."""
    observations: list
    missing: list
    cpi: dict
    fetched: bool = False

    def table(self) -> pd.DataFrame:
        if not self.observations:
            return pd.DataFrame()
        return pd.DataFrame([item.row() for item in self.observations])

    def get(self, name: str):
        for item in self.observations:
            if item.name == name:
                return item
        return None

    @property
    def sources(self) -> list:
        return sorted({item.source for item in self.observations})


def _cached(path: Path, today) -> list:
    if not path or not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if not {"name", "value", "unit", "period", "source", "fetched"} <= set(frame.columns):
        return []
    fresh = pd.Timestamp(today).tz_localize(None) - pd.Timedelta(days=CACHE_DAYS)
    rows = frame[pd.to_datetime(frame["fetched"], errors="coerce") >= fresh]
    return [Observation(str(row["name"]), float(row["value"]), str(row["unit"]),
                        str(row["period"]), str(row["source"]),
                        str(row.get("url", "")), int(row.get("decimals", 1)))
            for _, row in rows.iterrows()]


def _write_cache(path: Path, observations: list, today) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{"name": item.name, "value": item.value, "unit": item.unit,
                      "period": item.period, "source": item.source, "url": item.url,
                      "decimals": item.decimals},
                   "fetched": str(pd.Timestamp(today).date())}
                  for item in observations]).to_csv(path, index=False)


def _fetch(session, series: Series, timeout: int) -> tuple:
    """The latest observation of one World Bank series, and its history.

    Returns ``(Observation | None, {year: value})``; a series the API answers
    with no observations gives ``(None, {})`` rather than raising.
    """
    response = session.get(WORLD_BANK.format(code=series.code), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    history = {int(row["date"]): float(row["value"]) for row in rows or []
               if row.get("value") is not None and str(row.get("date", "")).isdigit()}
    if not history:
        return None, {}
    year = max(history)
    return Observation(series.name, history[year], series.unit, f"{year}-yil",
                       "Jahon banki, ochiq ma'lumotlar", WORLD_BANK_SOURCE,
                       series.decimals), history


def _pointers() -> list:
    """Indicators only the Uzbek institutions publish, from the knowledge file."""
    try:
        payload = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload.get("indicators") or [])


def gather(*, cache=None, session=None, progress=print, timeout: int = 20,
           today=None) -> Macro:
    """Every official indicator the report can stand on, with its source.

    The curated file is read first: a value filled in there is an official
    Uzbek figure and outranks anything else. The remaining series come from the
    World Bank API, or from the cache when the network is unavailable.
    """
    # tz-naive throughout: the cache stores plain dates and the two must compare.
    today = (pd.Timestamp(today) if today is not None
             else pd.Timestamp.utcnow().tz_localize(None)).normalize()
    cache = Path(cache) if cache else None
    observations, missing, cpi = [], [], {}

    for entry in _pointers():
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        value = entry.get("value")
        source = str(entry.get("source") or "").strip()
        if value is None:
            missing.append({"name": name, "source": source,
                            "url": str(entry.get("url") or "")})
            continue
        observations.append(Observation(
            name, float(value), str(entry.get("unit") or ""),
            str(entry.get("period") or ""), source, str(entry.get("url") or ""),
            int(entry.get("decimals", 1))))

    cached = _cached(cache, today)
    cpi = _cached_cpi(cache)
    have = {item.name for item in observations}
    fetched = False
    if len(cached) >= len(SERIES) - len(have):
        observations += [item for item in cached if item.name not in have]
    else:
        if session is None:
            import requests
            session = requests.Session()
        for series in SERIES:
            if series.name in have:
                continue
            try:
                observation, history = _fetch(session, series, timeout)
            except Exception as exc:
                progress(f"Rasmiy ko'rsatkich olinmadi ({series.name}): {exc}")
                break
            fetched = True
            if observation is None:
                missing.append({"name": series.name, "source": "Jahon banki",
                                "url": WORLD_BANK_SOURCE})
                continue
            observations.append(observation)
            if series.code == "FP.CPI.TOTL.ZG" and history:
                cpi = history
                _write_cpi(cache, history)
        world_bank = [item for item in observations
                      if item.source.startswith("Jahon banki")]
        if world_bank:
            _write_cache(cache, world_bank, today)
        else:
            observations += [item for item in cached
                             if item.name not in {o.name for o in observations}]

    if not cpi:
        cpi = _cpi_from(observations)
    observations.sort(key=lambda item: item.name)
    return Macro(observations, missing, cpi, fetched)


def _cpi_path(cache):
    """Where the inflation history is cached, beside the indicator cache.

    The deflator needs every year the index spans, not only the latest
    observation, so the whole series is kept rather than re-fetched.
    """
    return Path(cache).with_name("official_cpi.csv") if cache else None


def _cached_cpi(cache) -> dict:
    path = _cpi_path(cache)
    if not path or not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
        return {int(year): float(value)
                for year, value in zip(frame["year"], frame["inflation"])}
    except Exception:
        return {}


def _write_cpi(cache, history: dict) -> None:
    path = _cpi_path(cache)
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"year": sorted(history),
                  "inflation": [history[year] for year in sorted(history)]}
                 ).to_csv(path, index=False)


def _cpi_from(observations: list) -> dict:
    """One year of inflation is still a deflator for that year."""
    for item in observations:
        if item.name.startswith("Iste'mol narxlari") and item.period[:4].isdigit():
            return {int(item.period[:4]): item.value}
    return {}


def price_level(cpi: dict, quarters) -> dict:
    """A quarterly consumer-price level from annual inflation, base 1,0.

    Annual inflation is compounded into a yearly level and spread evenly across
    the four quarters of its year. That is an approximation — the report says so
    where it uses it — but it is enough to tell a nominal asking-price movement
    from a real one, which an unadjusted index cannot do at all.

    Official annual inflation is published with a lag, so the current year is
    usually missing. The last published rate is carried forward for those years
    rather than dropping the adjustment altogether; :func:`carried_years` names
    them so the text can say which part of the comparison is an assumption. A
    gap *before* the first observed year cannot be filled that way, and the
    adjustment is refused instead.
    """
    quarters = [pd.Period(q, freq="Q") for q in quarters]
    if not cpi or not quarters:
        return {}
    years = range(min(quarters).year, max(quarters).year + 1)
    known = dict(cpi)
    if min(years) < min(known):
        return {}
    last = known[max(known)]
    for year in years:
        known.setdefault(year, last)
    cpi = known
    level, yearly = 1.0, {}
    for year in years:
        yearly[year] = (level, level * (1 + cpi[year] / 100))
        level = yearly[year][1]
    out = {}
    for quarter in quarters:
        start, end = yearly[quarter.year]
        out[quarter] = start * (end / start) ** (quarter.quarter / 4)
    base = out[min(quarters)]
    return {quarter: value / base for quarter, value in out.items()}


def carried_years(cpi: dict, quarters) -> list:
    """Years whose inflation was assumed equal to the last published year."""
    quarters = [pd.Period(q, freq="Q") for q in quarters]
    if not cpi or not quarters:
        return []
    return [year for year in range(min(quarters).year, max(quarters).year + 1)
            if year not in cpi and year > max(cpi)]
