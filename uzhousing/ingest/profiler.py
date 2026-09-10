"""Work out what the loaded tables actually mean.

This is the "understand the data" step: it assigns a semantic role to every
column (date, region, price, transaction volume, mortgage rate, income, ...),
finds the time grain, and reshapes the best table into a tidy long frame that
the analysis modules can consume. Heuristics do the work; the LLM, when
available, is used only to review and correct the mapping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..llm import LLM, LLMUnavailable
from .loader import Dataset

LOGGER = logging.getLogger(__name__)

# Roles that carry a measured quantity we can plot and analyse.
METRIC_ROLES = {
    "price",
    "price_per_sqm",
    "volume",
    "supply",
    "mortgage",
    "rate",
    "income",
    "inflation",
    "fx",
    "area",
    "population",
    "value",
}

# name pattern -> (role, unit hint). Patterns cover English, Russian and Uzbek
# because national statistics for Uzbekistan appear in all three.
ROLE_PATTERNS: list[tuple[str, str, str]] = [
    # --- time -------------------------------------------------------------
    (r"^(date|period|time|month|quarter|year|yr|dt|sana|oy|chorak|yil|дата|период|месяц|квартал|год)$", "date", ""),
    (r"(date|period|_month$|_quarter$|_year$|yearmonth|ym)", "date", ""),
    # --- geography --------------------------------------------------------
    (r"(region|oblast|viloyat|province|district|tuman|rayon|область|регион|район|territory|hudud)", "region", ""),
    (r"^(city|shahar|town|город|локация|location|area_name|market)$", "region", ""),
    # --- segmentation -----------------------------------------------------
    (r"(segment|market_type|property_type|housing_type|dwelling|turi|тип|сегмент|category|klass|class)", "segment", ""),
    (r"(rooms?|xona|комнат|bedroom)", "segment", "rooms"),
    # --- prices -----------------------------------------------------------
    (r"(price_per_sq|per_sqm|per_m2|price_m2|sqm_price|m2_narx|narx_m2|цена_за_кв|стоимость_кв)", "price_per_sqm", "per m²"),
    (r"(price|narx|cost|стоимость|цена|value_usd|value_uzs|median_price|avg_price|mean_price)", "price", "currency"),
    (r"(rent|ijara|аренда|arenda)", "price", "currency/month"),
    # --- activity ---------------------------------------------------------
    (r"(transaction|deals?|sales?|bitim|sotuv|сделк|продаж|turnover|registrations?)", "volume", "count"),
    (r"(volume|count|qty|quantity|number_of|num_|soni|количество|listings?)", "volume", "count"),
    (r"(supply|commission|completion|built|construct|quril|foydalanishga|введен|ввод|строит|permits?|starts?)", "supply", "units"),
    (r"(vacancy|inventory|stock|zapas|запас)", "supply", "units"),
    # --- finance ----------------------------------------------------------
    # The exchange rate must be matched before the generic /rate/ pattern, or
    # "usd_uzs_rate" is read as an interest rate and then rejected for being too large.
    (r"(^fx|_fx$|exchange_rate|usd_uzs|uzs_usd|kurs|курс|som_rate|soum_rate)", "fx", "UZS/USD"),
    (r"(cpi|inflation|inflyatsiya|инфляц|deflator)", "inflation", "%"),
    # /rate/ before /mortgage/, so "mortgage_rate_pct" is an interest rate while
    # "mortgage_loans_issued" stays a lending volume.
    (r"(rate|stavka|ставка|percent|pct|yield|interest|foiz|процент)", "rate", "%"),
    (r"(mortgage|ipoteka|ипотек|home_loan|housing_loan)", "mortgage", ""),
    (r"(income|wage|salary|maosh|earning|зарплат|доход|daromad)", "income", "currency"),
    (r"(gdp|yalpi|ввп|валов)", "value", ""),
    # --- physical ---------------------------------------------------------
    (r"(area|sqm|sq_m|m2|square|maydon|площад)", "area", "m²"),
    (r"(population|aholi|населен)", "population", "people"),
    # --- keys -------------------------------------------------------------
    (r"^(id|_id|uuid|key|code|kod|index|row_number|no)$", "identifier", ""),
    (r"(_id$|_code$|_key$)", "identifier", ""),
    (r"(currency|valyuta|валют|unit|birlik|единиц|measure)", "unit", ""),
]

MONTHS = {
    "jan": 1, "january": 1, "yan": 1, "янв": 1,
    "feb": 2, "february": 2, "fev": 2, "фев": 2,
    "mar": 3, "march": 3, "мар": 3,
    "apr": 4, "april": 4, "апр": 4,
    "may": 5, "мая": 5, "май": 5,
    "jun": 6, "june": 6, "июн": 6,
    "jul": 7, "july": 7, "июл": 7,
    "aug": 8, "august": 8, "авг": 8,
    "sep": 9, "sept": 9, "september": 9, "сен": 9,
    "oct": 10, "october": 10, "окт": 10,
    "nov": 11, "november": 11, "ноя": 11,
    "dec": 12, "december": 12, "дек": 12,
}


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str
    unit: str = ""
    confidence: float = 0.0
    reason: str = ""
    non_null: int = 0
    missing_pct: float = 0.0
    unique: int = 0
    samples: list[Any] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    @property
    def is_metric(self) -> bool:
        return self.role in METRIC_ROLES


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list[ColumnProfile]
    date_col: str | None = None
    region_col: str | None = None
    segment_col: str | None = None
    metric_cols: list[str] = field(default_factory=list)
    grain: str = "unknown"
    period_start: pd.Timestamp | None = None
    period_end: pd.Timestamp | None = None
    score: float = 0.0

    def column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def role_of(self, name: str) -> str:
        col = self.column(name)
        return col.role if col else "unknown"


@dataclass
class Understanding:
    """The agent's model of the dataset."""

    dataset: Dataset
    profiles: dict[str, TableProfile]
    primary: str
    tidy: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    llm_reviewed: bool = False

    @property
    def primary_profile(self) -> TableProfile:
        return self.profiles[self.primary]

    @property
    def has_time(self) -> bool:
        return self.primary_profile.date_col is not None and not self.tidy.empty

    @property
    def regions(self) -> list[str]:
        """Real regional breakdowns only — the 'National' placeholder is excluded.

        Supporting tables without a region column are stamped 'National' when they
        are merged in, so scope this to the primary table and drop the placeholder.
        """
        if self.tidy.empty or "region" not in self.tidy.columns:
            return []
        primary_rows = self.tidy[self.tidy["source_table"] == self.primary]
        names = primary_rows["region"].dropna().astype(str).unique().tolist()
        return sorted(n for n in names if n != "National")

    @property
    def metrics(self) -> list[str]:
        if "metric" in self.tidy.columns:
            return sorted(self.tidy["metric"].dropna().astype(str).unique().tolist())
        return []

    def metric_frame(self, metric: str) -> pd.DataFrame:
        """Rows for one metric, sorted by date."""
        if self.tidy.empty:
            return pd.DataFrame()
        sub = self.tidy[self.tidy["metric"] == metric].copy()
        return sub.sort_values("date")

    def national_series(self, metric: str) -> pd.Series:
        """One value per period for a metric, aggregated across regions."""
        sub = self.metric_frame(metric)
        if sub.empty:
            return pd.Series(dtype=float)
        role = self.role_for_metric(metric)
        agg = "sum" if role in {"volume", "supply", "population"} else "mean"
        series = sub.groupby("date")["value"].agg(agg).sort_index()
        return series.astype(float)

    def role_for_metric(self, metric: str) -> str:
        prof = self.primary_profile
        col = prof.column(metric)
        if col:
            return col.role
        for profile in self.profiles.values():
            col = profile.column(metric)
            if col:
                return col.role
        return "value"

    def unit_for_metric(self, metric: str) -> str:
        for profile in self.profiles.values():
            col = profile.column(metric)
            if col and col.unit:
                return col.unit
        return ""


# ---------------------------------------------------------------------------
def understand(dataset: Dataset, llm: LLM | None = None) -> Understanding:
    """Profile every table, pick the most informative one, and tidy it."""
    profiles = {name: profile_table(name, df) for name, df in dataset.tables.items()}
    notes = list(dataset.notes)

    llm_reviewed = False
    if llm is not None and llm.available:
        try:
            _llm_review(profiles, dataset, llm)
            llm_reviewed = True
        except LLMUnavailable as exc:
            notes.append(f"LLM schema review skipped: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("LLM schema review failed: %s", exc)
            notes.append(f"LLM schema review failed: {exc}")

    primary = max(profiles, key=lambda n: profiles[n].score)
    tidy = to_tidy(dataset.tables[primary], profiles[primary])

    # Metrics living in sibling tables that share the same time grain are worth
    # merging in, because drivers (rates, income, FX) usually sit in their own table.
    extra = _merge_supporting_tables(tidy, dataset, profiles, primary)
    if not extra.empty:
        tidy = pd.concat([tidy, extra], ignore_index=True)
        notes.append(
            f"merged supporting metrics from {extra['source_table'].nunique()} additional table(s)"
        )

    if tidy.empty:
        notes.append("no usable time series was found; analysis will be cross-sectional only")

    return Understanding(dataset, profiles, primary, tidy, notes, llm_reviewed)


# ---------------------------------------------------------------------------
def profile_table(name: str, df: pd.DataFrame) -> TableProfile:
    columns = [_profile_column(df[col]) for col in df.columns]
    profile = TableProfile(name=name, rows=len(df), columns=columns)

    profile.date_col = _pick_date_column(df, columns)
    profile.region_col = _pick_by_role(df, columns, "region")
    profile.segment_col = _pick_by_role(df, columns, "segment")
    profile.metric_cols = [
        c.name
        for c in columns
        if c.is_metric
        and c.name not in {profile.date_col, profile.region_col, profile.segment_col}
        and pd.api.types.is_numeric_dtype(df[c.name])
    ]

    if profile.date_col:
        dates = parse_dates(df[profile.date_col], df)
        valid = dates.dropna()
        if not valid.empty:
            profile.period_start = valid.min()
            profile.period_end = valid.max()
            profile.grain = infer_grain(valid)

    profile.score = _score_table(profile)
    return profile


def safe_nunique(series: pd.Series) -> int:
    """``nunique`` that survives unhashable cells such as lists or dicts.

    The loader normally encodes containers as text before anything gets here, but
    ``understand`` is public API and may be handed a raw DataFrame.
    """
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().astype(str).nunique())


def safe_unique(series: pd.Series, limit: int = 5) -> list[Any]:
    try:
        return series.dropna().unique()[:limit].tolist()
    except TypeError:
        return series.dropna().astype(str).unique()[:limit].tolist()


def _profile_column(series: pd.Series) -> ColumnProfile:
    role, unit, confidence, reason = classify_column(series)
    non_null = int(series.notna().sum())
    samples = safe_unique(series)
    prof = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        role=role,
        unit=unit,
        confidence=confidence,
        reason=reason,
        non_null=non_null,
        missing_pct=round(float(series.isna().mean() * 100), 2),
        unique=safe_nunique(series),
        samples=[_jsonable(v) for v in samples],
    )
    if pd.api.types.is_numeric_dtype(series) and non_null:
        prof.minimum = float(np.nanmin(series.astype(float)))
        prof.maximum = float(np.nanmax(series.astype(float)))
        prof.mean = float(np.nanmean(series.astype(float)))
    return prof


def classify_column(series: pd.Series) -> tuple[str, str, float, str]:
    """Assign a semantic role from the column name, then sanity-check values."""
    name = str(series.name).lower()

    for pattern, role, unit in ROLE_PATTERNS:
        if re.search(pattern, name):
            # A "rate"-named column holding huge numbers is a level, not a percent.
            if role == "rate" and pd.api.types.is_numeric_dtype(series):
                finite = series.dropna()
                if not finite.empty and float(finite.abs().median()) > 200:
                    return "value", "", 0.5, f"name matched /{pattern}/ but values are too large for a rate"
            if role == "date" and not _plausible_dates(series):
                continue
            return role, unit, 0.85, f"column name matched /{pattern}/"

    # Nothing matched by name: fall back to the values themselves.
    if _plausible_dates(series) and safe_nunique(series) > 2:
        return "date", "", 0.5, "values parse as dates"
    if pd.api.types.is_numeric_dtype(series):
        return "value", "", 0.3, "unnamed numeric measure"
    if safe_nunique(series) <= max(20, len(series) * 0.05):
        return "category", "", 0.4, "low-cardinality text"
    return "text", "", 0.2, "free text"


def _plausible_dates(series: pd.Series) -> bool:
    sample = series.dropna().head(80)
    if sample.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    parsed = _parse_period_values(sample)
    return parsed.notna().mean() > 0.8


def _pick_date_column(df: pd.DataFrame, columns: list[ColumnProfile]) -> str | None:
    candidates = [c for c in columns if c.role == "date"]
    if not candidates:
        # Separate year / month columns are common in statistical exports.
        if _find_col(df, r"^(year|yil|год)$") and _find_col(df, r"^(month|oy|месяц|quarter|chorak|квартал)$"):
            return _find_col(df, r"^(year|yil|год)$")
        return None
    return max(candidates, key=lambda c: (c.confidence, safe_nunique(df[c.name]))).name


def _pick_by_role(df: pd.DataFrame, columns: list[ColumnProfile], role: str) -> str | None:
    candidates = [c for c in columns if c.role == role and 1 < c.unique <= max(200, len(df) // 2)]
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.confidence, -c.unique)).name


def _find_col(df: pd.DataFrame, pattern: str) -> str | None:
    for col in df.columns:
        if re.search(pattern, str(col).lower()):
            return col
    return None


def _score_table(profile: TableProfile) -> float:
    """How useful is this table as the centrepiece of the report?"""
    score = 0.0
    score += 40 if profile.date_col else 0
    score += 15 if profile.region_col else 0
    score += 8 if profile.segment_col else 0
    score += 10 * min(len(profile.metric_cols), 6)
    score += min(profile.rows / 50.0, 25)
    if any(profile.role_of(c) in {"price", "price_per_sqm"} for c in profile.metric_cols):
        score += 25
    if profile.grain in {"monthly", "quarterly"}:
        score += 10
    return round(score, 2)


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------
def parse_dates(series: pd.Series, frame: pd.DataFrame | None = None) -> pd.Series:
    """Parse a period column, combining with year/month siblings when needed."""
    parsed = _parse_period_values(series)

    if frame is not None and parsed.notna().mean() < 0.5:
        parsed = _combine_year_month(frame)
    elif frame is not None and str(series.name).lower() in {"year", "yil", "год"}:
        combined = _combine_year_month(frame)
        if combined.notna().mean() >= parsed.notna().mean():
            parsed = combined
    return parsed


def _parse_period_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series(pd.to_datetime(series), index=series.index)

    if pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        if not numeric.empty:
            lo, hi = float(numeric.min()), float(numeric.max())
            if 1900 <= lo and hi <= 2100:  # plain years
                return pd.to_datetime(series.astype("Int64").astype(str), format="%Y", errors="coerce")
            if 190001 <= lo and hi <= 210012:  # YYYYMM
                return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m", errors="coerce")
            if 19000101 <= lo and hi <= 21001231:  # YYYYMMDD
                return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        return pd.Series(pd.NaT, index=series.index)

    text = series.astype(str).str.strip()
    special = text.map(_parse_special_period)
    if special.notna().mean() > 0.6:
        return pd.Series(pd.to_datetime(special), index=series.index)

    parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
    if parsed.notna().mean() < 0.6:
        alt = pd.to_datetime(text, errors="coerce", dayfirst=True, format="mixed")
        if alt.notna().mean() > parsed.notna().mean():
            parsed = alt
    return parsed


def _parse_special_period(value: str) -> pd.Timestamp | None:
    """Handle 2024Q1, Q1 2024, 2024-M03, "Jan 2024", "Янв 2024"."""
    if not value or value.lower() in {"nan", "none", "nat", ""}:
        return None
    v = value.strip().lower().replace("_", "-").replace("/", "-")

    m = re.fullmatch(r"(\d{4})\s*-?\s*q([1-4])", v) or re.fullmatch(r"q([1-4])\s*-?\s*(\d{4})", v)
    if m:
        groups = m.groups()
        year, quarter = (groups[0], groups[1]) if len(groups[0]) == 4 else (groups[1], groups[0])
        return pd.Timestamp(year=int(year), month=(int(quarter) - 1) * 3 + 1, day=1)

    m = re.fullmatch(r"(\d{4})\s*-?\s*m?(\d{1,2})", v)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)

    m = re.fullmatch(r"([a-zа-я]{3,12})\.?\s*-?\s*(\d{4})", v)
    if m:
        month = _month_number(m.group(1))
        if month:
            return pd.Timestamp(year=int(m.group(2)), month=month, day=1)

    m = re.fullmatch(r"(\d{4})\s*-?\s*([a-zа-я]{3,12})", v)
    if m:
        month = _month_number(m.group(2))
        if month:
            return pd.Timestamp(year=int(m.group(1)), month=month, day=1)
    return None


def _month_number(token: str) -> int | None:
    token = token.strip(". ")
    for key, num in MONTHS.items():
        if token.startswith(key):
            return num
    return None


def _combine_year_month(frame: pd.DataFrame) -> pd.Series:
    year_col = _find_col(frame, r"^(year|yil|год)$")
    if not year_col:
        return pd.Series(pd.NaT, index=frame.index)
    years = pd.to_numeric(frame[year_col], errors="coerce")

    month_col = _find_col(frame, r"^(month|oy|месяц|mon)$")
    quarter_col = _find_col(frame, r"^(quarter|chorak|квартал|qtr|q)$")

    if month_col is not None:
        months = _to_month_number(frame[month_col])
    elif quarter_col is not None:
        quarters = pd.to_numeric(
            frame[quarter_col].astype(str).str.extract(r"(\d)")[0], errors="coerce"
        )
        months = (quarters - 1) * 3 + 1
    else:
        months = pd.Series(1, index=frame.index)

    valid = years.notna() & months.notna()
    out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if valid.any():
        out.loc[valid] = pd.to_datetime(
            {
                "year": years[valid].astype(int),
                "month": months[valid].astype(int).clip(1, 12),
                "day": 1,
            }
        )
    return out


def _to_month_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric
    return series.astype(str).str.lower().map(_month_number)


def infer_grain(dates: pd.Series) -> str:
    unique = pd.Series(sorted(pd.Series(dates).dropna().unique()))
    if len(unique) < 2:
        return "single period"
    gaps = unique.diff().dropna().dt.days
    median = float(gaps.median())
    if median <= 2:
        return "daily"
    if median <= 10:
        return "weekly"
    if median <= 45:
        return "monthly"
    if median <= 135:
        return "quarterly"
    if median <= 200:
        return "semi-annual"
    return "annual"


PANDAS_FREQ = {
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QS",
    "semi-annual": "2QS",
    "annual": "YS",
}

PERIODS_PER_YEAR = {
    "daily": 365,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "semi-annual": 2,
    "annual": 1,
}


# ---------------------------------------------------------------------------
# Tidying
# ---------------------------------------------------------------------------
def to_tidy(df: pd.DataFrame, profile: TableProfile) -> pd.DataFrame:
    """Melt a wide table into date / region / segment / metric / value rows."""
    if not profile.date_col or not profile.metric_cols:
        return pd.DataFrame(columns=["date", "region", "segment", "metric", "value", "source_table"])

    work = df.copy()
    work["__date"] = parse_dates(work[profile.date_col], work)
    work = work[work["__date"].notna()]
    if work.empty:
        return pd.DataFrame(columns=["date", "region", "segment", "metric", "value", "source_table"])

    id_vars = ["__date"]
    if profile.region_col:
        work["__region"] = work[profile.region_col].astype(str).str.strip()
        id_vars.append("__region")
    if profile.segment_col:
        work["__segment"] = work[profile.segment_col].astype(str).str.strip()
        id_vars.append("__segment")

    metrics = [c for c in profile.metric_cols if c in work.columns]
    long = work[id_vars + metrics].melt(id_vars=id_vars, var_name="metric", value_name="value")
    long = long.rename(columns={"__date": "date", "__region": "region", "__segment": "segment"})

    if "region" not in long.columns:
        long["region"] = "National"
    if "segment" not in long.columns:
        long["segment"] = "All"

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["source_table"] = profile.name
    return long[["date", "region", "segment", "metric", "value", "source_table"]].reset_index(drop=True)


def _merge_supporting_tables(
    tidy: pd.DataFrame,
    dataset: Dataset,
    profiles: dict[str, TableProfile],
    primary: str,
) -> pd.DataFrame:
    """Pull time-indexed metrics out of the other tables so drivers are available."""
    frames: list[pd.DataFrame] = []
    existing = set(tidy["metric"].unique()) if not tidy.empty else set()

    for name, profile in profiles.items():
        if name == primary or not profile.date_col or not profile.metric_cols:
            continue
        other = to_tidy(dataset.tables[name], profile)
        if other.empty:
            continue
        other = other[~other["metric"].isin(existing)]
        if other.empty:
            continue
        frames.append(other)
        existing.update(other["metric"].unique())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Optional LLM review of the mapping
# ---------------------------------------------------------------------------
SCHEMA_SYSTEM = (
    "You are a data engineer specialising in housing-market and macroeconomic statistics "
    "for Uzbekistan and Central Asia. You map raw columns to semantic roles precisely."
)


def _llm_review(profiles: dict[str, TableProfile], dataset: Dataset, llm: LLM) -> None:
    payload = {
        name: {
            "rows": prof.rows,
            "guessed_date_column": prof.date_col,
            "guessed_region_column": prof.region_col,
            "guessed_segment_column": prof.segment_col,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "guessed_role": c.role,
                    "unique_values": c.unique,
                    "samples": c.samples,
                }
                for c in prof.columns
            ],
        }
        for name, prof in profiles.items()
    }

    prompt = f"""Below is the inferred schema of a dataset a housing-market analyst uploaded.

{_compact_json(payload)}

Allowed roles: date, region, segment, price, price_per_sqm, volume, supply, mortgage,
rate, income, inflation, fx, area, population, value, identifier, unit, category, text.

Meaning of the trickier roles:
- price_per_sqm: price expressed per square metre
- volume: transaction / deal / listing counts
- supply: newly built, commissioned or permitted housing
- mortgage: mortgage lending amounts or counts (a mortgage *interest rate* is "rate")
- value: a numeric measure that does not fit any other role
- identifier: a key with no analytical meaning

Return JSON of this exact shape, listing only columns whose role should CHANGE:
{{"corrections": [{{"table": "...", "column": "...", "role": "...", "unit": "...", "why": "..."}}],
  "primary_table": "<table best suited to be the centrepiece of a housing market report>",
  "dataset_summary": "<two sentences on what this dataset measures>"}}"""

    result = llm.complete_json(prompt, system=SCHEMA_SYSTEM, max_tokens=2000)
    if not isinstance(result, dict):
        return

    valid_roles = METRIC_ROLES | {"date", "region", "segment", "identifier", "unit", "category", "text"}
    for fix in result.get("corrections", []) or []:
        table, column = fix.get("table"), fix.get("column")
        role = str(fix.get("role", "")).strip()
        if table not in profiles or role not in valid_roles:
            continue
        col = profiles[table].column(column)
        if col is None:
            continue
        LOGGER.info("LLM re-labelled %s.%s: %s -> %s", table, column, col.role, role)
        col.role = role
        col.unit = str(fix.get("unit", col.unit))
        col.confidence = 0.95
        col.reason = f"LLM review: {fix.get('why', 'role corrected')}"

    # Roles changed, so recompute the derived fields for every table.
    for name, prof in profiles.items():
        df = dataset.tables[name]
        prof.date_col = _pick_date_column(df, prof.columns)
        prof.region_col = _pick_by_role(df, prof.columns, "region")
        prof.segment_col = _pick_by_role(df, prof.columns, "segment")
        prof.metric_cols = [
            c.name
            for c in prof.columns
            if c.is_metric
            and c.name not in {prof.date_col, prof.region_col, prof.segment_col}
            and pd.api.types.is_numeric_dtype(df[c.name])
        ]
        prof.score = _score_table(prof)

    preferred = result.get("primary_table")
    if preferred in profiles:
        profiles[preferred].score += 100


def _compact_json(obj: Any, limit: int = 6000) -> str:
    import json

    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    return text[:limit] + ("\n... (truncated)" if len(text) > limit else "")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)[:80] if not isinstance(value, (int, float, bool, type(None))) else value
