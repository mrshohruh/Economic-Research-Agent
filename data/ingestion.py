"""
Stage 1 groundwork: raw XLSX ingestion. Makes no assumption about a fixed
Excel structure -- inspects every sheet, infers the date column, frequency,
numeric/categorical variables, and flags ambiguity instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATE_NAME_HINTS = ("date", "period", "month", "year", "quarter", "time", "sana", "дата")


@dataclass
class SheetData:
    sheet_name: str
    df: pd.DataFrame
    date_column: str | None = None
    frequency: str | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)


def load_workbook(file_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load every sheet of the workbook as-is (no coercion)."""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheets: dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        header_row = _detect_header_row(xls, name)
        df = xls.parse(name, header=header_row)
        # Blank header cells come back as literal float NaN column labels.
        # These aren't real variables and their NaN label breaks pandas
        # column lookups downstream (KeyError: '[nan] not in index'), so
        # drop them here before they can enter the pipeline.
        df = df.loc[:, df.columns.notna()]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        sheets[name] = df
    return sheets


def _detect_header_row(xls: pd.ExcelFile, sheet_name: str, max_scan: int = 15) -> int:
    """Real-world economic spreadsheets often carry title/notes rows above the
    actual column headers. Read the top of the sheet with no header and pick
    the first row that looks like a header: mostly non-empty, mostly text, and
    followed by rows that are more numeric than it is."""
    raw = xls.parse(sheet_name, header=None, nrows=max_scan + 5)
    if raw.empty:
        return 0
    best_row, best_score = 0, -1.0
    for i in range(min(max_scan, len(raw) - 1)):
        row = raw.iloc[i]
        non_null = row.notna()
        if non_null.sum() < 2:
            continue
        text_frac = sum(isinstance(v, str) for v in row[non_null]) / non_null.sum()
        below = raw.iloc[i + 1: i + 6]
        below_numeric = pd.to_numeric(below.stack(), errors="coerce").notna().mean() if not below.empty else 0.0
        score = non_null.mean() + text_frac + below_numeric
        if score > best_score:
            best_row, best_score = i, score
    return best_row


def _looks_like_date_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # object dtype, or the string/str dtype that newer pandas assigns to text
    # columns -- both can hold dates stored as text.
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(20)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format=None)
        return parsed.notna().mean() > 0.8
    return False


def detect_date_column(df: pd.DataFrame) -> str | None:
    # Prefer a column whose name hints at a date/period.
    for col in df.columns:
        name = str(col).strip().lower()
        if any(hint in name for hint in DATE_NAME_HINTS) and _looks_like_date_series(df[col]):
            return col
    # Otherwise, scan for any column that parses cleanly as dates.
    for col in df.columns:
        if _looks_like_date_series(df[col]):
            return col
    return None


def infer_frequency(dates: pd.Series) -> str | None:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values().unique()
    if len(d) < 3:
        return None
    diffs = pd.Series(d).diff().dropna().dt.days
    median_gap = diffs.median()
    if median_gap <= 2:
        return "daily"
    if 6 <= median_gap <= 8:
        return "weekly"
    if 27 <= median_gap <= 32:
        return "monthly"
    if 88 <= median_gap <= 95:
        return "quarterly"
    if 350 <= median_gap <= 380:
        return "annual"
    return "irregular"


def classify_columns(df: pd.DataFrame, date_col: str | None) -> tuple[list[str], list[str]]:
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col == date_col:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() > 0.9 and s.notna().sum() > 0:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols


def analyze_sheet(sheet_name: str, df: pd.DataFrame) -> SheetData:
    date_col = detect_date_column(df)
    freq = None
    work = df.copy()
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.sort_values(date_col)
        freq = infer_frequency(work[date_col])
    numeric_cols, categorical_cols = classify_columns(work, date_col)
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    return SheetData(
        sheet_name=sheet_name,
        df=work,
        date_column=date_col,
        frequency=freq,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )


def load_and_analyze(file_path: str | Path) -> dict[str, SheetData]:
    raw = load_workbook(file_path)
    return {name: analyze_sheet(name, df) for name, df in raw.items()}


def choose_primary_sheet(sheets: dict[str, SheetData]) -> str:
    """Pick the sheet most likely to hold the main time series: has a date
    column, the most numeric columns, and the most rows."""
    best_name, best_score = None, -1.0
    for name, sd in sheets.items():
        score = len(sd.numeric_columns) * 10 + len(sd.df) * 0.01
        if sd.date_column is not None:
            score += 100
        if score > best_score:
            best_name, best_score = name, score
    return best_name or next(iter(sheets))
