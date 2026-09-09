"""
Stage 2: Data Validation. Runs objective checks and reports issues; never
silently modifies questionable data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.ingestion import SheetData
from models.schemas import ValidationIssue, ValidationReport


def validate_sheet(sd: SheetData) -> ValidationReport:
    issues: list[ValidationIssue] = []
    df = sd.df
    n = len(df)

    issues.append(ValidationIssue(level="ok", message=f"{n} observations loaded from sheet '{sd.sheet_name}'."))

    if sd.date_column:
        issues.append(ValidationIssue(level="ok", message=f"Date column identified: '{sd.date_column}'."))
        dates = pd.to_datetime(df[sd.date_column], errors="coerce")
        n_bad_dates = int(dates.isna().sum())
        if n_bad_dates:
            issues.append(ValidationIssue(
                level="warning",
                message=f"{n_bad_dates} row(s) have a date value that could not be parsed.",
            ))
        dup_dates = int(dates.duplicated(keep=False).sum())
        if dup_dates:
            issues.append(ValidationIssue(
                level="warning",
                message=f"{dup_dates} row(s) share a duplicate date.",
                details="Duplicate timestamps can indicate merged data sources or data-entry errors.",
            ))
        if not dates.dropna().is_monotonic_increasing:
            issues.append(ValidationIssue(
                level="warning",
                message="Observations are not sorted in ascending date order in the source file.",
            ))
        if sd.frequency:
            issues.append(ValidationIssue(level="ok", message=f"{sd.frequency.capitalize()} frequency detected."))
            full_range = pd.date_range(dates.min(), dates.max(),
                                        freq={"monthly": "MS", "quarterly": "QS", "annual": "AS",
                                              "daily": "D", "weekly": "W"}.get(sd.frequency))
            if sd.frequency in ("monthly", "quarterly", "annual", "daily", "weekly") and len(full_range) > 0:
                missing_periods = len(full_range) - dates.dropna().nunique()
                if missing_periods > 0:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"{missing_periods} missing observation period(s) detected relative to a "
                                f"regular {sd.frequency} calendar.",
                    ))
        else:
            issues.append(ValidationIssue(level="warning", message="Frequency could not be reliably determined "
                                                                     "(irregular spacing between observations)."))
    else:
        issues.append(ValidationIssue(level="warning", message="No date/period column could be identified."))

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append(ValidationIssue(level="warning", message=f"{dup_rows} fully duplicate row(s) detected."))

    for col in sd.numeric_columns:
        s = pd.to_numeric(df[col], errors="coerce")
        missing = int(s.isna().sum())
        if missing:
            issues.append(ValidationIssue(
                level="warning",
                message=f"Column '{col}' has {missing} missing value(s) ({missing / n * 100:.1f}% of rows).",
            ))
        finite = s.dropna()
        if len(finite) >= 8:
            q1, q3 = finite.quantile(0.25), finite.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                outliers = int(((finite < lo) | (finite > hi)).sum())
                if outliers:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Column '{col}' has {outliers} statistical outlier(s) (beyond 3x IQR).",
                        details="Flagged, not removed. Review before treating as data error vs. genuine event.",
                    ))
            # crude structural break heuristic: rolling mean shift > 2 std
            if len(finite) >= 20:
                roll = finite.rolling(max(4, len(finite) // 8)).mean()
                diffs = roll.diff().abs()
                thresh = diffs.std(skipna=True) * 3 if diffs.std(skipna=True) else None
                if thresh and (diffs > thresh).sum() > 0:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Possible structural break detected in '{col}' "
                                f"(large shift in rolling mean).",
                    ))

    if not sd.numeric_columns:
        issues.append(ValidationIssue(level="error", message="No numeric variables could be identified on this sheet."))
    else:
        issues.append(ValidationIssue(level="ok", message=f"{len(sd.numeric_columns)} numeric variable(s) validated."))

    return ValidationReport(
        n_observations=n,
        frequency_detected=sd.frequency,
        date_column=sd.date_column,
        issues=issues,
    )
