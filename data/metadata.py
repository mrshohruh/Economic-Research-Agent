"""
Stage 1: Data Understanding. Builds the structured DatasetMetadata object.
Unit/definition inference is heuristic-only unless an LLM is available; when
ambiguous, variables are explicitly flagged rather than guessed (per
requirement: "do not invent definitions").
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from agents.llm_interface import LLM
from data.ingestion import SheetData
from models.schemas import DatasetMetadata, SheetMeta, VariableMeta

UNIT_PATTERNS = [
    (r"%|percent|pct|rate|inflation|growth|yoy|mom|qoq", "percent"),
    (r"index|idx", "index"),
    (r"usd|\$|dollar", "USD"),
    (r"uzs|som", "UZS"),
    (r"eur|euro", "EUR"),
    (r"million", "millions"),
    (r"billion", "billions"),
    (r"thousand", "thousands"),
    (r"ratio", "ratio"),
]


def _guess_unit(col_name: str) -> tuple[str | None, str]:
    name = col_name.lower()
    for pattern, unit in UNIT_PATTERNS:
        if re.search(pattern, name):
            return unit, "medium"
    return None, "low"


def _guess_definition(col_name: str) -> str | None:
    name = col_name.replace("_", " ").strip()
    if not name:
        return None
    return name[0].upper() + name[1:]


def build_variable_meta(sd: SheetData, col: str) -> VariableMeta:
    s = sd.df[col]
    is_numeric = col in sd.numeric_columns
    kind = "numeric" if is_numeric else ("categorical" if col in sd.categorical_columns else "unknown")
    n = len(s)
    missing = int(s.isna().sum())
    unit, unit_conf = (_guess_unit(str(col)) if is_numeric else (None, "low"))
    confidence = "high" if unit and unit_conf == "medium" else "low"
    numeric_s = pd.to_numeric(s, errors="coerce") if is_numeric else None
    return VariableMeta(
        name=str(col),
        column=str(col),
        unit=unit,
        dtype=str(s.dtype),
        kind=kind,
        definition=_guess_definition(str(col)),
        confidence=confidence,
        missing_count=missing,
        missing_pct=round(missing / n * 100, 2) if n else 0.0,
        n_obs=int(s.notna().sum()),
        min_value=float(numeric_s.min()) if is_numeric and numeric_s.notna().any() else None,
        max_value=float(numeric_s.max()) if is_numeric and numeric_s.notna().any() else None,
        mean_value=float(numeric_s.mean()) if is_numeric and numeric_s.notna().any() else None,
        notes=([] if unit else ["Unit could not be confidently inferred from the column name."]),
    )


def build_dataset_metadata(file_path: str | Path, sheets: dict[str, SheetData],
                            primary_sheet: str) -> DatasetMetadata:
    sheet_metas: list[SheetMeta] = []
    ambiguous: list[str] = []

    for name, sd in sheets.items():
        variables = [build_variable_meta(sd, c) for c in list(sd.numeric_columns) + list(sd.categorical_columns)]
        for v in variables:
            if v.confidence == "low" and v.kind == "numeric":
                ambiguous.append(f"{name}.{v.name}")

        period_start = period_end = None
        if sd.date_column is not None:
            dates = pd.to_datetime(sd.df[sd.date_column], errors="coerce").dropna()
            if not dates.empty:
                period_start = str(dates.min().date())
                period_end = str(dates.max().date())

        sheet_metas.append(SheetMeta(
            sheet_name=name,
            n_rows=len(sd.df),
            n_cols=len(sd.df.columns),
            date_column=sd.date_column,
            frequency=sd.frequency,
            period_start=period_start,
            period_end=period_end,
            variables=variables,
        ))

    meta = DatasetMetadata(
        dataset_name=Path(file_path).stem,
        file_path=str(file_path),
        sheets=sheet_metas,
        primary_sheet=primary_sheet,
        ambiguous_variables=ambiguous,
    )
    return meta


def refine_definitions_with_llm(meta: DatasetMetadata, topic: str) -> DatasetMetadata:
    """Optional LLM pass to improve variable definitions/units for ambiguous
    columns given the research topic. Never fabricates: if the LLM is
    unavailable, metadata from heuristics is returned unchanged."""
    if not LLM.enabled or not meta.ambiguous_variables:
        return meta

    var_list = "\n".join(
        f"- {v.column} (sheet: {sh.sheet_name}, sample stats: min={v.min_value}, max={v.max_value}, mean={v.mean_value})"
        for sh in meta.sheets for v in sh.variables if f"{sh.sheet_name}.{v.name}" in meta.ambiguous_variables
    )
    system = ("You are a meticulous data analyst. Given column names and summary statistics from an "
               "economic dataset, propose the most likely economic definition and unit for each column. "
               "Only use standard economic knowledge; if truly unclear, say 'unclear'. "
               "Respond as compact JSON: {\"column_name\": {\"definition\": str, \"unit\": str, \"confidence\": "
               "\"high|medium|low\"}}")
    user = f"Research topic: {topic}\n\nColumns:\n{var_list}"
    result = LLM.complete_json(system, user)
    if not isinstance(result, dict):
        return meta

    for sh in meta.sheets:
        for v in sh.variables:
            info = result.get(v.column)
            if isinstance(info, dict) and info.get("definition") and info.get("definition") != "unclear":
                v.definition = info["definition"]
                v.unit = info.get("unit", v.unit)
                v.confidence = info.get("confidence", v.confidence)
    return meta
