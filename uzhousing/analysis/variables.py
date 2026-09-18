"""Offline, exploratory selection of property attributes associated with price."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from ..ingest.profiler import classify_column
from ..ingest.relevance import STRONG_EXCLUSIONS

ATTRIBUTE = re.compile(
    r"(^|_)(rooms?|bedrooms?|floor|floors|total_floors|ceiling_height|"
    r"building_type|condition|repairs|furnished|layout|bathroom|balcony|"
    r"parking|elevator|garden|heating|year_built|construction_year|"
    r"total_area|living_area|kitchen_area)(_|$)"
)


def assess_variables(frame: pd.DataFrame, target: str) -> list[dict]:
    """Audit columns using meaning, coverage and descriptive price associations."""
    decisions = []
    y = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name in frame.columns:
        s = frame[name]
        role, _, _, _ = classify_column(s)
        item = {"name": str(name), "role": role, "selected": False,
                "reason": "", "observations": 0, "method": "", "association": None,
                "groups": []}
        decisions.append(item)
        if name == target or role in {"price", "price_per_sqm"}:
            item["reason"] = "Outcome or another price measure; excluded to avoid target leakage."
            continue
        if role in {"identifier", "date", "unit"} or any(
            re.search(pattern, str(name).lower()) for pattern, _ in STRONG_EXCLUSIONS
        ):
            item["reason"] = "Identifier, time, unit, or platform metadata."
            continue
        if role not in {"area", "region", "segment"} and not ATTRIBUTE.search(str(name).lower()):
            item["reason"] = "Meaning is not established as a property attribute; needs review."
            continue
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan) if numeric else s
        paired = pd.DataFrame({"x": x, "y": y}).dropna()
        item["observations"] = len(paired)
        if len(paired) < 30 or len(paired) < 0.2 * y.notna().sum():
            item["reason"] = "Insufficient paired coverage (30 rows and 20% coverage required)."
            continue
        if paired.x.nunique() < 2 or paired.y.nunique() < 2:
            item["reason"] = "No variation in the attribute or price."
            continue
        if numeric:
            rho = paired.x.rank().corr(paired.y.rank())
            if not np.isfinite(rho):
                item["reason"] = "Association could not be estimated."
                continue
            item.update(method="Spearman rank correlation", association=round(float(rho), 4))
            item["selected"] = bool(abs(rho) >= 0.15)
            item["reason"] = (
                "Candidate for further analysis: absolute rank correlation is at least 0.15."
                if item["selected"] else
                "Weak marginal association; retained as context, not evidence of irrelevance."
            )
        else:
            if paired.x.nunique() > 50:
                item["reason"] = "Too many categories for a stable automatic comparison."
                continue
            grouped = paired.groupby("x").y.agg(["size", "median"])
            grouped = grouped[grouped["size"] >= 15]
            if len(grouped) < 2:
                item["reason"] = "Fewer than two categories with at least 15 observations each."
                continue
            item["groups"] = [
                {"category": str(k), "observations": int(v["size"]), "median_price": float(v["median"])}
                for k, v in grouped.sort_values("median").iterrows()
            ]
            item.update(selected=True, method="Category medians",
                        reason="Supported category comparison; differences are descriptive.")
    return decisions
