"""
Generates a SYNTHETIC macroeconomic dataset for testing the AI Economic
Research Agent. Values are artificially constructed and are NOT real
Uzbekistan statistics -- do not present them as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "synthetic_inflation_data.xlsx"


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", "2026-08-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    # Headline inflation: starts high (~11%), declines with noise, with a
    # deliberate acceleration bump in food inflation mid-sample to exercise
    # the anomaly/finding-detection logic.
    headline = 11.5 - 0.045 * t + rng.normal(0, 0.35, n)
    headline = np.clip(headline, 3, None)

    food = 13.0 - 0.05 * t + rng.normal(0, 0.6, n)
    bump_start, bump_end = 30, 36
    food[bump_start:bump_end] += np.linspace(0, 4.0, bump_end - bump_start)
    food[bump_end:bump_end + 6] += np.linspace(4.0, 0, 6)

    core = 10.0 - 0.035 * t + rng.normal(0, 0.3, n)
    services = 9.0 - 0.02 * t + rng.normal(0, 0.4, n)

    exchange_rate = 10800 + 25 * t + rng.normal(0, 40, n)  # local-currency units per USD, synthetic
    policy_rate = np.piecewise(t.astype(float), [t < 18, (t >= 18) & (t < 40), t >= 40],
                                [16.0, 14.0, 13.5]) + rng.normal(0, 0.05, n)
    wage_growth = 14 - 0.02 * t + rng.normal(0, 0.5, n)
    industrial_production = 100 + 0.6 * t + 3 * np.sin(t / 6) + rng.normal(0, 1.2, n)
    imports = 900 + 6 * t + rng.normal(0, 25, n)
    exports = 750 + 5 * t + rng.normal(0, 22, n)

    df = pd.DataFrame({
        "date": dates,
        "headline_inflation": np.round(headline, 2),
        "food_inflation": np.round(food, 2),
        "core_inflation": np.round(core, 2),
        "services_inflation": np.round(services, 2),
        "exchange_rate": np.round(exchange_rate, 1),
        "policy_rate": np.round(policy_rate, 2),
        "wage_growth": np.round(wage_growth, 2),
        "industrial_production_index": np.round(industrial_production, 2),
        "imports_musd": np.round(imports, 1),
        "exports_musd": np.round(exports, 1),
    })
    return df


def main():
    df = generate()
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="monthly_data", index=False)
        notes = pd.DataFrame({
            "Notes": [
                "SYNTHETIC DATA -- for testing the AI Economic Research Agent only.",
                "All values are artificially generated and do NOT represent real Uzbekistan statistics.",
                "headline/food/core/services_inflation: %, year-on-year (synthetic).",
                "exchange_rate: local currency units per USD (synthetic level).",
                "policy_rate: %, central bank policy rate (synthetic).",
                "wage_growth: %, year-on-year nominal wage growth (synthetic).",
                "industrial_production_index: 2021-01=100 (synthetic).",
                "imports_musd / exports_musd: USD millions (synthetic).",
            ]
        })
        notes.to_excel(writer, sheet_name="readme", index=False)
    print(f"Synthetic dataset written to {OUT_PATH}")


if __name__ == "__main__":
    main()
