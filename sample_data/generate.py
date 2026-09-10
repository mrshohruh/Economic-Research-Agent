"""Generate a realistic synthetic Uzbek housing dataset for demos and tests.

The numbers are synthetic. The *shape* is deliberately realistic: a long upward
trend in nominal prices, a COVID interruption, a post-2021 boom, a cooling phase,
month-of-year seasonality in transactions, and regional divergence with Tashkent
city pulling ahead. That gives the analysis engine something real to find.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

REGIONS: dict[str, dict[str, float]] = {
    # name: base USD price per m², relative growth multiplier, market size weight
    "Tashkent City": {"base": 620, "growth": 1.28, "size": 1.00},
    "Tashkent Region": {"base": 380, "growth": 1.12, "size": 0.42},
    "Samarkand": {"base": 340, "growth": 1.15, "size": 0.36},
    "Bukhara": {"base": 300, "growth": 1.05, "size": 0.22},
    "Fergana": {"base": 285, "growth": 0.98, "size": 0.30},
    "Andijan": {"base": 270, "growth": 0.95, "size": 0.26},
    "Namangan": {"base": 260, "growth": 0.96, "size": 0.24},
    "Khorezm": {"base": 240, "growth": 0.92, "size": 0.18},
}

SEGMENTS = {"Primary (new build)": 1.12, "Secondary": 1.00}

# Month-of-year multipliers for transaction activity: quiet winters, busy autumn.
SEASONAL = [0.78, 0.82, 0.95, 1.03, 1.07, 1.05, 0.98, 1.02, 1.12, 1.15, 1.08, 0.95]

START_YEAR, START_MONTH = 2018, 1
END_YEAR, END_MONTH = 2026, 6


def _months() -> list[tuple[int, int]]:
    out = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (END_YEAR, END_MONTH):
        out.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def _cycle(t: int, total: int) -> float:
    """Nominal price path as a multiplier on the base, in dollar terms."""
    years = t / 12.0
    # Steady underlying appreciation.
    level = 1.0 + 0.052 * years
    # 2020 COVID interruption.
    level *= 1 - 0.075 * math.exp(-(((t - 27) / 6.0) ** 2))
    # 2021-2023 boom: remittance inflow, migration, mortgage expansion.
    level *= 1 + 0.20 / (1 + math.exp(-(t - 46) / 5.0))
    # 2024-2025 cooling as rates stay high and supply catches up.
    level *= 1 - 0.085 / (1 + math.exp(-(t - 76) / 6.0))
    return level


def _policy_rate(t: int) -> float:
    """Central bank policy rate path, in percent."""
    if t < 24:
        return 16.0
    if t < 30:
        return 15.0 + 0.5 * math.sin(t / 3.0)
    if t < 50:
        return 14.0
    if t < 62:
        return 17.0        # 2022 tightening
    if t < 80:
        return 14.5
    return 13.5


def build() -> dict:
    rng = random.Random(20260910)
    months = _months()
    total = len(months)

    housing_rows = []
    macro_rows = []

    for t, (year, month) in enumerate(months):
        cycle = _cycle(t, total)
        policy_rate = _policy_rate(t)
        mortgage_rate = policy_rate + 3.6 + 0.4 * math.sin(t / 7.0) + rng.gauss(0, 0.15)
        subsidised_rate = max(6.0, mortgage_rate - 8.0)
        inflation = max(
            5.0,
            14.5 - 0.055 * t + 2.2 * math.sin(t / 9.0) + rng.gauss(0, 0.35)
            + (2.6 if 74 <= t <= 84 else 0.0),  # energy tariff pass-through
        )
        fx = 8100 * (1 + 0.035 * (t / 12.0)) * (1 + 0.02 * math.sin(t / 11.0)) + rng.gauss(0, 45)
        wage = 265 * (1 + 0.085 * (t / 12.0)) * (1 + 0.03 * math.sin((t - 2) / 6.0)) + rng.gauss(0, 5)

        macro_rows.append(
            {
                "date": f"{year:04d}-{month:02d}-01",
                "policy_rate_pct": round(policy_rate, 2),
                "mortgage_rate_pct": round(mortgage_rate, 2),
                "subsidised_mortgage_rate_pct": round(subsidised_rate, 2),
                "cpi_inflation_yoy_pct": round(inflation, 2),
                "usd_uzs_rate": round(fx, 1),
                "avg_monthly_wage_usd": round(wage, 1),
                "mortgage_loans_issued_bn_uzs": round(
                    max(0.4, 2.1 * (1 + 0.10 * (t / 12.0)) * (1 - 0.030 * (mortgage_rate - 17.0))
                        * SEASONAL[month - 1] + rng.gauss(0, 0.12)),
                    2,
                ),
            }
        )

        for region, spec in REGIONS.items():
            regional_drift = 1 + (spec["growth"] - 1) * (t / total)
            for segment, premium in SEGMENTS.items():
                price = (
                    spec["base"] * premium * cycle * regional_drift
                    * (1 + rng.gauss(0, 0.011))
                )
                # Demand responds to the mortgage rate with a lag and to the season.
                lagged_rate = _policy_rate(max(0, t - 4)) + 3.6
                demand = (
                    1.0
                    * SEASONAL[month - 1]
                    * (1 - 0.045 * (lagged_rate - 17.5))
                    * (1 + 0.055 * (t / 12.0))
                )
                if 25 <= t <= 30:
                    demand *= 0.62  # lockdown
                transactions = max(
                    40,
                    int(1150 * spec["size"] * (0.55 if segment == "Primary (new build)" else 0.45)
                        * demand * (1 + rng.gauss(0, 0.07))),
                )
                commissioned = max(
                    0,
                    int(transactions * (0.68 if segment == "Primary (new build)" else 0.0)
                        * (1 + rng.gauss(0, 0.14))),
                )
                housing_rows.append(
                    {
                        "date": f"{year:04d}-{month:02d}-01",
                        "region": region,
                        "segment": segment,
                        "avg_price_per_sqm_usd": round(price, 1),
                        "median_deal_price_usd": round(price * rng.uniform(52, 68), 0),
                        "transactions": transactions,
                        "new_units_commissioned": commissioned,
                        "avg_area_sqm": round(rng.uniform(54, 72), 1),
                    }
                )

    return {
        "metadata": {
            "title": "Uzbekistan housing market — synthetic demonstration dataset",
            "note": "Synthetic data generated for testing. Not official statistics.",
            "currency": "USD",
            "frequency": "monthly",
            "coverage": f"{START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}",
        },
        "regional_housing": housing_rows,
        "macro_indicators": macro_rows,
    }


def write_sample(path: Path | None = None) -> Path:
    path = Path(path or HERE / "uz_housing_sample.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_sql_sample(path: Path | None = None) -> Path:
    """The same data as a SQL script, to exercise the SQL ingestion path."""
    path = Path(path or HERE / "uz_housing_sample.sql")
    payload = build()

    lines = [
        "-- Synthetic Uzbekistan housing market data (demo only, not official statistics).",
        "DROP TABLE IF EXISTS regional_housing;",
        "CREATE TABLE regional_housing (",
        "  date TEXT, region TEXT, segment TEXT,",
        "  avg_price_per_sqm_usd REAL, median_deal_price_usd REAL,",
        "  transactions INTEGER, new_units_commissioned INTEGER, avg_area_sqm REAL",
        ");",
        "DROP TABLE IF EXISTS macro_indicators;",
        "CREATE TABLE macro_indicators (",
        "  date TEXT, policy_rate_pct REAL, mortgage_rate_pct REAL,",
        "  subsidised_mortgage_rate_pct REAL, cpi_inflation_yoy_pct REAL,",
        "  usd_uzs_rate REAL, avg_monthly_wage_usd REAL, mortgage_loans_issued_bn_uzs REAL",
        ");",
    ]

    for row in payload["regional_housing"]:
        lines.append(
            "INSERT INTO regional_housing VALUES ("
            f"'{row['date']}', '{row['region'].replace(chr(39), chr(39) * 2)}', "
            f"'{row['segment'].replace(chr(39), chr(39) * 2)}', "
            f"{row['avg_price_per_sqm_usd']}, {row['median_deal_price_usd']}, "
            f"{row['transactions']}, {row['new_units_commissioned']}, {row['avg_area_sqm']});"
        )
    for row in payload["macro_indicators"]:
        lines.append(
            "INSERT INTO macro_indicators VALUES ("
            f"'{row['date']}', {row['policy_rate_pct']}, {row['mortgage_rate_pct']}, "
            f"{row['subsidised_mortgage_rate_pct']}, {row['cpi_inflation_yoy_pct']}, "
            f"{row['usd_uzs_rate']}, {row['avg_monthly_wage_usd']}, "
            f"{row['mortgage_loans_issued_bn_uzs']});"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"JSON: {write_sample()}")
    print(f"SQL:  {write_sql_sample()}")
