"""Why did the target move? Correlation, lead/lag and regression attribution.

Everything here is association, not proof of causation, and the wording of the
outputs is deliberately careful about that — the report repeats these strings.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# How a driver is normally expected to move with house prices. Used only to flag
# results that contradict the usual textbook sign, never to change the numbers.
EXPECTED_SIGN = {
    "rate": -1,       # higher mortgage/policy rates -> weaker prices
    "mortgage": 1,    # more mortgage lending -> stronger prices
    "income": 1,
    "volume": 1,
    "supply": -1,     # more completions -> softer prices
    "inflation": 1,
    "fx": 1,
    "population": 1,
}

# Roles that measure the same underlying thing. A second measure of the target is
# not an explanation of it, and two members of one family in a regression split a
# single effect across two coefficients.
ROLE_FAMILY = {
    "price": "price",
    "price_per_sqm": "price",
    "rate": "rate",
    "mortgage": "credit",
    "volume": "activity",
    "supply": "supply",
    "income": "income",
    "inflation": "prices_general",
    "fx": "fx",
    "population": "demography",
    "area": "physical",
}


def _family(role: str) -> str:
    return ROLE_FAMILY.get(role, f"other:{role}")


@dataclass
class DriverLink:
    driver: str
    driver_label: str = ""
    driver_role: str = "value"
    correlation: float | None = None
    p_value: float | None = None
    n: int = 0
    best_lag: int = 0
    best_lag_correlation: float | None = None
    strength: str = "negligible"
    direction: str = "none"
    significant: bool = False
    consistent_with_theory: bool | None = None
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class RegressionResult:
    target: str = ""
    formula: str = ""
    r_squared: float | None = None
    adj_r_squared: float | None = None
    n: int = 0
    coefficients: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class DriverAnalysis:
    target: str
    target_label: str = ""
    target_role: str = "value"
    basis: str = "period-over-period growth"
    basis_note: str = ""
    links: list[DriverLink] = field(default_factory=list)
    regression: RegressionResult | None = None
    excluded: list[str] = field(default_factory=list)
    correlation_matrix: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_label": self.target_label or self.target,
            "target_role": self.target_role,
            "basis": self.basis,
            "basis_note": self.basis_note,
            "excluded": self.excluded,
            "links": [link.to_dict() for link in self.links],
            "regression": self.regression.to_dict() if self.regression else None,
        }

    @property
    def significant_links(self) -> list[DriverLink]:
        return [link for link in self.links if link.significant]


# ---------------------------------------------------------------------------
def build_panel(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Align every metric onto one date index."""
    frames = []
    for name, series in series_map.items():
        s = pd.Series(series).dropna().astype(float)
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        frames.append(s.rename(name))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def analyse_drivers(
    panel: pd.DataFrame,
    target: str,
    roles: dict[str, str] | None = None,
    max_lag: int = 6,
    periods_per_year: int = 12,
    target_label: str = "",
    labels: dict[str, str] | None = None,
) -> DriverAnalysis:
    """Relate every other column in the panel to the target."""
    roles = roles or {}
    labels = labels or {}
    target_role = roles.get(target, "value")
    target_label = target_label or labels.get(target, target)
    analysis = DriverAnalysis(
        target=target, target_label=target_label or target, target_role=target_role
    )
    if panel.empty or target not in panel.columns:
        return analysis

    # Work in growth rates: levels of two trending series correlate spuriously.
    growth, analysis.basis, analysis.basis_note = _to_growth(panel, periods_per_year)
    analysis.correlation_matrix = growth.corr(min_periods=4).round(3)

    for column in growth.columns:
        if column == target:
            continue
        role = roles.get(column, "value")
        # A second measure of the same thing is not an explanation of the first:
        # median deal price does not "drive" price per square metre.
        if role != "value" and _family(role) == _family(target_role):
            analysis.excluded.append(column)
            continue
        link = _link(growth, target, column, role, max_lag, target_role,
                     labels.get(column, column), target_label)
        if link is not None:
            analysis.links.append(link)

    analysis.links.sort(key=lambda link: abs(link.correlation or 0), reverse=True)
    analysis.regression = _regression(growth, target, analysis.links, labels=labels,
                                      target_label=target_label)
    if analysis.excluded and analysis.regression:
        analysis.regression.note += (
            " Excluded from the analysis as another measure of the target itself: "
            + ", ".join(labels.get(c, c) for c in analysis.excluded) + "."
        )
    if analysis.regression and analysis.basis.startswith("year-on-year"):
        analysis.regression.note += " " + analysis.basis_note
    return analysis


def _to_growth(panel: pd.DataFrame, periods_per_year: int = 12) -> tuple[pd.DataFrame, str, str]:
    """Convert levels to growth rates.

    Year-on-year is preferred wherever the history supports it: it is the basis
    housing analysts actually use, it strips out seasonality, and month-on-month
    changes in this kind of data are mostly measurement noise. The cost is that
    overlapping windows are serially correlated, which the note records.
    """
    lag = periods_per_year if periods_per_year > 1 and len(panel) >= 2 * periods_per_year + 4 else 1

    if lag > 1:
        basis = f"year-on-year growth ({lag}-period change)"
        note = (
            "Year-on-year windows overlap, so successive observations are serially correlated. "
            "The p-values are therefore indicative rather than exact and overstate significance "
            "somewhat; treat them as a ranking device."
        )
    else:
        basis = "period-over-period growth"
        note = ""

    out = {}
    for column in panel.columns:
        s = panel[column].astype(float)
        median = float(s.abs().median()) if s.notna().any() else 0.0
        if 0 < median < 100 and s.min() > -100:
            # Small values: already a rate or percentage. Take the change in
            # percentage points rather than the percent change of a percent.
            out[column] = s.diff(lag)
        else:
            out[column] = s.pct_change(lag).replace([np.inf, -np.inf], np.nan) * 100
    return pd.DataFrame(out, index=panel.index), basis, note


def _link(growth: pd.DataFrame, target: str, driver: str, role: str, max_lag: int,
          target_role: str = "value", driver_label: str = "",
          target_label: str = "") -> DriverLink | None:
    from scipy import stats

    pair = growth[[target, driver]].dropna()
    if len(pair) < 5:
        return None

    y = pair[target].to_numpy(dtype=float)
    x = pair[driver].to_numpy(dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, pvalue = stats.pearsonr(x, y)

    link = DriverLink(
        driver=driver,
        driver_label=driver_label or driver,
        driver_role=role,
        correlation=round(float(corr), 3),
        p_value=round(float(pvalue), 5),
        n=len(pair),
        significant=bool(pvalue < 0.05 and abs(corr) >= 0.3),
    )

    # Lead/lag: does the driver move first?
    best_corr, best_lag = corr, 0
    for lag in range(1, min(max_lag, len(pair) // 4) + 1):
        lagged = growth[[target]].join(growth[[driver]].shift(lag), rsuffix="_lag").dropna()
        if len(lagged) < 5:
            continue
        candidate = float(lagged.corr().iloc[0, 1])
        if np.isfinite(candidate) and abs(candidate) > abs(best_corr):
            best_corr, best_lag = candidate, lag
    link.best_lag = best_lag
    link.best_lag_correlation = round(float(best_corr), 3)

    link.strength = _strength(abs(corr))
    link.direction = "positive" if corr > 0 else "negative" if corr < 0 else "none"

    # EXPECTED_SIGN describes how a driver should move with *house prices*, so the
    # check only means anything when the target is a price series.
    expected = EXPECTED_SIGN.get(role) if target_role in {"price", "price_per_sqm"} else None
    if expected is not None and abs(corr) >= 0.2:
        link.consistent_with_theory = bool(np.sign(corr) == expected)

    link.interpretation = _interpret(link, target_label or target)
    return link


def _strength(value: float) -> str:
    if value >= 0.7:
        return "strong"
    if value >= 0.5:
        return "moderate"
    if value >= 0.3:
        return "weak but visible"
    return "negligible"


def _interpret(link: DriverLink, target: str) -> str:
    name = link.driver_label or link.driver
    if not link.significant:
        return (
            f"{name} shows no statistically reliable co-movement with {target} in this sample "
            f"(r = {link.correlation}, p = {link.p_value})."
        )

    lead = ""
    if link.best_lag > 0 and abs(link.best_lag_correlation or 0) > abs(link.correlation or 0) + 0.05:
        lead = (
            f" The relationship is strongest when {name} is led by {link.best_lag} period(s) "
            f"(r = {link.best_lag_correlation}), which suggests it turns before {target} does and "
            "can be read as an early-warning indicator."
        )

    theory = ""
    if link.consistent_with_theory is False:
        theory = (
            " Note that the sign is the opposite of what standard housing-market theory predicts, "
            "so the association is probably picking up a common third factor rather than a direct effect."
        )
    elif link.consistent_with_theory is True:
        theory = " The direction is what housing-market theory would predict."

    return (
        f"{name} has a {link.strength} {link.direction} association with {target} "
        f"(r = {link.correlation}, p = {link.p_value}, n = {link.n}).{lead}{theory} "
        "Association is not proof of causation."
    )


def _regression(growth: pd.DataFrame, target: str, links: list[DriverLink],
                collinearity_limit: float = 0.9, labels: dict[str, str] | None = None,
                target_label: str = "") -> RegressionResult | None:
    """Multivariate OLS on the strongest drivers, sample size permitting."""
    labels = labels or {}

    def name(column: str) -> str:
        return labels.get(column, column)

    dropped: list[str] = []
    candidates: list[str] = []
    chosen_families: dict[str, str] = {}
    for link in links:
        if abs(link.correlation or 0) < 0.25 or len(candidates) >= 4:
            continue
        # Two regressors that move together cannot be told apart: including both
        # inflates the standard errors and splits one effect across two coefficients.
        family = _family(link.driver_role)
        if link.driver_role != "value" and family in chosen_families:
            dropped.append(f"{name(link.driver)} (same family as {name(chosen_families[family])})")
            continue
        redundant = next(
            (
                chosen for chosen in candidates
                if abs(float(growth[[chosen, link.driver]].corr().iloc[0, 1] or 0)) > collinearity_limit
            ),
            None,
        )
        if redundant:
            dropped.append(f"{name(link.driver)} (moves with {name(redundant)})")
            continue
        candidates.append(link.driver)
        chosen_families[family] = link.driver

    if not candidates:
        return None

    data = growth[[target] + candidates].dropna()
    # Keep at least ~6 observations per regressor to avoid over-fitting.
    while len(candidates) > 1 and len(data) < 6 * len(candidates):
        candidates = candidates[:-1]
        data = growth[[target] + candidates].dropna()
    if len(data) < 8:
        return RegressionResult(
            target=target,
            note=f"Too few aligned observations ({len(data)}) to estimate a reliable regression.",
        )

    try:
        import statsmodels.api as sm

        y = data[target]
        X = sm.add_constant(data[candidates])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X).fit()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("regression failed: %s", exc)
        return RegressionResult(target=target, note=f"Regression could not be estimated: {exc}")

    coefficients = []
    for variable in model.params.index:
        coefficients.append(
            {
                "variable": "Intercept" if variable == "const" else name(variable),
                "coefficient": round(float(model.params[variable]), 4),
                "std_error": round(float(model.bse[variable]), 4),
                "t_stat": round(float(model.tvalues[variable]), 3),
                "p_value": round(float(model.pvalues[variable]), 5),
                "significant": bool(model.pvalues[variable] < 0.05),
            }
        )

    note = (
        "Estimated on growth rates rather than levels, to avoid the spurious correlation that "
        "two independently trending series always produce. Coefficients describe conditional "
        "association within this sample, not causal effects."
    )
    if dropped:
        note += " Excluded as collinear with a variable already in the model: " + "; ".join(dropped) + "."

    return RegressionResult(
        target=target,
        formula=f"{target_label or target} ~ " + " + ".join(name(c) for c in candidates),
        r_squared=round(float(model.rsquared), 4),
        adj_r_squared=round(float(model.rsquared_adj), 4),
        n=int(model.nobs),
        coefficients=coefficients,
        note=note,
    )
