"""A data-quality audit of the pooled listing cross-section.

Listing sites are not statistical registers. The same flat is advertised on two
sites, a monthly rent is posted in the sale category, a studio is filed with
nine rooms, and a place name arrives in a spelling no region list knows. Left
alone, each of those lands in a median and then in a headline sentence.

This module screens the cleaned frame before any table is built. Nothing is
dropped silently: every screen reports how many records it touched and what it
did with them, and the report prints that as a table of its own. Where a value
is merely suspicious rather than impossible, the per-square-metre figure is
dropped and the record is kept, so the listing still counts towards the breadth
of supply without pricing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ingest.price_history import CANONICAL_REGIONS, UNKNOWN

#: Plausible asking prices per square metre. Sales below the floor are rents
#: filed in the wrong category or prices quoted for a share of a dwelling;
#: monthly rents above the ceiling are sale prices filed as rents.
SALE_BAND_USD = (80.0, 20_000.0)
RENT_BAND_USD = (0.3, 200.0)

#: A dwelling with more rooms than this is a mis-entered field, not a home the
#: rental-yield strata should be keyed on.
MAX_ROOMS = 12

#: How far into the tail of a group's own distribution a price may sit before
#: it is treated as a data error rather than an expensive flat. Measured in
#: median absolute deviations of the logarithm, which a skewed price
#: distribution tolerates far better than a standard deviation.
MAD_LIMIT = 4.0
MIN_GROUP_FOR_MAD = 40

#: Gross yields outside this band are artefacts of a mismatched stratum, not
#: returns a landlord could earn.
YIELD_BAND = (1.0, 25.0)

#: Keys that identify the same dwelling advertised twice.
DUPLICATE_KEYS = ["kind", "property", "region", "district", "rooms", "_area", "_price"]


@dataclass
class Finding:
    """One screen: what it looked for, what it found and what it did."""
    code: str
    check: str
    affected: int
    action: str
    detail: str = ""
    severity: str = "info"


@dataclass
class Audit:
    frame: pd.DataFrame
    findings: list = field(default_factory=list)

    @property
    def flagged(self) -> list:
        return [finding for finding in self.findings if finding.affected]

    @property
    def serious(self) -> list:
        return [finding for finding in self.flagged if finding.severity == "warn"]

    def table(self) -> pd.DataFrame:
        """The audit as the report prints it, screens that found nothing included.

        A screen that found nothing is evidence too: it says the check ran.
        """
        if not self.findings:
            return pd.DataFrame()
        return pd.DataFrame({
            "Tekshiruv": [finding.check for finding in self.findings],
            "Yozuvlar": [finding.affected for finding in self.findings],
            "Qaror": [finding.action for finding in self.findings]})

    def lines(self) -> list:
        """The screens that found something, as sentences for a callout."""
        return [finding.detail for finding in self.flagged if finding.detail]


def _finding(code, check, affected, action, detail="", severity="info") -> Finding:
    return Finding(code, check, int(affected), action, detail, severity)


def audit(frame: pd.DataFrame, *, screened: int = 0) -> Audit:
    """Screen the pooled cross-section and report every screen that ran.

    ``screened`` is how many records the price, currency and area cleaner had
    already rejected upstream, so the audit accounts for the whole distance
    between what was collected and what is priced here.
    """
    frame = frame.copy()
    findings = []

    if screened:
        findings.append(_finding(
            "screened", "Narx, valyuta va maydon tekshiruvi", screened,
            "tahlildan chiqarildi",
            f"Narxi, valyutasi yoki maydoni yaroqsiz {screened} ta yozuv tozalash "
            "bosqichida chiqarib tashlandi."))

    unknown_region = ~frame["region"].isin(CANONICAL_REGIONS)
    frame["place_ok"] = ~unknown_region
    names = sorted({str(value) for value in frame.loc[unknown_region, "region"]
                    if str(value) != UNKNOWN})[:5]
    findings.append(_finding(
        "place", "Hudud nomining ro'yxatga mosligi", int(unknown_region.sum()),
        "hududiy jadvallardan chiqarildi",
        (f"Hudud nomi ma'lum ro'yxatga tushmagan {int(unknown_region.sum())} ta e'lon "
         "hududiy medianalarga kiritilmadi"
         + (f": {', '.join(names)}." if names else ".")) if unknown_region.any() else "",
        severity="warn" if unknown_region.mean() > 0.1 else "info"))

    frame["_area"] = frame["total_area"].round(0)
    frame["_price"] = frame["price_usd"].round(-2)
    priced = frame.dropna(subset=["_area", "_price"])
    cross = pd.Series(False, index=frame.index)
    if "source" in frame and frame["source"].nunique() > 1 and not priced.empty:
        sources = priced.groupby(DUPLICATE_KEYS, dropna=False)["source"].transform("nunique")
        cross.loc[priced.index] = (sources > 1) & priced.duplicated(DUPLICATE_KEYS,
                                                                    keep="first")
    findings.append(_finding(
        "cross_source", "Manbalar o'rtasidagi takroriy e'lonlar", int(cross.sum()),
        "bir marta hisobga olindi",
        f"Ikki saytda bir xil narx, maydon, xona soni va joylashuv bilan berilgan "
        f"{int(cross.sum())} ta yozuv bir marta hisobga olindi." if cross.any() else ""))

    within = int(priced[~cross.reindex(priced.index, fill_value=False)]
                 .duplicated(DUPLICATE_KEYS + ["source"], keep="first").sum())
    findings.append(_finding(
        "repeat", "Bir manbadagi bir xil takliflar", within,
        "sanoqda qoldirildi",
        f"Bitta saytda bir xil xususiyatlar bilan {within} ta e'lon uchradi. Bular "
        "qayta joylashtirilgan takliflar bo'lishi mumkin; identifikatorlari har xil "
        "bo'lgani uchun olib tashlanmadi, lekin e'lonlar soni taklif hajmini biroz "
        "oshirib ko'rsatadi." if within else ""))

    frame = frame[~cross].copy()

    rooms = pd.to_numeric(frame["rooms"], errors="coerce")
    bad_rooms = rooms.notna() & ((rooms < 1) | (rooms > MAX_ROOMS) | (rooms % 1 != 0))
    frame.loc[bad_rooms, "rooms"] = np.nan
    findings.append(_finding(
        "rooms", "Xonalar sonining maqbulligi", int(bad_rooms.sum()),
        "xona maydoni bo'sh qoldirildi",
        f"Xonalar soni 1–{MAX_ROOMS} oralig'idan tashqarida bo'lgan "
        f"{int(bad_rooms.sum())} ta e'lon rentabellik qatlamlariga olinmadi."
        if bad_rooms.any() else ""))

    impossible = pd.Series(False, index=frame.index)
    for kind, (low, high) in (("sale", SALE_BAND_USD), ("rent", RENT_BAND_USD)):
        rows = (frame["kind"] == kind) & frame["sqm_usd"].notna()
        impossible |= rows & ~frame["sqm_usd"].between(low, high)
    findings.append(_finding(
        "classification", "Sotuv va ijara tasnifi", int(impossible.sum()),
        "m² narxi hisobga olinmadi",
        f"{int(impossible.sum())} ta e'londa m² narxi o'z toifasi uchun mumkin bo'lgan "
        "oraliqdan chiqib ketdi — ijara sotuv toifasida yoki aksincha joylashtirilgan "
        "bo'lishi mumkin; bu yozuvlar medianaga kiritilmadi." if impossible.any() else "",
        severity="warn" if impossible.sum() > 0.05 * max(len(frame), 1) else "info"))

    outlier = _tail_outliers(frame) & ~impossible
    findings.append(_finding(
        "outlier", "m² narxining taqsimotdagi o'rni", int(outlier.sum()),
        "m² narxi hisobga olinmadi",
        f"O'z guruhi taqsimotining {MAD_LIMIT:.0f} medianaviy og'ishidan uzoqda turgan "
        f"{int(outlier.sum())} ta e'lon medianaga kiritilmadi." if outlier.any() else ""))

    frame.loc[impossible | outlier, ["sqm_usd", "sqm_uzs"]] = np.nan
    return Audit(frame.drop(columns=["_area", "_price"]), findings)


def _tail_outliers(frame: pd.DataFrame) -> pd.Series:
    """Prices sitting implausibly far into their own group's log tail."""
    flags = pd.Series(False, index=frame.index)
    priced = frame.dropna(subset=["sqm_usd"])
    if priced.empty:
        return flags
    for _, group in priced.groupby(["kind", "property"], observed=True):
        if len(group) < MIN_GROUP_FOR_MAD:
            continue
        logged = np.log(group["sqm_usd"].clip(lower=1e-6))
        centre = logged.median()
        spread = (logged - centre).abs().median()
        if not spread:
            continue
        flags.loc[group.index] = (logged - centre).abs() > MAD_LIMIT * spread * 1.4826
    return flags


def screen_yields(table: pd.DataFrame, column: str):
    """Yields inside the plausible band, and the strata held back with them.

    A stratum whose gross yield lands outside the band is matching a rent and a
    sale that are not the same kind of dwelling — a mismatch, not a return.
    """
    if table is None or table.empty or column not in table:
        return table, pd.DataFrame()
    low, high = YIELD_BAND
    keep = table[column].between(low, high)
    return table[keep].reset_index(drop=True), table[~keep].reset_index(drop=True)
