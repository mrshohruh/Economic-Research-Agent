"""Quarter-end official USD/UZS rates from the Central Bank of Uzbekistan.

The archive's asking prices are dollar-linked, so reporting them in so'm needs
the rate that applied at the end of each quarter, not today's rate. The
Central Bank serves a dated rate at
``cbu.uz/uz/arkhiv-kursov-valyut/json/USD/<YYYY-MM-DD>/``.

Three traps are handled here rather than left to the caller. The same endpoint
also accepts ``dd.mm.yyyy``, but silently answers **today's** rate for it, so a
response is accepted only when the ``Date`` it carries is the date that was
asked for. Several quarters end on a weekend or an Uzbek public holiday, when
no rate is published at all, so the search walks back to the last rate that was
in force before the quarter closed and records which date that was. And a
quarter that has not ended yet has no quarter-end rate; those are left out
rather than filled with the nearest one.

Every rate is cached, so the archive is priced from one small local file after
the first run. A fetch that fails returns what is already known: the bulletin
then reports dollars and says why, which is preferable to pricing a 2022 advert
at a 2026 rate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The dated endpoint. Only the ISO form returns the rate for a past date.
CBU_DATED = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/USD/{date}/"
CBU_SOURCE = "https://cbu.uz/uz/arkhiv-kursov-valyut/"

#: Cache file, written next to the price-history cache.
CACHE_NAME = "cbu_quarter_end_rates.csv"

#: Days to walk back from a quarter end before giving up. Uzbekistan's longest
#: run of consecutive public holidays is well inside this.
LOOKBACK_DAYS = 10


def _read_cache(path: Path) -> dict:
    """Cached rates as ``{quarter: (rate, date)}``."""
    if not path or not path.exists():
        return {}
    frame = pd.read_csv(path)
    if not {"quarter", "rate", "date"} <= set(frame.columns):
        return {}  # an older cache without the effective date is rebuilt
    return {pd.Period(q, freq="Q"): (float(r), str(d))
            for q, r, d in zip(frame["quarter"], frame["rate"], frame["date"])
            if float(r) > 0}


def _write_cache(path: Path, rates: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(rates)
    pd.DataFrame({"quarter": [str(q) for q in order],
                  "date": [rates[q][1] for q in order],
                  "rate": [rates[q][0] for q in order]}).to_csv(path, index=False)


def _fetch_one(session, date: pd.Timestamp) -> float | None:
    """The official rate for exactly ``date``, or None if it is not served.

    A response dated anything other than what was asked for is the endpoint
    falling back to the current rate, and is refused.
    """
    wanted = date.strftime("%Y-%m-%d")
    response = session.get(CBU_DATED.format(date=wanted), timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if str(row.get("Ccy", "USD")).upper() != "USD":
            continue
        # The API answers dd.mm.yyyy; today's rate for an unrecognised date.
        if str(row.get("Date", "")).strip() != date.strftime("%d.%m.%Y"):
            return None
        rate = float(row.get("Rate") or 0)
        return rate if rate > 0 else None
    return None


def _fetch_in_force(session, quarter_end: pd.Timestamp):
    """The last rate published on or before ``quarter_end``, with its date.

    Quarter ends landing on a weekend or a public holiday carry no rate of
    their own; the rate in force on that day is the previous published one.
    """
    for back in range(LOOKBACK_DAYS + 1):
        date = quarter_end - pd.Timedelta(days=back)
        rate = _fetch_one(session, date)
        if rate is not None:
            return rate, date.strftime("%Y-%m-%d")
    return None


def quarter_end_rates(quarters, *, cache=None, session=None, progress=print,
                      today=None) -> dict:
    """Official USD/UZS rate at the end of each requested quarter.

    Returns ``{quarter: (rate, date)}``, where ``date`` is the day the rate was
    actually published on — the quarter's last day, or the last banking day
    before it. Quarters that have not ended, and quarters the Central Bank does
    not serve at all, are absent rather than approximated.
    """
    cache = Path(cache) if cache else None
    rates = _read_cache(cache)
    # tz-naive throughout: a Period's end_time is naive and the two must compare.
    today = (pd.Timestamp(today) if today is not None
             else pd.Timestamp.utcnow().tz_localize(None)).normalize()
    wanted = [pd.Period(q, freq="Q") for q in quarters]
    missing = [q for q in dict.fromkeys(wanted)
               if q not in rates and q.end_time.normalize() <= today]
    if not missing:
        return {q: rates[q] for q in wanted if q in rates}
    if session is None:
        import requests
        session = requests.Session()
    fetched = 0
    for quarter in missing:
        date = quarter.end_time.normalize()
        try:
            found = _fetch_in_force(session, date)
        except Exception as exc:  # network, HTTP or malformed payload
            progress(f"Markaziy bank kursi olinmadi ({date:%Y-%m-%d}): {exc}")
            break
        if found is None:
            progress(f"Markaziy bank {quarter} chorak oxiri uchun kurs bermadi.")
            continue
        rates[quarter] = found
        fetched += 1
    if fetched:
        progress(f"Markaziy bankdan {fetched} ta chorak oxiri kursi olindi.")
        _write_cache(cache, rates)
    return {q: rates[q] for q in wanted if q in rates}
