"""Research themes, query templates, and the editable local policy record.

Nothing here asserts a number that the report will present as fact without a
confidence flag. The heavy lifting is done by live web research; this module
decides *what to ask*, and holds a small file of structural background that the
user can extend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_POLICY_FILE = Path(__file__).resolve().parent.parent.parent / "knowledge" / "policy_events.json"


@dataclass
class Theme:
    """One research question, with the queries that answer it."""

    key: str
    title: str
    purpose: str
    queries: list[str] = field(default_factory=list)


def themes(period_start: str = "", period_end: str = "", regions: list[str] | None = None) -> list[Theme]:
    """Build the research agenda, tailored to the period the data covers."""
    year_from = period_start[:4] if period_start else ""
    year_to = period_end[:4] if period_end else ""
    span = f"{year_from}-{year_to}" if year_from and year_to else "recent years"
    recent = year_to or "2026"

    agenda = [
        Theme(
            key="market_state",
            title="Current state of the Uzbek housing market",
            purpose="Establish where prices, demand and supply stand right now.",
            queries=[
                f"Uzbekistan housing market {recent} prices apartments analysis",
                f"цены на жилье в Узбекистане {recent} анализ рынка недвижимости",
                f"Tashkent apartment prices per square meter {recent}",
                f"O'zbekiston uy-joy bozori {recent} narxlar tahlili",
            ],
        ),
        Theme(
            key="policy",
            title="Housing and construction policy",
            purpose="Identify decrees, programmes and regulations that shape the market.",
            queries=[
                f"Uzbekistan housing policy decree {recent} presidential resolution construction",
                f"Узбекистан постановление жилищное строительство {recent} программа",
                "Uzbekistan state housing programme subsidised construction rural housing",
                f"Uzbekistan urban planning code land allocation reform {recent}",
            ],
        ),
        Theme(
            key="mortgage",
            title="Mortgage market and subsidy programmes",
            purpose="Explain the credit channel: availability, rates and state support.",
            queries=[
                f"Uzbekistan mortgage lending {recent} subsidised mortgage programme interest rate",
                f"Узбекистан ипотека {recent} льготная ипотека ставка условия",
                "Uzbekistan Mortgage Refinancing Company UzMRC housing finance",
                f"Uzbekistan mortgage portfolio growth banks {recent} statistics",
            ],
        ),
        Theme(
            key="monetary",
            title="Monetary policy and inflation",
            purpose="Link the policy rate and inflation path to housing demand and prices.",
            queries=[
                f"Central Bank of Uzbekistan policy rate decision {recent} inflation",
                f"Центральный банк Узбекистана ставка {recent} инфляция решение",
                f"Uzbekistan inflation rate {recent} forecast CBU",
            ],
        ),
        Theme(
            key="macro",
            title="Macroeconomic and demographic drivers",
            purpose="Cover incomes, GDP, remittances, migration and urbanisation.",
            queries=[
                f"Uzbekistan GDP growth wages remittances {recent} IMF World Bank",
                f"Узбекистан денежные переводы {recent} доходы населения",
                "Uzbekistan urbanisation population growth Tashkent internal migration housing demand",
                f"Uzbekistan construction sector output {recent} statistics",
            ],
        ),
        Theme(
            key="supply",
            title="Construction supply, costs and materials",
            purpose="Explain the cost side: materials, energy tariffs, developer capacity.",
            queries=[
                f"Uzbekistan construction materials prices cement {recent} increase",
                f"Узбекистан стоимость строительства {recent} цены на стройматериалы",
                f"Uzbekistan energy tariff reform {recent} impact construction households",
            ],
        ),
        Theme(
            key="risks",
            title="Risks, imbalances and outlook",
            purpose="Surface affordability stress, oversupply, credit risk and forecasts.",
            queries=[
                f"Uzbekistan housing affordability {recent} risk oversupply bubble",
                f"Узбекистан рынок недвижимости прогноз {recent} риски",
                f"Uzbekistan real estate market outlook {recent} forecast",
            ],
        ),
    ]

    if regions:
        focus = ", ".join(regions[:4])
        agenda.append(
            Theme(
                key="regional",
                title="Regional housing dynamics",
                purpose="Explain why individual regions in the data diverge.",
                queries=[
                    f"Uzbekistan regional housing prices {focus} {recent}",
                    f"Узбекистан цены на жилье регионы {focus} {recent}",
                ],
            )
        )

    if span != "recent years":
        agenda.append(
            Theme(
                key="history",
                title=f"Historical context {span}",
                purpose="Explain the structural breaks visible in the data.",
                queries=[
                    f"Uzbekistan housing market history {span} price growth reform",
                    f"Узбекистан рынок жилья {span} динамика реформы",
                ],
            )
        )
    return agenda


# ---------------------------------------------------------------------------
def load_policy_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the local, user-editable policy file. Missing file is not an error."""
    path = Path(path or DEFAULT_POLICY_FILE)
    if not path.exists():
        LOGGER.info("no local policy file at %s", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", path, exc)
        return []

    events = payload.get("events", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict) and e.get("date") and not e.get("example")]


def events_in_window(
    events: list[dict[str, Any]], start: str | None, end: str | None
) -> list[dict[str, Any]]:
    """Policies that fall inside the data window, plus anything still in force."""
    if not start and not end:
        return events
    selected = []
    for event in events:
        date = str(event.get("date", ""))[:10]
        if start and date < str(start)[:10]:
            # Still relevant if it set the structural backdrop and was not repealed.
            if event.get("still_in_force"):
                selected.append(event)
            continue
        if end and date > str(end)[:10]:
            continue
        selected.append(event)
    return sorted(selected, key=lambda e: str(e.get("date", "")))
