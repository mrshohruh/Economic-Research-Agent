"""Runtime configuration, loaded from the environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing is not fatal
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANGUAGES = {"en": "English", "ru": "Russian", "uz": "Uzbek"}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass
class Settings:
    """Everything the pipeline needs to know about how to run."""

    anthropic_api_key: str = ""
    model: str = "claude-opus-5"
    web_research: bool = True
    search_results_per_query: int = 6
    pages_to_read: int = 3
    language: str = "en"
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    policy_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "knowledge" / "policy_events.json"
    )
    request_timeout: int = 20
    # Uzbek property is advertised in both som and dollar-linked "у.е.", so a
    # single rate is needed to put every listing on one currency. Set
    # UZS_PER_USD to the rate that applied when the data was collected;
    # otherwise a rerun of an old file is priced at today's rate.
    uzs_per_usd: float = 12_650.0
    # A folder of quarterly dumps can hold several million adverts, which is
    # more than a laptop can hold in memory at once. Rows above this budget are
    # thinned evenly across the source and the report says so. 0 means no cap.
    max_rows: int = 400_000

    @classmethod
    def from_env(cls) -> "Settings":
        out = os.getenv("OUTPUT_DIR", "").strip() or "outputs"
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        lang = (os.getenv("REPORT_LANGUAGE", "en") or "en").strip().lower()
        if lang not in LANGUAGES:
            lang = "en"
        return cls(
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY", "") or "").strip(),
            model=(os.getenv("ANTHROPIC_MODEL", "") or "claude-opus-5").strip(),
            web_research=_bool_env("WEB_RESEARCH", True),
            search_results_per_query=_int_env("SEARCH_RESULTS_PER_QUERY", 6),
            pages_to_read=_int_env("PAGES_TO_READ", 3),
            language=lang,
            output_dir=out_path,
            uzs_per_usd=_float_env("UZS_PER_USD", 12_650.0),
            max_rows=max(0, _int_env("MAX_ROWS", 400_000)),
        )

    # -- convenience ---------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir,
            self.figures_dir,
            self.reports_dir,
            self.runs_dir,
            self.tables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
