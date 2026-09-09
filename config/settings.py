"""
Central configuration. Never hard-code API keys — everything comes from
environment variables / .env. If no LLM key is present, the system falls
back to deterministic template-based generation so the pipeline still runs
end-to-end (per V1 scope requirement).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower()

LLM_ENABLED = bool(ANTHROPIC_API_KEY)

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RUNS_DIR = OUTPUTS_DIR / "runs"
DB_PATH = ROOT_DIR / "storage" / "research_agent.db"

for d in (OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, RUNS_DIR, DB_PATH.parent):
    d.mkdir(parents=True, exist_ok=True)

PRIORITY_SOURCES = [
    "cbu.uz", "stat.uz", "gov.uz", "mineconomy.uz", "lex.uz",
    "imf.org", "worldbank.org", "adb.org", "ebrd.com",
    "reuters.com", "bloomberg.com", "ft.com",
]

MAX_SEARCH_RESULTS_PER_QUERY = 5
MAX_TOTAL_QUERIES = 14
