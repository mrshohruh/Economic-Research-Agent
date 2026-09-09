"""SQLite storage for run logs (Section 32)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config.settings import DB_PATH, RUNS_DIR
from models.schemas import RunLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    file_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    report_path TEXT,
    run_json_path TEXT,
    created_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def save_run(run_log: RunLog) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RUNS_DIR / f"run_{ts}.json"
    json_path.write_text(run_log.model_dump_json(indent=2), encoding="utf-8")

    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO runs (topic, file_name, started_at, report_path, run_json_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_log.topic, run_log.file_name, run_log.started_at, run_log.report_path,
             str(json_path), datetime.now().isoformat(timespec="seconds")),
        )
    conn.close()
    return json_path


def list_runs(limit: int = 20) -> list[dict]:
    conn = _connect()
    cur = conn.execute("SELECT id, topic, file_name, started_at, report_path, created_at FROM runs "
                        "ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    conn.close()
    return rows
