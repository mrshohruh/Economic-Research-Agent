"""Append-only observations; absence from a bounded crawl is never a sale."""
import json
import math
import sqlite3
from pathlib import Path


def save_observations(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS collection_runs (
                captured_at TEXT PRIMARY KEY, metadata_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS listing_snapshots (
                listing_id TEXT NOT NULL, captured_at TEXT NOT NULL,
                collection_category TEXT NOT NULL, price REAL, currency TEXT,
                status TEXT NOT NULL, housing_json TEXT NOT NULL,
                PRIMARY KEY (listing_id, captured_at, collection_category));
            CREATE VIEW IF NOT EXISTS listing_observation_history AS
                SELECT listing_id, MIN(captured_at) AS first_seen,
                       MAX(captured_at) AS last_seen,
                       COUNT(DISTINCT captured_at) AS observations
                FROM listing_snapshots GROUP BY listing_id;
        """)
        metadata = {key: value for key, value in payload.items() if key != "data"}
        db.execute("INSERT OR IGNORE INTO collection_runs VALUES (?, ?)",
                   (payload["collected_at"], json.dumps(metadata, ensure_ascii=False)))
        for offer in payload["data"]:
            price, currency = None, None
            for param in offer.get("params") or []:
                if param.get("key") == "price" and isinstance(param.get("value"), dict):
                    value = param["value"]
                    currency = value.get("currency")
                    try:
                        number = float(str(value.get("value")).replace(" ", "").replace(",", "."))
                        price = number if math.isfinite(number) else None
                    except (TypeError, ValueError):
                        pass
            db.execute("INSERT OR IGNORE INTO listing_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(offer["id"]), payload["collected_at"], offer["collection_category"],
                 price, currency, "observed", json.dumps(offer, ensure_ascii=False)))
