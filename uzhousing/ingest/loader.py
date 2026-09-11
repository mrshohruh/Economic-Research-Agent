"""Load a dataset from whatever the user has: JSON, SQL, SQLite, CSV or Excel.

The goal is that the caller says "here is my file" and gets back a dictionary of
clean :class:`pandas.DataFrame` objects plus a human-readable description of
where they came from. A folder works too: every supported file inside it is
read and the tables that share a shape are stacked into one.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import listings

LOGGER = logging.getLogger(__name__)

RECORD_KEYS = ("data", "records", "rows", "items", "result", "results", "values", "table")

TEXT_SUFFIXES = {".json", ".jsonl", ".ndjson", ".sql", ".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
PARQUET_SUFFIXES = {".parquet"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | SQLITE_SUFFIXES | PARQUET_SUFFIXES


@dataclass
class Dataset:
    """A loaded collection of tables."""

    tables: dict[str, pd.DataFrame]
    source: str
    kind: str
    notes: list[str] = field(default_factory=list)
    # Set when the source was a marketplace listing feed, so the analysis knows
    # each row is one advert rather than an aggregated statistic.
    listing_type: str = ""
    uzs_per_usd: float = 0.0
    # Rows the source actually held, before any thinning for the memory budget.
    # Zero when nothing was thinned, so it equals ``total_rows``.
    source_rows: int = 0

    @property
    def total_rows(self) -> int:
        return int(sum(len(df) for df in self.tables.values()))

    def largest_table(self) -> tuple[str, pd.DataFrame]:
        name = max(self.tables, key=lambda k: (len(self.tables[k]), self.tables[k].shape[1]))
        return name, self.tables[name]


class LoadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load(
    path: str | Path | None = None,
    *,
    raw_text: str | None = None,
    filename: str | None = None,
    connection_url: str | None = None,
    query: str | None = None,
    uzs_per_usd: float = listings.DEFAULT_UZS_PER_USD,
    max_rows: int | None = None,
) -> Dataset:
    """Load data from a path, a folder, an in-memory blob, or a database connection.

    ``max_rows`` caps how many rows are held in memory at once. A folder of
    quarterly database dumps easily runs to millions of adverts, which is more
    than a laptop can hold; when the cap bites the rows are thinned evenly
    across the source rather than truncated, and a note records that it happened.
    """
    if connection_url:
        return _load_from_connection(connection_url, query)

    if raw_text is not None:
        name = filename or "uploaded"
        return _load_from_text(raw_text, name, uzs_per_usd=uzs_per_usd)

    if path is None:
        raise LoadError("nothing to load: pass a path, raw text, or a connection URL")

    p = Path(path)
    if not p.exists():
        raise LoadError(f"file not found: {p}")

    if p.is_dir():
        dataset = _load_directory(p, query=query, uzs_per_usd=uzs_per_usd, max_rows=max_rows)
    else:
        dataset = _load_file(p, query=query, uzs_per_usd=uzs_per_usd, max_rows=max_rows)

    # One note for the whole load, however many files it took: fifteen identical
    # "this file was sampled" warnings tell the reader nothing extra.
    _note_sampling(dataset)
    return dataset


def _load_file(
    p: Path,
    *,
    query: str | None = None,
    uzs_per_usd: float = listings.DEFAULT_UZS_PER_USD,
    max_rows: int | None = None,
) -> Dataset:
    suffix = p.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        dataset = _load_from_text(
            p.read_text(encoding="utf-8-sig"), p.name, origin=str(p), uzs_per_usd=uzs_per_usd
        )
    elif suffix in EXCEL_SUFFIXES:
        dataset = _load_excel(p)
    elif suffix in SQLITE_SUFFIXES:
        # SQLite can do the thinning itself, so the full table never reaches memory.
        return _load_sqlite_file(p, query, max_rows=max_rows)
    elif suffix in PARQUET_SUFFIXES:
        notes: list[str] = []
        dataset = Dataset({p.stem: _clean(pd.read_parquet(p), notes)}, str(p), "parquet", notes)
    else:
        raise LoadError(f"unsupported file type '{suffix}' for {p.name}")

    _apply_budget(dataset, max_rows)
    return dataset


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def _load_directory(
    directory: Path,
    *,
    query: str | None,
    uzs_per_usd: float,
    max_rows: int | None,
) -> Dataset:
    """Read every supported file in a folder and stack matching tables together."""
    files = _discover_files(directory)
    if not files:
        others = sorted({f.suffix.lower() or "(no extension)" for f in directory.rglob("*") if f.is_file()})
        found = f" It contains files of type: {', '.join(others[:10])}." if others else " It is empty."
        raise LoadError(
            f"no supported data files in {directory}."
            f" Looked for {', '.join(sorted(SUPPORTED_SUFFIXES))}.{found}"
        )

    quotas = _plan_quotas(files, max_rows)

    loaded: list[Dataset] = []
    failures: list[str] = []
    for file in files:
        LOGGER.info("loading %s", file)
        try:
            loaded.append(
                _load_file(file, query=query, uzs_per_usd=uzs_per_usd, max_rows=quotas.get(file))
            )
        except Exception as exc:  # one bad file must not sink the whole folder
            LOGGER.warning("could not read %s: %s", file.name, exc)
            failures.append(f"{file.name}: {exc}")

    if not loaded:
        detail = "\n  ".join(failures[:10])
        raise LoadError(f"none of the {len(files)} files in {directory} could be read:\n  {detail}")

    return _merge_datasets(loaded, directory, files, failures, max_rows)


def _discover_files(directory: Path) -> list[Path]:
    """Supported data files inside a folder, hidden and helper paths excluded."""
    found: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(directory)
        if any(part.startswith((".", "~$")) for part in relative.parts):
            continue
        found.append(path)
    return sorted(found)


def _plan_quotas(files: list[Path], max_rows: int | None) -> dict[Path, int | None]:
    """Split a row budget between files in proportion to how big each one is.

    Counting rows up front is only possible for SQLite; anything else is given an
    equal share, which it applies once loaded.
    """
    if not max_rows or max_rows <= 0:
        return {}

    counts = {f: _count_sqlite_rows(f) for f in files}
    known = {f: n for f, n in counts.items() if n}
    known_total = sum(known.values())
    unknown = [f for f in files if f not in known]

    # Reserve the unknown files an even slice; the rest is shared by row count.
    unknown_share = int(max_rows * len(unknown) / len(files)) if unknown else 0
    known_budget = max_rows - unknown_share

    quotas: dict[Path, int | None] = {}
    for file in unknown:
        quotas[file] = max(1, unknown_share // len(unknown))
    for file, count in known.items():
        quotas[file] = max(1, int(known_budget * count / known_total)) if known_total else None
    return quotas


def _count_sqlite_rows(path: Path) -> int | None:
    if path.suffix.lower() not in SQLITE_SUFFIXES:
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        total = 0
        for name in _sqlite_table_names(conn):
            total += int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        return total or None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _merge_datasets(
    datasets: list[Dataset],
    directory: Path,
    files: list[Path],
    failures: list[str],
    max_rows: int | None,
) -> Dataset:
    """Stack tables that share a shape; keep the rest side by side."""
    groups: dict[str, list[pd.DataFrame]] = {}
    origins: dict[str, list[str]] = {}

    for dataset in datasets:
        stem = _safe_name(Path(dataset.source).stem) or "file"
        for name, frame in dataset.tables.items():
            key = name
            if key in groups and not _same_shape(groups[key][0], frame):
                key = f"{name}_{stem}"
                suffix_n = 2
                while key in groups and not _same_shape(groups[key][0], frame):
                    key = f"{name}_{stem}_{suffix_n}"
                    suffix_n += 1
            groups.setdefault(key, []).append(frame)
            origins.setdefault(key, []).append(Path(dataset.source).name)

    tables = {
        name: (frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False))
        for name, frames in groups.items()
    }

    notes = _folder_notes(datasets, files, failures, origins, tables)
    listing_type = next((d.listing_type for d in datasets if d.listing_type), "")
    uzs_per_usd = next((d.uzs_per_usd for d in datasets if d.uzs_per_usd), 0.0)

    kinds = sorted({d.kind for d in datasets})
    kind = (
        f"folder of {len(datasets)} {kinds[0]} files"
        if len(kinds) == 1
        else f"folder of {len(datasets)} files ({', '.join(kinds[:4])})"
    )

    merged = Dataset(
        tables, str(directory), kind, notes, listing_type, uzs_per_usd,
        source_rows=sum(d.source_rows or d.total_rows for d in datasets),
    )
    _apply_budget(merged, max_rows)
    return merged


def _same_shape(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """True when two tables are the same thing recorded at different times."""
    a, b = set(left.columns), set(right.columns)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6


def _folder_notes(
    datasets: list[Dataset],
    files: list[Path],
    failures: list[str],
    origins: dict[str, list[str]],
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    notes: list[str] = []
    names = ", ".join(Path(d.source).name for d in datasets)
    notes.append(
        f"Read {len(datasets)} of {len(files)} data files in the folder and combined them: {names}."
    )
    for name, sources in origins.items():
        if len(sources) > 1:
            notes.append(
                f"Table '{name}' was stacked from {len(sources)} files into "
                f"{len(tables[name]):,} rows."
            )
    if failures:
        notes.append(
            f"{len(failures)} file(s) could not be read and are missing from this analysis: "
            + "; ".join(failures[:5])
        )

    # The same warning from fifteen quarterly files is one warning, not fifteen.
    tally: dict[str, int] = {}
    for dataset in datasets:
        for note in dict.fromkeys(dataset.notes):
            tally[note] = tally.get(note, 0) + 1
    for note, count in tally.items():
        notes.append(note if count == 1 else f"{note} (in {count} of {len(datasets)} files)")
    return notes


# ---------------------------------------------------------------------------
# Row budget
# ---------------------------------------------------------------------------
def _apply_budget(dataset: Dataset, max_rows: int | None) -> None:
    """Thin a dataset down to ``max_rows`` rows, evenly across each table."""
    if not max_rows or max_rows <= 0 or dataset.total_rows <= max_rows:
        return

    total = dataset.total_rows
    dataset.source_rows = max(dataset.source_rows, total)
    for name, frame in list(dataset.tables.items()):
        quota = max(1, int(max_rows * len(frame) / total))
        thinned, step = _thin(frame, quota)
        if step > 1:
            dataset.tables[name] = thinned


def _note_sampling(dataset: Dataset) -> None:
    """Say once, plainly, that the analysis ran on a sample rather than the lot."""
    kept = dataset.total_rows
    available = dataset.source_rows
    # A handful of blank rows dropped while cleaning is not a sample; only a real
    # shortfall is worth warning the reader about.
    if not kept or not available or kept > available * 0.95:
        return
    share = 100.0 * kept / available
    dataset.notes.insert(
        0,
        f"The source holds {available:,} rows, more than fits in memory at once. Every "
        f"{_ordinal(round(available / kept))} row was read instead, spread evenly across the data, "
        f"leaving a {kept:,}-row sample ({share:.1f}% of the total). Averages, shares, ratios and "
        f"trends are unaffected by this. Counts and totals are roughly "
        f"{available / kept:.0f}x lower than the true figures and should be read as sample counts. "
        f"Pass --max-rows 0 to analyse every row."
    )


def _thin(frame: pd.DataFrame, quota: int | None) -> tuple[pd.DataFrame, int]:
    if not quota or quota <= 0 or len(frame) <= quota:
        return frame, 1
    step = math.ceil(len(frame) / quota)
    return frame.iloc[::step].reset_index(drop=True), step


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


# ---------------------------------------------------------------------------
def _load_from_text(
    text: str,
    name: str,
    origin: str | None = None,
    uzs_per_usd: float = listings.DEFAULT_UZS_PER_USD,
) -> Dataset:
    origin = origin or name
    suffix = Path(name).suffix.lower()

    if suffix == ".sql" or _looks_like_sql(text):
        return _load_sql_script(text, origin)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(text, origin, uzs_per_usd)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else None
        notes: list[str] = []
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        return Dataset({Path(name).stem: _clean(df, notes)}, origin, "csv", notes)

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        return _load_json_text(text, origin, Path(name).stem, uzs_per_usd)
    if "\n" in text and ("," in text or ";" in text or "\t" in text):
        notes = []
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        return Dataset({Path(name).stem: _clean(df, notes)}, origin, "delimited text", notes)

    raise LoadError(f"could not work out the format of {name}")


def _looks_like_sql(text: str) -> bool:
    head = text[:4000].upper()
    return bool(re.search(r"\b(CREATE\s+TABLE|INSERT\s+INTO|SELECT\s+.+\s+FROM)\b", head))


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _load_json_text(
    text: str, origin: str, stem: str, uzs_per_usd: float = listings.DEFAULT_UZS_PER_USD
) -> Dataset:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        # A file of concatenated JSON objects is common enough to be worth a retry.
        try:
            return _load_jsonl(text, origin, uzs_per_usd)
        except Exception:
            raise LoadError(f"invalid JSON: {exc}") from exc

    candidates = obj if isinstance(obj, list) else [obj]
    marketplace = _as_listing_dataset(candidates, origin, "json", uzs_per_usd)
    if marketplace is not None:
        return marketplace

    tables, notes = _frames_from_json(obj, stem or "data")
    if not tables:
        raise LoadError("the JSON contained no tabular records")
    return Dataset(tables, origin, "json", notes)


def _load_jsonl(
    text: str, origin: str, uzs_per_usd: float = listings.DEFAULT_UZS_PER_USD
) -> Dataset:
    records = []
    skipped = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
            LOGGER.warning("skipping unparseable JSONL line %s", line_no)
    if not records:
        raise LoadError("no valid JSON lines found")

    # A truncated or interrupted crawl leaves broken lines behind. Silently
    # dropping them would overstate how complete the sample is.
    damaged = (
        [
            f"{skipped:,} of {skipped + len(records):,} lines in the source file were not valid "
            "JSON and were skipped, so the records they held are missing from this analysis."
        ]
        if skipped
        else []
    )

    marketplace = _as_listing_dataset(records, origin, "jsonl", uzs_per_usd)
    if marketplace is not None:
        marketplace.notes.extend(damaged)
        return marketplace

    notes: list[str] = list(damaged)
    frame = _clean(pd.json_normalize(records), notes)
    return Dataset({Path(origin).stem or "data": frame}, origin, "jsonl", notes)


def _as_listing_dataset(
    records: list[Any], origin: str, kind: str, uzs_per_usd: float
) -> Dataset | None:
    """Return a one-row-per-advert dataset when the records are a listing feed.

    Without this, ``pd.json_normalize`` reads a paged feed as one row per *page*
    and buries every advert in an unhashable cell, which leaves the report with
    no price or region column to analyse.
    """
    if not listings.looks_like_listings(records):
        return None
    try:
        table = listings.flatten(records, uzs_per_usd=uzs_per_usd)
    except Exception as exc:  # pragma: no cover - unexpected feed variant
        LOGGER.warning("listing feed detected but could not be flattened: %s", exc)
        return None
    if table is None:
        return None

    LOGGER.info("flattened %s adverts from a %s listing feed", len(table.frame), kind)
    return Dataset(
        tables={"listings": _clean(table.frame, table.notes)},
        source=origin,
        kind=f"{kind} property listings",
        notes=table.notes,
        listing_type=table.listing_type,
        uzs_per_usd=table.uzs_per_usd,
    )


def _frames_from_json(obj: Any, name: str, depth: int = 0) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Turn an arbitrary JSON structure into one or more flat tables."""
    notes: list[str] = []
    tables: dict[str, pd.DataFrame] = {}

    if depth > 3:
        return tables, notes

    if isinstance(obj, list):
        if not obj:
            return tables, notes
        if all(isinstance(item, dict) for item in obj):
            tables[name] = _clean(pd.json_normalize(obj), notes)
        else:
            tables[name] = _clean(pd.DataFrame({name: obj}), notes)
        return tables, notes

    if not isinstance(obj, dict):
        return tables, notes

    # {"data": [...], "meta": {...}} -> unwrap the payload, remember the metadata.
    for key in RECORD_KEYS:
        if key in obj and isinstance(obj[key], (list, dict)):
            sub, sub_notes = _frames_from_json(obj[key], name, depth + 1)
            if sub:
                other = {k: v for k, v in obj.items() if k != key and not isinstance(v, (list, dict))}
                if other:
                    notes.append("metadata in file: " + json.dumps(other, ensure_ascii=False)[:400])
                return sub, notes + sub_notes

    # {"table_a": [...], "table_b": [...]} -> several named tables.
    list_children = {k: v for k, v in obj.items() if isinstance(v, list) and v}
    if list_children and all(all(isinstance(i, dict) for i in v) for v in list_children.values()):
        for key, value in list_children.items():
            tables[_safe_name(key)] = _clean(pd.json_normalize(value), notes)
        skipped = [k for k in obj if k not in list_children]
        if skipped:
            notes.append(f"ignored non-tabular top-level keys: {', '.join(skipped[:10])}")
        return tables, notes

    # {"col_a": [...], "col_b": [...]} -> a single column-oriented table.
    if list_children and len({len(v) for v in list_children.values()}) == 1:
        try:
            tables[name] = _clean(pd.DataFrame(list_children), notes)
            return tables, notes
        except Exception:
            pass

    # {"2020": {...}, "2021": {...}} -> index-oriented table.
    dict_children = {k: v for k, v in obj.items() if isinstance(v, dict)}
    if dict_children and len(dict_children) == len(obj):
        try:
            df = pd.DataFrame.from_dict(dict_children, orient="index")
            df.index.name = "key"
            tables[name] = _clean(df.reset_index(), notes)
            return tables, notes
        except Exception:
            pass

    # Last resort: nested objects each become their own table.
    for key, value in obj.items():
        if isinstance(value, (list, dict)):
            sub, sub_notes = _frames_from_json(value, _safe_name(key), depth + 1)
            tables.update(sub)
            notes.extend(sub_notes)
    if not tables:
        flat = pd.json_normalize(obj)
        if not flat.empty:
            tables[name] = _clean(flat, notes)
    return tables, notes


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
def _load_sql_script(text: str, origin: str) -> Dataset:
    """Replay a .sql dump into an in-memory SQLite database and read it back."""
    conn = sqlite3.connect(":memory:")
    notes: list[str] = []
    script = _sanitise_sql(text)

    executed = 0
    failed = 0
    for statement in _split_statements(script):
        if not statement.strip():
            continue
        try:
            conn.execute(statement)
            executed += 1
        except sqlite3.Error as exc:
            failed += 1
            if failed <= 5:
                notes.append(f"skipped a statement SQLite rejected: {exc}")
    conn.commit()

    if failed:
        notes.append(f"{failed} of {executed + failed} statements could not be replayed in SQLite.")

    tables, _ = _read_all_sqlite_tables(conn, notes)

    # A script that is only SELECTs has no tables; run the final SELECT instead.
    if not tables:
        selects = [s for s in _split_statements(script) if s.strip().upper().startswith("SELECT")]
        if selects:
            try:
                tables["query_result"] = _clean(pd.read_sql_query(selects[-1], conn), notes)
            except Exception as exc:
                raise LoadError(f"could not run the SELECT statement: {exc}") from exc

    conn.close()
    if not tables:
        raise LoadError("the SQL script produced no tables")
    return Dataset(tables, origin, "sql script", notes)


def _sanitise_sql(text: str) -> str:
    """Make common MySQL / PostgreSQL dumps palatable to SQLite."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"^\s*--.*$", "", text, flags=re.MULTILINE)
    text = text.replace("`", '"')
    text = re.sub(r"\bAUTO_INCREMENT\b(=\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAUTOINCREMENT\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bENGINE\s*=\s*\w+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDEFAULT\s+CHARSET\s*=\s*[\w]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCOLLATE\s+[\w]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bUNSIGNED\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSERIAL\b", "INTEGER", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTIMESTAMP\s+WITHOUT\s+TIME\s+ZONE\b", "TIMESTAMP", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(SET|LOCK TABLES|UNLOCK TABLES|USE|START TRANSACTION|COMMIT)\b.*?;",
                  "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return text


def _split_statements(script: str) -> list[str]:
    """Split on semicolons that are not inside a quoted string."""
    statements: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(script):
        ch = script[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                if i + 1 < len(script) and script[i + 1] == quote:  # escaped quote
                    buf.append(script[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ";":
            statements.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if "".join(buf).strip():
        statements.append("".join(buf))
    return statements


def _sqlite_table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%'"
        )
    ]


def _read_all_sqlite_tables(
    conn: sqlite3.Connection,
    notes: list[str] | None = None,
    max_rows: int | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Read every table, thinning in SQL if needed. Also reports rows available."""
    tables: dict[str, pd.DataFrame] = {}
    names = _sqlite_table_names(conn)

    counts: dict[str, int] = {}
    if max_rows and max_rows > 0:
        for name in names:
            try:
                counts[name] = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            except sqlite3.Error:
                counts[name] = 0
    grand_total = sum(counts.values())

    for name in names:
        quota = None
        if max_rows and grand_total > max_rows:
            quota = max(1, int(max_rows * counts.get(name, 0) / grand_total))
        try:
            df = _read_sqlite_table(conn, name, counts.get(name), quota)
        except Exception as exc:
            LOGGER.warning("could not read table %s: %s", name, exc)
            continue
        if not df.empty:
            tables[_safe_name(name)] = _clean(df, notes)
    return tables, grand_total


def _read_sqlite_table(
    conn: sqlite3.Connection, name: str, count: int | None, quota: int | None
) -> pd.DataFrame:
    """Read a table, letting SQLite skip rows when the quota is smaller than it.

    Sampling in SQL matters: a table too big for memory cannot be loaded first
    and thinned afterwards.
    """
    if quota and count and count > quota:
        step = math.ceil(count / quota)
        LOGGER.info("reading every %s row of %s (%s rows)", _ordinal(step), name, f"{count:,}")
        try:
            return pd.read_sql_query(f'SELECT * FROM "{name}" WHERE rowid % {step} = 0', conn)
        except Exception as exc:  # WITHOUT ROWID tables have no rowid to step over
            LOGGER.info("cannot sample %s by rowid (%s); reading it in full", name, exc)
            df, _ = _thin(pd.read_sql_query(f'SELECT * FROM "{name}"', conn), quota)
            return df
    return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)


def _load_sqlite_file(path: Path, query: str | None, max_rows: int | None = None) -> Dataset:
    conn = sqlite3.connect(str(path))
    notes: list[str] = []
    source_rows = 0
    try:
        if query:
            tables = {"query_result": _clean(pd.read_sql_query(query, conn), notes)}
        else:
            tables, source_rows = _read_all_sqlite_tables(conn, notes, max_rows=max_rows)
    finally:
        conn.close()
    if not tables:
        raise LoadError(f"no readable tables in {path.name}")
    dataset = Dataset(tables, str(path), "sqlite database", notes, source_rows=source_rows)
    if query:
        _apply_budget(dataset, max_rows)
    return dataset


def _load_from_connection(url: str, query: str | None) -> Dataset:
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError as exc:  # pragma: no cover
        raise LoadError("SQLAlchemy is required for database URLs") from exc

    engine = create_engine(url)
    tables: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    with engine.connect() as conn:
        if query:
            tables["query_result"] = _clean(pd.read_sql_query(query, conn), notes)
        else:
            for name in inspect(engine).get_table_names():
                try:
                    df = pd.read_sql_table(name, conn)
                except Exception as exc:
                    LOGGER.warning("skipping table %s: %s", name, exc)
                    continue
                if not df.empty:
                    tables[_safe_name(name)] = _clean(df, notes)
    if not tables:
        raise LoadError("the database connection returned no rows")
    return Dataset(tables, _redact(url), "database", notes)


def _load_excel(path: Path) -> Dataset:
    sheets = pd.read_excel(path, sheet_name=None)
    notes: list[str] = []
    tables = {_safe_name(k): _clean(v, notes) for k, v in sheets.items() if not v.empty}
    if not tables:
        raise LoadError(f"no non-empty sheets in {path.name}")
    return Dataset(tables, str(path), "excel workbook", notes)


# ---------------------------------------------------------------------------
# Shared cleaning
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame, notes: list[str] | None = None) -> pd.DataFrame:
    """Normalise column names, flatten container cells, coerce numeric text."""
    df = df.copy()
    df.columns = [_safe_name(str(c)) for c in df.columns]

    # De-duplicate column names.
    seen: dict[str, int] = {}
    cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            cols.append(col)
    df.columns = cols

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    df, exploded = _explode_parallel_arrays(df)
    if exploded and notes is not None:
        notes.append(
            "Expanded parallel arrays into one row per element: "
            + ", ".join(exploded)
            + f". The table now has {len(df):,} rows."
        )

    flattened = _flatten_containers(df)
    if flattened and notes is not None:
        notes.append(
            "Encoded as text because the column holds lists or nested objects rather than "
            "single values: " + ", ".join(flattened[:8])
            + (f" (and {len(flattened) - 8} more)" if len(flattened) > 8 else "")
            + ". These columns are described but not analysed numerically."
        )

    for col in df.columns:
        if _is_textual(df[col]):
            converted = _maybe_numeric(df[col])
            if converted is not None:
                df[col] = converted
    return df.reset_index(drop=True)


def _explode_parallel_arrays(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Expand records that store parallel arrays into one row per element.

    ``{"region": "Tashkent", "date": [...], "price": [...]}`` is a common export
    shape. Stringifying those arrays would leave the report with no measures at
    all, so where two or more list columns agree on their per-row length they are
    exploded together instead.

    A single list column is deliberately left alone: exploding one array (say
    ``tags``) would duplicate every other value in the row and silently
    double-count the actual measures.
    """
    list_cols = [
        col for col in df.columns
        if df[col].map(lambda v: isinstance(v, list)).any()
    ]
    if len(list_cols) < 2:
        return df, []

    # Every list column present in a row must have the same number of elements.
    lengths = pd.DataFrame(
        {col: df[col].map(lambda v: len(v) if isinstance(v, list) else None) for col in list_cols}
    )
    if not lengths.nunique(axis=1, dropna=True).le(1).all():
        return df, []

    try:
        expanded = df.explode(list_cols, ignore_index=True)
    except (ValueError, KeyError) as exc:  # pragma: no cover - ragged input
        LOGGER.info("could not expand parallel arrays: %s", exc)
        return df, []

    # Nested lists survive an explode; that is not the shape we are handling.
    if any(expanded[col].map(lambda v: isinstance(v, (list, dict))).any() for col in list_cols):
        return df, []

    return expanded, list_cols


def _flatten_containers(df: pd.DataFrame) -> list[str]:
    """Encode list / dict / set cells as compact JSON text, in place.

    ``pd.json_normalize`` flattens nested objects but leaves arrays as Python
    lists, which are unhashable. Anything downstream that calls ``unique``,
    ``nunique``, ``groupby`` or ``drop_duplicates`` on such a column raises
    ``TypeError: unhashable type: 'list'``, so containers are converted here,
    at the one place every loader path passes through.
    """
    changed: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if not series.map(_is_container).any():
            continue
        df[col] = series.map(_encode_container)
        changed.append(str(col))
    return changed


def _is_container(value: Any) -> bool:
    return isinstance(value, (list, dict, set, tuple))


def _encode_container(value: Any) -> Any:
    if not _is_container(value):
        return value
    if isinstance(value, (set, tuple)):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic objects
        return str(value)


def _is_textual(series: pd.Series) -> bool:
    """True for text columns.

    pandas 3 gives text columns a dedicated string dtype rather than ``object``,
    so testing ``dtype == object`` alone silently skips every string column.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _maybe_numeric(series: pd.Series) -> pd.Series | None:
    """Convert "1 234,5" / "12%" / "$1,200" style strings to floats when safe."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return None
    cleaned = (
        sample.str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.notna().mean() < 0.9:
        return None
    full = (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(full, errors="coerce")


def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\wЀ-ӿ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower() or "column"


def _redact(url: str) -> str:
    return re.sub(r"//[^@/]+@", "//***@", url)
