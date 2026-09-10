"""Load a dataset from whatever the user has: JSON, SQL, SQLite, CSV or Excel.

The goal is that the caller says "here is my file" and gets back a dictionary of
clean :class:`pandas.DataFrame` objects plus a human-readable description of
where they came from.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

RECORD_KEYS = ("data", "records", "rows", "items", "result", "results", "values", "table")


@dataclass
class Dataset:
    """A loaded collection of tables."""

    tables: dict[str, pd.DataFrame]
    source: str
    kind: str
    notes: list[str] = field(default_factory=list)

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
) -> Dataset:
    """Load data from a path, an in-memory blob, or a database connection."""
    if connection_url:
        return _load_from_connection(connection_url, query)

    if raw_text is not None:
        name = filename or "uploaded"
        return _load_from_text(raw_text, name)

    if path is None:
        raise LoadError("nothing to load: pass a path, raw text, or a connection URL")

    p = Path(path)
    if not p.exists():
        raise LoadError(f"file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson", ".sql", ".csv", ".tsv", ".txt"}:
        return _load_from_text(p.read_text(encoding="utf-8-sig"), p.name, origin=str(p))
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return _load_excel(p)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return _load_sqlite_file(p, query)
    if suffix in {".parquet"}:
        notes: list[str] = []
        return Dataset({p.stem: _clean(pd.read_parquet(p), notes)}, str(p), "parquet", notes)

    raise LoadError(f"unsupported file type '{suffix}' for {p.name}")


# ---------------------------------------------------------------------------
def _load_from_text(text: str, name: str, origin: str | None = None) -> Dataset:
    origin = origin or name
    suffix = Path(name).suffix.lower()

    if suffix == ".sql" or _looks_like_sql(text):
        return _load_sql_script(text, origin)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(text, origin)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else None
        notes: list[str] = []
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        return Dataset({Path(name).stem: _clean(df, notes)}, origin, "csv", notes)

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        return _load_json_text(text, origin, Path(name).stem)
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
def _load_json_text(text: str, origin: str, stem: str) -> Dataset:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        # A file of concatenated JSON objects is common enough to be worth a retry.
        try:
            return _load_jsonl(text, origin)
        except Exception:
            raise LoadError(f"invalid JSON: {exc}") from exc

    tables, notes = _frames_from_json(obj, stem or "data")
    if not tables:
        raise LoadError("the JSON contained no tabular records")
    return Dataset(tables, origin, "json", notes)


def _load_jsonl(text: str, origin: str) -> Dataset:
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            LOGGER.warning("skipping unparseable JSONL line %s", line_no)
    if not records:
        raise LoadError("no valid JSON lines found")
    notes: list[str] = []
    frame = _clean(pd.json_normalize(records), notes)
    return Dataset({Path(origin).stem or "data": frame}, origin, "jsonl", notes)


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

    tables = _read_all_sqlite_tables(conn, notes)

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


def _read_all_sqlite_tables(
    conn: sqlite3.Connection, notes: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    names = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%'",
        conn,
    )["name"].tolist()
    for name in names:
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
        except Exception as exc:
            LOGGER.warning("could not read table %s: %s", name, exc)
            continue
        if not df.empty:
            tables[_safe_name(name)] = _clean(df, notes)
    return tables


def _load_sqlite_file(path: Path, query: str | None) -> Dataset:
    conn = sqlite3.connect(str(path))
    notes: list[str] = []
    try:
        if query:
            tables = {"query_result": _clean(pd.read_sql_query(query, conn), notes)}
        else:
            tables = _read_all_sqlite_tables(conn, notes)
    finally:
        conn.close()
    if not tables:
        raise LoadError(f"no readable tables in {path.name}")
    return Dataset(tables, str(path), "sqlite database", notes)


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
