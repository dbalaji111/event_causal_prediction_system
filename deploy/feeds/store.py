"""SQLite store for the live terminal: time series (one row per instrument and day) and the news feed (one row per
headline, with its analysis once the worker has run it). On Railway, put DATA_DIR on a volume so it survives redeploys."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS series_meta (
  ric TEXT PRIMARY KEY, name TEXT, field TEXT, source TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS series_points (
  ric TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY (ric, date));
CREATE TABLE IF NOT EXISTS news (
  story_id TEXT PRIMARY KEY, ts TEXT NOT NULL, headline TEXT NOT NULL, source TEXT, query TEXT,
  received_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'new', analysis TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS news_ts ON news (ts DESC);
CREATE INDEX IF NOT EXISTS news_status ON news (status);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ series
    def upsert_series(self, ric: str, points: pd.DataFrame, name: Optional[str] = None, field: Optional[str] = None,
                      source: str = "refinitiv") -> int:
        rows = [(ric, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in zip(points.iloc[:, 0], points.iloc[:, 1])
                if pd.notna(d) and pd.notna(v)]
        with self.lock:
            self.db.execute("BEGIN")
            self.db.executemany("INSERT OR REPLACE INTO series_points (ric, date, value) VALUES (?, ?, ?)", rows)
            self.db.execute("""INSERT INTO series_meta (ric, name, field, source, updated_at) VALUES (?, ?, ?, ?, ?)
                               ON CONFLICT(ric) DO UPDATE SET name=COALESCE(excluded.name, series_meta.name),
                               field=COALESCE(excluded.field, series_meta.field), source=excluded.source,
                               updated_at=excluded.updated_at""", (ric, name, field, source, _now()))
            self.db.execute("COMMIT")
        return len(rows)

    def list_series(self) -> List[dict]:
        q = """SELECT m.ric, m.name, m.field, m.source, m.updated_at, COUNT(p.date), MIN(p.date), MAX(p.date)
               FROM series_meta m LEFT JOIN series_points p ON p.ric = m.ric GROUP BY m.ric ORDER BY m.ric"""
        with self.lock:
            rows = self.db.execute(q).fetchall()
        return [{"ric": r[0], "name": r[1] or r[0], "field": r[2], "source": r[3], "updated_at": r[4],
                 "points": r[5], "first": r[6], "last": r[7]} for r in rows]

    def get_series(self, ric: str, start: Optional[str] = None) -> pd.DataFrame:
        q, args = "SELECT date, value FROM series_points WHERE ric = ?", [ric]
        if start:
            q += " AND date >= ?"
            args.append(start)
        with self.lock:
            rows = self.db.execute(q + " ORDER BY date", args).fetchall()
        df = pd.DataFrame(rows, columns=["Date", "value"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def has_series(self, ric: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM series_meta WHERE ric = ?", (ric,)).fetchone() is not None

    # ------------------------------------------------------------------ news
    def add_news(self, items: Iterable[dict]) -> int:
        added = 0
        with self.lock:
            self.db.execute("BEGIN")
            for it in items:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO news (story_id, ts, headline, source, query, received_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(it["story_id"]), str(it["ts"]), str(it["headline"])[:1000], it.get("source", ""),
                     it.get("query", ""), _now()))
                added += cur.rowcount
            self.db.execute("COMMIT")
        return added

    def pending(self, limit: int = 5) -> List[dict]:
        with self.lock:
            rows = self.db.execute("SELECT story_id, ts, headline FROM news WHERE status = 'new' ORDER BY ts DESC LIMIT ?",
                                   (limit,)).fetchall()
        return [{"story_id": r[0], "ts": r[1], "headline": r[2]} for r in rows]

    def set_result(self, story_id: str, analysis: Optional[dict] = None, error: Optional[str] = None):
        with self.lock:
            self.db.execute("UPDATE news SET status = ?, analysis = ?, error = ? WHERE story_id = ?",
                            ("done" if error is None else "error", json.dumps(analysis) if analysis else None,
                             error, story_id))

    def feed(self, limit: int = 50, before: Optional[str] = None) -> List[dict]:
        q, args = "SELECT story_id, ts, headline, source, status, analysis, error FROM news", []
        if before:
            q += " WHERE ts < ?"
            args.append(before)
        with self.lock:
            rows = self.db.execute(q + " ORDER BY ts DESC LIMIT ?", args + [limit]).fetchall()
        return [{"story_id": r[0], "ts": r[1], "headline": r[2], "source": r[3], "status": r[4],
                 "summary": json.loads(r[5]) if r[5] else None, "error": r[6]} for r in rows]

    def news_counts(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT status, COUNT(*), MAX(ts) FROM news GROUP BY status").fetchall()
        return {"by_status": {r[0]: r[1] for r in rows}, "latest": max((r[2] for r in rows), default=None)}

    # ------------------------------------------------------------------ small state
    def set_kv(self, k: str, v: str):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, v))

    def get_kv(self, k: str) -> Optional[str]:
        with self.lock:
            r = self.db.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return r[0] if r else None
