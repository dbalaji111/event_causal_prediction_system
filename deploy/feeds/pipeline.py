"""The retrieval loop shared by the collector (runs next to Refinitiv, pushes over HTTPS) and the server (pulls
directly when it has its own Refinitiv session). A sink decides where the data goes."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Iterable, List, Optional, Sequence

import pandas as pd

log = logging.getLogger("feeds")
SERIES_CHUNK = 5000


class HttpSink:
    """Pushes to the live server's ingest endpoints with the INGEST_TOKEN."""

    def __init__(self, server: str, token: str, timeout: int = 60):
        self.base = server.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last = None
        for attempt in range(4):
            req = urllib.request.Request(self.base + path, data=data, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode() or "{}")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 422):
                    raise RuntimeError(f"server refused {path}: HTTP {e.code} {e.read().decode()[:300]}") from e
                last = e
            except urllib.error.URLError as e:
                last = e
            time.sleep(2 ** attempt)
        raise RuntimeError(f"could not reach {self.base}{path}: {last}")

    def push_news(self, items: List[dict]) -> int:
        return int(self._post("/api/ingest/news", {"items": items}).get("added", 0)) if items else 0

    def push_series(self, ric: str, points: pd.DataFrame, name: Optional[str], field: Optional[str]) -> int:
        n = 0
        for i in range(0, len(points), SERIES_CHUNK):
            part = points.iloc[i:i + SERIES_CHUNK]
            body = {"ric": ric, "name": name, "field": field,
                    "points": [[pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)] for d, v in zip(part["Date"], part["value"])]}
            n += int(self._post("/api/ingest/series", body).get("stored", 0))
        return n


class StoreSink:
    """Writes straight into the server's store."""

    def __init__(self, store):
        self.store = store

    def push_news(self, items: List[dict]) -> int:
        return self.store.add_news(items)

    def push_series(self, ric, points, name, field) -> int:
        return self.store.upsert_series(ric, points, name=name, field=field)


class FeedLoop:
    """state: any object with get_kv / set_kv (the store, or the collector's JSON state file)."""

    def __init__(self, source, sink, state, queries: Sequence[str], rics: Sequence[str],
                 history_start: str = "1999-01-01", news_lookback_hours: int = 24, names: Optional[dict] = None):
        self.source, self.sink, self.state = source, sink, state
        self.queries = [q for q in queries if q.strip()]
        self.rics = [r for r in rics if r.strip()]
        self.history_start = history_start
        self.lookback = news_lookback_hours
        self.names = names or {}

    def pull_news(self) -> int:
        added = 0
        now = pd.Timestamp.now("UTC").tz_localize(None)
        for q in self.queries:
            key = f"news_last::{q}"
            last = self.state.get_kv(key)
            start = pd.Timestamp(last) - pd.Timedelta(minutes=5) if last else now - pd.Timedelta(hours=self.lookback)
            items = self.source.headlines(q, start=start, end=now, count=100)
            if items:
                added += self.sink.push_news(items)
                self.state.set_kv(key, max(i["ts"] for i in items))
            log.info("news '%s': %d fetched, %d new", q, len(items), added)
        return added

    def pull_series(self, rics: Optional[Iterable[str]] = None) -> int:
        stored = 0
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        for ric in (rics or self.rics):
            key = f"series_last::{ric}"
            last = self.state.get_kv(key)
            start = pd.Timestamp(last) - pd.Timedelta(days=10) if last else pd.Timestamp(self.history_start)
            df = self.source.history(ric, start=start, end=today)
            n = self.sink.push_series(ric, df, self.names.get(ric), df.attrs.get("field"))
            stored += n
            if len(df):
                self.state.set_kv(key, df["Date"].max().strftime("%Y-%m-%d"))
            log.info("series %s: %d values from %s", ric, n, start.date())
        return stored

    def run(self, news_every: int = 60, series_every: int = 3600, once: bool = False, stop=None):
        next_series = 0.0
        while True:
            if stop is not None and stop.is_set():
                return
            t = time.monotonic()
            try:
                if t >= next_series:
                    self.pull_series()
                    next_series = t + series_every
                self.pull_news()
            except Exception as e:  # keep the loop alive through network blips; log and retry next round
                log.warning("feed round failed: %s", e)
            if once or (stop is not None and stop.is_set()):
                return
            if stop is not None:
                stop.wait(news_every)
            else:
                time.sleep(news_every)
