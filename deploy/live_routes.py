"""The live terminal: Refinitiv news feed scored as it arrives, any stored time series as the KDE's price source,
and the ingest endpoints the collector pushes to.

Viewer pages and APIs (/live, /api/live/*) need LIVE_USERNAME + LIVE_PASSWORD (HTTP basic auth): Refinitiv content
may not be shown to the public under a standard licence. Ingest endpoints (/api/ingest/*) need INGEST_TOKEN.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from feeds.pipeline import FeedLoop, StoreSink
from feeds.store import Store
from redflag_kit.series import series_prices

log = logging.getLogger("live")
LIVE_PAGE = Path(__file__).resolve().parent / "live_dashboard.html"
BUNDLED = "BUNDLED:XAUUSD"
MAX_PLOT_POINTS = 4000


class LiveAnalyzeIn(BaseModel):
    headline: str = Field(min_length=10, max_length=1000)
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    series: Optional[str] = Field(default=None, max_length=64)
    horizons: Optional[List[int]] = Field(default=None, max_length=4)


class NewsItem(BaseModel):
    story_id: str = Field(min_length=1, max_length=200)
    ts: str = Field(min_length=10, max_length=40)
    headline: str = Field(min_length=1, max_length=2000)
    source: Optional[str] = ""
    query: Optional[str] = ""


class NewsIn(BaseModel):
    items: List[NewsItem] = Field(max_length=1000)


class SeriesIn(BaseModel):
    ric: str = Field(min_length=1, max_length=64)
    name: Optional[str] = Field(default=None, max_length=120)
    field: Optional[str] = Field(default=None, max_length=64)
    points: List[List] = Field(max_length=10000)


def compact(analysis: dict) -> dict:
    """What the feed list shows for a scored headline."""
    cone = analysis.get("forecast_cone") or {}
    return {"event_type": analysis.get("event_type"),
            "score": ((analysis.get("score") or {}).get("composite") or {}).get("score"),
            "red_flag": any(c.get("red_flag") for c in cone.values()),
            "cone": {h: {k: c.get(k) for k in ("p5", "median", "p95", "signal", "red_flag", "is_bimodal")} for h, c in cone.items()},
            "n_analogues": len(analysis.get("analogues") or []),
            "series": (analysis.get("live") or {}).get("series")}


class LiveService:
    def __init__(self, state: dict, busy: threading.BoundedSemaphore, allow: Callable[[str], bool], data_root: Path,
                 data_dir: Optional[Path] = None, live_user: Optional[str] = None, live_password: Optional[str] = None,
                 ingest_token: Optional[str] = None, source_factory: Optional[Callable] = None,
                 default_series: Optional[str] = None, worker_interval: float = 3.0):
        env = os.environ.get
        self.state, self.busy, self.allow, self.data_root = state, busy, allow, Path(data_root)
        self.data_dir = Path(data_dir or env("DATA_DIR") or Path(__file__).resolve().parent / "runtime")
        self.live_user = live_user if live_user is not None else env("LIVE_USERNAME")
        self.live_password = live_password if live_password is not None else env("LIVE_PASSWORD")
        self.ingest_token = ingest_token if ingest_token is not None else env("INGEST_TOKEN")
        self.source_factory = source_factory
        self.default_series = default_series or env("DEFAULT_SERIES") or BUNDLED
        self.worker_interval = worker_interval
        self.queries = [q for q in env("NEWS_QUERIES", "Topic:GOL AND Language:LEN").split("|") if q.strip()]
        self.rics = [r for r in env("SERIES_RICS", "XAU=").split(",") if r.strip()]
        self.store: Optional[Store] = None
        self.stop = threading.Event()
        self.threads: List[threading.Thread] = []
        self.cache: dict = {}
        self.refinitiv_ok: Optional[bool] = None
        self.last_ingest: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self.store = Store(self.data_dir / "live.sqlite3")
        self._seed_bundled()
        self._spawn(self._worker, "scorer")
        if self.source_factory is not None:
            self._spawn(self._puller, "refinitiv")
        log.info("live: store %s, default series %s, server-side Refinitiv %s", self.store.path, self.default_series,
                 "on" if self.source_factory else "off")

    def shutdown(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5)

    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self.threads.append(t)

    def _seed_bundled(self):
        f = self.data_root / "data" / "gold_prices.csv"
        if self.store.has_series(BUNDLED) or not f.exists():
            return
        src = self.data_root / "price_source.json"
        kind = json.loads(src.read_text()).get("source", "bundled") if src.exists() else "bundled"
        df = pd.read_csv(f)
        if "XAUUSD" in df.columns:
            self.store.upsert_series(BUNDLED, df[[df.columns[0], "XAUUSD"]], name=f"Gold spot (built-in, {kind})",
                                     field="XAUUSD", source=kind)

    # ------------------------------------------------------------------ series
    def prices_for(self, ric: Optional[str]):
        """(PriceData, name) for a stored series; None means use the engine's own corpus prices."""
        ric = ric or self.default_series
        meta = next((s for s in self.store.list_series() if s["ric"] == ric), None)
        if meta is None:
            raise HTTPException(404, f"No data stored for {ric}. Add it to the collector's --rics list.")
        key = (ric, meta["updated_at"], meta["points"])
        if key not in self.cache:
            if len(self.cache) > 16:
                self.cache.clear()
            self.cache[key] = series_prices(self.state["engine"].cfg, self.store.get_series(ric))
        return self.cache[key], meta["name"]

    # ------------------------------------------------------------------ background work
    def _worker(self):
        """Scores every new headline once, newest first, sharing the analysis slots with interactive requests."""
        while not self.stop.is_set():
            items = self.store.pending(limit=5)
            if not items:
                self.stop.wait(self.worker_interval)
                continue
            for it in items:
                if self.stop.is_set():
                    return
                try:
                    px, name = self.prices_for(None)
                except HTTPException:
                    px, name = None, "corpus"
                when = pd.Timestamp(it["ts"]).normalize()
                if not self.busy.acquire(timeout=60):
                    break
                try:
                    out = self.state["engine"].analyze(it["headline"], when, prices=px, series_name=name)
                    self.store.set_result(it["story_id"], compact(out))
                except Exception as e:  # one bad headline must not stop the feed
                    self.store.set_result(it["story_id"], error=str(e)[:300])
                finally:
                    self.busy.release()

    def _puller(self):
        try:
            src = self.source_factory().open()
            self.refinitiv_ok = True
        except Exception as e:
            self.refinitiv_ok = False
            log.warning("server-side Refinitiv session failed: %s", e)
            return
        try:
            FeedLoop(src, StoreSink(self.store), self.store, self.queries, self.rics).run(
                news_every=int(os.environ.get("NEWS_POLL_SECONDS", "60")),
                series_every=int(os.environ.get("SERIES_POLL_SECONDS", "3600")), stop=self.stop)
        finally:
            src.close()

    # ------------------------------------------------------------------ auth
    def viewer(self, request: Request):
        if not (self.live_user and self.live_password):
            raise HTTPException(503, "The live terminal is switched off: set LIVE_USERNAME and LIVE_PASSWORD on the server.")
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.lower().startswith("basic "):
            try:
                user, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                ok = secrets.compare_digest(user, self.live_user) and secrets.compare_digest(pw, self.live_password)
            except Exception:
                ok = False
        if not ok:
            raise HTTPException(401, "Sign in to use the live terminal.", headers={"WWW-Authenticate": 'Basic realm="Brahmanda Live"'})

    def ingester(self, request: Request):
        if not self.ingest_token:
            raise HTTPException(503, "Ingest is switched off: set INGEST_TOKEN on the server.")
        auth = request.headers.get("authorization", "")
        if not (auth.lower().startswith("bearer ") and secrets.compare_digest(auth[7:], self.ingest_token)):
            raise HTTPException(401, "Invalid ingest token.")

    # ------------------------------------------------------------------ routes
    def register(self, app: FastAPI):
        V, I = Depends(self.viewer), Depends(self.ingester)

        @app.get("/live", include_in_schema=False, dependencies=[V])
        def live_page():
            return FileResponse(LIVE_PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})

        @app.get("/api/live/status", dependencies=[V])
        def status():
            return {"default_series": self.default_series, "series": len(self.store.list_series()),
                    "news": self.store.news_counts(), "last_ingest": self.last_ingest,
                    "server_refinitiv": None if self.source_factory is None else self.refinitiv_ok,
                    "memory_range": self.state["info"]["memory_range"], "horizon_max": 90}

        @app.get("/api/live/series", dependencies=[V])
        def list_series():
            return {"default": self.default_series, "series": self.store.list_series()}

        @app.get("/api/live/series/{ric:path}", dependencies=[V])
        def get_series(ric: str, start: Optional[str] = None):
            if not self.store.has_series(ric):
                raise HTTPException(404, f"No data stored for {ric}.")
            df = self.store.get_series(ric, start)
            step = max(1, len(df) // MAX_PLOT_POINTS)
            df = df.iloc[::step]
            return {"ric": ric, "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(), "values": df["value"].round(6).tolist()}

        @app.get("/api/live/feed", dependencies=[V])
        def feed(limit: int = 60, before: Optional[str] = None):
            return {"items": self.store.feed(min(max(limit, 1), 200), before)}

        @app.post("/api/live/analyze", dependencies=[V])
        def live_analyze(body: LiveAnalyzeIn, request: Request):
            engine = self.state["engine"]
            if not self.allow(request.client.host if request.client else "unknown"):
                raise HTTPException(429, "Too many analyses in a minute. Try again shortly.", headers={"Retry-After": "60"})
            when = None
            if body.date:
                try:
                    when = pd.Timestamp(body.date)
                except ValueError:
                    raise HTTPException(422, f"{body.date} is not a valid date.")
                if when > pd.Timestamp.now("UTC").tz_localize(None):
                    raise HTTPException(422, "The date cannot be in the future.")
            px, name = self.prices_for(body.series)
            if not self.busy.acquire(timeout=30):
                raise HTTPException(503, "The server is busy. Try again in a few seconds.")
            try:
                return engine.analyze(body.headline, when, prices=px, horizons=body.horizons, density=True,
                                      series_name=name)
            except ValueError as e:
                raise HTTPException(422, str(e))
            finally:
                self.busy.release()

        @app.post("/api/ingest/news", dependencies=[I])
        def ingest_news(body: NewsIn):
            items = []
            for it in body.items:
                ts = pd.to_datetime(it.ts, errors="coerce", utc=True)
                if pd.isna(ts):
                    raise HTTPException(422, f"bad timestamp {it.ts!r} for story {it.story_id}")
                items.append({**it.model_dump(), "ts": ts.tz_convert(None).isoformat(timespec="seconds")})
            added = self.store.add_news(items)
            self.last_ingest = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            return {"received": len(items), "added": added}

        @app.post("/api/ingest/series", dependencies=[I])
        def ingest_series(body: SeriesIn):
            try:
                df = pd.DataFrame(body.points, columns=["Date", "value"])
                df["Date"] = pd.to_datetime(df["Date"], errors="raise")
                df["value"] = pd.to_numeric(df["value"], errors="raise")
            except Exception:
                raise HTTPException(422, "points must be [[\"YYYY-MM-DD\", number], ...]")
            stored = self.store.upsert_series(body.ric, df, name=body.name, field=body.field)
            self.last_ingest = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            return {"ric": body.ric, "stored": stored}
