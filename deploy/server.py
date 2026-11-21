"""Serves the gold event risk terminal and analyses new headlines live (POST /api/analyze)."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

KIT_DIR = Path(os.environ.get("KIT_DIR", Path(__file__).resolve().parent.parent / "redflag_market_kit"))
sys.path.insert(0, str(KIT_DIR))
from redflag_kit.config import MarketConfig  # noqa: E402
from redflag_kit.live import LiveEngine  # noqa: E402

MARKET_FILE = KIT_DIR / "markets" / "gold.json"
DATA_ROOT = KIT_DIR / "demo" / "data_root_gold_real"
PAGE = KIT_DIR / "redflag_dashboard_gold_static.html"
SNAPSHOT = KIT_DIR / "static_snapshot.json"
MAP = KIT_DIR / "embeddings_viz.json"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("terminal")


def minilm_embedder(name: str) -> Callable:
    """The model the article corpus was embedded with, so new headlines land in the same space."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    model = SentenceTransformer(name, device="cpu")
    return lambda texts: model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)


class AnalyzeIn(BaseModel):
    headline: str = Field(min_length=10, max_length=1000)
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


def create_app(embed: Optional[Callable] = None, rate_per_minute: Optional[int] = None,
               max_concurrent: Optional[int] = None) -> FastAPI:
    rate = rate_per_minute if rate_per_minute is not None else int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
    busy = threading.BoundedSemaphore(max_concurrent or int(os.environ.get("MAX_CONCURRENT_ANALYSES", "2")))
    hits: dict = defaultdict(deque)
    hits_lock = threading.Lock()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app):
        t0 = time.perf_counter()
        cfg = MarketConfig.load(MARKET_FILE, DATA_ROOT)
        pca = json.loads(MAP.read_text(encoding="utf-8")).get("pca") if MAP.exists() else None
        engine = LiveEngine(cfg, embed or minilm_embedder(EMBED_MODEL), pca)
        src_file = DATA_ROOT / "price_source.json"
        src = json.loads(src_file.read_text()) if src_file.exists() else {"source": "unknown"}
        state["engine"] = engine
        state["info"] = {
            "market": cfg["display_name"], "embedding_model": EMBED_MODEL if embed is None else "custom",
            "memory_events": engine.memory_size,
            "memory_range": [str(engine.memory_range[0].date()), str(engine.memory_range[1].date())],
            "price_source": src.get("source"), "price_range": [src.get("first_date"), src.get("last_date")],
            "analogue_maturity_days": engine.maturity_days,
        }
        log.info("ready in %.1fs: %s", time.perf_counter() - t0, state["info"])
        yield

    app = FastAPI(title="Gold Event Risk Terminal", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def allow(ip: str) -> bool:
        now = time.monotonic()
        with hits_lock:
            q = hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= rate:
                return False
            q.append(now)
            if len(hits) > 10_000:
                for k in [k for k, v in hits.items() if not v]:
                    del hits[k]
            return True

    @app.get("/api/health")
    def health():
        if "engine" not in state:
            raise HTTPException(503, "starting")
        return {"status": "ok", **state["info"]}

    @app.post("/api/analyze")
    def analyze(body: AnalyzeIn, request: Request):
        engine = state["engine"]
        if not allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "Too many analyses from your address. Try again in a minute.",
                                headers={"Retry-After": "60"})
        when = None
        if body.date:
            try:
                when = pd.Timestamp(body.date)
            except ValueError:
                raise HTTPException(422, f"{body.date} is not a valid date.")
            if when > pd.Timestamp.now("UTC").tz_localize(None):
                raise HTTPException(422, "The date cannot be in the future.")
            if when < engine.memory_range[0]:
                raise HTTPException(422, f"The event history starts on {engine.memory_range[0].date()}.")
        if not busy.acquire(timeout=30):
            raise HTTPException(503, "The server is busy. Try again in a few seconds.")
        try:
            out = engine.analyze(body.headline, when)
        except ValueError as e:
            raise HTTPException(422, str(e))
        finally:
            busy.release()
        log.info("analyze: %d analogues, %d horizons, %d ms", len(out["analogues"]),
                 len(out["forecast_cone"]), out["live"]["elapsed_ms"])
        return out

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/static_snapshot.json", include_in_schema=False)
    def snapshot():
        return FileResponse(SNAPSHOT, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})

    @app.get("/embeddings_viz.json", include_in_schema=False)
    def embedding_map():
        return FileResponse(MAP, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})

    return app


app = create_app()
