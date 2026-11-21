"""Run after deploy/build.py (needs the built event memory): python -m pytest deploy/test_server.py
The embedding model is replaced by stand-ins here, so no model download is needed."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402
from redflag_kit.config import MarketConfig  # noqa: E402
from redflag_kit.corpus import load_records  # noqa: E402
from redflag_kit.live import LiveEngine  # noqa: E402

pytestmark = pytest.mark.skipif(not (server.DATA_ROOT / ".real_ok").exists(), reason="run deploy/build.py first")


def hash_embed(texts):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=384)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


@pytest.fixture(scope="module")
def cfg():
    return MarketConfig.load(server.MARKET_FILE, server.DATA_ROOT)


@pytest.fixture(scope="module")
def anchor(cfg):
    train = load_records(cfg.path("train_corpus"), cfg)
    return next(r for r in sorted(train, key=lambda r: r["date"]) if r["date"] >= pd.Timestamp("2015-06-01"))


def engine_returning(cfg, vec):
    pca = json.loads(server.MAP.read_text())["pca"]
    return LiveEngine(cfg, lambda texts: np.asarray([vec] * len(texts), dtype=np.float32), pca)


def test_finds_the_matching_event_once_its_outcome_has_matured(cfg, anchor):
    eng = engine_returning(cfg, anchor["embedding"])
    when = anchor["date"] + pd.Timedelta(days=400)
    out = eng.analyze("Gold climbs as the dollar weakens", when)
    assert out["analogues"][0]["event_id"] == anchor["event_id"]
    assert out["analogues"][0]["similarity"] > 0.999
    cutoff = when - pd.Timedelta(days=eng.maturity_days)
    assert all(pd.Timestamp(a["date"]) < cutoff for a in out["analogues"])


def test_never_uses_an_event_whose_outcome_overlaps_the_new_one(cfg, anchor):
    eng = engine_returning(cfg, anchor["embedding"])
    when = anchor["date"] + pd.Timedelta(days=10)
    out = eng.analyze("Gold climbs as the dollar weakens", when)
    assert anchor["event_id"] not in {a["event_id"] for a in out["analogues"]}
    cutoff = when - pd.Timedelta(days=eng.maturity_days)
    assert all(pd.Timestamp(a["date"]) < cutoff for a in out["analogues"])


def test_result_matches_the_snapshot_event_shape(cfg, anchor):
    eng = engine_returning(cfg, anchor["embedding"])
    out = eng.analyze("Central banks add to gold reserves again", anchor["date"] + pd.Timedelta(days=400))
    snap_event = json.loads(server.SNAPSHOT.read_text())["events"][0]
    assert set(snap_event) - {"forecast_cone"} <= set(out)
    assert set(out["analogues"][0]) >= set(snap_event["analogues"][0])
    assert out["forecast_cone"], "expected a KDE posterior for a well-covered date"
    for c in out["forecast_cone"].values():
        assert c["p5"] <= c["median"] <= c["p95"]
    assert len(out["map_xyz"]) == 3
    json.dumps(out, allow_nan=False)


def test_api(cfg):
    from fastapi.testclient import TestClient
    with TestClient(server.create_app(embed=hash_embed, rate_per_minute=4)) as client:
        h = client.get("/api/health").json()
        assert h["status"] == "ok" and h["memory_events"] > 5000 and h["price_source"] == "illustrative"
        assert client.get("/").headers["content-type"].startswith("text/html")
        assert client.get("/static_snapshot.json").json()["events"]

        r = client.post("/api/analyze", json={"headline": "Gold jumps as investors seek safety after missile strikes"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["analogues"] and body["score"]["composite"]["score"] >= 0 and body["live"]["elapsed_ms"] >= 0

        assert client.post("/api/analyze", json={"headline": "too short"}).status_code == 422
        assert client.post("/api/analyze", json={"headline": "Gold rallies on Fed cut hopes", "date": "2999-01-01"}).status_code == 422
        r = client.post("/api/analyze", json={"headline": "Gold rallies on Fed cut hopes", "date": "2025-02-30"})
        assert r.status_code == 422 and "not a valid date" in r.json()["detail"]
        assert client.post("/api/analyze", json={"headline": "Gold rallies on Fed cut hopes"}).status_code == 200
        r = client.post("/api/analyze", json={"headline": "Gold rallies on Fed cut hopes"})
        assert r.status_code == 429 and r.headers["retry-after"] == "60"
