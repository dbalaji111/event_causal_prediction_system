"""Live terminal tests: python -m pytest deploy/test_live.py (after deploy/build.py). Refinitiv is replaced by a
fake source, the embedding model by a hash embedder."""
from __future__ import annotations

import base64
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402
from feeds.pipeline import FeedLoop, StoreSink  # noqa: E402
from feeds.refinitiv import normalise_headlines, pick_price_column  # noqa: E402
from feeds.store import Store  # noqa: E402
from test_server import hash_embed  # noqa: E402

pytestmark = pytest.mark.skipif(not (server.DATA_ROOT / ".real_ok").exists(), reason="run deploy/build.py first")
AUTH = {"Authorization": "Basic " + base64.b64encode(b"viewer:secret").decode()}
TOKEN = {"Authorization": "Bearer ingest-123"}


def fake_series(start="2000-01-01", end=None, seed=0, level=60.0):
    end = end or pd.Timestamp.now().normalize()
    d = pd.bdate_range(start, end)
    r = np.random.default_rng(seed).normal(0.0002, 0.018, len(d))
    return pd.DataFrame({"Date": d, "value": level * np.exp(np.cumsum(r))})


class FakeSource:
    def __init__(self):
        self.calls = 0

    def open(self):
        return self

    def close(self):
        pass

    def headlines(self, query, start=None, end=None, count=100):
        self.calls += 1
        now = pd.Timestamp.now("UTC").tz_localize(None).floor("s")
        return [{"story_id": f"s{self.calls}-{i}", "ts": (now - pd.Timedelta(minutes=i)).isoformat(),
                 "headline": f"Gold rallies as central banks add reserves, report {self.calls}-{i}", "source": "RTRS",
                 "query": query} for i in range(3)]

    def history(self, ric, start, end=None, field=None):
        df = fake_series(start=start, end=end, seed=len(ric))
        df.attrs["field"] = "TRDPRC_1"
        return df


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    app = server.create_app(embed=hash_embed, rate_per_minute=100, live={
        "data_dir": tmp_path, "live_user": "viewer", "live_password": "secret", "ingest_token": "ingest-123",
        "worker_interval": 0.2, "source_factory": None})
    with TestClient(app) as c:
        yield c


def test_auth_is_required(client):
    assert client.get("/live").status_code == 401
    assert client.get("/live").headers["www-authenticate"].startswith("Basic")
    assert client.get("/live", headers=AUTH).headers["content-type"].startswith("text/html")
    assert client.post("/api/ingest/news", json={"items": []}).status_code == 401
    assert client.get("/api/health").json()["live_enabled"] is True


def test_series_ingest_and_dynamic_kde(client):
    pts = fake_series(level=80.0)
    body = {"ric": "LCOc1", "name": "Brent front month", "field": "TRDPRC_1",
            "points": [[d.strftime("%Y-%m-%d"), float(v)] for d, v in zip(pts["Date"], pts["value"])][:9000]}
    assert client.post("/api/ingest/series", json=body, headers=TOKEN).json()["stored"] > 6000
    lst = client.get("/api/live/series", headers=AUTH).json()["series"]
    assert {s["ric"] for s in lst} >= {"LCOc1", "BUNDLED:XAUUSD"}
    s = client.get("/api/live/series/LCOc1", headers=AUTH).json()
    assert len(s["dates"]) == len(s["values"]) > 1000

    req = {"headline": "Gold jumps as investors seek safety after missile strikes", "date": "2020-03-02"}
    gold = client.post("/api/live/analyze", json={**req, "series": "BUNDLED:XAUUSD"}, headers=AUTH).json()
    brent = client.post("/api/live/analyze", json={**req, "series": "LCOc1", "horizons": [5, 20]}, headers=AUTH).json()
    assert gold["live"]["series"].startswith("Gold spot") and brent["live"]["series"] == "Brent front month"
    assert set(brent["forecast_cone"]) == {"5", "20"} and set(brent["densities"]) == {"5", "20"}
    assert [a["event_id"] for a in gold["analogues"]] == [a["event_id"] for a in brent["analogues"]]
    a0 = brent["analogues"][0]
    assert set(a0["outcomes"]) == {"5", "20"} and a0["outcomes"] != gold["analogues"][0]["outcomes"]
    cutoff = pd.Timestamp("2020-03-02") - pd.Timedelta(days=brent["live"]["maturity_days"])
    assert all(pd.Timestamp(a["date"]) < cutoff for a in brent["analogues"])
    d = brent["densities"]["20"]
    assert len(d["x"]) == len(d["y"]) > 100 and abs(np.trapz(d["y"], d["x"]) - 1) < 0.1
    assert client.post("/api/live/analyze", json={**req, "series": "NOPE"}, headers=AUTH).status_code == 404
    assert client.post("/api/live/analyze", json={**req, "horizons": [0]}, headers=AUTH).status_code == 422


def test_news_feed_is_scored(client):
    items = [{"story_id": f"n{i}", "ts": f"2026-11-2{i}T08:00:00Z", "headline": f"Gold climbs as the dollar weakens {i}",
              "source": "RTRS"} for i in range(3)]
    r = client.post("/api/ingest/news", json={"items": items + items[:1]}, headers=TOKEN).json()
    assert r == {"received": 4, "added": 3}
    deadline = time.time() + 60
    while time.time() < deadline:
        feed = client.get("/api/live/feed", headers=AUTH).json()["items"]
        if all(it["status"] != "new" for it in feed):
            break
        time.sleep(0.3)
    assert [it["story_id"] for it in feed] == ["n2", "n1", "n0"]
    assert all(it["status"] == "done" and it["summary"]["cone"] for it in feed), feed
    st = client.get("/api/live/status", headers=AUTH).json()
    assert st["news"]["by_status"]["done"] == 3


def test_feed_loop_with_store(tmp_path):
    store = Store(tmp_path / "x.sqlite3")
    loop = FeedLoop(FakeSource(), StoreSink(store), store, ["Topic:GOL"], ["XAU=", "LCOc1"], history_start="2024-01-01")
    loop.run(once=True)
    assert {s["ric"] for s in store.list_series()} == {"XAU=", "LCOc1"}
    assert store.news_counts()["by_status"]["new"] == 3
    first = store.get_kv("series_last::XAU=")
    loop.run(once=True)                                  # second round: incremental, duplicates ignored
    assert store.news_counts()["by_status"]["new"] == 6 and store.get_kv("series_last::XAU=") == first
    stop = threading.Event(); stop.set()
    loop.run(stop=stop)                                  # a set stop event returns at once


def test_refinitiv_frames_are_normalised():
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-20T10:00:00Z", "2026-11-20T09:00:00Z"]), name="versionCreated")
    h = pd.DataFrame({"headline": ["Gold  up", "Gold down"], "storyId": ["a", "b"], "sourceCode": ["NS:RTRS", "NS:RTRS"]}, index=idx)
    out = normalise_headlines(h, "q")
    assert out[0] == {"story_id": "a", "ts": "2026-11-20T10:00:00", "headline": "Gold up", "source": "NS:RTRS", "query": "q"}
    hist = pd.DataFrame({"TRDPRC_1": [None, None, 1.0], "MID_PRICE": [2000.5, 2001.0, 2002.0]},
                        index=pd.DatetimeIndex(["2026-11-18", "2026-11-19", "2026-11-20"], name="Date"))
    df, col = pick_price_column(hist)
    assert col == "MID_PRICE" and df["value"].tolist() == [2000.5, 2001.0, 2002.0]
