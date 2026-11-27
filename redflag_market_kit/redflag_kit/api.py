"""Market-agnostic Red Flag API (FastAPI). Same routes the v6 dashboard calls; data comes from the market folder."""
from __future__ import annotations
import json, os, threading, uuid
from pathlib import Path
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from . import kde as kde_mod
from . import scoring
from . import scenarios as scen
from .backtest import run_backtest
from .config import MarketConfig
from .prices import PriceData
from .retrieval import Retriever


def create_app(cfg: MarketConfig) -> FastAPI:
    app = FastAPI(title=f"Red Flag API - {cfg['display_name']}", version="3.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    C = {}; bt = {"status": "idle", "progress": 0, "total": 0, "message": "", "results": None}; lock = threading.Lock()
    kde_dir = lambda: cfg.path("kde_dir")

    def retr() -> Retriever:
        if "r" not in C: C["r"] = Retriever.from_config(cfg)
        return C["r"]

    def prices() -> PriceData:
        if "p" not in C: C["p"] = PriceData(cfg)
        return C["p"]

    def kde_results(horizon: int = 15) -> pd.DataFrame:
        key = f"k{horizon}"
        f = kde_dir() / f"kde_v3_results_h{horizon}.csv"
        if key not in C and f.exists(): C[key] = pd.read_csv(f)
        return C.get(key, pd.DataFrame())

    def event(eid):
        e = retr().by_id.get(eid)
        if e is None: raise HTTPException(404, f"{eid} not found in test corpus")
        return e

    def train_text() -> dict:
        if "tt" not in C: C["tt"] = {r["event_id"]: r["text"] for r in retr().train}
        return C["tt"]

    def enrich(analogues: list) -> list:
        tt = train_text()
        return [{**a, "summary": tt.get(a["event_id"], "")[:140]} for a in analogues]

    @app.get("/")
    def root(): return {"name": f"Red Flag API - {cfg['display_name']}", "market": cfg.market, "status": "operational"}

    @app.get("/health")
    def health():
        r = retr(); v = prices().df
        return {"status": "operational", "market": cfg.market, "events_loaded": len(r.test), "corpus_emb_shape": list(r.M.shape),
                "kde_dir_exists": (kde_dir() / "per_event").exists(), "ovx_loaded": "VOL" in v.columns, "vol_loaded": "VOL" in v.columns,
                "calibrated": cfg.get("status", {}).get("calibrated"), "uncalibrated_params": cfg.uncalibrated()}

    @app.get("/config")
    def config():
        return {"market": cfg.market, "ticker_label": cfg["ticker_label"], "vol_label": cfg["dashboard"]["vol_label"], "dashboard": cfg["dashboard"],
                "horizon": cfg["backtest"]["horizon"]}

    @app.get("/events")
    def events(q: str = "", type: str = "", limit: int = 100):
        ev = sorted(retr().test, key=lambda r: r["date"], reverse=True)
        if q: ev = [e for e in ev if q.lower() in e["text"].lower()]
        if type and type != "All": ev = [e for e in ev if e["event_type"] == type]
        out = [{"event_id": e["event_id"], "text": e["text"][:300], "headline": e["text"][:200], "date": e["date"].isoformat(), "source": "corpus",
                "Event_type": e["event_type"], "event_type": e["event_type"],
                "kde_available": (kde_dir() / "per_event" / f"{e['event_id']}_kde_v3.json").exists()} for e in ev[:limit]]
        return {"events": out, "total": len(out)}

    @app.post("/analyze")
    def analyze(payload: dict):
        h, eid = payload.get("headline", ""), payload.get("event_id", "")
        if not h: raise HTTPException(400, "headline required")
        cl = scoring.classify_headline(cfg, h); sev = cl["severity"]; rf = scoring.red_flag_score(cfg, h, sev)
        row, similar, cone, e = None, [], {}, retr().by_id.get(eid) if eid else None
        if e is not None:
            # live retrieval + KDE for every configured horizon, computed fresh in-process
            # (no precomputed-file dependency) - this is what makes similar_events and
            # forecast_cone real instead of the heuristic fallback below.
            res = retr().retrieve_adaptive(e, cfg["retrieval"]["tau"], cfg["kde"]["min_analogues"])
            similar = enrich(res.get("analogues", []))
            for hz in cfg["kde"]["horizons"]:
                kr, _, _ = kde_mod.run_event(cfg, eid, e["date"], res.get("analogues", []), prices(), hz)
                if "error" not in kr:
                    cone[str(hz)] = {"p5": kr["kde_p5"], "median": kr["kde_median"], "p95": kr["kde_p95"],
                                      "is_bimodal": bool(kr.get("kde_is_bimodal"))}
                    if hz == cfg["backtest"]["horizon"]: row = kr
        sig = 4 + sev * 1.5
        src = "precomputed" if row is not None else "heuristic"
        g = lambda c, d: float(row[c]) if row is not None and c in row and pd.notna(row[c]) else d
        return {"headline": h, "event_id": eid, **cl, "severity_label": cfg["severity_labels"][str(sev)], "red_flag_score": rf,
                "red_flag_label": scoring.label_for(cfg, rf), "kde_source": src,
                "kde_p5": g("kde_p5", round(-sig * 1.65, 1)), "kde_median": g("kde_median", round(sev * 0.8 - 1.2, 1)),
                "kde_p95": g("kde_p95", round(sig * 1.65, 1)), "tail_weight": g("kde_tail_weight", round(0.1 + sev * 0.06, 3)),
                "is_bimodal": bool(g("kde_is_bimodal", 0)), "similar_events": similar, "similar_events_found": len(similar),
                "forecast_cone": cone}

    @app.post("/retrieve")
    def retrieve(payload: dict):
        e = event(payload.get("event_id", ""))
        res = retr().retrieve_adaptive(e, float(payload.get("tau", 0.85)), int(payload.get("k_min", 25)))
        res["analogues"] = enrich(res["analogues"])
        return res

    @app.get("/kde/{event_id}")
    def kde(event_id: str):
        p = kde_dir() / "per_event" / f"{event_id}_kde_v3.json"
        if not p.exists(): raise HTTPException(404, f"No KDE for {event_id}")
        d = json.loads(p.read_text()); x, y = d.get("x_grid", []), d.get("density", [])
        if len(x) > 300: s = len(x) // 300; x, y = x[::s], y[::s]
        return {"event_id": event_id, "posterior_stats": d.get("posterior_stats", {}), "garch": d.get("garch", {}), "x_grid": x, "density": y,
                "base_date": d.get("base_date", "")}

    @app.get("/ovx")
    @app.get("/vol")
    def vol(days: int = 90):
        v = prices().df
        if "VOL" not in v.columns: raise HTTPException(404, "no volatility index loaded")
        s = v["VOL"].dropna().tail(days)
        return {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "values": [round(float(x), 2) for x in s], "latest": round(float(s.iloc[-1]), 2)}

    @app.get("/score/{event_id}")
    def score(event_id: str):
        e = event(event_id); res = retr().retrieve_adaptive(e, 0.85, 25)
        p = kde_dir() / "per_event" / f"{event_id}_kde_v3.json"
        ks = json.loads(p.read_text()).get("posterior_stats") if p.exists() else None
        return {"event_id": event_id, **scoring.score_event(cfg, e["text"], res, ks, vol=prices().vol_at(e["date"]) if not ks else None)}

    @app.post("/report")
    def report(payload: dict):
        eid = payload.get("event_id", ""); k = kde_results(); a = {}
        if eid and not k.empty:
            m = k[k["event_id"] == eid]
            if not m.empty:
                r = m.iloc[0]
                a = {"red_flag": int(r.get("red_flag", 0)), "p5": float(r.get("kde_p5", 0)), "median": float(r.get("kde_median", 0)),
                     "p95": float(r.get("kde_p95", 0)), "bimodal": bool(r.get("kde_is_bimodal", 0)), "tail_weight": float(r.get("kde_tail_weight", 0))}
        md = (f"## Event risk report - {cfg['display_name']}\n**Event:** {eid or 'manual'}\n\n- P5 {a.get('p5', 0):+.1f}%  median {a.get('median', 0):+.1f}%  "
              f"P95 {a.get('p95', 0):+.1f}%\n- Red flag: {'YES' if a.get('red_flag') else 'no'}; "
              f"{'bimodal' if a.get('bimodal') else 'unimodal'}; tail weight {a.get('tail_weight', 0) * 100:.0f}%\n\n*Template report; no LLM configured.*")
        return {"report_id": str(uuid.uuid4())[:8], "event_id": eid, "engine": "template", "markdown": md}

    @app.post("/backtest/run")
    def bt_run(payload: dict):
        with lock:
            if bt["status"] == "running": raise HTTPException(409, "A backtest is already running")
            bt.update(status="running", progress=0, message="Starting...", results=None)
        c = {**cfg["backtest"], **payload}

        def work():
            try:
                k = kde_results(c.get("horizon", cfg["backtest"]["horizon"]))
                if k.empty: raise ValueError(f"No KDE results for horizon {c.get('horizon')} - run `kde --horizons {c.get('horizon')}` first")
                r = run_backtest(k, c, headlines=train_text() | {t["event_id"]: t["text"] for t in retr().test})
                with lock: bt.update(status="done", progress=r["n_events"], total=r["n_events"], message="Complete", results=r)
            except Exception as e:
                with lock: bt.update(status="error", message=str(e))
        threading.Thread(target=work, daemon=True).start()
        return {"status": "started", "config": c}

    @app.get("/backtest/status")
    def bt_status():
        with lock: return {k: v for k, v in bt.items() if k != "results"}

    @app.get("/backtest/results")
    def bt_results():
        with lock:
            if bt["results"] is None: raise HTTPException(404, "No results - run a backtest first")
            return bt["results"]

    # ── backtest scenario tests (redflag_kit.scenarios), opened in a new window from the dashboard ──
    kit = Path(__file__).resolve().parent.parent
    sjobs: dict = {}; sbusy = threading.Semaphore(1)

    @app.get("/scenarios/catalog")
    def scenarios_catalog():
        ready = any((kde_dir() / f"kde_v3_results_h{h}.csv").exists() for h in cfg["kde"]["horizons"])
        return {"scenarios": scen.CATALOG, "data": {"ready": ready, "kde_dir": str(kde_dir())}}

    @app.post("/scenarios/run")
    def scenarios_run(payload: dict):
        items = payload.get("scenarios") or []
        if not items or not isinstance(items, list):
            raise HTTPException(422, "scenarios: give a list of {id, params}")
        bad = [i.get("id") if isinstance(i, dict) else i for i in items
               if (i.get("id") if isinstance(i, dict) else i) not in scen.RUNNERS]
        if bad: raise HTTPException(422, f"unknown scenario(s): {bad}")
        btc = {**cfg["backtest"], **(payload.get("config") or {})}
        jid = str(uuid.uuid4())[:12]
        if len(sjobs) > 20:
            for old in [k for k, v in sjobs.items() if v["status"] in ("done", "error")][:len(sjobs) - 20]: sjobs.pop(old, None)
        sjobs[jid] = {"status": "queued", "message": "Waiting for the previous run to finish..."}

        def work():
            with sbusy:
                sjobs[jid].update(status="running", message="Loading data...")
                try:
                    frames = {h: kde_results(h) for h in (7, 15, 30) if not kde_results(h).empty}
                    res = scen.run_scenarios(cfg, btc, items, prices(), frames, progress=lambda m: sjobs[jid].update(message=m))
                    sjobs[jid].update(status="done", message="Done", result=res)
                except Exception as e:
                    sjobs[jid].update(status="error", message=f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()
        return {"job_id": jid}

    @app.get("/scenarios/status/{jid}")
    def scenarios_status(jid: str):
        j = sjobs.get(jid)
        if not j: raise HTTPException(404, "unknown job")
        return {"status": j["status"], "message": j["message"]}

    @app.get("/scenarios/result/{jid}")
    def scenarios_result(jid: str):
        j = sjobs.get(jid)
        if not j: raise HTTPException(404, "unknown job")
        if j["status"] != "done": raise HTTPException(409, f"job is {j['status']}")
        return j["result"]

    @app.get("/scenarios/page", include_in_schema=False)
    @app.get("/redflag_scenarios_gold.html", include_in_schema=False)
    def scenarios_page():
        return FileResponse(kit / "redflag_scenarios_gold.html", media_type="text/html")

    @app.get("/redflag_dashboard_gold.html", include_in_schema=False)
    def dashboard_file():
        return FileResponse(kit / "redflag_dashboard_gold.html", media_type="text/html")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        f = cfg.path("dashboard")
        if f and f.exists(): return HTMLResponse(f.read_text(encoding="utf-8"))
        raise HTTPException(404, f"dashboard not found: {f} (run scripts/port_dashboard.py)")
    return app
