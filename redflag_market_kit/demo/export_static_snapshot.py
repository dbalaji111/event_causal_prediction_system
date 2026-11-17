"""Exports a self-contained JSON snapshot of the real gold corpus + precomputed retrieval/KDE/score
results, for a fully static (no backend) build of the dashboard. Every number in it comes from a
real run of redflag_kit against the real article corpus (see build_real_gold_corpus.py) - this
script only serializes what's already on disk, it doesn't compute anything new except the v6 score
(scoring.score_event, same function the live /analyze endpoint calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from redflag_kit import scoring
from redflag_kit.config import MarketConfig
from redflag_kit.prices import PriceData
from redflag_kit.retrieval import Retriever

ROOT = Path(__file__).resolve().parent / "data_root_gold_real"
MARKET_JSON = Path(__file__).resolve().parent.parent / "markets" / "gold.json"


def main():
    cfg = MarketConfig.load(MARKET_JSON, root=ROOT)
    retr = Retriever.from_config(cfg)
    px = PriceData(cfg)
    kde_dir = cfg.path("kde_dir")

    train_text = {r["event_id"]: r["text"] for r in retr.train}

    h_csv = {h: pd.read_csv(kde_dir / f"kde_v3_results_h{h}.csv") for h in (7, 15, 30)}
    h_by_id = {h: df.set_index("event_id") for h, df in h_csv.items()}

    events = []
    backtest_rows = {7: [], 15: [], 30: []}
    referenced_train_ids = set()

    for e in sorted(retr.test, key=lambda r: r["date"], reverse=True):
        eid = e["event_id"]
        per_event_f = kde_dir / "per_event" / f"{eid}_kde_v3.json"
        cl = scoring.classify_headline(cfg, e["text"])

        retrieval_res = retr.retrieve_adaptive(e, cfg["retrieval"]["tau"], cfg["kde"]["min_analogues"])
        analogues = []
        for a in retrieval_res["analogues"][:40]:
            referenced_train_ids.add(a["event_id"])
            analogues.append({k: a[k] for k in
                               ("event_id", "date", "similarity", "recency_weight", "quantile_cat",
                                "event_type", "r7", "r15", "r30")})

        cone = {}
        for h in (7, 15, 30):
            if eid in h_by_id[h].index and pd.notna(h_by_id[h].loc[eid].get("kde_p5")):
                row = h_by_id[h].loc[eid]
                cone[str(h)] = {
                    "p5": float(row["kde_p5"]), "median": float(row["kde_median"]),
                    "p95": float(row["kde_p95"]), "is_bimodal": bool(row.get("kde_is_bimodal", 0)),
                    "tail_weight": float(row.get("kde_tail_weight", 0)),
                    "n_analogues": int(row.get("n_analogues", 0)),
                    "tau_effective": float(row.get("tau_effective", 0) or 0),
                    "red_flag": int(row["red_flag"]) if pd.notna(row.get("red_flag")) else 0,
                    "signal": str(row.get("signal", "")) if pd.notna(row.get("signal")) else "NEUTRAL",
                    "realised_r": None if pd.isna(row.get("realised_r")) else float(row["realised_r"]),
                    "covers_90": None if pd.isna(row.get("covers_90")) else int(row["covers_90"]),
                }
                backtest_rows[h].append({
                    "event_id": eid, "base_date": str(e["date"].date()),
                    "kde_median": float(row["kde_median"]),
                    "kde_p_down10": float(row.get("kde_p_down10", 0.1)),
                    "realised_r": cone[str(h)]["realised_r"],
                    "red_flag": int(row["red_flag"]) if pd.notna(row.get("red_flag")) else 0,
                    "covers_90": cone[str(h)]["covers_90"],
                    "headline": e["text"],
                })

        ks = None
        if per_event_f.exists():
            ks = json.loads(per_event_f.read_text())["posterior_stats"]
        score = scoring.score_event(cfg, e["text"], retrieval_res, ks,
                                     vol=px.vol_at(e["date"]) if not ks else None)

        events.append({
            "event_id": eid, "date": str(e["date"].date()), "text": e["text"],
            "event_type": e["event_type"],
            "classify": cl,
            "analogues": analogues,
            "forecast_cone": cone,
            "score": score,
        })

    # price & vol history (compact columnar arrays)
    pdf = px.df.reset_index()
    price_series = {
        "dates": [d.strftime("%Y-%m-%d") for d in pdf["Date"]],
        "px": [round(float(v), 2) for v in pdf["PX"]],
        "spread": [None if pd.isna(v) else round(float(v), 3) for v in pdf.get("SPREAD", pd.Series(dtype=float))],
        "vol": [None if pd.isna(v) else round(float(v), 2) for v in pdf.get("VOL", pd.Series(dtype=float))],
    }

    train_headlines = {eid: train_text.get(eid, "")[:160] for eid in referenced_train_ids}

    snapshot = {
        "market": {"display_name": cfg["display_name"], "ticker_label": cfg["ticker_label"],
                   "price_unit": cfg["price_unit"], "vol_label": cfg["dashboard"]["vol_label"],
                   "severity_labels": cfg["severity_labels"], "uncalibrated_params": cfg.uncalibrated()},
        "events": events,
        "train_headlines": train_headlines,
        "price_series": price_series,
        "backtest_rows": backtest_rows,
        "meta": {"n_train": len(retr.train), "n_test": len(retr.test),
                 "generated_note": "Real redflag_kit output over the real gold article corpus; "
                                    "price/vol are an illustrative historical approximation, not a feed "
                                    "(see markets/gold.json status.notes)."},
    }

    out = Path(__file__).resolve().parent.parent / "static_snapshot.json"
    # allow_nan=False: fail loudly here rather than ship a NaN/Infinity token that breaks the
    # browser's strict JSON.parse (Python's json module writes them by default; JS's doesn't accept them).
    out.write_text(json.dumps(snapshot, default=str, allow_nan=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) - {len(events)} events, "
          f"{len(train_headlines)} referenced train headlines")


if __name__ == "__main__":
    main()
