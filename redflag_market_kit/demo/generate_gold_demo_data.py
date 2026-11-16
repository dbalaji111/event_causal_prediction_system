"""Synthetic gold market folder matching markets/gold.json's `paths` exactly, built from gold.json's
own event_keywords so classify_headline() and the v6 scoring produce varied, on-theme results.
Mirrors generate_demo_data.py (oil) but gold has its own column names, file formats (CSV not XLSX,
train is JSON/test is JSONL at different paths) and a GVZ-style (not OVX-style) vol index.
For demo purposes only — gold.json is explicitly flagged uncalibrated (see markets/gold.json "status").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_LABELS = [
    (5, "5%"), (20, "5%-20%"), (30, "20%-30%"), (40, "30%-40%"),
    (60, "40%-60%"), (80, "60%-80%"), (90, "80%-90%"), (101, "90%+"),
]

HEADLINE_TEMPLATES = {
    "Geopolitical Risk": [
        "Gold Jumps As Safe-Haven Demand Surges On Military Escalation",
        "War Fears Drive Investors Into Gold As Conflict Widens",
        "Sanctions Standoff Fuels Safe-Haven Bid For Bullion",
        "Gold Rallies As Geopolitical Tension Spurs Risk-Off Flows",
    ],
    "Monetary Policy": [
        "Fed Signals Rate Cut Path, Gold Extends Gains On Lower Real Yields",
        "Powell's Hawkish Tone Pressures Gold As Dollar Firms",
        "Treasury Yields Slide, Gold Climbs On Inflation Print",
        "ECB Rate Decision Lifts Gold Against Weaker Dollar",
    ],
    "Central Bank Buying": [
        "PBOC Adds To Gold Reserves For Sixth Straight Month",
        "Central Banks Accelerate Bullion Purchases Amid Reserve Diversification",
        "Record Central Bank Gold Buying Underpins Price Floor",
    ],
    "Demand Signal": [
        "Indian Jewellery Demand Rebounds Ahead Of Festival Season",
        "Gold ETF Holdings See Largest Weekly Inflow In a Year",
        "China Gold Demand Cools As Consumers Pull Back",
        "ETF Outflows Weigh On Gold Despite Firm Physical Demand",
    ],
    "Price Movement": [
        "Gold Surges To Record High On Broad-Based Buying",
        "Bullion Slumps From Highs As Profit-Taking Accelerates",
        "Gold Spikes Then Reverses In Volatile Session",
    ],
    "Market News": [
        "Gold Steady Ahead Of Key US Inflation Data",
        "Traders Await Jobs Report For Fed Rate Cues",
        "Bullion Markets Quiet In Thin Holiday Trading",
    ],
}
TYPES = list(HEADLINE_TEMPLATES)
PRICE_NODE_OFFSETS = [(-30, "price_t_minus_30"), (-15, "price_t_minus_15"), (-7, "price_t_minus_7"),
                       (7, "price_t_plus_7"), (15, "price_t_plus_15"), (30, "price_t_plus_30")]


def _quantile_label(pct: float) -> str:
    return next(label for hi, label in QUANTILE_LABELS if pct < hi)


def build_gold_demo_market(root: Path, paths: dict, n_train=450, n_test=40, seed=11) -> Path:
    """Writes the gold.json-shaped files under `root` if they don't already exist. Returns `root`."""
    root = Path(root)
    marker = root / ".demo_ok"
    if marker.exists():
        return root
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2010-01-01", "2026-06-30")
    xau = 1150.0 * np.exp(np.cumsum(rng.normal(0.00022, 0.009, len(dates))))
    gc_m1 = xau * (1 + rng.normal(0.0015, 0.0015, len(dates)))
    gc_m2 = gc_m1 * (1 + rng.normal(0.0012, 0.001, len(dates)))  # gold futures: usually mild contango

    prices_df = pd.DataFrame({"Date": dates, "XAUUSD": xau, "GC_M1": gc_m1, "GC_M2": gc_m2})
    price_file = root / paths["prices"]
    price_file.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(price_file, index=False)

    gvz = 13 + 7 * np.abs(rng.standard_t(4, len(dates))) + 4 * np.abs(np.diff(np.log(xau), prepend=np.log(xau[0]))) * 100
    vol_df = pd.DataFrame({"Date": dates.strftime("%m/%d/%Y"), "Price": np.clip(gvz, 8, 90)})
    vol_file = root / paths["vol_index"]
    vol_file.parent.mkdir(parents=True, exist_ok=True)
    vol_df.to_csv(vol_file, index=False)

    px = pd.Series(xau, index=dates)
    n_sub = 10
    type_centers = rng.normal(size=(len(TYPES), 768))
    sub_centers = type_centers[:, None, :] + 0.9 * rng.normal(size=(len(TYPES), n_sub, 768))

    def price_near(d, offset_days):
        t = d + pd.Timedelta(days=offset_days)
        k = max(0, min(len(px) - 1, px.index.searchsorted(t, side="right") - 1))
        return float(px.iloc[k])

    def make(n, lo, hi, start_i):
        pool = [d for d in dates if pd.Timestamp(lo) <= d <= pd.Timestamp(hi)]
        picks = sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))
        out = []
        for j, ix in enumerate(picks):
            d = pool[ix]
            type_i = int(rng.integers(len(TYPES)))
            etype = TYPES[type_i]
            sub_i = int(rng.integers(n_sub))
            emb = sub_centers[type_i, sub_i] + 0.4 * rng.normal(size=768)
            emb = emb / np.linalg.norm(emb)
            hist = px.loc[d - pd.Timedelta(days=5 * 365):d]
            pct = float((hist < px[d]).mean() * 100) if len(hist) >= 50 else 50.0
            rec = {
                "event_id": f"EVT_GOLD_{d.year}_{start_i + j:05d}",
                "date": d.strftime("%Y-%m-%d"),
                "bert_embeddings": [round(float(x), 5) for x in emb],
                "Event type": etype,
                "Quantile_Category": _quantile_label(pct),
                "Polarity": str(rng.choice(["Bullish", "Bearish", "Neutral"])),
                "text": str(rng.choice(HEADLINE_TEMPLATES[etype])),
                "price": float(px[d]),
            }
            for off, name in PRICE_NODE_OFFSETS:
                rec[name] = price_near(d, off)
            out.append(rec)
        return out

    train = make(n_train, "2011-06-01", "2024-06-30", 0)
    test = make(n_test, "2024-09-01", "2026-05-31", 10_000)

    train_file = root / paths["train_corpus"]
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_text(json.dumps(train), encoding="utf-8")

    test_file = root / paths["test_corpus"]
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("\n".join(json.dumps(r) for r in test) + "\n", encoding="utf-8")

    marker.write_text("ok", encoding="utf-8")
    return root
