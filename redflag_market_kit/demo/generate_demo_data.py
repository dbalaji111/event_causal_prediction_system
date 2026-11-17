"""Synthetic oil market folder matching markets/oil.json's `paths` and `corpus_fields` exactly, so the
Red Flag pipeline (retrieval -> KDE -> scoring -> backtest) runs end-to-end without the real corpus/price
files (those live on the user's own machine — see oil.json's `paths` section).

Random walk prices + invented headlines built from oil.json's own event_keywords, so classify_headline()
and the behavioural/structural scoring produce varied, on-theme results. For demo purposes only.
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
        "Iran Tensions Escalate Near Strait Of Hormuz",
        "Russia Sanctions Widen As Military Conflict Drags On",
        "OPEC+ Nations Clash Over Embargo Strategy",
        "Hezbollah Attack Raises Alarm Over Regional Oil Supply",
        "Hormuz Blockade Fears Resurface Amid War Escalation",
    ],
    "Supply Shock": [
        "OPEC Announces Surprise Production Cut Amid Oversupply Concerns",
        "Major Pipeline Shutdown Disrupts Crude Barrel Supply",
        "Quota Cuts Deepen As Production Disruption Spreads",
        "Unplanned Refinery Shutdown Tightens Supply Outlook",
        "OPEC+ Extends Supply Cuts Into Next Quarter",
    ],
    "Demand Signal": [
        "China Manufacturing PMI Signals Slowdown In Oil Demand",
        "Global Recession Fears Weigh On Crude Demand Outlook",
        "Weak GDP Growth Clouds Energy Demand Forecasts",
        "China Factory Slowdown Deepens Demand Concerns",
    ],
    "Monetary Policy": [
        "Fed Rate Hike Lifts Dollar, Pressures Oil Prices",
        "ECB Signals Hawkish Pivot As Inflation Persists",
        "Dollar Rally On Fed Rate Path Weighs On Crude",
        "Central Bank Inflation Fight Pressures Commodity Prices",
    ],
    "Price Movement": [
        "Crude Oil Prices Surge On Volatile Trading Session",
        "Oil Plunges As Market Volatility Spikes",
        "Crude Rallies Sharply After Session Of Sharp Swings",
        "Oil Prices Crash On Thin, Volatile Trading",
    ],
    "Market News": [
        "Weekly Inventory Report Shows Mixed Signals For Crude",
        "Traders Await EIA Stockpile Data",
        "Crude Markets Steady Ahead Of Weekly Inventory Report",
        "Oil Edges Lower In Quiet Overnight Session",
    ],
}
TYPES = list(HEADLINE_TEMPLATES)
PRICE_NODE_OFFSETS = [(-30, "price_t_minus_30"), (-15, "price_t_minus_15"), (-7, "price_t_minus_7"),
                       (7, "price_t_plus_7"), (15, "price_t_plus_15"), (30, "price_t_plus_30")]


def _quantile_label(pct: float) -> str:
    return next(label for hi, label in QUANTILE_LABELS if pct < hi)


def build_demo_market(root: Path, paths: dict, n_train=450, n_test=40, seed=7) -> Path:
    """Writes the oil.json-shaped files under `root` if they don't already exist. Returns `root`."""
    root = Path(root)
    marker = root / ".demo_ok"
    if marker.exists():
        return root
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2008-01-01", "2026-06-30")
    wti = 70.0 * np.exp(np.cumsum(rng.normal(0.00015, 0.018, len(dates))))
    wti_close = wti * (1 + rng.normal(0, 0.002, len(dates)))
    wti_spread = rng.normal(0.0, 0.35, len(dates)).cumsum() * 0.02  # slow-drifting contango/backwardation
    wti_spread = np.clip(wti_spread - wti_spread.mean(), -2.5, 2.5)
    brent_m1 = wti * (1 + rng.normal(0.01, 0.01, len(dates)))
    brent_m2 = brent_m1 * (1 - rng.normal(0.002, 0.004, len(dates)))

    prices_df = pd.DataFrame({
        "Date": dates,
        "Crude Oil-WTI Spot Cushing U$/BBL": wti,
        "Crude Oil WTI NYMEX Close M U$/BBL": wti_close,
        "NYMEX Crude Oil WTI M1-M2 Spread": wti_spread,
        "Crude Oil Brent US M1 U$/BBL": brent_m1,
        "Crude Oil Brent US M2 U$/BBL": brent_m2,
    })
    price_file = root / paths["prices"]
    price_file.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_excel(price_file, index=False)

    ovx = 18 + 14 * np.abs(rng.standard_t(4, len(dates))) + 6 * np.abs(np.diff(np.log(wti), prepend=np.log(wti[0]))) * 100
    vol_df = pd.DataFrame({"Date": dates.strftime("%m/%d/%Y"), "Price": np.clip(ovx, 9, 180)})
    vol_file = root / paths["vol_index"]
    vol_file.parent.mkdir(parents=True, exist_ok=True)
    vol_df.to_csv(vol_file, index=False)

    px = pd.Series(wti, index=dates)
    # Two-level structure (type center -> several sub-topic centers -> per-event noise) so each
    # type forms a handful of visible sub-blobs under PCA instead of one point per type, while
    # same-type cosine similarity still clears the retrieval tau thresholds most of the time.
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
                "event_id": f"EVT_OIL_{d.year}_{start_i + j:05d}",
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

    train = make(n_train, "2009-06-01", "2023-06-30", 0)
    test = make(n_test, "2023-09-01", "2026-05-31", 10_000)

    train_file = root / paths["train_corpus"]
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_text(json.dumps(train), encoding="utf-8")

    test_file = root / paths["test_corpus"]
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("\n".join(json.dumps(r) for r in test) + "\n", encoding="utf-8")

    marker.write_text("ok", encoding="utf-8")
    return root
