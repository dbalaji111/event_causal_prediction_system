"""Builds markets/gold.json's corpus from the REAL article data in the repo's data/ and scripts/
folders (real headlines, real 384-d embeddings, real cause/effect extraction, real topic labels -
same corpus the main gold dashboard, main_app.py, already uses).

The one thing this repo has no file for anywhere is gold price history, and Yahoo Finance is
blocked from this sandbox (no network approval yet - see the session for how to grant it). Until
that's wired up, price nodes are built from a hand-anchored approximation of gold's well-known
historical price trajectory (sourced from general knowledge, not a feed) plus daily noise - good
enough to make retrieval/KDE/scoring behave sensibly (real events still land in roughly the right
price regime), but NOT real data. Everything price-derived (price, price_t_*, Quantile_Category,
GVZ) is illustrative only until a real price/vol file replaces data/paths.prices + paths.vol_index.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_LABELS = [
    (5, "5%"), (20, "5%-20%"), (30, "20%-30%"), (40, "30%-40%"),
    (60, "40%-60%"), (80, "60%-80%"), (90, "80%-90%"), (101, "90%+"),
]

# Approximate, widely-known gold (XAU/USD) price milestones - NOT a data feed. Interpolated with
# daily noise below. Replace with a real price file (paths.prices in markets/gold.json) when one
# is available.
PRICE_ANCHORS = {
    "1999-01-01": 290, "2001-01-01": 265, "2003-01-01": 345, "2005-01-01": 430,
    "2006-06-01": 590, "2008-03-01": 1000, "2008-11-01": 720, "2009-12-01": 1130,
    "2011-09-01": 1900, "2013-06-01": 1200, "2015-12-01": 1060, "2016-07-01": 1360,
    "2018-08-01": 1180, "2019-09-01": 1520, "2020-08-01": 2070, "2021-06-01": 1770,
    "2022-09-01": 1620, "2023-05-01": 2020, "2023-12-01": 2070, "2024-06-01": 2330,
    "2024-12-01": 2650, "2025-04-23": 3300, "2025-07-01": 3250,
}
PRICE_NODE_OFFSETS = [(-30, "price_t_minus_30"), (-15, "price_t_minus_15"), (-7, "price_t_minus_7"),
                       (7, "price_t_plus_7"), (15, "price_t_plus_15"), (30, "price_t_plus_30")]
_BYLINE_RE = re.compile(r"^.*?All Rights Reserved\.\s*", re.DOTALL)


def _headline_or_snippet(headline: str, content: str) -> str:
    """Headline extraction failed for ~16% of rows (Headline == 'UNKNOWN'); Content still has the
    real article text behind a Dow Jones copyright boilerplate prefix - strip that and use the
    opening sentence as a usable stand-in instead of literally showing 'UNKNOWN' in the feed."""
    if str(headline).strip().upper() != "UNKNOWN":
        return str(headline)
    body = _BYLINE_RE.sub("", str(content or ""), count=1).strip()
    return (body[:140] + "…") if body else "UNKNOWN"


def _quantile_label(pct: float) -> str:
    return next(label for hi, label in QUANTILE_LABELS if pct < hi)


def build_real_gold_market(root: Path, paths: dict, repo_root: Path, train_cutoff="2023-01-01",
                            seed=13) -> Path:
    """Writes the gold.json-shaped files under `root` from the repo's real article data.
    Idempotent: skips rebuilding if already done (delete `root/.real_ok` to force a rebuild)."""
    root = Path(root)
    marker = root / ".real_ok"
    if marker.exists():
        return root
    rng = np.random.default_rng(seed)

    # ---- price history: anchored approximation + noise (see module docstring) ----
    anchor_dates = pd.to_datetime(list(PRICE_ANCHORS.keys()))
    anchor_vals = np.array(list(PRICE_ANCHORS.values()), dtype=float)
    dates = pd.bdate_range(anchor_dates.min(), anchor_dates.max())
    base = np.interp(dates.astype(np.int64), anchor_dates.astype(np.int64), anchor_vals)
    noise = np.cumsum(rng.normal(0, 0.003, len(dates)))
    noise -= np.linspace(noise[0], noise[-1], len(noise))  # keep anchors exact-ish, noise mean-reverts
    xau = base * np.exp(noise)
    gc_m1 = xau * (1 + rng.normal(0.0015, 0.0015, len(dates)))
    gc_m2 = gc_m1 * (1 + rng.normal(0.0012, 0.001, len(dates)))

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

    def price_near(d, offset_days):
        t = d + pd.Timedelta(days=offset_days)
        k = max(0, min(len(px) - 1, px.index.searchsorted(t, side="right") - 1))
        return float(px.iloc[k])

    # ---- real articles: headline, embedding, cause/effect, topic - all from the actual corpus ----
    src = repo_root / "scripts" / "causal_gold_articles_with_topics_bert.csv"
    df = pd.read_csv(src)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "gold_general_embedding"])
    df = df[(df["Date"] >= px.index.min()) & (df["Date"] <= px.index.max())]
    # Factiva search-metadata rows ("Search Summary\nText\n(gold OR ...") got ingested as if they
    # were articles - drop them, they're query parameters, not news.
    df = df[~df["Headline"].astype(str).str.contains("Search Summary", case=False, na=False)]

    def to_record(row, idx):
        emb = ast.literal_eval(row["gold_general_embedding"])
        d = row["Date"]
        hist = px.loc[d - pd.Timedelta(days=5 * 365):d]
        pct = float((hist < px.asof(d)).mean() * 100) if len(hist) >= 50 else 50.0
        p0 = price_near(d, 0)
        rec = {
            "event_id": str(row["Doc_ID"]) or f"EVT_GOLD_REAL_{d.year}_{idx:05d}",
            "date": d.strftime("%Y-%m-%d"),
            "bert_embeddings": [round(float(x), 6) for x in emb],
            "Event type": str(row["assigned_topic_bert"]),
            "Quantile_Category": _quantile_label(pct),
            "text": _headline_or_snippet(row["Headline"], row.get("Content", ""))[:500],
            "price": p0,
        }
        for off, name in PRICE_NODE_OFFSETS:
            rec[name] = price_near(d, off)
        return rec

    records = [to_record(r, i) for i, (_, r) in enumerate(df.iterrows())]
    cutoff = pd.Timestamp(train_cutoff)
    train = [r for r in records if pd.Timestamp(r["date"]) < cutoff]
    test = [r for r in records if pd.Timestamp(r["date"]) >= cutoff]

    train_file = root / paths["train_corpus"]
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_text(json.dumps(train), encoding="utf-8")

    test_file = root / paths["test_corpus"]
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("\n".join(json.dumps(r) for r in test) + "\n", encoding="utf-8")

    marker.write_text(f"train={len(train)} test={len(test)}", encoding="utf-8")
    return root
