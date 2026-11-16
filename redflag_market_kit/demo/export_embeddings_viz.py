"""Exports a compact JSON for the 3D embedding map panel of the static gold terminal: the exact
same PCA-3D projection, KMeans topic clusters, and KDE-based temporal bursts already computed in
scripts/main_app.py's "Clusters & Retrieval" tab (and streamlit_gold_embeddings_visualisation.py),
over the same real 384-d BERT embeddings - just baked to a file instead of recomputed by Streamlit.

event_id here is str(Doc_ID), the same key build_real_gold_corpus.py uses for the redflag_kit
corpus, so every event the gold terminal can select has a matching point in this file.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "scripts" / "causal_gold_articles_with_topics_bert.csv"
OUT = Path(__file__).resolve().parent.parent / "embeddings_viz.json"

N_CLUSTERS = 6          # same default as main_app.py's "cluster_k" slider
BURST_WINDOW_DAYS = 10  # same default as main_app.py's "burst_window" slider

_BYLINE_RE = re.compile(r"^.*?All Rights Reserved\.\s*", re.DOTALL)


def _headline_or_snippet(headline: str, content: str) -> str:
    if str(headline).strip().upper() != "UNKNOWN":
        return str(headline)
    body = _BYLINE_RE.sub("", str(content or ""), count=1).strip()
    return (body[:140] + "…") if body else "UNKNOWN"


def main():
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "gold_general_embedding"])
    df = df[~df["Headline"].astype(str).str.contains("Search Summary", case=False, na=False)]
    df = df[df["Doc_ID"].astype(str).str.match(r"^[A-Za-z0-9]{10,}$")]  # drops 2 malformed rows
    df = df.sort_values("Date").reset_index(drop=True)

    embeddings = np.vstack(df["gold_general_embedding"].apply(ast.literal_eval).apply(np.array).values)

    # same PCA(3) projection as main_app.py's load_event_data()
    xyz = PCA(n_components=3, random_state=42).fit_transform(embeddings)

    # same KMeans over full 384-d embeddings as compute_embedding_clusters()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    topic_cluster = km.fit_predict(embeddings)

    # same KDE-over-dates burst detection as compute_temporal_density(), then assign each event
    # to its nearest burst peak if within BURST_WINDOW_DAYS, else -1 ("background", no burst)
    day_nums = df["Date"].values.astype("datetime64[D]").astype(float)
    kde = gaussian_kde(day_nums)
    grid = np.linspace(day_nums.min(), day_nums.max(), 1500)
    density = kde(grid)
    peak_idx, _ = find_peaks(density, distance=15, prominence=density.max() * 0.08)
    peak_days = grid[peak_idx]
    peak_dates = [pd.Timestamp("1970-01-01") + pd.Timedelta(days=d) for d in peak_days]

    if len(peak_days):
        dist_to_peak = np.abs(day_nums[:, None] - peak_days[None, :])
        nearest = dist_to_peak.argmin(axis=1)
        within = dist_to_peak[np.arange(len(day_nums)), nearest] <= BURST_WINDOW_DAYS
        temporal_cluster = np.where(within, nearest, -1)
    else:
        temporal_cluster = np.full(len(df), -1)

    points = []
    for i, row in df.iterrows():
        points.append({
            "event_id": str(row["Doc_ID"]),
            "date": row["Date"].strftime("%Y-%m-%d"),
            "headline": _headline_or_snippet(row["Headline"], row.get("Content", ""))[:160],
            "topic": str(row["assigned_topic_bert"]),
            "relevance": None if pd.isna(row.get("gold_relevance_score")) else round(float(row["gold_relevance_score"]), 3),
            "x": round(float(xyz[i, 0]), 4), "y": round(float(xyz[i, 1]), 4), "z": round(float(xyz[i, 2]), 4),
            "topic_cluster": int(topic_cluster[i]),
            "temporal_cluster": int(temporal_cluster[i]),
            # the KDE burst detector (same params as main_app.py) finds only the dominant
            # multi-year burst over a corpus spanning 1999-2025 - year buckets give a second,
            # coarser temporal grouping that stays informative across the whole span.
            "year_bin": int(row["Date"].year),
        })

    out_obj = {
        "points": points,
        "topic_clusters": {str(c): sorted(set(p["topic"] for p in points if p["topic_cluster"] == c))
                            for c in range(N_CLUSTERS)},
        "temporal_clusters": [
            {"id": int(j), "peak_date": pd_ts.strftime("%Y-%m-%d"),
             "n_events": int((temporal_cluster == j).sum())}
            for j, pd_ts in enumerate(peak_dates)
        ],
        "meta": {
            "n_points": len(points), "n_topic_clusters": N_CLUSTERS,
            "n_temporal_clusters": len(peak_dates), "burst_window_days": BURST_WINDOW_DAYS,
            "year_range": [int(df["Date"].dt.year.min()), int(df["Date"].dt.year.max())],
            "source": "scripts/causal_gold_articles_with_topics_bert.csv — same corpus and PCA/"
                       "KMeans/KDE logic as scripts/main_app.py's Clusters & Retrieval tab.",
        },
    }
    OUT.write_text(json.dumps(out_obj, allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) - {len(points)} points, "
          f"{N_CLUSTERS} topic clusters, {len(peak_dates)} temporal bursts")


if __name__ == "__main__":
    main()
