"""Builds the causal-chain dataset for the "Causal Chain Analyses" artifact: a directed graph linking
each event's extracted gold_effect text to the gold_cause text of later events, so a selected event shows
both what historically led to it and what it has historically tended to lead to - a QUALITATIVE read
("what kind of event tends to follow"), not the numerical return forecasting redflag_dashboard_gold_
static.html already does.

LINKING METHOD - READ BEFORE TRUSTING THE OUTPUT: edges are scored by TF-IDF cosine similarity between
one event's gold_effect text and a later event's gold_cause text. This is a classical lexical-overlap
heuristic, not semantic causal understanding, and it is a deliberate placeholder: the real linking logic
is meant to come from a BART-based causal model, run locally (outside this sandbox, which has no network
access to huggingface.co to even download a transformer here - see ingest_news.py's docstring for the
same constraint). Swap `score_links()` for a call into that model and nothing else here needs to change -
the output schema (events/edges/transition_table) stays the same either way.

Output: redflag_market_kit/causal_chain_snapshot.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "scripts" / "causal_gold_articles_with_topics_bert.csv"
OUT = Path(__file__).resolve().parent.parent / "causal_chain_snapshot.json"

MAX_LAG_DAYS = 120   # how far forward a cause is allowed to "follow" an effect and still count as linked
TOP_K_SUCCESSORS = 4  # outgoing edges kept per event, highest-scoring first
MIN_SCORE = 0.12      # TF-IDF cosine floor below which a link isn't shown as a real connection

_BYLINE_RE = re.compile(r"^.*?All Rights Reserved\.\s*", re.DOTALL)


def _headline_or_snippet(headline: str, content: str) -> str:
    if str(headline).strip().upper() != "UNKNOWN":
        return str(headline)
    body = _BYLINE_RE.sub("", str(content or ""), count=1).strip()
    return (body[:140] + "…") if body else "UNKNOWN"


def load_events() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "gold_general_embedding"])
    df = df[~df["Headline"].astype(str).str.contains("Search Summary", case=False, na=False)]
    df = df[df["Doc_ID"].astype(str).str.match(r"^[A-Za-z0-9]{10,}$")]  # same filters export_embeddings_viz.py uses
    df = df.dropna(subset=["gold_cause", "gold_effect"])
    bad_cause = df["gold_cause"].astype(str).str.contains("No gold cause identified", na=False)
    bad_effect = df["gold_effect"].astype(str).str.contains("No gold effect identified", na=False)
    df = df[~bad_cause & ~bad_effect]
    return df.sort_values("Date").reset_index(drop=True)


def score_links(df: pd.DataFrame):
    """Returns (effect_vectors, cause_vectors): TF-IDF rows for every event's effect text and cause
    text, in one shared vocabulary so effect_i . cause_j is a valid cosine-similarity comparison.
    THIS is the one function a real causal model replaces - everything around it (the date-window
    restriction, top-k selection, transition table) is linking-method-agnostic."""
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    both = pd.concat([df["gold_effect"], df["gold_cause"]], ignore_index=True)
    vec.fit(both)
    effect_vecs = vec.transform(df["gold_effect"])
    cause_vecs = vec.transform(df["gold_cause"])
    return effect_vecs, cause_vecs


def build_edges(df: pd.DataFrame, effect_vecs, cause_vecs) -> list[dict]:
    dates = df["Date"].values
    n = len(df)
    edges = []
    for i in range(n):
        d0 = dates[i]
        lo = np.searchsorted(dates, d0, side="right")  # strictly after i's date
        hi = np.searchsorted(dates, d0 + np.timedelta64(MAX_LAG_DAYS, "D"), side="right")
        if lo >= hi:
            continue
        window_causes = cause_vecs[lo:hi]
        sims = (effect_vecs[i] @ window_causes.T).toarray().ravel()  # cosine: TF-IDF rows are L2-normalized
        order = np.argsort(-sims)
        kept = 0
        for off in order:
            if kept >= TOP_K_SUCCESSORS:
                break
            score = float(sims[off])
            if score < MIN_SCORE:
                break  # order is descending, so nothing further clears the bar either
            j = lo + off
            edges.append({
                "source": df.iloc[i]["Doc_ID"], "target": df.iloc[j]["Doc_ID"],
                "score": round(score, 4),
                "days_gap": int((dates[j] - d0) / np.timedelta64(1, "D")),
            })
            kept += 1
    return edges


MIN_BART_SCORE = 0.5   # BART link probability below which a link isn't shown


def build_edges_from_bart(df: pd.DataFrame, scores_csv: Path) -> list[dict]:
    """Edges from the BART causal-link model (bart_causal/bart_causal_pipeline.py score): for each source event, its
    TOP_K_SUCCESSORS highest-probability later events within MAX_LAG_DAYS, kept when the probability is at least
    MIN_BART_SCORE. Only pairs the pipeline scored are considered."""
    sc = pd.read_csv(scores_csv)
    ids = set(df["Doc_ID"])
    sc = sc[sc["src"].isin(ids) & sc["tgt"].isin(ids) & (sc["days_gap"] > 0) & (sc["days_gap"] <= MAX_LAG_DAYS)
            & (sc["score"] >= MIN_BART_SCORE)]
    sc = sc.sort_values(["src", "score"], ascending=[True, False]).groupby("src").head(TOP_K_SUCCESSORS)
    return [{"source": r.src, "target": r.tgt, "score": round(float(r.score), 4), "days_gap": int(r.days_gap)}
            for r in sc.itertuples()]


def build_transition_table(df: pd.DataFrame, edges: list[dict]) -> dict:
    """7x7 'given a source event's type, what type does the linked successor tend to be' - aggregated
    over every kept edge, weighted by its link score, normalized to a probability row per source type.
    This is the aggregate, corpus-wide view; the per-event Sankey in the dashboard is the same signal
    narrowed to one selected event's own outgoing edges."""
    type_by_id = dict(zip(df["Doc_ID"], df["assigned_topic_bert"]))
    types = sorted(df["assigned_topic_bert"].unique())
    tbl = {t: {t2: 0.0 for t2 in types} for t in types}
    for e in edges:
        st, tt = type_by_id.get(e["source"]), type_by_id.get(e["target"])
        if st and tt:
            tbl[st][tt] += e["score"]
    for t in types:
        total = sum(tbl[t].values())
        if total > 0:
            tbl[t] = {k: round(v / total, 4) for k, v in tbl[t].items()}
    return {"types": types, "table": tbl}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build causal_chain_snapshot.json")
    ap.add_argument("--bart-scores", type=Path, default=None,
                    help="bart_link_scores.csv from bart_causal/bart_causal_pipeline.py; default is the TF-IDF placeholder")
    args = ap.parse_args(argv)
    df = load_events()
    print(f"{len(df)} events with usable cause/effect text, {df['Date'].min().date()} to {df['Date'].max().date()}")
    if args.bart_scores:
        edges = build_edges_from_bart(df, args.bart_scores)
        print(f"{len(edges)} edges kept (BART probability >= {MIN_BART_SCORE}, lag <= {MAX_LAG_DAYS}d, top {TOP_K_SUCCESSORS}/event)")
    else:
        effect_vecs, cause_vecs = score_links(df)
        edges = build_edges(df, effect_vecs, cause_vecs)
        print(f"{len(edges)} edges kept (score >= {MIN_SCORE}, lag <= {MAX_LAG_DAYS}d, top {TOP_K_SUCCESSORS}/event)")
    transitions = build_transition_table(df, edges)

    events = [{
        "id": r.Doc_ID, "date": r.Date.strftime("%Y-%m-%d"),
        "text": _headline_or_snippet(r.Headline, r.Content)[:200],
        "event_type": r.assigned_topic_bert,
        "cause": str(r.gold_cause)[:300], "effect": str(r.gold_effect)[:300],
    } for r in df.itertuples()]

    snapshot = {
        "events": events,
        "edges": edges,
        "transition_table": transitions,
        "meta": {
            "n_events": len(events), "n_edges": len(edges),
            "link_method": "bart_cross_encoder_effect_to_cause" if args.bart_scores else "tfidf_cosine_effect_to_cause",
            "link_method_note": (
                "BART cross-encoder (bart_causal/bart_causal_pipeline.py): probability that one event's effect "
                "plausibly led to a later event's cause, trained on judge labels; see that pipeline's eval.json."
                if args.bart_scores else
                "Lexical-overlap heuristic (TF-IDF cosine between one event's gold_effect "
                "text and a later event's gold_cause text), not semantic causal "
                "understanding. A placeholder for the BART causal model meant to replace "
                "it - see export_causal_chains.py's module docstring."),
            "max_lag_days": MAX_LAG_DAYS, "top_k_successors": TOP_K_SUCCESSORS,
            "min_score": MIN_BART_SCORE if args.bart_scores else MIN_SCORE,
        },
    }
    OUT.write_text(json.dumps(snapshot, default=str, allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
