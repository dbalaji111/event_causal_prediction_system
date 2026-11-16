"""Ingests new articles into the gold event repository (scripts/causal_gold_articles_with_topics_bert.csv),
embeds them, assigns a topic, and appends them in the exact existing schema - so every downstream script
(build_real_gold_corpus.py, export_static_snapshot.py, export_embeddings_viz.py, and main_app.py /
streamlit_gold_embeddings_visualisation.py) keeps working unchanged on the enlarged repository.

    python ingest_news.py --source local --input new_articles.json      # test the pipeline, no network
    python ingest_news.py --source refinitiv --query "gold OR bullion" --start 2026-01-01 --end 2026-01-07

Two deliberate simplifications vs. the original notebooks (scripts/gold_relvant_data_extraction.ipynb,
scripts/gold_bert_topics.ipynb), both just to avoid new dependencies rather than changing the method:

1. No Ollama step. The notebooks ran a local Ollama LLM to produce `gold_general_summary` (a 2-3 sentence
   gold-specific summary) before embedding it. Ollama isn't available in this sandbox, so new rows embed
   `headline + opening content` directly instead, and `gold_general_summary` is set to that same text
   rather than an LLM summary. If you run this where Ollama *is* available, swap `_fallback_summary()`
   for a real call to it and nothing else here needs to change.

2. One embedding model instead of two. The notebooks used `all-MiniLM-L6-v2` (384-d) for the stored
   `gold_general_embedding` column, but a *second*, separate model (`bert-base-nli-mean-tokens`) just for
   scoring the 7 topic prototypes and the relevance prototype. This script reuses the same MiniLM
   embedding for both, so new rows stay in the exact same 384-d space the whole retrieval/KDE/scoring
   pipeline already keys off of, at the cost of the topic/relevance scores not being bit-identical to the
   original notebook's numbers (they're the same method, same prototypes, just scored in a different but
   equally valid embedding space).

Refinitiv access: this targets the Refinitiv Data Platform (RDP) News API (api.refinitiv.com), a cloud
REST API reachable from a server - NOT the Eikon/Workspace Data API, which requires the Eikon desktop
app running locally as a proxy and cannot be reached from a cloud session at all. If what you actually
have is Eikon/Workspace access, this script needs a different integration path (running next to Eikon,
not in this sandbox) - say so before wiring in real credentials.

Needs, once you're ready to pull real news (none of this is required for --source local):
  - REFINITIV_CLIENT_ID / REFINITIV_CLIENT_SECRET (or REFINITIV_APP_KEY, if your tenant uses API-key auth
    instead of OAuth2 client-credentials - see _refinitiv_token())
  - network access to api.refinitiv.com from wherever this runs
  - network access to huggingface.co (to download all-MiniLM-L6-v2 the first time; cached after that)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPO_CSV = REPO_ROOT / "scripts" / "causal_gold_articles_with_topics_bert.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Verbatim from scripts/gold_bert_topics.ipynb - the 7 topics every dashboard in this repo colors by.
TOPIC_PROTOTYPES = {
    "Safe-Haven Demand": "Gold demand rises due to geopolitical tensions and war uncertainty",
    "Inflation Hedge": "Gold surges as inflation rises and central banks cut interest rates",
    "Central Bank Buying": "Central banks increase gold reserves amid de-dollarization efforts",
    "ETF & Retail Sentiment": "Retail investors shift into gold ETFs during market turbulence",
    "Trade War Tariffs": "Gold gains as Trump announces new tariffs and trade war escalates",
    "Volatility Hedging": "Gold options and VIX surge due to market fear and instability",
    "Asset Rotation": "Investors rotate from equities to gold in risk-off macro environment",
}
# From scripts/gold_relvant_data_extraction.ipynb's summarize_and_score() - scores how gold-relevant an
# article is at all (separate from *which* gold topic it is).
RELEVANCE_PROTOTYPE = "gold price, bullion, inflation, gold futures, safe haven"

REPO_COLUMNS = ["Doc_ID", "Date", "Headline", "Content", "gold_cause", "gold_effect",
                 "gold_cause_effect_summary", "gold_general_summary", "gold_general_embedding",
                 "gold_relevance_score", "causal_only", "assigned_topic", "topic_similarity",
                 "assigned_topic_bert"]


def _safe_doc_id(raw_id: str, fallback_date: str, idx: int) -> str:
    """export_embeddings_viz.py filters Doc_ID against ^[A-Za-z0-9]{10,}$ (same rule it uses to drop 2
    malformed rows in the real corpus) - an ID with any other character would silently vanish from the
    3D embedding map export. Refinitiv story IDs in particular commonly contain ':'/'-' (e.g. RDP's
    urn:newsml:... form), so always route IDs through this rather than using them raw."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_id or ""))
    if len(cleaned) >= 10:
        return cleaned
    stamp = re.sub(r"[^A-Za-z0-9]", "", str(fallback_date or ""))
    return f"SRC{stamp}{idx:04d}{cleaned}"


def _repo_date_format(date_value) -> str:
    """Every existing row's Date is the string form 'D Month YYYY' (e.g. '2 January 2019' - Factiva's
    export format). pd.to_datetime(df["Date"], errors="coerce"), used with no explicit `format` by
    build_real_gold_corpus.py, export_embeddings_viz.py and the Streamlit scripts alike, infers one
    format from the column and coerces whatever doesn't match to NaT - verified: appending ISO dates
    ('2026-01-05') alongside the existing format silently turned those new rows' Date to NaT, which
    then drops them from every one of those scripts' dropna(subset=["Date"]). So new rows must match
    this exact format, not just be "a valid date"."""
    ts = pd.Timestamp(date_value)
    return f"{ts.day} {ts.strftime('%B')} {ts.year}"


# ─────────────────────────── sources ───────────────────────────

def load_local_source(path: Path) -> pd.DataFrame:
    """Reads a JSON array or CSV of {Doc_ID?, Date, Headline, Content} for testing the embed+append
    pipeline with no network and no credentials. Doc_ID is generated from Date+Headline if missing."""
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = pd.read_csv(path).to_dict("records")
    for i, r in enumerate(rows):
        r["Doc_ID"] = _safe_doc_id(r.get("Doc_ID", ""), r.get("Date", ""), i) if r.get("Doc_ID") else \
            f"LOCAL{pd.Timestamp(r['Date']).strftime('%Y%m%d')}{i:04d}"
        r["Date"] = _repo_date_format(r["Date"])
        r.setdefault("Content", r.get("Headline", ""))
    return pd.DataFrame(rows)


def _refinitiv_token(client_id: str, client_secret: str) -> str:
    """OAuth2 client-credentials grant against RDP's auth service. Untested in this sandbox (no
    credentials, no network to api.refinitiv.com here yet) - the request shape matches RDP's documented
    /auth/oauth2/v1/token endpoint, but confirm the scope/grant_type your tenant expects before relying
    on this, since RDP auth details do vary by subscription (some tenants use a signon_control /
    refresh-token flow instead of plain client_credentials)."""
    import requests
    resp = requests.post(
        "https://api.refinitiv.com/auth/oauth2/v1/token",
        data={"grant_type": "client_credentials", "scope": "trapi"},
        auth=(client_id, client_secret),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_refinitiv_news(query: str, start: str, end: str, client_id: str, client_secret: str,
                          limit: int = 100) -> pd.DataFrame:
    """Pulls headlines + full story text from the RDP News API over [start, end]. Untested end-to-end
    (needs REFINITIV_CLIENT_ID/SECRET and network access to api.refinitiv.com, neither available in this
    sandbox right now) - the endpoint paths/params match RDP's documented News Headlines + News Story
    resources as of this writing, but RDP's exact response shape can differ slightly by tenant/version,
    so treat this as a working skeleton to validate against your own account rather than a guarantee."""
    import requests
    token = _refinitiv_token(client_id, client_secret)
    headers = {"Authorization": f"Bearer {token}"}

    hl_resp = requests.get(
        "https://api.refinitiv.com/data/news/v1/headlines",
        headers=headers,
        params={"query": query, "dateFrom": start, "dateTo": end, "limit": limit},
        timeout=30,
    )
    hl_resp.raise_for_status()
    headlines = hl_resp.json().get("data", [])

    rows = []
    for i, h in enumerate(headlines):
        story_id = h.get("storyId") or h.get("id")
        content = h.get("text") or h.get("title", "")
        if story_id:
            try:
                st_resp = requests.get(f"https://api.refinitiv.com/data/news/v1/stories/{story_id}",
                                        headers=headers, timeout=30)
                st_resp.raise_for_status()
                content = st_resp.json().get("newsItem", {}).get("contentSet", {}) \
                    .get("inlineData", [{}])[0].get("$", content)
            except Exception as e:
                print(f"  (story fetch failed for {story_id}: {e} - keeping headline text only)", file=sys.stderr)
        date = h.get("versionCreated") or h.get("firstCreated") or start
        rows.append({
            "Doc_ID": _safe_doc_id(story_id or h.get("title", ""), date, i),
            "Date": _repo_date_format(date),
            "Headline": h.get("title", ""),
            "Content": content,
        })
    return pd.DataFrame(rows)


# ─────────────────────────── embed + classify ───────────────────────────

class _FakeEmbedder:
    """Deterministic hash-based stand-in for SentenceTransformer, used only with --fake-embeddings to
    test the fetch/classify/append mechanics without network access or the real model. The vectors carry
    no real semantic meaning - never use this path for data that feeds the actual dashboards."""
    DIM = 384

    def encode(self, texts, convert_to_numpy=True):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.DIM)
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return np.asarray(out)


def _fallback_summary(headline: str, content: str) -> str:
    """Stand-in for the notebooks' Ollama-generated gold_general_summary (see module docstring §1)."""
    text = (content or headline or "").strip()
    return text[:280] if len(text) > 280 else text or "No gold-related content found."


def _normalize_similarity(cos_sim: float) -> float:
    """Verbatim from gold_relvant_data_extraction.ipynb: maps cosine [-1,1] -> relevance [0.1,1.0]."""
    norm = (cos_sim + 1) / 2
    return round(0.1 + 0.9 * norm, 4)


def embed_and_classify(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Adds every repository column that depends on embeddings: gold_general_embedding,
    gold_general_summary, gold_relevance_score, assigned_topic(_bert), topic_similarity, gold_cause/
    effect placeholders, causal_only. `model` is injectable for tests; defaults to a real
    SentenceTransformer(all-MiniLM-L6-v2), which needs network access to huggingface.co the first time."""
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)

    summaries = [_fallback_summary(r.Headline, r.Content) for r in df.itertuples()]
    doc_embs = np.asarray(model.encode(summaries, convert_to_numpy=True))

    topic_names = list(TOPIC_PROTOTYPES.keys())
    proto_embs = np.asarray(model.encode(list(TOPIC_PROTOTYPES.values()), convert_to_numpy=True))
    rel_emb = np.asarray(model.encode([RELEVANCE_PROTOTYPE], convert_to_numpy=True))[0]

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    topics, topic_sims, rel_scores = [], [], []
    for emb in doc_embs:
        sims = [cos(emb, p) for p in proto_embs]
        bi = int(np.argmax(sims))
        topics.append(topic_names[bi]); topic_sims.append(round(float(sims[bi]), 4))
        rel_scores.append(_normalize_similarity(cos(emb, rel_emb)))

    out = df.copy()
    out["gold_general_summary"] = summaries
    out["gold_general_embedding"] = [str([round(float(x), 6) for x in e]) for e in doc_embs]
    out["gold_relevance_score"] = rel_scores
    out["assigned_topic"] = topics
    out["assigned_topic_bert"] = topics
    out["topic_similarity"] = topic_sims
    out["gold_cause"] = "No gold cause identified. (no Ollama extraction step - see module docstring)"
    out["gold_effect"] = "No gold effect identified. (no Ollama extraction step - see module docstring)"
    out["gold_cause_effect_summary"] = "No gold causality found."
    out["causal_only"] = False
    return out[REPO_COLUMNS]


# ─────────────────────────── repository ───────────────────────────

def append_to_repository(new_rows: pd.DataFrame, repo_csv: Path, dry_run: bool = False) -> dict:
    """Appends new_rows to repo_csv in-place (unless dry_run), de-duplicating on Doc_ID - new rows never
    overwrite an existing Doc_ID, matching build_real_gold_corpus.py's assumption that event_id (= Doc_ID)
    is a stable key other exported files already reference."""
    existing = pd.read_csv(repo_csv) if repo_csv.exists() else pd.DataFrame(columns=REPO_COLUMNS)
    before = len(existing)
    existing_ids = set(existing["Doc_ID"].astype(str)) if before else set()
    fresh = new_rows[~new_rows["Doc_ID"].astype(str).isin(existing_ids)]
    combined = pd.concat([existing, fresh], ignore_index=True)
    if not dry_run:
        combined.to_csv(repo_csv, index=False)
    return {"before": before, "submitted": len(new_rows), "duplicates_skipped": len(new_rows) - len(fresh),
            "appended": len(fresh), "after": before + len(fresh)}


# ─────────────────────────── CLI ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["local", "refinitiv"], required=True)
    ap.add_argument("--input", type=Path, help="[local] JSON or CSV of articles to ingest")
    ap.add_argument("--query", default="gold OR bullion OR XAU", help="[refinitiv] news query")
    ap.add_argument("--start", help="[refinitiv] YYYY-MM-DD")
    ap.add_argument("--end", help="[refinitiv] YYYY-MM-DD")
    ap.add_argument("--client-id", default=None, help="[refinitiv] or set REFINITIV_CLIENT_ID")
    ap.add_argument("--client-secret", default=None, help="[refinitiv] or set REFINITIV_CLIENT_SECRET")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO_CSV)
    ap.add_argument("--dry-run", action="store_true", help="embed/classify but don't write the repository")
    ap.add_argument("--fake-embeddings", action="store_true",
                     help="skip loading the real model (no huggingface.co access needed) - for testing "
                          "the fetch/classify/append mechanics only; vectors are meaningless, never for real data")
    args = ap.parse_args()

    if args.source == "local":
        if not args.input:
            ap.error("--source local requires --input")
        raw = load_local_source(args.input)
    else:
        import os
        client_id = args.client_id or os.environ.get("REFINITIV_CLIENT_ID")
        client_secret = args.client_secret or os.environ.get("REFINITIV_CLIENT_SECRET")
        if not (client_id and client_secret and args.start and args.end):
            ap.error("--source refinitiv requires --start/--end and REFINITIV_CLIENT_ID/SECRET "
                     "(env or --client-id/--client-secret)")
        raw = fetch_refinitiv_news(args.query, args.start, args.end, client_id, client_secret)

    if raw.empty:
        print("No articles fetched."); return
    print(f"Fetched {len(raw)} article(s) from {args.source}. Embedding + classifying...")
    model = _FakeEmbedder() if args.fake_embeddings else None
    classified = embed_and_classify(raw, model=model)
    stats = append_to_repository(classified, args.repo, dry_run=args.dry_run)
    print(f"{'[dry-run] would append' if args.dry_run else 'Appended'} {stats['appended']} new row(s) "
          f"({stats['duplicates_skipped']} duplicate Doc_ID(s) skipped) to {args.repo} "
          f"({stats['before']} -> {stats['after']} rows).")


if __name__ == "__main__":
    main()
