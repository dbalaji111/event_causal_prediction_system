"""Builds markets/gold.json's corpus from the REAL article data in the repo's data/ and scripts/
folders (real headlines, real 384-d embeddings, real cause/effect extraction, real topic labels -
same corpus the main gold dashboard, main_app.py, already uses).

Prices: pass `prices_csv` (columns Date, XAUUSD, and optionally GC_M1 / GC_M2) and optionally
`vol_csv` (columns Date in MM/DD/YYYY, Price) to build from real data. Without them, price nodes come
from a hand-anchored approximation of gold's historical trajectory plus daily noise - NOT real data,
and everything price-derived (price, price_t_*, Quantile_Category, GVZ) is then illustrative only.
The choice is recorded in <root>/price_source.json so every output can say which one it used.
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


_ID_DATE_RE = re.compile(r"^[Jj]0{6,7}(\d{4})(\d{2})(\d{2})")
MAX_SAME_EMBEDDING = 5


def clean_gold_articles(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Removes rows that are not usable events. Shared by every script that reads the gold CSV, so the
    corpus, the dashboards and the embedding map all leave out the same rows:
    - Factiva search-metadata rows ("Search Summary ...") ingested as if they were articles;
    - rows whose embedding is shared by more than MAX_SAME_EMBEDDING rows, or whose summary says there
      is no gold content: that vector embeds a placeholder summary, not the article, so these rows all
      match one another at similarity 1.0;
    - rows dated more than 30 days after the load date inside their own Doc_ID (J0000000YYYYMMDD...):
      an article cannot be dated after it was archived; these carry the export date instead."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "gold_general_embedding"])
    n0 = len(df)
    search = df["Headline"].astype(str).str.contains("Search Summary", case=False, na=False)
    shared = df["gold_general_embedding"].map(df["gold_general_embedding"].value_counts()) > MAX_SAME_EMBEDDING
    no_gold = df["gold_general_summary"].astype(str).str.contains("No gold-related content found", na=False)
    m = df["Doc_ID"].astype(str).str.extract(_ID_DATE_RE)
    id_date = pd.to_datetime(m[0] + "-" + m[1] + "-" + m[2], errors="coerce")
    misdated = (df["Date"] - id_date).dt.days > 30          # unreadable ids give NaT -> False -> kept
    drop = search | shared | no_gold | misdated
    log(f"[clean] {n0} dated rows: -{int(search.sum())} search-summary, -{int((shared | no_gold).sum())} placeholder "
        f"summary/embedding, -{int(misdated.sum())} dated after their own archive date -> {int((~drop).sum())} kept")
    return df[~drop]


def headline_text(headline: str, content: str, doc_id: str = "") -> str:
    """The article's title. Most Headline cells are 'Doc_ID\\ntitle\\nsection\\ntitle': the title is the
    first line that is not the id. Headline extraction failed for ~16% of rows ('UNKNOWN'); Content still
    has the real text behind a Dow Jones copyright prefix, so its opening is used as a stand-in."""
    lines = [l.strip() for l in str(headline).splitlines() if l.strip()]
    lines = [l for l in lines if l.lower() != str(doc_id).strip().lower()]
    if lines and lines[0].upper() != "UNKNOWN":
        return lines[0]
    body = _BYLINE_RE.sub("", str(content or ""), count=1).strip()
    return (body[:140] + "…") if body else "UNKNOWN"


def _load_real_prices(prices_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(prices_csv)
    if "Date" not in df.columns or "XAUUSD" not in df.columns:
        raise ValueError(f"{prices_csv} needs columns 'Date' and 'XAUUSD' (optional: GC_M1, GC_M2); found {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "XAUUSD"]).drop_duplicates("Date", keep="last").sort_values("Date")
    if len(df) < 250:
        raise ValueError(f"{prices_csv} has only {len(df)} usable rows; need at least a year of daily prices")
    return df[[c for c in ("Date", "XAUUSD", "GC_M1", "GC_M2") if c in df.columns]].reset_index(drop=True)


def _quantile_label(pct: float) -> str:
    return next(label for hi, label in QUANTILE_LABELS if pct < hi)


def build_real_gold_market(root: Path, paths: dict, repo_root: Path, train_cutoff="2023-01-01",
                            seed=13, prices_csv: Path | None = None, vol_csv: Path | None = None) -> Path:
    """Writes the gold.json-shaped files under `root` from the repo's real article data.
    Idempotent: skips rebuilding if already done (delete `root/.real_ok` to force a rebuild)."""
    root = Path(root)
    marker = root / ".real_ok"
    if marker.exists():
        return root
    rng = np.random.default_rng(seed)
    price_file = root / paths["prices"]
    price_file.parent.mkdir(parents=True, exist_ok=True)
    vol_file = root / paths["vol_index"]
    vol_file.parent.mkdir(parents=True, exist_ok=True)

    if prices_csv is not None:
        prices_df = _load_real_prices(Path(prices_csv))
        prices_df.to_csv(price_file, index=False)
        if vol_csv is not None:
            vol_file.write_bytes(Path(vol_csv).read_bytes())
        elif vol_file.exists():
            vol_file.unlink()          # never pair real prices with the synthetic GVZ
        source = {"source": "real", "prices_file": Path(prices_csv).name,
                  "vol_file": Path(vol_csv).name if vol_csv is not None else None}
    else:
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
        prices_df.to_csv(price_file, index=False)

        gvz = 13 + 7 * np.abs(rng.standard_t(4, len(dates))) + 4 * np.abs(np.diff(np.log(xau), prepend=np.log(xau[0]))) * 100
        vol_df = pd.DataFrame({"Date": dates.strftime("%m/%d/%Y"), "Price": np.clip(gvz, 8, 90)})
        vol_df.to_csv(vol_file, index=False)
        source = {"source": "illustrative", "prices_file": None, "vol_file": None}

    px = pd.Series(prices_df["XAUUSD"].to_numpy(dtype=float), index=pd.DatetimeIndex(prices_df["Date"]))
    source["first_date"], source["last_date"] = str(px.index.min().date()), str(px.index.max().date())
    (root / "price_source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")

    def price_near(d, offset_days):
        """Last price on or before d + offset; None outside the price history (never a stale stand-in)."""
        t = d + pd.Timedelta(days=offset_days)
        if t < px.index[0] or t > px.index[-1]:
            return None
        return float(px.iloc[px.index.searchsorted(t, side="right") - 1])

    # ---- real articles: headline, embedding, cause/effect, topic - all from the actual corpus ----
    src = repo_root / "scripts" / "causal_gold_articles_with_topics_bert.csv"
    df = clean_gold_articles(pd.read_csv(src))
    df = df[(df["Date"] >= px.index.min()) & (df["Date"] <= px.index.max())]

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
            "text": headline_text(row["Headline"], row.get("Content", ""), row["Doc_ID"])[:500],
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
