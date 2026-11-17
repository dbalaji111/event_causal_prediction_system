"""Event-corpus reader for the project's native format: JSON array or JSONL of records with bert_embeddings + price nodes."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from .config import MarketConfig


def _strip_ansi(s): return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)


def parse_embedding(raw) -> Optional[np.ndarray]:
    if raw is None: return None
    if isinstance(raw, (list, np.ndarray)):
        a = np.array(raw, dtype=np.float32).ravel()
        return a if len(a) >= 100 else None
    if isinstance(raw, str):
        nums = []
        for t in _strip_ansi(raw).replace("[", "").replace("]", "").replace("\n", " ").split():
            try: nums.append(float(t.rstrip(",")))
            except ValueError: pass
        return np.array(nums, dtype=np.float32) if len(nums) >= 100 else None
    return None


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _first(row, keys, default=""):
    for k in ([keys] if isinstance(keys, str) else keys):
        if row.get(k) not in (None, ""):
            return row[k]
    return default


def to_record(row: dict, idx: int, cfg: MarketConfig) -> Optional[dict]:
    cf = cfg["corpus_fields"]; dim = cf["embedding_dim"]
    emb = parse_embedding(row.get(cf["embedding"]))
    if emb is None: return None
    emb = np.pad(emb, (0, dim - len(emb))) if len(emb) < dim else emb[:dim]
    try: date = pd.to_datetime(str(row.get(cf["date"]) or ""), errors="raise")
    except Exception: return None
    p0 = _f(row.get(cf["price_nodes"]["0"]))
    nodes = {}
    for off, col in cf["price_nodes"].items():
        v = row.get(col)
        nodes[int(off)] = _f(v, np.nan) if v not in (None, "") else np.nan
    s0 = _f(row.get(cf["spread_nodes"]["0"]), np.nan) if "spread_nodes" in cf else np.nan
    return {"event_id": str(row.get(cf["event_id"]) or f"EVT_{date.year}_{idx:05d}"), "date": date,
            "quantile_cat": str(row.get(cf["quantile"], "")).strip(),
            "event_type": str(_first(row, cf["event_type"])).strip(),
            "polarity": str(row.get(cf["polarity"], "Neutral")).strip(),
            "text": str(_first(row, cf["text"])), "embedding": emb.astype(np.float32),
            "price_nodes": nodes, "price0": p0, "spread0": s0}


def load_records(path: Path, cfg: MarketConfig) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    txt = path.read_text(encoding="utf-8", errors="replace")
    if txt.lstrip().startswith("["):
        rows = json.loads(txt)
    else:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
    recs = [r for r in (to_record(row, i, cfg) for i, row in enumerate(rows)) if r]
    return recs


def price_path(rec: dict, nodes) -> Dict[int, Optional[float]]:
    """Percent return of each node price versus the event-date price (None where the node price is missing)."""
    p0 = rec["price_nodes"].get(0)
    p0 = p0 if p0 and np.isfinite(p0) else rec["price0"]
    out = {}
    for n in nodes:
        pt = rec["price_nodes"].get(n, np.nan)
        out[n] = None if (not p0 or not np.isfinite(pt)) else float((pt - p0) / abs(p0) * 100)
    return out
