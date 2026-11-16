"""Analogue retrieval: cosine similarity on L2-normalised embeddings, temporal filter, event-type exclusion.
Writes one <event_id>_analogues.json per base event (schema: redflag_kit.schema.RETRIEVAL_FILE)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
import numpy as np, pandas as pd
from .config import MarketConfig
from .corpus import load_records, price_path

PATH_NODES = (-30, -15, -7, 7, 15, 30)


class Retriever:
    def __init__(self, cfg: MarketConfig, train: List[dict], test: List[dict]):
        self.cfg, self.train, self.test = cfg, train, test
        rc = cfg["retrieval"]
        ex = {s.lower() for s in rc["exclude_event_types"]}
        self.keep = np.array([r["event_type"].lower() not in ex for r in train], dtype=bool)
        M = np.stack([r["embedding"] for r in train])
        self.M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
        self.dates = np.array([r["date"].to_datetime64() for r in train])
        self.by_id = {r["event_id"]: r for r in test}

    @classmethod
    def from_config(cls, cfg: MarketConfig) -> "Retriever":
        return cls(cfg, load_records(cfg.path("train_corpus"), cfg), load_records(cfg.path("test_corpus"), cfg))

    def similarities(self, base: dict) -> np.ndarray:
        q = base["embedding"] / (np.linalg.norm(base["embedding"]) + 1e-10)
        return self.M @ q

    def candidates(self, base: dict, tau: float):
        rc = self.cfg["retrieval"]
        sims = self.similarities(base)
        cutoff = (base["date"] - pd.Timedelta(days=rc["embargo_days"])).to_datetime64()
        ok = self.keep & (self.dates < cutoff)               # only the past, never the event's own future
        idx = np.where(ok & (sims >= tau))[0]
        return idx[np.argsort(-sims[idx])], sims

    def retrieve(self, base: dict, tau: Optional[float] = None, top_k: Optional[int] = None) -> dict:
        rc = self.cfg["retrieval"]
        tau = rc["tau"] if tau is None else tau
        top_k = rc["top_k"] if top_k is None else top_k
        idx, sims = self.candidates(base, tau)
        out = []
        for i in idx[:top_k]:
            a = self.train[i]
            pp = price_path(a, PATH_NODES)
            age = max((base["date"] - a["date"]).days / 365.25, 0.0)
            out.append({"event_id": a["event_id"], "date": str(a["date"].date()), "similarity": round(float(sims[i]), 6),
                        "recency_weight": round(float(np.exp(-rc["recency_lambda"] * age)), 6),
                        "quantile_cat": a["quantile_cat"], "event_type": a["event_type"], "polarity": a["polarity"],
                        "price": a["price0"], "price_path": {str(k): v for k, v in pp.items()},
                        "r7": pp[7], "r15": pp[15], "r30": pp[30]})
        base_pp = price_path(base, PATH_NODES)
        have = all(v is not None for v in base_pp.values())
        return {"base_event_id": base["event_id"], "base_date": str(base["date"].date()), "tau": tau, "top_k": top_k,
                "n_candidates": int(len(idx)), "n_analogues": len(out), "analogues": out,
                "realised_path": {str(k): v for k, v in base_pp.items()} if have else None,
                "realised_r30": base_pp[30] if have else None}

    def retrieve_adaptive(self, base: dict, tau: float, k_min: int, top_k: int = 50, ladder=(0.82, 0.80, 0.77, 0.75, 0.70, 0.65, 0.0)):
        """As the API's /retrieve: relax tau down a ladder until at least k_min candidates exist."""
        used = 0.0
        for t in [tau, *ladder]:
            idx, _ = self.candidates(base, t)
            if len(idx) >= k_min:
                used = t; break
        res = self.retrieve(base, used, top_k)
        res["tau_used"] = used
        return res

    def run_batch(self, out_dir: Path, tau=None, top_k=None, log=print) -> int:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for rec in self.test:
            res = self.retrieve(rec, tau, top_k)
            (out_dir / f"{rec['event_id']}_analogues.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
            n += 1
        log(f"[retrieve] wrote {n} files to {out_dir}")
        return n
