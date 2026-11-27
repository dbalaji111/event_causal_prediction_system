"""Live analysis of a headline that is not in the corpus: embed it, retrieve historical analogues from the whole
event memory, run the KDE at every configured horizon and score it. The result has the same shape as an event in
static_snapshot.json, so the dashboard renders a live result and a precomputed one the same way."""
from __future__ import annotations

import copy
import hashlib
import time
from collections import Counter
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import kde as kde_mod
from . import scoring
from .config import MarketConfig
from .corpus import load_records
from .prices import PriceData
from .retrieval import Retriever
from .series import repath

Embedder = Callable[[Sequence[str]], np.ndarray]
MAX_ANALOGUES_SHOWN = 40
MAX_HORIZON = 90
DENSITY_POINTS = 240
ANALOGUE_FIELDS = ("event_id", "date", "similarity", "recency_weight", "quantile_cat", "event_type", "r7", "r15", "r30")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


class LiveEngine:
    def __init__(self, cfg: MarketConfig, embed: Embedder, pca: Optional[dict] = None):
        self.horizons = list(cfg["kde"]["horizons"])
        # An analogue only counts once its longest outcome window closed before the event being analysed;
        # PriceData.forward_return may look up to forward_max_days past the nominal end of that window.
        self.maturity_days = max(self.horizons) + int(cfg["prices"]["forward_max_days"]) + 1
        d = copy.deepcopy(cfg.d)
        d["retrieval"]["embargo_days"] = max(int(d["retrieval"].get("embargo_days", 0)), self.maturity_days)
        self.cfg = MarketConfig(d, cfg.root)

        seen, memory = set(), []
        for r in load_records(cfg.path("train_corpus"), cfg) + load_records(cfg.path("test_corpus"), cfg):
            if r["event_id"] not in seen:
                seen.add(r["event_id"])
                memory.append(r)
        if not memory:
            raise ValueError("the event memory is empty")
        self.retriever = Retriever(self.cfg, memory, [])
        self.text = {r["event_id"]: r["text"] for r in memory}
        self.prices = PriceData(self.cfg)
        self.embed = embed
        self.pca = None if pca is None else (np.asarray(pca["mean"], float), np.asarray(pca["components"], float))
        dates = [r["date"] for r in memory]
        self.memory_range = (min(dates), max(dates))
        self.memory_size = len(memory)
        self.dim = self.retriever.M.shape[1]

    def analyze(self, headline: str, when: Optional[pd.Timestamp] = None, prices: Optional[PriceData] = None,
                horizons: Optional[Sequence[int]] = None, density: bool = False, series_name: Optional[str] = None) -> dict:
        """prices: the series to measure outcomes on (default: the corpus series). horizons: days ahead (default: the
        configured ones). density: also return each horizon's KDE curve for plotting."""
        t0 = time.perf_counter()
        text = " ".join(str(headline).split())
        if not text:
            raise ValueError("the headline is empty")
        when = (pd.Timestamp.now("UTC").tz_localize(None) if when is None else pd.Timestamp(when)).normalize()
        hs = sorted({int(h) for h in horizons}) if horizons else list(self.horizons)
        if not hs or hs[0] < 1 or hs[-1] > MAX_HORIZON:
            raise ValueError(f"horizons must be between 1 and {MAX_HORIZON} days")
        custom = prices is not None
        px = prices if custom else self.prices
        maturity = max(max(hs) + int(self.cfg["prices"]["forward_max_days"]) + 1, self.maturity_days)

        emb = np.asarray(self.embed([text]), dtype=np.float32).reshape(1, -1)[0]
        if emb.shape[0] != self.dim:
            raise RuntimeError(f"the embedding model returns {emb.shape[0]} numbers per text, "
                               f"the event memory was built with {self.dim}")
        base = {"event_id": "LIVE-" + hashlib.sha1(f"{when.date()}|{text}".encode()).hexdigest()[:12],
                "date": when, "text": text, "embedding": emb, "event_type": "", "quantile_cat": "",
                "polarity": "Neutral", "price_nodes": {}, "price0": None, "spread0": np.nan}

        k = self.cfg["kde"]
        res = self.retriever.retrieve_adaptive(base, self.cfg["retrieval"]["tau"], k["min_analogues"])
        cutoff = when - pd.Timedelta(days=maturity)
        found = [a for a in res["analogues"] if pd.Timestamp(a["date"]) < cutoff]
        if custom:
            found = repath(found, px)

        cone, rows, curves = {}, {}, {}
        for h in hs:
            row, _, grid = kde_mod.run_event(self.cfg, base["event_id"], when, found, px, h)
            if "error" in row:
                continue
            rows[h] = row
            cone[str(h)] = {
                "p5": row["kde_p5"], "p25": row["kde_p25"], "median": row["kde_median"], "p75": row["kde_p75"],
                "p95": row["kde_p95"], "mean": row["kde_mean"],
                "p_down10": row.get("kde_p_down10"), "p_up10": row.get("kde_p_up10"), "p_down5": row.get("kde_p_down5"),
                "is_bimodal": bool(row.get("kde_is_bimodal", 0)), "tail_weight": row.get("kde_tail_weight", 0),
                "n_analogues": row.get("n_analogues", 0), "tau_effective": row.get("tau_effective") or 0,
                "red_flag": row.get("red_flag", 0), "signal": row.get("signal") or "NEUTRAL",
                "realised_r": row.get("realised_r"), "covers_90": row.get("covers_90"),
            }
            if density and grid is not None:
                x, y = grid
                step = max(1, len(x) // DENSITY_POINTS)
                curves[str(h)] = {"x": [round(float(v), 3) for v in x[::step]], "y": [float(v) for v in y[::step]]}
        ks = rows.get(self.cfg["backtest"]["horizon"])
        score = scoring.score_event(self.cfg, text, res, ks, vol=None if ks else px.vol_at(when))

        topic_weight = Counter()
        for a in found[:10]:
            topic_weight[a["event_type"]] += float(a["similarity"])
        topic = topic_weight.most_common(1)[0][0] if topic_weight else "Unclassified"

        xyz = None
        if self.pca is not None and self.pca[0].shape[0] == emb.shape[0]:
            xyz = [round(float(v), 4) for v in (emb - self.pca[0]) @ self.pca[1].T]

        def outcomes(a):
            ad = pd.Timestamp(a["date"])
            return {str(h): (None if (r := px.forward_return(ad, h)) is None else round(r, 3)) for h in hs}

        out = {
            "event_id": base["event_id"], "date": when, "text": text, "event_type": topic,
            "classify": scoring.classify_headline(self.cfg, text),
            "analogues": [{**{f: a.get(f) for f in ANALOGUE_FIELDS}, "outcomes": outcomes(a),
                           "summary": self.text.get(a["event_id"], "")[:160]} for a in found[:MAX_ANALOGUES_SHOWN]],
            "forecast_cone": cone, "score": score, "map_xyz": xyz,
            "live": {"price_context": px.px_at(when) is not None, "maturity_days": maturity,
                     "tau_used": res.get("tau_used"), "n_candidates": res.get("n_candidates"),
                     "memory_size": self.memory_size, "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                     "series": series_name or "corpus", "series_value": px.px_at(when), "horizons": hs},
        }
        if density:
            out["densities"] = curves
        return _jsonable(out)
