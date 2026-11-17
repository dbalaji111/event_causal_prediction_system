"""Market configuration: every number or name that differs between markets lives in markets/<market>.json."""
from __future__ import annotations
import json
from pathlib import Path

REQUIRED = ["market", "display_name", "ticker_label", "paths", "prices", "corpus_fields", "retrieval", "kde",
            "dashboard", "backtest", "event_keywords", "causal_keywords"]
DTW_MODES = ("legacy_full_path", "pre_event_only")
QUANTILE_MODES = ("legacy_fixed_mid", "price_percentile")


class MarketConfig:
    def __init__(self, d: dict, root):
        self.d, self.root = d, Path(root)

    @classmethod
    def load(cls, path, root=None) -> "MarketConfig":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), root or path.parent.parent)

    def __getitem__(self, k): return self.d[k]
    def get(self, k, default=None): return self.d.get(k, default)
    @property
    def market(self): return self.d["market"]

    def path(self, key):
        v = self.d["paths"].get(key)
        return None if v is None else self.root / v

    def uncalibrated(self) -> list:
        """Names of parameters flagged '_calibrate' (placeholders until scripts/calibrate is run and reviewed)."""
        out = []
        def walk(node, prefix):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "_calibrate":
                        out.extend(f"{prefix}.{n}" for n in v)
                    else:
                        walk(v, f"{prefix}.{k}" if prefix else k)
        walk(self.d, "")
        return out

    def validate(self) -> list:
        d, p = self.d, []
        p += [f"missing section '{k}'" for k in REQUIRED if k not in d]
        if p:
            return p
        k = d["kde"]
        if k["dtw_mode"] not in DTW_MODES: p.append(f"kde.dtw_mode must be one of {DTW_MODES}")
        if k["quantile_mode"] not in QUANTILE_MODES: p.append(f"kde.quantile_mode must be one of {QUANTILE_MODES}")
        if min(k["w_sim"], k["w_quantile"], k["w_dtw"]) < 0: p.append("kde weights must be >= 0")
        if sorted(k["tau_fallbacks"], reverse=True) != k["tau_fallbacks"]: p.append("kde.tau_fallbacks must be descending")
        nodes = k["price_nodes"]
        if 0 not in nodes: p.append("kde.price_nodes must include 0")
        if k["dtw_mode"] == "pre_event_only" and not [n for n in nodes if n < 0]:
            p.append("pre_event_only needs at least one negative node")
        for h in k["horizons"]:
            if h not in nodes and h <= 0: p.append(f"horizon {h} invalid")
        miss = [n for n in nodes if str(n) not in d["corpus_fields"]["price_nodes"]]
        if miss: p.append(f"corpus_fields.price_nodes lacks nodes {miss}")
        w = d["dashboard"]["composite_weights"]
        if abs(sum(w.values()) - 1.0) > 1e-6: p.append("dashboard.composite_weights must sum to 1")
        if "PX" not in d["prices"]["columns"].values() and "PX_ALT" not in d["prices"]["columns"].values():
            p.append("prices.columns must map one raw column to 'PX'")
        for need in ("train_corpus", "test_corpus", "prices"):
            if not d["paths"].get(need): p.append(f"paths.{need} is required")
        return p
