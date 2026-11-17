"""Pre-flight checks on a market folder: run before retrieve/kde so problems show up as one list, not as a crash mid-run."""
from __future__ import annotations
import numpy as np, pandas as pd
from .config import MarketConfig
from .corpus import load_records, price_path
from .prices import PriceData


def validate_market(cfg: MarketConfig, log=print) -> dict:
    errors, warns, info = list(cfg.validate()), [], {}
    if errors:
        return {"errors": errors, "warnings": warns, "info": info}
    for key in ("train_corpus", "test_corpus", "prices"):
        if not cfg.path(key).exists(): errors.append(f"missing file paths.{key}: {cfg.path(key)}")
    vi = cfg.path("vol_index") if cfg["paths"].get("vol_index") else None
    if vi is None: warns.append("no vol_index configured: structural score loses its volatility term")
    elif not vi.exists(): warns.append(f"vol_index file not found: {vi}")
    if errors: return {"errors": errors, "warnings": warns, "info": info}
    unc = cfg.uncalibrated()
    if unc: warns.append(f"{len(unc)} parameter(s) still placeholders (run calibrate): {', '.join(unc)}")
    try:
        px = PriceData(cfg)
        info["price_range"] = f"{px.df.index[0].date()} -> {px.df.index[-1].date()} ({len(px.df)} rows)"
        info["price_cols"] = [c for c in ("PX", "SPREAD", "VOL", "FRONT", "SECOND") if c in px.df.columns]
        if "SPREAD" not in px.df.columns: warns.append("no spread (FRONT-SECOND): structural curve term unavailable")
        if "VOL" not in px.df.columns: warns.append("no volatility index joined")
    except Exception as e:
        errors.append(f"price file: {e}"); return {"errors": errors, "warnings": warns, "info": info}
    tr, te = load_records(cfg.path("train_corpus"), cfg), load_records(cfg.path("test_corpus"), cfg)
    info.update(train=len(tr), test=len(te))
    if len(tr) < 100: warns.append(f"only {len(tr)} training events - analogue pools will be thin (oil has 11,983)")
    if not te: errors.append("no valid test records (need bert_embeddings and a parseable date)")
    for name, recs in (("train", tr), ("test", te)):
        ids = [r["event_id"] for r in recs]
        if len(set(ids)) != len(ids): errors.append(f"{name}: duplicate event_id values")
    if tr and te:
        if max(r["date"] for r in tr) >= min(r["date"] for r in te): warns.append("train and test periods overlap (test should start after train ends)")
        info["train_range"] = f"{min(r['date'] for r in tr).date()} -> {max(r['date'] for r in tr).date()}"
        info["test_range"] = f"{min(r['date'] for r in te).date()} -> {max(r['date'] for r in te).date()}"
        qm = set(cfg["kde"]["quantile_midpoints"])
        unk = {r["quantile_cat"] for r in tr} - qm - {""}
        if unk: warns.append(f"quantile labels not in quantile_midpoints (treated as 50th pct): {sorted(unk)}")
        if not any(r["quantile_cat"] for r in tr): warns.append("no Quantile_Category in training records")
        need = [n for n in cfg["kde"]["price_nodes"] if n < 0] + [h for h in cfg["kde"]["horizons"]]
        miss = {}
        for r in tr:
            for n, v in price_path(r, [n for n in cfg["kde"]["price_nodes"] if n != 0]).items():
                if v is None: miss[n] = miss.get(n, 0) + 1
        bad = {n: f"{c}/{len(tr)}" for n, c in miss.items() if c > 0.05 * len(tr)}
        if bad: warns.append(f"training records missing node prices (>5%): {bad}")
        covered = [r for r in te if px.px_at(r["date"]) is not None]
        if len(covered) < len(te): warns.append(f"{len(te) - len(covered)}/{len(te)} test events have no price within {px.near} days")
        dim = tr[0]["embedding"].shape[0]
        info["embedding_dim"] = dim
        if dim != cfg["corpus_fields"]["embedding_dim"]: errors.append("embedding dimension differs from config")
    for e in errors: log("ERROR  " + e)
    for w in warns: log("WARN   " + w)
    log("INFO   " + "; ".join(f"{k}={v}" for k, v in info.items()))
    return {"errors": errors, "warnings": warns, "info": info}
