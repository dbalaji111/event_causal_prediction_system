"""Event backtest on a KDE results CSV (port of redflag_api_v5.run_backtest_thread, same rules, no I/O).
Return-based and per-event: no roll correction, no benchmark, Sharpe assumes independent events (see schema known issues)."""
from __future__ import annotations
import uuid
import numpy as np, pandas as pd


def run_backtest(kdf: pd.DataFrame, cfg: dict) -> dict:
    h = cfg.get("horizon", 15)
    s, e = pd.Timestamp(cfg.get("start_date", "2024-01-01")), pd.Timestamp(cfg.get("end_date", "2026-04-28"))
    band, veto, cost = cfg.get("no_trade_band", 0.5), cfg.get("tail_veto", 0.25), cfg.get("cost_bps", 5) / 10000 * 100
    kdf = kdf.copy(); kdf["base_date"] = pd.to_datetime(kdf["base_date"], errors="coerce")
    df = kdf[(kdf["base_date"] >= s) & (kdf["base_date"] <= e)].dropna(subset=["realised_r", "kde_median"]).reset_index(drop=True)
    pnl, sig = [], []
    for _, r in df.iterrows():
        med, pd10 = float(r["kde_median"]), float(r.get("kde_p_down10", 0.1))
        sg = 0 if pd10 > veto else 1 if med > band else -1 if med < -band else 0
        sig.append(sg); pnl.append(sg * float(r["realised_r"]) - abs(sg) * cost if sg else 0.0)
    pnl, sig = np.array(pnl), np.array(sig); tr = sig != 0; n = int(tr.sum())
    mean = float(pnl[tr].mean()) if n else 0.0; sd = float(pnl[tr].std()) if n > 1 else 0.0
    cum = np.cumsum(pnl)
    cov = df["covers_90"].dropna() if "covers_90" in df else pd.Series(dtype=float)
    flagged = df[df["red_flag"] == 1] if "red_flag" in df else df.iloc[0:0]
    unfl = df[df["red_flag"] == 0] if "red_flag" in df else df.iloc[0:0]
    return {"run_id": str(uuid.uuid4())[:8], "config": cfg, "n_events": len(df), "metrics": {
        "calibration": {"coverage_90pct": round(float(cov.mean()) * 100, 1) if len(cov) else 0},
        "direction": {"n_trades": n, "n_long": int((sig == 1).sum()), "n_short": int((sig == -1).sum()), "n_flat": int((sig == 0).sum()),
                      "win_rate": round(float((pnl[tr] > 0).mean()) * 100, 1) if n else 0},
        "trading": {"cumulative_return_pct": round(float(pnl.sum()), 2), "mean_trade_return_pct": round(mean, 3),
                    "sharpe_ratio": round(mean / sd * np.sqrt(252 / h), 3) if sd > 0 else 0,
                    "max_drawdown_pct": round(float((cum - np.maximum.accumulate(cum)).min()), 2) if len(cum) else 0},
        "red_flag_utility": {"n_flagged": int(len(flagged)), "flagged_mean_r": round(float(flagged["realised_r"].mean()), 2) if len(flagged) else 0,
                             "unflagged_mean_r": round(float(unfl["realised_r"].mean()), 2) if len(unfl) else 0}},
        "equity_curve": [round(float(v), 3) for v in cum], "dates": [str(d)[:10] for d in df["base_date"]]}
