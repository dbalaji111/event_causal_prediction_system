"""Transfer oil's hand-picked constants to another market by PERCENTILE, not by copying numbers.
Idea: oil's tail threshold (10%) sits at some percentile of oil's own 15-day |return| distribution; the equivalent gold threshold is the
same percentile of gold's distribution. Same for the vol-index range and the curve-spread bands. Return/bandwidth thresholds are scaled by
the ratio of 15-day return standard deviations (a heuristic - flagged as such). Needs the oil price data as the reference."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, pandas as pd
from .config import MarketConfig
from .prices import PriceData

H = 15


def _ret(px: PriceData) -> pd.Series:
    s = px.df["PX"].dropna()
    fut = s.reindex(s.index + pd.Timedelta(days=H), method="nearest", tolerance=pd.Timedelta(days=5))
    r = (fut.values / s.values - 1) * 100
    r = pd.Series(r, index=s.index).dropna()
    return r[r.index <= s.index[-1] - pd.Timedelta(days=H)]


def _rank(x: pd.Series, v: float) -> float:
    return float((x <= v).mean())


def _series(px: PriceData):
    d = px.df
    return {"ret15": _ret(px).abs(), "ret15_signed": _ret(px), "vol": d["VOL"].dropna() if "VOL" in d else None,
            "spread": d["SPREAD"].dropna().abs() if "SPREAD" in d else None}


def describe(px: PriceData) -> dict:
    out = {}
    for k, s in _series(px).items():
        if s is not None and len(s) > 20 and k != "ret15_signed":
            out[k] = {f"p{p}": round(float(np.percentile(s, p)), 3) for p in (10, 25, 50, 75, 90, 95)}
    return out


def propose(target: MarketConfig, reference: MarketConfig, tpx: PriceData = None, rpx: PriceData = None) -> dict:
    tpx = tpx or PriceData(target); rpx = rpx or PriceData(reference)
    t, r = _series(tpx), _series(rpx)
    rk, rd = reference["kde"], reference["dashboard"]
    out = {}

    def transfer(name, ref_val, ref_series, tgt_series, rnd):
        if ref_series is None or tgt_series is None or len(ref_series) < 50 or len(tgt_series) < 50:
            out[name] = {"status": "not_computed", "reason": "series missing or too short"}; return
        q = _rank(ref_series, ref_val)
        out[name] = {"value": round(float(np.quantile(tgt_series, q)) / rnd) * rnd, "method": "percentile_transfer", "reference_value": ref_val,
                     "reference_percentile": round(q, 3)}

    transfer("kde.tail_threshold_pct", rk["tail_threshold_pct"], r["ret15"], t["ret15"], 0.5)
    transfer("dashboard.structural.curve_flat_band", rd["structural"]["curve_flat_band"], r["spread"], t["spread"], 0.01)
    transfer("dashboard.structural.curve_mag_scale", rd["structural"]["curve_mag_scale"], r["spread"], t["spread"], 0.05)
    if r["vol"] is not None and t["vol"] is not None and len(t["vol"]) > 50:
        s = rd["structural"]
        q_lo, q_hi = _rank(r["vol"], s["vol_lo"]), _rank(r["vol"], s["vol_lo"] + s["vol_span"])
        lo, hi = float(np.quantile(t["vol"], q_lo)), float(np.quantile(t["vol"], q_hi))
        out["dashboard.structural.vol_lo"] = {"value": round(lo, 1), "method": "percentile_transfer", "reference_percentile": round(q_lo, 3)}
        out["dashboard.structural.vol_span"] = {"value": round(hi - lo, 1), "method": "percentile_transfer", "reference_percentile": round(q_hi, 3)}
    else:
        for n in ("vol_lo", "vol_span"): out[f"dashboard.structural.{n}"] = {"status": "not_computed", "reason": "no vol index series"}
    ratio = float(t["ret15_signed"].std() / r["ret15_signed"].std())
    out["_vol_ratio"] = {"value": round(ratio, 3), "note": f"std of {H}-day returns, target / reference"}
    for name, ref_val, rnd in (("kde.signal.median_abs", rk["signal"]["median_abs"], 0.1), ("kde.signal.p5_floor", rk["signal"]["p5_floor"], 0.1),
                               ("kde.signal.p95_cap", rk["signal"]["p95_cap"], 0.1), ("kde.min_bandwidth_pct", rk["min_bandwidth_pct"], 0.05),
                               ("dashboard.ticket.trade_median", rd["ticket"]["trade_median"], 0.05)):
        out[name] = {"value": round(ref_val * ratio / rnd) * rnd, "method": "volatility_ratio_scaling (heuristic)", "reference_value": ref_val}
    return out


def _set(d: dict, dotted: str, v):
    keys = dotted.split("."); cur = d
    for k in keys[:-1]: cur = cur[k]
    cur[keys[-1]] = v
    parent = cur
    flagged = parent.get("_calibrate")
    if flagged and keys[-1] in flagged: flagged.remove(keys[-1])
    if flagged == []: parent.pop("_calibrate")


def apply(cfg_path: Path, proposals: dict) -> dict:
    cfg_path = Path(cfg_path)
    d = json.loads(cfg_path.read_text())
    cfg_path.with_suffix(".json.bak").write_text(json.dumps(d, indent=1))
    done = []
    for k, v in proposals.items():
        if k.startswith("_") or "value" not in v: continue
        _set(d, k, v["value"]); done.append(k)
    left = MarketConfig(d, ".").uncalibrated()
    d["status"]["calibrated"] = not left
    d["status"]["calibration"] = {"applied": done, "proposals": {k: v for k, v in proposals.items() if not k.startswith("_")}}
    cfg_path.write_text(json.dumps(d, indent=1))
    return {"applied": done, "still_uncalibrated": left}
