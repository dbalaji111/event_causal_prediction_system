"""Bayesian-mixture KDE engine: generalised from kde_engine_v3.py.
Numeric core (dtw_increments, three-rank weight, analogue prior, mixture posterior, red-flag and signal rules) is kept as in the
oil engine; market constants come from the config. Two switches change behaviour on purpose - see kde.dtw_mode / kde.quantile_mode."""
from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from scipy.stats import norm as scipy_norm
from .config import MarketConfig
from .prices import PriceData

_trapz = getattr(np, "trapezoid", None) or np.trapz
warnings.filterwarnings("ignore")


def dtw_increments(path_a: np.ndarray, path_b: np.ndarray, radius: int = 2) -> float:
    """DTW on first differences, Sakoe-Chiba band, squared cost (as kde_engine_v3.dtw_increments)."""
    a, b = np.diff(path_a.astype(float)), np.diff(path_b.astype(float))
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i - j) > radius: continue
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def three_rank_weight(k: dict, similarity, base_q_pct, analogue_label, base_path, analogue_path, all_dtw) -> float:
    """w = sim^a * quantile^b * dtw^c  (a,b,c = 0.5,0.2,0.3 for oil)."""
    q_anal = k["quantile_midpoints"].get(str(analogue_label).strip(), 50.0)
    quantile_score = float(np.exp(-0.5 * ((base_q_pct - q_anal) / k["quantile_sigma"]) ** 2))
    d = dtw_increments(base_path, analogue_path, k["dtw_radius"])
    med = np.median(all_dtw) if len(all_dtw) else 1.0
    med = 1.0 if med < 1e-6 else med
    dtw_score = float(np.exp(-d / med))
    return float((float(similarity) ** k["w_sim"]) * (quantile_score ** k["w_quantile"]) * (dtw_score ** k["w_dtw"]))


def analogue_prior(realised_r: float, path: np.ndarray, min_bw: float) -> Tuple[float, float]:
    """N(mu = realised return, h = max(std of path increments, min_bw))."""
    return float(realised_r), max(float(np.std(np.diff(path.astype(float)))), min_bw)


def mixture_posterior(k: dict, mus, sigmas, weights):
    w = weights / weights.sum()
    T = k["tail_threshold_pct"]
    lo, hi = min(mus - 3 * sigmas) - 5, max(mus + 3 * sigmas) + 5
    x = np.linspace(lo, hi, k["n_grid"]); dx = x[1] - x[0]
    dens = np.zeros(len(x))
    for mu, s, wi in zip(mus, sigmas, w):
        dens += wi * scipy_norm.pdf(x, mu, s)
    cdf = np.clip(np.cumsum(dens) * dx / max(np.cumsum(dens)[-1] * dx, 1e-10), 0, 1)
    q = lambda p: float(x[min(int(np.searchsorted(cdf, p)), len(x) - 1)])
    tr = lambda m: float(_trapz(dens[m], x[m])) if m.any() else 0.0
    p_down, p_up, p_down5 = tr(x <= -T), tr(x >= T), tr(x <= -5.0)
    tail, mid = np.abs(mus) > T, np.abs(mus) <= T
    mean = float(np.average(mus, weights=w))
    n_modes = int(np.sum(np.diff(np.sign(np.gradient(dens))) < 0))
    stats = {"p5": round(q(.05), 3), "p25": round(q(.25), 3), "median": round(q(.5), 3), "p75": round(q(.75), 3), "p95": round(q(.95), 3),
             "mean": round(mean, 3), "std": round(float(np.sqrt(np.average((mus - mean) ** 2 + sigmas ** 2, weights=w))), 3),
             "p_down10": round(max(0, min(1, p_down)), 4), "p_up10": round(max(0, min(1, p_up)), 4),
             "p_down5": round(max(0, min(1, p_down5)), 4), "n_analogues": len(mus),
             "n_tail_analogues": int(tail.sum()), "n_medium_analogues": int(mid.sum()),
             "tail_weight": round(float(w[tail].sum()), 4), "medium_weight": round(float(w[mid].sum()), 4),
             "tail_mu": round(float(np.average(mus[tail], weights=w[tail])), 3) if tail.any() else None,
             "medium_mu": round(float(np.average(mus[mid], weights=w[mid])), 3) if mid.any() else None,
             "is_bimodal": int(n_modes >= 2)}
    stats["band_width"] = round(stats["p95"] - stats["p5"], 3)
    return x, dens, stats


def red_flag_and_signal(k: dict, st: dict):
    s = k["signal"]
    rf = int(st["p_down10"] > k["red_flag_p_down"])
    if st["median"] > s["median_abs"] and st["p5"] > s["p5_floor"]: sig = "LONG"
    elif st["median"] < -s["median_abs"] and st["p95"] < s["p95_cap"]: sig = "SHORT"
    else: sig = "NEUTRAL"
    return rf, sig


def garch_forecast(prices: PriceData, date, horizon) -> Optional[Dict]:
    try:
        from arch import arch_model
    except ImportError:
        return None
    s = prices.df.loc[date - pd.Timedelta(days=400):date, "PX"].dropna()
    if len(s) < 100: return None
    try:
        res = arch_model(s.pct_change().dropna() * 100, vol="Garch", p=1, q=1).fit(disp="off", show_warning=False)
        sig = float(np.sqrt(res.forecast(horizon=horizon, reindex=False).variance.values[-1, -1] * horizon))
        return {"garch_sigma": round(sig, 4), "garch_p5": round(float(scipy_norm.ppf(.05, 0, sig)), 3),
                "garch_p95": round(float(scipy_norm.ppf(.95, 0, sig)), 3),
                "garch_p_down10": round(float(scipy_norm.cdf(-10, 0, sig)), 4), "garch_p_up10": round(float(1 - scipy_norm.cdf(10, 0, sig)), 4)}
    except Exception:
        return None


def _base_path_and_nodes(cfg, prices, base_date, nodes):
    """Path used for DTW. legacy_full_path = oil v3 behaviour (uses realised FUTURE returns: leakage). pre_event_only = nodes <= 0 only."""
    k = cfg["kde"]
    if k["dtw_mode"] == "legacy_full_path":
        arr = np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            r = prices.forward_return(base_date, abs(n) if n != 0 else 1)
            if r is not None:
                arr[i] = r if n > 0 else -r if n < 0 else 0.0
        return arr, list(range(len(nodes)))
    keep = [i for i, n in enumerate(nodes) if n <= 0]
    arr = np.zeros(len(nodes))
    for i in keep:
        n = nodes[i]
        r = prices.backward_return(base_date, abs(n)) if n < 0 else 0.0
        arr[i] = 0.0 if r is None else r
    return arr, keep


def run_event(cfg: MarketConfig, event_id, base_date: pd.Timestamp, analogues: List[dict], prices: PriceData, horizon: int):
    k = cfg["kde"]; nodes = k["price_nodes"]
    realised = prices.forward_return(base_date, horizon)
    excl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in k["exclude_windows"]]
    pool, tau_eff = [], None
    for tau in k["tau_fallbacks"]:
        cand = []
        for a in analogues:
            if float(a.get("similarity", 0)) < tau: continue
            ad = pd.to_datetime(a.get("date", ""), errors="coerce")
            if pd.isna(ad) or ad >= base_date: continue
            if any(s <= ad <= e for s, e in excl): continue
            cand.append(a)
        if len(cand) >= k["min_analogues"]:
            pool, tau_eff = cand, tau; break
    if not pool:
        pool = [a for a in analogues if pd.to_datetime(a.get("date", ""), errors="coerce") < base_date]; tau_eff = 0.0
    head = {"event_id": event_id, "base_date": str(base_date.date()), "horizon": horizon, "tau_effective": tau_eff,
            "realised_r": realised}
    if len(pool) < 5:
        return {**head, "n_analogues": len(pool), "error": "insufficient_analogues"}, [], None

    def a_path(a):
        pp = a.get("price_path") or {}
        return np.array([0.0 if n == 0 else float(pp.get(str(n), 0) or 0) for n in nodes])

    base_path, keep = _base_path_and_nodes(cfg, prices, base_date, nodes)
    if k["quantile_mode"] == "legacy_fixed_mid":
        base_q = k["quantile_midpoints"].get(k["legacy_base_quantile"], 50.0)
    else:
        pct = prices.price_percentile(base_date)
        base_q = 50.0 if pct is None else pct
    bp = base_path[keep]
    all_dtw = [dtw_increments(bp, a_path(a)[keep], k["dtw_radius"]) for a in pool]
    mus, sigs, ws, det = [], [], [], []
    for a in pool:
        ad = pd.to_datetime(a["date"])
        r = prices.forward_return(ad, horizon)
        if r is None:
            key = {7: "r7", 15: "r15", 30: "r30"}.get(horizon)
            stored = a.get(key) if key else None      # no stored outcome for other horizons: skip, never assume 0
            try: r = None if stored is None else float(stored)
            except (TypeError, ValueError): r = None
        if r is None or not np.isfinite(r): continue
        ap = a_path(a)
        w = three_rank_weight(k, a.get("similarity", 0.5), base_q, a.get("quantile_cat", ""), bp, ap[keep], all_dtw)
        if w <= 0: continue
        mu, h = analogue_prior(r, ap, k["min_bandwidth_pct"])
        mus.append(mu); sigs.append(h); ws.append(w)
        det.append({"event_id": a.get("event_id", ""), "date": str(ad.date()), "similarity": round(float(a.get("similarity", 0)), 4),
                    "quantile": a.get("quantile_cat", ""), "realised_r": round(r, 4), "prior_mu": round(mu, 4), "prior_sigma": round(h, 4),
                    "weight": round(w, 6), "is_tail": int(abs(r) > k["tail_threshold_pct"]),
                    "scenario": "tail" if abs(r) > k["tail_threshold_pct"] else "medium"})
    if len(mus) < 5:
        return {**head, "n_analogues": len(mus), "error": "insufficient_valid_analogues"}, [], None
    x, dens, st = mixture_posterior(k, np.array(mus), np.array(sigs), np.array(ws))
    c90 = c50 = None
    if realised is not None and np.isfinite(realised):
        c90, c50 = int(st["p5"] <= realised <= st["p95"]), int(st["p25"] <= realised <= st["p75"])
    rf, sig = red_flag_and_signal(k, st)
    bpx, bsp, bvol = prices.px_at(base_date), prices.spread_at(base_date), prices.vol_at(base_date)
    res = {**head, "tau_relaxed": int(tau_eff < 0.90), "n_analogues": len(mus), "covers_90": c90, "covers_50": c50, "red_flag": rf, "signal": sig,
           "base_px": bpx, "base_spread": bsp, "base_vol": bvol, "base_pctile": prices.price_percentile(base_date),
           # oil-era column names kept as aliases so the dashboard / API / strategy layer read the file unchanged
           "base_wti": bpx, "base_spread_M1M2": bsp, "base_ovx": bvol,
           **{f"kde_{kk}": v for kk, v in st.items()}}
    g = garch_forecast(prices, base_date, horizon)
    if g:
        res.update(g)
        if st["std"] > 0 and g["garch_sigma"] > 0: res["FEI"] = round(g["garch_sigma"] / st["std"], 4)
        if realised is not None and np.isfinite(realised): res["garch_covers_90"] = int(g["garch_p5"] <= realised <= g["garch_p95"])
    return res, det, (x, dens)


def run_batch(cfg: MarketConfig, analogues_dir: Path, prices: PriceData, out_dir: Path, horizon: int, log=print) -> pd.DataFrame:
    out_dir = Path(out_dir); per = out_dir / "per_event"; per.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(analogues_dir).glob("*_analogues.json"))
    if not files: raise FileNotFoundError(f"no *_analogues.json in {analogues_dir} - run retrieve first")
    rows = []
    for jf in files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        eid = data.get("base_event_id", jf.stem.replace("_analogues", ""))
        bd = pd.to_datetime(data.get("base_date", ""), errors="coerce")
        if pd.isna(bd) or not data.get("analogues"): continue
        res, det, grid = run_event(cfg, eid, bd, data["analogues"], prices, horizon)
        rows.append(res)
        if "error" in res: continue
        (per / f"{eid}_kde_v3.json").write_text(json.dumps({
            "event_id": eid, "base_date": res["base_date"], "horizon": horizon,
            "posterior_stats": {kk: v for kk, v in res.items() if kk.startswith("kde_") or kk in ("base_px", "base_spread", "base_vol", "base_pctile", "base_wti", "base_spread_M1M2", "base_ovx")},
            "garch": {kk: v for kk, v in res.items() if kk.startswith("garch_")},
            "analogue_details": det, "x_grid": grid[0].tolist(), "density": grid[1].tolist()}, indent=1, default=str), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"kde_v3_results_h{horizon}.csv", index=False)
    ok = df.dropna(subset=["kde_p5"]) if "kde_p5" in df else df
    log(f"[kde] h{horizon}: {len(ok)}/{len(df)} events produced a posterior -> {out_dir}")
    return df
