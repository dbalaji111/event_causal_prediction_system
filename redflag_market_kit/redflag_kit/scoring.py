"""Python port of the dashboard v6 scoring (redflag_dashboard_v6.html: scoreInformational/Behavioural/Structural, composite, ticket).
Lets researchers get the same scores without a browser. JS is the source of truth; tests/test_kit.py compares both through node."""
from __future__ import annotations
import math, zlib
from typing import Dict, List, Optional
from .config import MarketConfig

clamp = lambda v, a, b: max(a, min(b, v))
js_round = lambda x: int(math.floor(x + 0.5))        # JS Math.round, not Python banker's rounding


def classify_headline(cfg: MarketConfig, headline: str) -> Dict:
    """Keyword event type + severity (first matching group wins, in config order), as redflag_api_v5.score_headline."""
    h = headline.lower()
    for name, g in cfg["event_keywords"].items():
        if any(w in h for w in g["words"]):
            return {"event_type": name, "severity": g["severity"]}
    d = cfg["default_event_type"]
    return {"event_type": d["name"], "severity": d["severity"]}


def red_flag_score(cfg: MarketConfig, headline: str, severity: int) -> int:
    """Keyword red-flag score. The oil API jitters the base with Python hash(), which differs on every restart; here crc32 is used."""
    base = {1: 15, 2: 38, 3: 62, 4: 84}[severity]
    return int(min(100, max(0, base + (zlib.crc32(headline.encode()) % 12) - 6)))


def label_for(cfg: MarketConfig, score: float) -> str:
    c = cfg["dashboard"]["label_cutoffs"]
    return "Critical" if score >= c["critical"] else "High" if score >= c["high"] else "Monitor" if score >= c["monitor"] else "Ignore"


def informational(cfg, severity: Optional[float], n_analogues: int, sims: List[float], tau: Optional[float]) -> Dict:
    c = cfg["dashboard"]["informational"]
    sev = 1 if severity is None else severity
    sims = [s for s in sims if s]
    m = sum(sims) / len(sims) if sims else 0
    tau = tau or 0
    total = js_round((sev / 4) * c["sev_pts"] + clamp((m - c["sim_floor"]) / c["sim_span"], 0, 1) * c["sim_pts"]
                     + clamp(n_analogues / c["count_ref"], 0, 1) * c["count_pts"]
                     + clamp((tau - c["tau_floor"]) / c["tau_span"], 0, 1) * c["tau_pts"])
    return {"total": total, "mean_sim": m, "novelty": 1 - m, "n": n_analogues, "tau": tau}


def behavioural(cfg, analogues: List[Dict]) -> Dict:
    c = cfg["dashboard"]["behavioural"]
    P = [a for a in analogues if a.get("r7") is not None and a.get("r15") is not None]
    if not P:
        return {"total": 0, "over": 0, "under": 0, "agree": 0, "mode": "—", "n": 0}
    over = under = up = 0
    for a in P:
        a7, a15 = abs(a["r7"]), abs(a["r15"])
        if a7 > a15 * c["fade_ratio"]: over += 1
        elif a15 > a7 * c["fade_ratio"]: under += 1
        if a["r15"] > 0: up += 1
    n = len(P)
    oR, uR = over / n, under / n
    agree = max(up, n - up) / n
    skew = abs(oR - uR)
    total = js_round(clamp(skew / c["skew_ref"], 0, 1) * c["skew_pts"] + clamp((agree - c["agree_floor"]) / c["agree_span"], 0, 1) * c["agree_pts"])
    mode = "Fade" if oR > uR * c["fade_ratio"] else "Follow" if uR > oR * c["fade_ratio"] else "Mixed"
    return {"total": total, "over": oR, "under": uR, "agree": agree, "n": n, "mode": mode, "up": up / n}


def structural(cfg, spread: Optional[float], vol: Optional[float]) -> Dict:
    c = cfg["dashboard"]["structural"]
    regime, reg_pts = "—", 0
    if spread is not None:
        if spread > c["curve_flat_band"]: regime, reg_pts = "Backwardation", c["backwardation_pts"]
        elif spread < -c["curve_flat_band"]: regime, reg_pts = "Contango", c["contango_pts"]
        else: regime, reg_pts = "Flat", c["flat_pts"]
    vol_pts = clamp((vol - c["vol_lo"]) / c["vol_span"], 0, 1) * c["vol_pts"] if vol is not None else 0
    mag_pts = clamp(abs(spread) / c["curve_mag_scale"], 0, 1) * c["mag_pts"] if spread is not None else 0
    return {"total": js_round(reg_pts + vol_pts + mag_pts), "regime": regime, "spread": spread, "vol": vol,
            "partial": spread is None or vol is None}


def composite(cfg, info_total, behav_total, struct_total) -> Dict:
    w = cfg["dashboard"]["composite_weights"]
    mis = js_round(info_total * w["info"] + behav_total * w["behav"] + struct_total * w["struct"])
    cv = cfg["dashboard"]["conviction"]
    return {"score": mis, "label": "strong" if mis >= 70 else "moderate" if mis >= 45 else "weak" if mis >= 25 else "none",
            "conviction": "High" if mis >= cv["high"] else "Medium" if mis >= cv["medium"] else "Low"}


def ticket_action(cfg, bimodal: bool, mode: str, median: Optional[float], score: float) -> str:
    t = cfg["dashboard"]["ticket"]; m = t["trade_median"]
    med = median
    if bimodal: return "BUY STRADDLE"
    if mode == "Fade" and med is not None and med > m: return "FADE / SELL"
    if mode == "Fade" and med is not None and med < -m: return "FADE / BUY"
    if mode == "Follow" and med is not None and med > m: return "BUY"
    if mode == "Follow" and med is not None and med < -m: return "SELL"
    if score >= t["hedge_score"]: return "HEDGE"
    return "NO TRADE"


def score_event(cfg, headline, retrieval: Dict, kde_stats: Optional[Dict], spread=None, vol=None) -> Dict:
    """Everything the dashboard shows for one event, from the same inputs the API gives it."""
    cl = classify_headline(cfg, headline)
    an = retrieval.get("analogues", [])
    inf = informational(cfg, cl["severity"], retrieval.get("n_analogues", len(an)), [a["similarity"] for a in an], retrieval.get("tau_used"))
    beh = behavioural(cfg, an)
    ks = kde_stats or {}
    spread = spread if spread is not None else ks.get("base_spread_M1M2", ks.get("base_spread"))
    vol = vol if vol is not None else ks.get("base_ovx", ks.get("base_vol"))
    st = structural(cfg, spread, vol)
    comp = composite(cfg, inf["total"], beh["total"], st["total"])
    return {**cl, "informational": inf, "behavioural": beh, "structural": st, "composite": comp,
            "action": ticket_action(cfg, bool(ks.get("kde_is_bimodal") == 1), beh["mode"], ks.get("kde_median"), comp["score"])}
