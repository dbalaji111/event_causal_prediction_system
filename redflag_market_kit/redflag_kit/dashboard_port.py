"""Patch redflag_dashboard_v6.html for another market. Every edit is an exact-string anchor that must occur exactly once; if your copy of the
dashboard has drifted from the one this was written against, the script stops and names the anchor instead of producing a half-patched page."""
from __future__ import annotations
import json
from .config import MarketConfig

ORACLE_UNPATCHED = {
    "informational": {"sev_pts": 45, "sim_floor": 0.70, "sim_span": 0.28, "sim_pts": 30, "count_ref": 50, "count_pts": 15, "tau_floor": 0.65, "tau_span": 0.25, "tau_pts": 10},
    "behavioural": {"fade_ratio": 1.15, "skew_ref": 0.45, "skew_pts": 55, "agree_floor": 0.5, "agree_span": 0.45, "agree_pts": 45}}

ANCHORS = [  # (name, exact old text, new text template using {…} from `vals`)
 ("api_url", 'const API = "http://localhost:8000";',
  'const API = @@api_url_js@@;\nconst MKT = @@mkt_js@@;'),
 ("title", "<title>Brahmanda v6 — Market Inefficiency Platform</title>", "<title>Brahmanda v6 — @@display_name@@ — Market Inefficiency Platform</title>"),
 ("ticker", '<div class="tk-sym">CRUDE · WTI</div>', '<div class="tk-sym">@@ticker_label@@</div>'),
 ("vol_row", '<span class="n">Implied vol (OVX)</span>', '<span class="n">Implied vol (@@vol_label@@)</span>'),
 ("vol_head", '<div class="l">OVX</div>', '<div class="l">@@vol_label@@</div>'),
 ("vol_pts", "const volPts = ovx!=null ? clamp((ovx-22)/26,0,1)*40 : 0;",
  "const volPts = ovx!=null ? clamp((ovx-MKT.vol_lo)/MKT.vol_span,0,1)*MKT.vol_pts : 0;"),
 ("backw", "if(spread >  0.10){ regime='Backwardation'; regPts=45; }", "if(spread >  MKT.flat_band){ regime='Backwardation'; regPts=MKT.backw_pts; }"),
 ("conta", "else if(spread < -0.10){ regime='Contango'; regPts=40; }", "else if(spread < -MKT.flat_band){ regime='Contango'; regPts=MKT.conta_pts; }"),
 ("flat", "else { regime='Flat'; regPts=18; }", "else { regime='Flat'; regPts=MKT.flat_pts; }"),
 ("mag", "const magPts = spread!=null ? clamp(Math.abs(spread)/1.5,0,1)*15 : 0;",
  "const magPts = spread!=null ? clamp(Math.abs(spread)/MKT.mag_scale,0,1)*MKT.mag_pts : 0;"),
 ("composite", "const mis = Math.round(i.total*0.50 + b.total*0.30 + s.total*0.20);",
  "const mis = Math.round(i.total*MKT.w_info + b.total*MKT.w_behav + s.total*MKT.w_struct);"),
 ("fade_up", "beh.mode==='Fade'   && med> 1", "beh.mode==='Fade'   && med> MKT.trade_median"),
 ("fade_dn", "beh.mode==='Fade'   && med<-1", "beh.mode==='Fade'   && med<-MKT.trade_median"),
 ("follow_up", "beh.mode==='Follow' && med> 1", "beh.mode==='Follow' && med> MKT.trade_median"),
 ("follow_dn", "beh.mode==='Follow' && med<-1", "beh.mode==='Follow' && med<-MKT.trade_median"),
 ("hedge", "else if(S.score>=70){", "else if(S.score>=MKT.hedge_score){"),
 ("conviction", "const conv = S.score>=70 ? 'High' : S.score>=45 ? 'Medium' : 'Low';",
  "const conv = S.score>=MKT.conv_high ? 'High' : S.score>=MKT.conv_med ? 'Medium' : 'Low';"),
]
BANNER_ANCHOR = '<div id="synthBanner"></div>'


def market_js(cfg: MarketConfig) -> dict:
    s, d, t = cfg["dashboard"]["structural"], cfg["dashboard"], cfg["dashboard"]["ticket"]
    w = d["composite_weights"]
    return {"vol_lo": s["vol_lo"], "vol_span": s["vol_span"], "vol_pts": s["vol_pts"], "flat_band": s["curve_flat_band"],
            "backw_pts": s["backwardation_pts"], "conta_pts": s["contango_pts"], "flat_pts": s["flat_pts"], "mag_scale": s["curve_mag_scale"],
            "mag_pts": s["mag_pts"], "w_info": w["info"], "w_behav": w["behav"], "w_struct": w["struct"], "trade_median": t["trade_median"],
            "hedge_score": t["hedge_score"], "conv_high": d["conviction"]["high"], "conv_med": d["conviction"]["medium"]}


def port(src: str, cfg: MarketConfig, allow_unpatched=False) -> str:
    diffs = [k for k, v in ORACLE_UNPATCHED.items() if cfg["dashboard"][k] != v]
    if diffs and not allow_unpatched:
        raise ValueError(f"dashboard.{diffs} differ from the oil defaults, but the v6 JavaScript hard-codes them and this porter does not patch them. "
                         "Either keep the oil values or pass allow_unpatched (the Python scoring would then disagree with the page).")
    vals = {"api_url_js": json.dumps(cfg["dashboard"]["api_url"]), "mkt_js": json.dumps(market_js(cfg)), "display_name": cfg["display_name"],
            "ticker_label": cfg["ticker_label"], "vol_label": cfg["dashboard"]["vol_label"]}
    out = src
    for name, old, new in ANCHORS:
        n = out.count(old)
        if n != 1:
            raise ValueError(f"anchor '{name}' found {n} times (expected 1): {old!r}. Your dashboard differs from the version this porter targets.")
        for k, v in vals.items():
            new = new.replace(f"@@{k}@@", str(v))
        out = out.replace(old, new)
    unc = cfg.uncalibrated()
    if unc:
        msg = f"{cfg['display_name']}: {len(unc)} parameter(s) are uncalibrated placeholders ({', '.join(unc[:4])}{'…' if len(unc) > 4 else ''}). Scores are not comparable with oil until calibrated."
        if out.count(BANNER_ANCHOR) != 1: raise ValueError("banner anchor not found")
        out = out.replace(BANNER_ANCHOR, BANNER_ANCHOR + f'<div class="synth" style="background:#2A2410;border-color:var(--warn);color:#F2D98B">{msg}</div>')
    return out
