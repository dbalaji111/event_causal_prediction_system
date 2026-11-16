"""Machine-readable contracts for every file and endpoint of the Red Flag system. `python -m redflag_kit.cli schema` writes schema/redflag_schema.json.
check(schema, obj) is a tiny JSON-Schema-subset validator (type, required, properties, items, enum) used by tests and validate."""
from __future__ import annotations
import json

NUM, STR, INT, BOOL, NULL = "number", "string", "integer", "boolean", "null"
NN = [NUM, NULL]

ANALOGUE = {"type": "object", "required": ["event_id", "date", "similarity", "price_path", "quantile_cat", "r7", "r15", "r30"], "properties": {
    "event_id": {"type": STR}, "date": {"type": STR}, "similarity": {"type": NUM}, "recency_weight": {"type": NUM},
    "quantile_cat": {"type": STR}, "event_type": {"type": STR}, "polarity": {"type": STR}, "price": {"type": NN},
    "price_path": {"type": "object", "description": "node offset (string, e.g. '-30','7') -> percent move vs event-date price; null if node price missing"},
    "r7": {"type": NN}, "r15": {"type": NN}, "r30": {"type": NN}}}

RETRIEVAL_FILE = {"title": "retrieval_output/per_event/<event_id>_analogues.json", "type": "object",
    "required": ["base_event_id", "base_date", "tau", "top_k", "n_candidates", "n_analogues", "analogues"], "properties": {
    "base_event_id": {"type": STR}, "base_date": {"type": STR}, "tau": {"type": NUM}, "top_k": {"type": INT},
    "n_candidates": {"type": INT}, "n_analogues": {"type": INT}, "analogues": {"type": "array", "items": ANALOGUE},
    "realised_path": {"type": ["object", NULL], "description": "outcome fields: present only when the base event's own forward prices are in the record (back-test), never used by the engine in pre_event_only mode"},
    "realised_r30": {"type": NN}, "tau_used": {"type": NUM, "description": "only on live /retrieve"}}}

POSTERIOR_STATS_KEYS = ["kde_p5", "kde_p25", "kde_median", "kde_p75", "kde_p95", "kde_mean", "kde_std", "kde_p_down10", "kde_p_up10", "kde_p_down5",
                        "kde_n_analogues", "kde_n_tail_analogues", "kde_n_medium_analogues", "kde_tail_weight", "kde_medium_weight",
                        "kde_tail_mu", "kde_medium_mu", "kde_is_bimodal", "kde_band_width"]
KDE_EVENT_FILE = {"title": "kde_output/per_event/<event_id>_kde_v3.json", "type": "object",
    "required": ["event_id", "base_date", "horizon", "posterior_stats", "analogue_details", "x_grid", "density"], "properties": {
    "event_id": {"type": STR}, "base_date": {"type": STR}, "horizon": {"type": INT},
    "posterior_stats": {"type": "object", "required": POSTERIOR_STATS_KEYS + ["base_spread_M1M2", "base_ovx"]},
    "garch": {"type": "object"}, "x_grid": {"type": "array", "items": {"type": NUM}}, "density": {"type": "array", "items": {"type": NUM}},
    "analogue_details": {"type": "array", "items": {"type": "object", "required": ["event_id", "date", "similarity", "realised_r", "prior_mu", "prior_sigma", "weight", "scenario"]}}}}

KDE_CSV = {"title": "kde_output/kde_v3_results_h<H>.csv", "one_row_per": "base event",
    "required": ["event_id", "base_date", "horizon", "tau_effective", "n_analogues", "realised_r", "covers_90", "covers_50", "red_flag", "signal",
                 "base_px", "base_spread", "base_vol", "base_wti", "base_spread_M1M2", "base_ovx", *POSTERIOR_STATS_KEYS],
    "optional": ["garch_sigma", "garch_p5", "garch_p95", "garch_p_down10", "garch_p_up10", "FEI", "garch_covers_90", "tau_relaxed", "base_pctile", "error"],
    "notes": "rows with `error` (insufficient_analogues) carry only the head columns. base_wti/base_spread_M1M2/base_ovx are oil-era aliases of base_px/base_spread/base_vol."}

CORPUS_RECORD = {"title": "memory_*/price_event_memory.json (JSON array or JSONL), one record per event", "required_fields": [
    "event_id", "date", "bert_embeddings (>=100 floats, padded/truncated to embedding_dim)", "price (event-date price)",
    "price_t_minus_30", "price_t_minus_15", "price_t_minus_7", "Quantile_Category", "Event type"],
    "optional_fields": ["price_t_plus_7/15/30 (needed only for back-testing; absent for live events)", "Spread_M1_M2 and _t_plus_/_t_minus_ variants",
                        "Polarity", "Entity", "Influence_Score", "RSI", "text|combined_text|headline"],
    "notes": "Field names are remappable in markets/<m>.json -> corpus_fields."}

ENDPOINTS = [
 {"method": "GET", "path": "/health", "returns": "status, events_loaded, corpus sizes, kde_dir_exists, vol_loaded"},
 {"method": "GET", "path": "/config", "returns": "market config the dashboard needs (labels, thresholds)"},
 {"method": "GET", "path": "/events?q=&type=&limit=", "returns": "{events:[{event_id,text,headline,date,source,Event_type,event_type,kde_available}],total}"},
 {"method": "POST", "path": "/analyze", "body": {"headline": STR, "event_id": STR}, "returns": "keyword event_type, severity(1-4), red_flag_score, label, kde_* (precomputed if available else heuristic, see kde_source)"},
 {"method": "POST", "path": "/retrieve", "body": {"event_id": STR, "tau": NUM, "k_min": INT}, "returns": "RETRIEVAL_FILE + tau_used"},
 {"method": "GET", "path": "/kde/{event_id}", "returns": "posterior_stats, garch, x_grid (<=300 pts), density, base_date"},
 {"method": "GET", "path": "/ovx?days=90  (alias /vol)", "returns": "{dates, values, latest} of the volatility index"},
 {"method": "GET", "path": "/score/{event_id}", "returns": "Python port of the dashboard scores: informational, behavioural, structural, composite, action"},
 {"method": "POST", "path": "/report", "body": {"event_id": STR, "use_llm": BOOL}, "returns": "{report_id, event_id, engine, markdown}"},
 {"method": "POST", "path": "/backtest/run", "body": {"horizon": INT, "start_date": STR, "end_date": STR, "no_trade_band": NUM, "tail_veto": NUM, "cost_bps": NUM}, "returns": "{status:'started'}; 409 if running"},
 {"method": "GET", "path": "/backtest/status", "returns": "{status: idle|running|done|error, progress, total, message}"},
 {"method": "GET", "path": "/backtest/results", "returns": "{run_id, config, n_events, metrics{calibration,direction,trading,red_flag_utility}, equity_curve, dates}"},
 {"method": "GET", "path": "/dashboard", "returns": "the market's dashboard HTML"}]

FORMULAS = {
 "weight": "w_i = sim_i^w_sim * quantile_score_i^w_quantile * dtw_score_i^w_dtw   (oil: 0.5, 0.2, 0.3)",
 "quantile_score": "exp(-0.5 * ((q_base - q_analogue)/quantile_sigma)^2); q in percentile points; unknown label -> 50",
 "dtw_score": "exp(-dtw_i / median(dtw over pool)); dtw = sqrt of squared-cost DTW on first differences, Sakoe-Chiba radius",
 "analogue_prior": "N(mu_i = realised h-day return of analogue, h_i = max(std of its path increments, min_bandwidth_pct))",
 "posterior": "f(r) = sum_i w_i/sum(w) * N(r | mu_i, h_i^2) on an n_grid grid; quantiles from the cumulative sum",
 "red_flag": "1 if P(r < -tail_threshold_pct) > red_flag_p_down",
 "signal": "LONG if median>m and p5>floor; SHORT if median<-m and p95<cap; else NEUTRAL",
 "informational": "round(sev/4*sev_pts + clamp((mean_sim-sim_floor)/sim_span)*sim_pts + clamp(n/count_ref)*count_pts + clamp((tau-tau_floor)/tau_span)*tau_pts)",
 "behavioural": "over = |r7|>|r15|*fade_ratio; under = |r15|>|r7|*fade_ratio; skew=|over%-under%|; agree=max(up,down)/n; round(clamp(skew/skew_ref)*skew_pts + clamp((agree-agree_floor)/agree_span)*agree_pts)",
 "structural": "regime pts (backwardation/contango/flat by spread vs curve_flat_band) + clamp((vol-vol_lo)/vol_span)*vol_pts + clamp(|spread|/curve_mag_scale)*mag_pts",
 "composite": "round(w_info*info + w_behav*behav + w_struct*struct)   (oil: 0.5/0.3/0.2); strong>=70, moderate>=45, weak>=25",
 "rank_label": "Ignore <40, Monitor 40-60, High 60-80, Critical >=80"}

SCHEMA = {"version": 1, "retrieval_file": RETRIEVAL_FILE, "kde_event_file": KDE_EVENT_FILE, "kde_results_csv": KDE_CSV,
          "corpus_record": CORPUS_RECORD, "endpoints": ENDPOINTS, "formulas": FORMULAS}


def _types(t): return t if isinstance(t, list) else [t]


def _is(v, t) -> bool:
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str), "boolean": isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool), "null": v is None,
            "number": isinstance(v, (int, float)) and not isinstance(v, bool)}[t]


def check(schema: dict, obj, path="$") -> list:
    errs = []
    if "type" in schema and not any(_is(obj, t) for t in _types(schema["type"])):
        return [f"{path}: expected {schema['type']}, got {type(obj).__name__}"]
    if isinstance(obj, dict):
        errs += [f"{path}: missing '{k}'" for k in schema.get("required", []) if k not in obj]
        for k, sub in schema.get("properties", {}).items():
            if k in obj: errs += check(sub, obj[k], f"{path}.{k}")
    if isinstance(obj, list) and "items" in schema:
        for i, it in enumerate(obj[:200]): errs += check(schema["items"], it, f"{path}[{i}]")
    return errs


def write(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SCHEMA, f, indent=1)
