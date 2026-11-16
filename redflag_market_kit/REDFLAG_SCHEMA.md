# Red Flag system - schema and implementation reference

Source of truth: `redflag_api_v5.py`, `redflag_dashboard_v6.html`, `redflag_strategy_layer.py`, `kde_engine_v3.py`,
`retrieval_engine_v2.py` (oil implementation). Where the kit changes behaviour this is called out (section 11).
Machine-readable contracts: `schema/redflag_schema.json` (regenerate: `python -m redflag_kit.cli --root . schema`).

## 1. Purpose
For a news event on date *t*, find historically similar events, build a probability distribution of the forward price return,
flag fat-tail downside risk, and present a score/trade ticket on a dashboard.

## 2. Flow
```
events (text + 768-d BERT embedding + price nodes)
  -> retrieval  : cosine sim, only events BEFORE t, tau threshold, top_k      -> <id>_analogues.json
  -> KDE        : per-analogue Gaussian prior, 3-factor weight, mixture       -> kde_v3_results_h{H}.csv + per_event/<id>_kde_v3.json
  -> API (FastAPI) : /analyze /retrieve /kde /score /backtest ...
  -> Dashboard (HTML/JS): informational + behavioural + structural scores -> composite -> ticket
```

## 3. Data contracts
**Corpus record** (train = JSON array, test = JSONL): `event_id`, `date`, `bert_embeddings` (768 floats), `Event type`,
`Quantile_Category` (label like `40%-60%`), `Polarity`, `text`, `price`, and `price_t_minus_30/-15/-7`, `price_t_plus_7/15/30`.
**Prices file** (csv/xlsx): date + spot (PX) + front (FRONT) + second (SECOND) contract; SPREAD = FRONT - SECOND. Column names are mapped in `prices.columns`.
**Vol index file**: date + value (OVX for oil, GVZ for gold), mapped to `VOL`.
**Retrieval file**: `base_event_id, base_date, tau, top_k, n_candidates, n_analogues, analogues[], realised_path, realised_r30`;
each analogue: `event_id, date, similarity, recency_weight, quantile_cat, event_type, polarity, price, price_path{-30..30}, r7, r15, r30`.
**KDE CSV/JSON**: event_id, base_date, horizon, tau_effective, n_analogues, realised_r, covers_90/50, red_flag, signal,
kde_{median,p5,p25,p75,p95,p_down10,p_up10,p_down5,tail_weight,is_bimodal}, base_px/spread/vol/pctile (+ oil aliases base_wti, base_spread_M1M2, base_ovx), garch_*.
Exact field lists and types: `schema/redflag_schema.json`.

## 4. Retrieval
cos(base, analogue); keep analogue.date < base.date (optional `embargo_days`), exclude `exclude_event_types`, sim >= tau (0.75), top_k 50.
`recency_weight = exp(-0.05 * age_years)`. The API's `/retrieve` relaxes tau down [0.82,0.80,0.77,0.75,0.70,0.65,0] until >= k_min candidates.

## 5. KDE (kde_engine_v3)
- Tau ladder [0.90,0.87,0.85,0.82,0.80,0.77,0.75,0.70,0.65,0] until >= 25 analogues; analogues in `exclude_windows` (oil: COVID 2020-02-15..2020-07-01) dropped.
- Per analogue: prior N(mu = realised forward return at horizon, sigma = max(std of path increments, min_bandwidth 1.0)).
- Weight `w = sim^0.5 * quantile_score^0.2 * dtw_score^0.3`; `quantile_score = exp(-0.5((q_base - q_anal)/20)^2)` (labels -> midpoints, unknown -> 50);
  `dtw_score = exp(-dtw/median_dtw)`; DTW on path increments, squared cost, Sakoe-Chiba radius 2, nodes [-30,-15,-7,0,7,15,30].
- Mixture over a 1000-point grid. Tail = |r| > 10%. Outputs p_down10, p_up10, p_down5, tail weight, modes (bimodal if >= 2).
- **Red flag** if p_down10 > 0.20. **Signal** LONG if median > 2 and p5 > -5; SHORT if median < -2 and p95 < 5; else NEUTRAL.
- Optional GARCH(1,1) benchmark and FEI (needs `arch`).

## 6. Dashboard v6 scoring (JS; ported 1:1 to `redflag_kit/scoring.py`, parity-tested against the JS in node)
- **Informational** = sev/4*45 + clamp((meanSim-0.70)/0.28)*30 + clamp(n/50)*15 + clamp((tau-0.65)/0.25)*10
- **Behavioural**: over/under-reaction from r7 vs r15 (ratio 1.15); clamp(skew/0.45)*55 + clamp((agree-0.5)/0.45)*45
- **Structural**: spread band +-0.10 -> Backwardation 45 / Contango 40 / Flat 18; vol pts clamp((vol-22)/26)*40; magnitude clamp(|spread|/1.5)*15
- **Composite** 0.5*info + 0.3*behav + 0.2*struct; labels strong >= 70, moderate >= 45, weak >= 25, else none; conviction 70/45
- **Ticket**: median thresholds +-1, hedge when score >= 70. Rank labels: Ignore < 40, Monitor 40-60, High 60-80, Critical >= 80.

## 7. API endpoints
`/health /events /analyze /retrieve /kde/{id} /ovx (alias /vol) /report /backtest/run|status|results /dashboard`,
strategy layer adds `/sankey /strategy /llama/report /backtest/regime`; kit adds `/config /score/{id}`.
(`/llama/report`, `/sankey`, `/strategy`, `/backtest/regime` are NOT implemented in the kit.)

## 8. Backtest
Per event at horizon 15: trade by KDE median with dead-band 0.5, veto if p_down10 > 0.25, flat 5 bps cost; report direction accuracy,
cumulative return, Sharpe, 90%/50% interval coverage.

## 9. Market parameters: oil -> gold
| Parameter | Oil | Gold treatment | Determined by |
|---|---|---|---|
| prices / vol file | WTI+Brent M1/M2, OVX | XAUUSD, GC M1/M2, GVZ | you supply |
| tail_threshold_pct | 10 | placeholder -> calibrate | percentile transfer |
| min_bandwidth_pct | 1.0 | placeholder -> calibrate | vol ratio |
| signal median/p5/p95 | 2 / -5 / 5 | placeholder -> calibrate | vol ratio |
| curve flat band / magnitude | 0.10 / 1.5 | placeholder -> calibrate | percentile transfer |
| vol_lo / vol_span | 22 / 26 | placeholder -> calibrate | percentile transfer |
| ticket trade_median | 1.0 | placeholder -> calibrate | vol ratio |
| dtw_mode | legacy_full_path | pre_event_only | leakage fix |
| quantile_mode | legacy_fixed_mid | price_percentile | leakage/bug fix |
| exclude_windows | COVID | none | your decision |
| keywords | oil | gold-flavoured draft | you must review |
| port | 8000 | 8010 | avoids clash |

## 10. Porting procedure -> script
1 scaffold `scripts/new_market.py` -> 2 data per `DATA_CHECKLIST.md` -> 3 `validate` -> 4 build gold corpus with the same embedding/event-typing pipeline as oil
-> 5 `retrieve` -> 6 `kde` -> 7 `calibrate [--apply]` and re-run `kde` -> 8 `port-dashboard` (fails loudly if an anchor text is missing/duplicated) -> 9 `serve` -> 10 review `score` output on known gold events.

## 11. Known issues in the oil implementation and what the kit does
| Issue | Kit |
|---|---|
| DTW base path uses realised future returns (outcome leakage) | `pre_event_only` mode for gold; `legacy_full_path` kept to reproduce oil |
| Base quantile fixed "40%-60%" | `price_percentile` mode for gold |
| `hash()` jitter in headline score is non-reproducible | crc32 jitter (reproducible across processes) |
| Heuristic KDE fallback in /analyze looks like real output | response includes `kde_source: precomputed|heuristic` |
| /retrieve r7/r15 come from straddling windows but are plotted as forward returns | kit uses forward returns |
| Dashboard reads `base_spread_M1M2`/`base_ovx` that KDE v3 does not write there | kit writes those alias fields |
| Backtest is return-based, flat 5 bps, Sharpe assumes independent events | unchanged, documented |
| `combined_score` formula in retrieval_engine.py not seen | not reproduced |

## 12. Known limits of this kit
- Tested on **synthetic** data only (29 tests). Not run on your real corpus or prices.
- Parity with your real `kde_engine_v3.py` is untested (not available to me); the kit was rebuilt from the source I read.
- Dashboard anchors were taken from the v6 text I read; if your file has drifted the porter will stop with an error naming the anchor.
- Gold needs your decisions: structural-score meaning (contango is normal carry in gold, so Backwardation/Contango points are a poor transplant), keyword lists, GVZ data, the gold event corpus pipeline, calibrated thresholds.
- Calibration by percentile transfer is a defensible starting point, not a validated one; check with a backtest on gold.
