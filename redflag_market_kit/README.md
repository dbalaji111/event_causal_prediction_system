# redflag_market_kit

A market-agnostic, config-driven version of the Red Flag stack (retrieval -> KDE -> scoring -> API -> dashboard).
Oil is `markets/oil.json` (as implemented); gold is `markets/gold.json` (leakage-safe defaults, **uncalibrated placeholders flagged**).
Full description of the system: **REDFLAG_SCHEMA.md**. Machine-readable contracts: `schema/redflag_schema.json`.

## Make a new market folder (e.g. gold) - 5 steps
```bash
python scripts/new_market.py --market gold --dest ../redflag_gold --dashboard-src /path/to/redflag_dashboard_v6.html
cd ../redflag_gold && cat DATA_CHECKLIST.md        # 1. put the data files where the checklist says
./run.sh validate                                   # 2. fix every ERROR it prints
./run.sh calibrate                                  # 3. review proposals; add --apply once you accept them
./run.sh all                                        # 4. retrieve -> kde (h7,h15,h30) -> dashboard
./run.sh serve                                      # 5. API + dashboard on the port in markets/gold.json (8010)
```
CLI directly: `python -m redflag_kit.cli --market gold --root . <validate|retrieve|kde|calibrate|port-dashboard|serve|score|schema>`
(`--market/--root` go **before** the subcommand).

## Tests
`python -m unittest discover -s tests -t .`  (29 tests, synthetic data only; needs `httpx`, and `node` for the JS parity test.)

## What is NOT verified
See "Known limits" in REDFLAG_SCHEMA.md section 12. Short version: tested on synthetic data only; parity with your real
`kde_engine_v3.py` and the exact v6 dashboard text was not checkable from here.
