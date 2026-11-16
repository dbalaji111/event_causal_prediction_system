"""python -m redflag_kit.cli --market markets/gold.json --root . <command>"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .config import MarketConfig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="redflag_kit")
    ap.add_argument("--market", required=False, help="markets/<market>.json")
    ap.add_argument("--root", default=".", help="market folder (paths in the config are relative to it)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate"); sub.add_parser("retrieve")
    k = sub.add_parser("kde"); k.add_argument("--horizons", default=None, help="e.g. 7,15,30 (default: from config)")
    c = sub.add_parser("calibrate"); c.add_argument("--reference", help="oil market json"); c.add_argument("--reference-root", help="folder holding the oil data")
    c.add_argument("--apply", action="store_true")
    d = sub.add_parser("port-dashboard"); d.add_argument("--src", required=True); d.add_argument("--allow-unpatched", action="store_true")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=None); s.add_argument("--host", default="127.0.0.1")
    sc = sub.add_parser("score"); sc.add_argument("event_id")
    sub.add_parser("schema")
    a = ap.parse_args(argv)
    if a.cmd == "schema":
        from . import schema; out = Path(a.root) / "schema" / "redflag_schema.json"; out.parent.mkdir(exist_ok=True); schema.write(out); print(out); return 0
    if not a.market: ap.error("--market is required")
    cfg = MarketConfig.load(a.market, a.root)
    if a.cmd == "validate":
        from .validate import validate_market
        r = validate_market(cfg); return 1 if r["errors"] else 0
    problems = cfg.validate()
    if problems: print("config problems:", *problems, sep="\n  "); return 1
    if a.cmd == "retrieve":
        from .retrieval import Retriever
        Retriever.from_config(cfg).run_batch(cfg.path("retrieval_dir")); return 0
    if a.cmd == "kde":
        from .kde import run_batch; from .prices import PriceData
        px = PriceData(cfg)
        for h in ([int(x) for x in a.horizons.split(",")] if a.horizons else cfg["kde"]["horizons"]):
            run_batch(cfg, cfg.path("retrieval_dir"), px, cfg.path("kde_dir"), h)
        return 0
    if a.cmd == "calibrate":
        from . import calibrate as cal; from .prices import PriceData
        tpx = PriceData(cfg)
        if not a.reference:
            print("No --reference: showing descriptive percentiles of this market's data only (pick thresholds yourself or supply oil data).")
            print(json.dumps(cal.describe(tpx), indent=1)); return 0
        ref = MarketConfig.load(a.reference, a.reference_root or a.root)
        prop = cal.propose(cfg, ref, tpx)
        print(json.dumps(prop, indent=1))
        if a.apply: print(cal.apply(Path(a.market), prop))
        else: print("\n(not applied - review, then re-run with --apply; a .bak copy of the config is written)")
        return 0
    if a.cmd == "port-dashboard":
        from .dashboard_port import port
        out = cfg.path("dashboard"); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(port(Path(a.src).read_text(encoding="utf-8"), cfg, a.allow_unpatched), encoding="utf-8"); print(f"wrote {out}"); return 0
    if a.cmd == "serve":
        import uvicorn; from .api import create_app
        port = a.port or int(cfg["dashboard"]["api_url"].rsplit(":", 1)[1])
        uvicorn.run(create_app(cfg), host=a.host, port=port); return 0
    if a.cmd == "score":
        from . import scoring; from .retrieval import Retriever
        r = Retriever.from_config(cfg); e = r.by_id[a.event_id]
        kp = cfg.path("kde_dir") / "per_event" / f"{a.event_id}_kde_v3.json"
        ks = json.loads(kp.read_text())["posterior_stats"] if kp.exists() else None
        print(json.dumps(scoring.score_event(cfg, e["text"], r.retrieve_adaptive(e, 0.85, 25), ks), indent=1, default=str)); return 0


if __name__ == "__main__":
    sys.exit(main())
