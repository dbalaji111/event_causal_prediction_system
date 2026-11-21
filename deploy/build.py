"""Image build step: turns the article CSV (plus an optional real price file) into everything the server needs -
the event memory, the precomputed feed with its KDE posteriors, the dashboard snapshot and the 3D embedding map.

Real prices: put deploy/data/gold_prices.csv (columns Date, XAUUSD, optional GC_M1, GC_M2) and optionally
deploy/data/gvz.csv (columns Date as MM/DD/YYYY, Price) in the repo before deploying. Without them the build uses
the illustrative price approximation and the dashboard says so."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
KIT = APP / "redflag_market_kit"
DATA = APP / "deploy" / "data"
sys.path.insert(0, str(KIT / "demo"))
from build_real_gold_corpus import build_real_gold_market  # noqa: E402

ROOT_REL = "demo/data_root_gold_real"


def run(*args: str) -> None:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=KIT, check=True)
    print(f"[build] {' '.join(args[:4])} … done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    root = KIT / ROOT_REL
    if (root / ".real_ok").exists():
        raise SystemExit(f"{root} already exists; delete it to rebuild from scratch")
    prices = DATA / "gold_prices.csv"
    vol = DATA / "gvz.csv"
    if vol.exists() and not prices.exists():
        raise SystemExit("deploy/data/gvz.csv was given without deploy/data/gold_prices.csv")
    cfg = json.loads((KIT / "markets" / "gold.json").read_text(encoding="utf-8"))
    build_real_gold_market(root, cfg["paths"], APP,
                           prices_csv=prices if prices.exists() else None,
                           vol_csv=vol if vol.exists() else None)
    print(f"[build] prices: {(root / 'price_source.json').read_text()}", flush=True)
    market = ["-m", "redflag_kit.cli", "--market", "markets/gold.json", "--root", ROOT_REL]
    run(*market, "retrieve")
    run(*market, "kde", "--horizons", ",".join(str(h) for h in cfg["kde"]["horizons"]))
    run("demo/export_static_snapshot.py")
    run("demo/export_embeddings_viz.py")


if __name__ == "__main__":
    main()
