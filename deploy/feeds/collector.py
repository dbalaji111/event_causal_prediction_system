"""Run this on the computer that has Refinitiv access (for example the one running Workspace). It pulls gold news
headlines and daily price history and pushes them to the live terminal, which scores every new headline.

    pip install lseg-data pandas
    set REFINITIV_APP_KEY=...            (Workspace: App Key Generator)
    set INGEST_TOKEN=...                 (the same value as the server's INGEST_TOKEN variable)

    python collector.py --check                                     # test the Refinitiv session only
    python collector.py --server https://terminal.brahmanda.com.au \\
        --rics "XAU=,GCc1,LCOc1,CLc1,.GVZ" --query "Topic:GOL AND Language:LEN"

It keeps running: news every --news-every seconds, prices every --series-every seconds. The first run sends each
series' full history from --history-start. Progress is kept in collector_state.json, so a restart carries on.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import FeedLoop, HttpSink  # noqa: E402
from refinitiv import RefinitivError, RefinitivSource  # noqa: E402

DEFAULT_QUERY = "Topic:GOL AND Language:LEN"


class JsonState:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def get_kv(self, k):
        return self.d.get(k)

    def set_kv(self, k, v):
        self.d[k] = v
        self.path.write_text(json.dumps(self.d, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("TERMINAL_URL"), help="the live terminal's address")
    ap.add_argument("--token", default=os.environ.get("INGEST_TOKEN"), help="or set INGEST_TOKEN")
    ap.add_argument("--query", action="append", help=f"news query; repeat for several (default: {DEFAULT_QUERY})")
    ap.add_argument("--rics", default="XAU=", help="comma-separated instruments, e.g. XAU=,GCc1,LCOc1,CLc1")
    ap.add_argument("--names", default="", help='optional display names, e.g. "XAU=:Gold spot,LCOc1:Brent front month"')
    ap.add_argument("--history-start", default="1999-01-01")
    ap.add_argument("--news-every", type=int, default=60, help="seconds between news checks")
    ap.add_argument("--series-every", type=int, default=3600, help="seconds between price updates")
    ap.add_argument("--mode", default=None, help="desktop (default), platform or config; or set REFINITIV_MODE")
    ap.add_argument("--state", default="collector_state.json")
    ap.add_argument("--once", action="store_true", help="one round, then stop")
    ap.add_argument("--check", action="store_true", help="test the Refinitiv session and print a sample, push nothing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queries = a.query or [DEFAULT_QUERY]
    rics = [r.strip() for r in a.rics.split(",") if r.strip()]
    names = dict(p.split(":", 1) for p in a.names.split(",") if ":" in p)
    try:
        src = RefinitivSource(mode=a.mode).open()
    except RefinitivError as e:
        sys.exit(f"Refinitiv: {e}")

    try:
        if a.check:
            for q in queries:
                items = src.headlines(q, count=5)
                print(f"\nnews '{q}': {len(items)} headlines")
                for it in items:
                    print(f"  {it['ts']}  {it['headline'][:110]}")
            for ric in rics:
                import pandas as pd
                df = src.history(ric, start=pd.Timestamp.now() - pd.Timedelta(days=20))
                print(f"\n{ric} ({df.attrs.get('field')}): {len(df)} values")
                print(df.tail(5).to_string(index=False))
            return
        if not a.server or not a.token:
            sys.exit("set --server and --token (or TERMINAL_URL and INGEST_TOKEN)")
        loop = FeedLoop(src, HttpSink(a.server, a.token), JsonState(Path(a.state)), queries, rics,
                        history_start=a.history_start, names=names)
        loop.run(news_every=a.news_every, series_every=a.series_every, once=a.once)
    finally:
        src.close()


if __name__ == "__main__":
    main()
