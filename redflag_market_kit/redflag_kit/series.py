"""Any daily time series as the price source for the KDE.

The event memory stores each event's price path on the series the corpus was built with (gold). To measure what
happened after the same analogues on another series (Brent, WTI, a gold future, an FX rate, a volatility index),
build a PriceData from that series and recompute every analogue's path and outcomes on it with `repath`."""
from __future__ import annotations

import copy
from typing import Iterable, List

import pandas as pd

from .config import MarketConfig
from .prices import PriceData

PATH_NODES = (-30, -15, -7, 7, 15, 30)


def series_prices(cfg: MarketConfig, points: pd.DataFrame) -> PriceData:
    """points: columns Date, value (one row per day). Returns a PriceData whose PX is that series, with no
    spread or volatility index attached."""
    if points is None or len(points) < 60:
        raise ValueError("a series needs at least 60 daily values to be used for the KDE")
    df = points.rename(columns={points.columns[0]: "Date", points.columns[1]: "VALUE"})[["Date", "VALUE"]].copy()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna()
    d = copy.deepcopy(cfg.d)
    d["prices"]["columns"] = {"VALUE": "PX"}
    d["prices"]["date_col"] = "Date"
    d["paths"]["vol_index"] = None
    return PriceData(MarketConfig(d, cfg.root), df)


def _round(v):
    return None if v is None else round(float(v), 3)


def repath(analogues: Iterable[dict], prices: PriceData, nodes=PATH_NODES) -> List[dict]:
    """Copies of the analogues with price_path, r7, r15 and r30 measured on `prices` instead of the corpus series.
    Every analogue passed in has already matured (the caller enforces it), so using its forward path is not look-ahead."""
    out = []
    for a in analogues:
        b = dict(a)
        ad = pd.Timestamp(a["date"])
        pp = {}
        for n in nodes:
            pp[str(n)] = _round(prices.backward_return(ad, -n) if n < 0 else prices.forward_return(ad, n))
        b["price_path"] = pp
        for h in (7, 15, 30):
            b[f"r{h}"] = pp.get(str(h))
        b["price"] = prices.px_at(ad)
        out.append(b)
    return out
