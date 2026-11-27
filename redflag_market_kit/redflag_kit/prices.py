"""Price + spread + volatility-index loader (generalised from kde_engine_v3.PriceData; same date logic)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
from .config import MarketConfig


class PriceData:
    def __init__(self, cfg: MarketConfig, df: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        pc = cfg["prices"]
        self.lim, self.near, self.fwd_max = pc["ffill_limit"], pc["nearest_max_days"], pc["forward_max_days"]
        self.lookback = pc["percentile_lookback_years"]
        if df is None:
            path = cfg.path("prices")
            df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
        dc = pc.get("date_col") or df.columns[0]
        df = df.rename(columns={dc: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.rename(columns={k: v for k, v in pc["columns"].items() if k in df.columns})
        if "PX" not in df.columns and "PX_ALT" in df.columns:
            df["PX"] = df["PX_ALT"]
        if "PX" not in df.columns:
            raise ValueError(f"price file has no column mapped to PX (mapped names: {list(pc['columns'])})")
        if "SPREAD" not in df.columns and {"FRONT", "SECOND"} <= set(df.columns):
            df["SPREAD"] = df["FRONT"] - df["SECOND"]
        df = df.ffill(limit=self.lim)
        vp = cfg.path("vol_index") if cfg["paths"].get("vol_index") else None
        if vp is not None and vp.exists():
            v = pd.read_csv(vp)
            v[pc["vol_date_col"]] = pd.to_datetime(v[pc["vol_date_col"]], format=pc["vol_date_format"], errors="coerce")
            v = v.dropna(subset=[pc["vol_date_col"]]).set_index(pc["vol_date_col"]).sort_index()
            v = v[~v.index.duplicated(keep="last")]
            df = df.join(v[[pc["vol_value_col"]]].rename(columns={pc["vol_value_col"]: "VOL"}), how="left")
        self.df = df.ffill(limit=self.lim)

    # ---- date helpers (calendar days, as in the oil engine) ----
    def _nearest(self, date, max_offset=None):
        max_offset = self.near if max_offset is None else max_offset
        if date in self.df.index:
            return date
        c = self.df.index[abs(self.df.index - date) <= pd.Timedelta(days=max_offset)]
        return c[abs(c - date).argmin()] if len(c) else None

    def get_val(self, date, col) -> Optional[float]:
        if col not in self.df.columns: return None
        t = self._nearest(date)
        if t is None: return None
        v = self.df.loc[t, col]
        return float(v) if np.isfinite(v) else None

    def px_at(self, d): return self.get_val(d, "PX")
    def spread_at(self, d): return self.get_val(d, "SPREAD")
    def vol_at(self, d): return self.get_val(d, "VOL")

    def forward_return(self, date, horizon) -> Optional[float]:
        t0 = self._nearest(date)
        if t0 is None: return None
        p0 = self.df.loc[t0, "PX"]
        if not np.isfinite(p0) or p0 == 0: return None
        target = date + pd.Timedelta(days=horizon)
        if self.df.index[-1] < target: return None        # the outcome window has not finished in this data yet
        t1 = self._nearest(target, self.fwd_max)
        if t1 is None or t1 <= t0: return None
        p1 = self.df.loc[t1, "PX"]
        return float((p1 - p0) / abs(p0) * 100) if np.isfinite(p1) else None

    def backward_return(self, date, days) -> Optional[float]:
        """Percent move from (date - days) to date, expressed like the corpus price_path: (P(t-d) - P(t0)) / P(t0) * 100."""
        t0 = self._nearest(date)
        tb = self._nearest(date - pd.Timedelta(days=days), self.fwd_max)
        if t0 is None or tb is None: return None
        p0, pb = self.df.loc[t0, "PX"], self.df.loc[tb, "PX"]
        return float((pb - p0) / abs(p0) * 100) if np.isfinite(p0) and np.isfinite(pb) and p0 else None

    def price_percentile(self, date) -> Optional[float]:
        t = self._nearest(date)
        if t is None: return None
        p = self.df.loc[t, "PX"]
        hist = self.df.loc[t - pd.Timedelta(days=self.lookback * 365):t, "PX"].dropna()
        return float((hist < p).mean() * 100) if len(hist) >= 50 else None
