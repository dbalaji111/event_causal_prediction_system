"""Refinitiv / LSEG Data Library adapter: news headlines and daily price history.

Works with either package, whichever is installed:
    pip install lseg-data          (current name)
    pip install refinitiv-data     (older name, same API)

Three ways to open a session (REFINITIV_MODE):
    desktop   Workspace (or Eikon) is running on the same computer. Needs REFINITIV_APP_KEY, created in Workspace
              with the App Key Generator. This is the usual university setup; run the collector on that computer.
    platform  Cloud login to the Data Platform, usable from a server. Needs REFINITIV_APP_KEY, REFINITIV_USERNAME and
              REFINITIV_PASSWORD, and a licence that includes Data Platform access.
    config    Use the library's own config file (lseg-data.config.json / refinitiv-data.config.json) in the
              working directory.
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

PRICE_FIELD_PREFERENCE = ("TRDPRC_1", "CLOSE", "SETTLE", "MID_PRICE", "BID", "OFF_CLOSE", "HST_CLOSE", "ASK")


class RefinitivError(RuntimeError):
    pass


def _library():
    try:
        import lseg.data as ld  # type: ignore
        return ld
    except ImportError:
        pass
    try:
        import refinitiv.data as rd  # type: ignore
        return rd
    except ImportError:
        raise RefinitivError("Neither lseg-data nor refinitiv-data is installed. Run: pip install lseg-data")


class RefinitivSource:
    def __init__(self, mode: Optional[str] = None, app_key: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        self.mode = (mode or os.environ.get("REFINITIV_MODE", "desktop")).lower()
        self.app_key = app_key or os.environ.get("REFINITIV_APP_KEY")
        self.username = username or os.environ.get("REFINITIV_USERNAME")
        self.password = password or os.environ.get("REFINITIV_PASSWORD")
        self.lib = None
        self.session = None

    # ------------------------------------------------------------------ session
    def open(self) -> "RefinitivSource":
        lib = _library()
        try:
            if self.mode == "config":
                lib.open_session()
            elif self.mode == "desktop":
                if not self.app_key:
                    raise RefinitivError("REFINITIV_APP_KEY is not set (create one in Workspace: App Key Generator)")
                lib.open_session(name="desktop.workspace", app_key=self.app_key)
            elif self.mode == "platform":
                if not (self.app_key and self.username and self.password):
                    raise RefinitivError("platform mode needs REFINITIV_APP_KEY, REFINITIV_USERNAME and REFINITIV_PASSWORD")
                platform = lib.session.platform
                self.session = platform.Definition(
                    app_key=self.app_key,
                    grant=platform.GrantPassword(username=self.username, password=self.password),
                ).get_session()
                self.session.open()
                lib.session.set_default(self.session)
            else:
                raise RefinitivError(f"unknown REFINITIV_MODE '{self.mode}' (use desktop, platform or config)")
        except RefinitivError:
            raise
        except Exception as e:  # the library raises many types; report them in one place
            raise RefinitivError(f"could not open a {self.mode} session: {e}") from e
        self.lib = lib
        return self

    def close(self):
        if self.lib is not None:
            try:
                self.lib.close_session()
            except Exception:
                pass
        self.lib = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ news
    def headlines(self, query: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
                  count: int = 100) -> List[dict]:
        """Newest first. Each item: story_id, ts (ISO, UTC), headline, source."""
        if self.lib is None:
            raise RefinitivError("session is not open")
        kw = {"query": query, "count": int(count)}
        if start is not None:
            kw["start"] = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S")
        if end is not None:
            kw["end"] = pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            df = self.lib.news.get_headlines(**kw)
        except Exception as e:
            raise RefinitivError(f"headline request failed for query '{query}': {e}") from e
        return normalise_headlines(df, query)

    # ------------------------------------------------------------------ prices
    def history(self, ric: str, start, end=None, field: Optional[str] = None) -> pd.DataFrame:
        """Daily values for one instrument: DataFrame with columns Date, value. Long ranges are requested in
        five-year pieces because the service caps the rows returned per request."""
        if self.lib is None:
            raise RefinitivError("session is not open")
        start = pd.Timestamp(start)
        end = pd.Timestamp.now("UTC").tz_localize(None).normalize() if end is None else pd.Timestamp(end)
        parts, chosen = [], field
        lo = start
        while lo <= end:
            hi = min(lo + pd.DateOffset(years=5) - pd.Timedelta(days=1), end)
            kw = {"universe": ric, "interval": "daily", "start": lo.strftime("%Y-%m-%d"), "end": hi.strftime("%Y-%m-%d")}
            if field:
                kw["fields"] = [field]
            try:
                df = self.lib.get_history(**kw)
            except Exception as e:
                raise RefinitivError(f"history request failed for {ric} {kw['start']}..{kw['end']}: {e}") from e
            if df is not None and len(df):
                part, chosen = pick_price_column(df, chosen)
                parts.append(part)
            lo = hi + pd.Timedelta(days=1)
        if not parts:
            raise RefinitivError(f"no history returned for {ric} (check the RIC and your entitlements)")
        out = pd.concat(parts).drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
        out.attrs["field"] = chosen
        return out


def normalise_headlines(df, query: str) -> List[dict]:
    if df is None or len(df) == 0:
        return []
    df = df.reset_index()
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("versioncreated") or cols.get("index") or df.columns[0]
    head_col = cols.get("headline") or cols.get("text")
    id_col = cols.get("storyid") or cols.get("story_id")
    src_col = cols.get("sourcecode") or cols.get("source")
    if head_col is None or id_col is None:
        raise RefinitivError(f"unexpected headline columns: {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        ts = pd.to_datetime(r[ts_col], errors="coerce", utc=True)
        if pd.isna(ts) or not str(r[head_col]).strip():
            continue
        out.append({"story_id": str(r[id_col]), "ts": ts.tz_convert(None).isoformat(timespec="seconds"),
                    "headline": " ".join(str(r[head_col]).split()), "source": str(r[src_col]) if src_col else "",
                    "query": query})
    return out


def pick_price_column(df: pd.DataFrame, field: Optional[str] = None):
    """The history frame has the date in its index and one column per field. Use `field` if given, otherwise
    the first preferred field that is mostly filled."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[-1] for c in d.columns]
    d = d.reset_index()
    date_col = next((c for c in d.columns if str(c).lower() in ("date", "timestamp", "index")), d.columns[0])
    if field and field in d.columns:
        col = field
    else:
        numeric = [c for c in d.columns if c != date_col and pd.to_numeric(d[c], errors="coerce").notna().mean() >= 0.5]
        if not numeric:
            raise RefinitivError(f"no numeric price field in history columns {list(d.columns)}")
        col = next((c for c in PRICE_FIELD_PREFERENCE if c in numeric), numeric[0])
    out = pd.DataFrame({"Date": pd.to_datetime(d[date_col], errors="coerce").dt.tz_localize(None).dt.normalize(),
                        "value": pd.to_numeric(d[col], errors="coerce")}).dropna()
    return out, col
