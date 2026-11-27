"""Robustness scenarios for the dashboard's event backtest (redflag_kit.backtest.run_backtest).

The tests follow the process in Kevin J. Davey, "Building Winning Algorithmic Trading Systems" (Wiley, 2014):
limited testing (Ch. 12), in-depth testing and walk-forward analysis (Ch. 13, 18), Monte Carlo analysis and incubation
(Ch. 14, 19), diversification (Ch. 15), position sizing (Ch. 16, 20) and the random "monkey" benchmark (Appendix A),
adapted to an event signal: an entry is a news event whose KDE forecast passes the rule, the exit is the holding period.

The P&L is the one the Validation panel uses: each traded event earns signal x realised return (%) minus the cost in
basis points, every event is its own trade (no position limit), and the baseline reproduces run_backtest exactly. Dollar
figures assume a fixed notional per trade (position % of capital). Pass marks are adjustable rules of thumb.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import MarketConfig
from .prices import PriceData

SEED = 11

CATALOG = [
    dict(id="baseline", title="Baseline", source="Reference run",
         what="The Validation panel's backtest with the settings on the dashboard. Every other test is compared with it.", params=[]),
    dict(id="entry_test", title="Entry test (fixed holds)", source="Davey Ch. 12, Limited testing",
         what="Keeps the same entries and swaps the exit for a range of holding periods. A real entry edge makes money for "
              "most holds, not only the one chosen.",
         params=[dict(key="holds", label="Holding periods (days)", default="1,3,5,7,10,15,20,30"),
                 dict(key="pass_share", label="Pass if this share of holds is profitable", default=0.7)]),
    dict(id="monkey", title="Monkey test (random trading)", source="Davey Appendix A, Monkey trading",
         what="Random 'monkeys' trade the same events: random events and direction with the same number of trades, and the "
              "real signals with random holds. The strategy should beat almost all of them.",
         params=[dict(key="runs", label="Monkey runs", default=1000),
                 dict(key="pass_pct", label="Pass if it beats this % of monkeys", default=90)]),
    dict(id="reverse", title="Reverse (opposite) system", source="Davey Ch. 1",
         what="Takes every signal in the opposite direction. With a real edge the opposite system loses.", params=[]),
    dict(id="sensitivity", title="Parameter sensitivity", source="Davey Ch. 13, In-depth testing",
         what="Re-runs the rule over a grid of no-trade bands and tail vetoes. Robust settings sit on a profitable plateau; "
              "a lone profitable peak suggests curve fitting.",
         params=[dict(key="bands", label="No-trade bands (%)", default="0.25,0.5,0.75,1.0,1.5"),
                 dict(key="vetoes", label="Tail vetoes P(<-10%)", default="0.15,0.2,0.25,0.35,1.01"),
                 dict(key="pass_share", label="Pass if this share of neighbours is profitable", default=0.7)]),
    dict(id="walk_forward", title="Walk-forward analysis", source="Davey Ch. 13 and 18",
         what="Chooses band, veto and hold on an in-sample window using only trades that had closed, trades the next "
              "out-of-sample window with them, then rolls forward. Only out-of-sample trades count.",
         params=[dict(key="is_months", label="In-sample window (months)", default=12),
                 dict(key="oos_months", label="Out-of-sample window (months)", default=3),
                 dict(key="anchored", label="Anchored (in-sample grows)", default=False),
                 dict(key="objective", label="Optimise for (net or sharpe)", default="net"),
                 dict(key="bands", label="Bands to search", default="0.25,0.5,1.0,1.5"),
                 dict(key="vetoes", label="Vetoes to search", default="0.15,0.25,0.35,1.01"),
                 dict(key="holds", label="Holds to search (days)", default="7,10,15,20"),
                 dict(key="pass_wfe", label="Pass if walk-forward efficiency is at least", default=0.5)]),
    dict(id="monte_carlo", title="Monte Carlo analysis", source="Davey Ch. 14 and 19",
         what="Resamples the trades into thousands of one-year sequences to estimate risk of ruin, typical drawdown and "
              "return, and the return-to-drawdown ratio.",
         params=[dict(key="sims", label="Simulations", default=2500),
                 dict(key="years", label="Years per simulation", default=1),
                 dict(key="quit_pct", label="Ruin if equity falls below (% of start)", default=60),
                 dict(key="max_ruin", label="Pass if risk of ruin below (%)", default=10),
                 dict(key="min_ret_dd", label="Pass if median return/drawdown above", default=2.0)]),
    dict(id="incubation", title="Incubation (out-of-time check)", source="Davey Ch. 14 and 19",
         what="Splits the period at a date. Trades before it set the expected range; trades after it are the incubation "
              "and should stay inside that range.",
         params=[dict(key="split_date", label="Incubation starts (YYYY-MM-DD; blank = last 25% of trades)", default=""),
                 dict(key="sims", label="Simulations for the band", default=2000)]),
    dict(id="position_sizing", title="Position sizing", source="Davey Ch. 16 and 20",
         what="Replays the trades with a fixed notional and with fixed-fractional sizing: risk a share of equity per trade, "
              "where the risk of one unit of notional is the largest % loss seen so far.",
         params=[dict(key="fractions", label="Fixed-fractional risk per trade (%)", default="1,2,5"),
                 dict(key="max_leverage", label="Most notional per trade (x equity)", default=2.0)]),
    dict(id="cost_stress", title="Cost stress", source="Davey Ch. 12 and 22",
         what="Raises the cost per trade step by step to see how much friction the edge survives.",
         params=[dict(key="costs", label="Cost per trade (basis points)", default="0,5,10,20,40,60")]),
    dict(id="entry_lag", title="Execution delay", source="Davey Ch. 22, Considerations before going live",
         what="Enters 0, 1, 2 or more trading days after the event, to see how fast the edge fades if execution is late.",
         params=[dict(key="lags", label="Entry lags (trading days)", default="0,1,2,3")]),
    dict(id="consistency", title="Consistency by period", source="Davey Ch. 23, Monitoring a live strategy",
         what="Breaks the result into years and quarters. A durable edge earns in most periods rather than in one burst.",
         params=[dict(key="pass_share", label="Pass if this share of quarters is profitable", default=0.6)]),
    dict(id="diversification", title="Diversification across horizons", source="Davey Ch. 15, Diversification",
         what="Runs the rule on the 7, 15 and 30-day forecasts as separate accounts, correlates their monthly P&L and "
              "combines them with equal capital, to see whether spreading lowers drawdown.", params=[]),
]
CATALOG_BY_ID = {c["id"]: c for c in CATALOG}


def _num_list(s, cast=float):
    if isinstance(s, (list, tuple)):
        return [cast(x) for x in s]
    return [cast(x) for x in str(s).replace(";", ",").split(",") if str(x).strip()]


def _p(params, key):
    d = next(p["default"] for p in CATALOG_BY_ID[params["_id"]]["params"] if p["key"] == key)
    v = params.get(key, d)
    if isinstance(d, bool):
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
    if isinstance(d, int):
        return int(float(v))
    if isinstance(d, float):
        return float(v)
    return v


def _r(v, n=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, n)


def verdict(status, text):
    return {"status": status, "text": text}


def table(title, columns, rows):
    return {"title": title, "columns": columns, "rows": [[_r(c) if isinstance(c, (float, np.floating)) else c for c in r] for r in rows]}


def chart(kind, title, **kw):
    return {"kind": kind, "title": title, **kw}


def trade_metrics(t: pd.DataFrame, years: float) -> dict:
    if t is None or len(t) == 0:
        return dict(trades=0, net=0.0, pct=0.0, win_pct=None, avg=None, pf=None, max_dd=0.0, sharpe_t=None)
    t = t.sort_values("exit")
    eq = t["net"].cumsum().values
    dd = float(np.min(eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]))
    wins, losses = t["net"][t["net"] > 0].sum(), -t["net"][t["net"] < 0].sum()
    sd = t["net"].std(ddof=1) if len(t) > 1 else np.nan
    return dict(trades=int(len(t)), net=float(t["net"].sum()), pct=float(t["pct"].sum()), win_pct=float((t["net"] > 0).mean() * 100),
                avg=float(t["net"].mean()), pf=float(wins / losses) if losses > 0 else None, max_dd=min(dd, 0.0),
                sharpe_t=float(t["net"].mean() / sd * np.sqrt(len(t) / max(years, 1e-9))) if sd and sd > 0 else None)


class Ctx:
    def __init__(self, cfg: MarketConfig, bt: Optional[dict] = None, prices: Optional[PriceData] = None, kde_frames: Optional[dict] = None):
        self.cfg = cfg
        b = {**cfg["backtest"], "start_date": "2024-01-01", "end_date": "2026-04-28", "capital": 100000.0,
             "position_pct": 100.0, "entry_lag": 0, **(bt or {})}
        self.bt = b
        self.h = int(b.get("horizon", 15))
        self.band, self.veto = float(b.get("no_trade_band", 0.5)), float(b.get("tail_veto", 0.25))
        self.cost_bps = float(b.get("cost_bps", 5))
        self.capital = float(b["capital"]); self.notional = self.capital * float(b["position_pct"]) / 100
        self.lag = int(b["entry_lag"])
        self.start, self.end = pd.Timestamp(b["start_date"]), pd.Timestamp(b["end_date"])
        if self.end <= self.start:
            raise ValueError("the end date must be after the start date")
        self.years = max((self.end - self.start).days / 365.25, 1 / 12)
        self.prices = prices or PriceData(cfg)
        self._raw = dict(kde_frames or {})
        self._kde, self._ev = {}, {}

    def kde_raw(self, h):
        if h not in self._raw:
            f = self.cfg.path("kde_dir") / f"kde_v3_results_h{int(h)}.csv"
            if not f.exists():
                raise FileNotFoundError(f"{f.name} not found: run `kde --horizons {h}` first")
            self._raw[h] = pd.read_csv(f)
        return self._raw[h]

    def kde(self, h):
        if h not in self._kde:
            k = self.kde_raw(h).copy()
            k["base_date"] = pd.to_datetime(k["base_date"], errors="coerce")
            k = k[(k["base_date"] >= self.start) & (k["base_date"] <= self.end)].dropna(subset=["kde_median"])
            if "kde_mean" not in k:
                k["kde_mean"] = k["kde_median"]
            if "kde_p_down10" not in k:
                k["kde_p_down10"] = 0.1                       # run_backtest's default when the column is absent
            self._kde[h] = k.reset_index(drop=True)          # keeps the CSV order, as run_backtest does
        return self._kde[h]

    def events(self, h=None, hold=None, lag=None):
        """One row per usable event: its forecast and the realised % move over the hold, entered `lag` trading days late.
        The Validation panel's own case (hold = forecast horizon, no lag) uses the file's realised_r, row for row."""
        h = self.h if h is None else int(h)
        hold = h if hold is None else int(hold)
        lag = self.lag if lag is None else int(lag)
        key = (h, hold, lag)
        if key in self._ev:
            return self._ev[key]
        k = self.kde(h)
        idx = self.prices.df.index
        rows = []
        for i, r in k.iterrows():
            d = r["base_date"]
            if hold == h and lag == 0 and "realised_r" in k:
                if pd.isna(r["realised_r"]):                 # run_backtest drops these rows too
                    continue
                entry, ret = d, float(r["realised_r"])
            else:
                j = int(idx.searchsorted(d, side="left")) + lag
                if j >= len(idx):
                    continue
                entry = idx[j] if lag else d
                ret = self.prices.forward_return(entry, hold)
                if ret is None:
                    continue
            rows.append((i, d, entry, entry + pd.Timedelta(days=hold), ret, r["kde_median"], r["kde_p_down10"], r["kde_mean"]))
        ev = pd.DataFrame(rows, columns=["kidx", "base_date", "entry", "exit", "r", "kde_median", "kde_p_down10", "kde_mean"])
        self._ev[key] = ev
        return ev

    def signals(self, ev, band=None, veto=None, rule="median_veto", flip=False):
        band = self.band if band is None else float(band)
        veto = self.veto if veto is None else float(veto)
        if rule == "every_event":
            s = np.ones(len(ev), dtype=int)
        else:
            med, pd10 = ev["kde_median"].values.astype(float), ev["kde_p_down10"].values.astype(float)
            with np.errstate(invalid="ignore"):
                s = np.where(pd10 > veto, 0, np.where(med > band, 1, np.where(med < -band, -1, 0)))
        return (-s if flip else s).astype(int)

    def trades(self, ev, s, cost_bps=None):
        c = (self.cost_bps if cost_bps is None else float(cost_bps)) / 100.0   # bps -> % of notional
        s = np.asarray(s, int)
        t = ev.loc[s != 0, ["kidx", "base_date", "entry", "exit", "r"]].copy()
        t["sig"] = s[s != 0]
        t["pct"] = t["sig"] * t["r"] - c
        t["net"] = t["pct"] * self.notional / 100.0
        return t.reset_index(drop=True)

    def base(self):
        ev = self.events()
        s = self.signals(ev)
        return ev, s, self.trades(ev, s)


COLS = ["Run", "Trades", "Net $", "Sum of trade returns %", "Win %", "Profit factor", "Max DD $", "Sharpe (per trade, annualised)"]


def _row(name, m):
    return [name, m["trades"], m["net"], m["pct"], m["win_pct"], m["pf"], m["max_dd"], m["sharpe_t"]]


def sc_baseline(ctx, p):
    k = ctx.kde_raw(ctx.h)
    ref = run_backtest(k, {**ctx.bt, "horizon": ctx.h})
    _, _, t = ctx.base()
    m = trade_metrics(t, ctx.years)
    tm = ref["metrics"]["trading"]; dm = ref["metrics"]["direction"]
    curve = t.sort_values("exit")
    st = "info" if m["net"] > 0 else "fail"
    return dict(verdict=verdict(st, f"{m['trades']} trades, {m['pct']:+.1f}% summed trade return (Validation panel: "
                                    f"{tm['cumulative_return_pct']:+.1f}%, {dm['n_trades']} trades), ${m['net']:,.0f} on a "
                                    f"${ctx.notional:,.0f} notional per trade."),
                tables=[table("Baseline", COLS, [_row(f"{ctx.h}-day forecast, band {ctx.band}, veto {ctx.veto}", m)]),
                        table("Validation panel metrics (run_backtest)", ["Measure", "Value"],
                              [["Cumulative return %", tm["cumulative_return_pct"]], ["Mean trade return %", tm["mean_trade_return_pct"]],
                               ["Sharpe", tm["sharpe_ratio"]], ["Max drawdown %", tm["max_drawdown_pct"]], ["Win rate %", dm["win_rate"]],
                               ["Trades", dm["n_trades"]]])],
                charts=[chart("lines", "Cumulative P&L (USD)", curves=[{"name": "baseline", "x": [d.strftime("%Y-%m-%d") for d in curve["exit"]],
                                                                         "y": [round(float(v), 0) for v in curve["net"].cumsum()]}], yprefix="$")])


def sc_entry_test(ctx, p):
    holds = _num_list(_p(p, "holds"), int)
    rows, nets = [], []
    for hd in holds:
        ev = ctx.events(hold=hd)
        m = trade_metrics(ctx.trades(ev, ctx.signals(ev)), ctx.years)
        rows.append(_row(f"{hd} days", m)); nets.append(m["net"])
    share = float(np.mean([n > 0 for n in nets])) if nets else 0.0
    need = _p(p, "pass_share")
    st = "pass" if share >= need else ("watch" if share >= need - 0.2 else "fail")
    return dict(verdict=verdict(st, f"{share:.0%} of {len(holds)} holding periods are profitable (pass at {need:.0%})."),
                tables=[table("Same entries, fixed holds", COLS, rows)],
                charts=[chart("bar", "Net profit by holding period (USD)", x=[r[0] for r in rows], series=[{"name": "net", "y": [round(n, 0) for n in nets]}], yprefix="$")])


def sc_monkey(ctx, p):
    runs, need = _p(p, "runs"), _p(p, "pass_pct")
    ev, s, t = ctx.base()
    real = trade_metrics(t, ctx.years)["net"]
    rng = np.random.default_rng(SEED)
    n_sig = int((s != 0).sum())
    if n_sig == 0:
        return dict(verdict=verdict("fail", "The rule takes no trades."), tables=[], charts=[])
    share_long = float((s[s != 0] > 0).mean())
    r = ev["r"].values; c = ctx.cost_bps / 100.0; f = ctx.notional / 100.0
    m_entry = np.empty(runs)
    for b in range(runs):
        pick = rng.choice(len(ev), size=n_sig, replace=False)
        sg = np.where(rng.random(n_sig) < share_long, 1, -1)
        m_entry[b] = float(((sg * r[pick]) - c).sum() * f)
    holds = sorted({1, 3, 5, 7, 10, 15, 20, 30} | {ctx.h})
    rmap = {hd: dict(zip(ctx.events(hold=hd)["kidx"], ctx.events(hold=hd)["r"])) for hd in holds}
    sig_k, sig_v = ev["kidx"].values[s != 0], s[s != 0]
    m_exit = np.empty(runs)
    for b in range(runs):
        hs = rng.choice(holds, size=len(sig_k))
        tot = 0.0
        for kk, sv, hd in zip(sig_k, sig_v, hs):
            rv = rmap[hd].get(kk)
            if rv is not None:
                tot += sv * rv - c
        m_exit[b] = tot * f
    pe, px_ = float((m_entry < real).mean() * 100), float((m_exit < real).mean() * 100)
    st = "pass" if min(pe, px_) >= need else ("watch" if min(pe, px_) >= need - 15 else "fail")
    q = lambda a: [_r(np.percentile(a, k_), 0) for k_ in (5, 50, 95)]
    return dict(verdict=verdict(st, f"Beats {pe:.0f}% of random-entry monkeys and {px_:.0f}% of random-exit monkeys (pass at {need:.0f}%)."),
                tables=[table("Strategy vs monkeys (net USD)", ["Run", "Net $", "Monkey 5th pct", "Monkey median", "Monkey 95th pct", "Beats %"],
                              [["Random events + direction", real, *q(m_entry), pe], ["Random exit (hold)", real, *q(m_exit), px_]])],
                charts=[chart("hist", "Random-entry monkeys: net profit (USD)", values=[round(v, 0) for v in m_entry], marker=round(real, 0), xprefix="$"),
                        chart("hist", "Random-exit monkeys: net profit (USD)", values=[round(v, 0) for v in m_exit], marker=round(real, 0), xprefix="$")])


def sc_reverse(ctx, p):
    ev = ctx.events()
    a = trade_metrics(ctx.trades(ev, ctx.signals(ev)), ctx.years)
    b = trade_metrics(ctx.trades(ev, ctx.signals(ev, flip=True)), ctx.years)
    st = "pass" if a["net"] > 0 and b["net"] < 0 else ("watch" if a["net"] > b["net"] else "fail")
    txt = (f"Strategy ${a['net']:,.0f}, opposite ${b['net']:,.0f}. " +
           ("The opposite loses, as a real edge should." if st == "pass" else
            "Both make money: the result may come from market drift, not the forecast." if a["net"] > 0 and b["net"] > 0 else
            "The opposite does better: the signal direction is not helping."))
    return dict(verdict=verdict(st, txt), tables=[table("Strategy vs its opposite", COLS, [_row("Strategy", a), _row("Opposite", b)])], charts=[])


def sc_sensitivity(ctx, p):
    bands, vetoes = _num_list(_p(p, "bands")), _num_list(_p(p, "vetoes"))
    ev = ctx.events()
    Z = np.array([[trade_metrics(ctx.trades(ev, ctx.signals(ev, band=b, veto=v)), ctx.years)["net"] for b in bands] for v in vetoes])
    bi = int(np.argmin([abs(b - ctx.band) for b in bands])); vi = int(np.argmin([abs(v - ctx.veto) for v in vetoes]))
    nb = Z[max(vi - 1, 0):vi + 2, max(bi - 1, 0):bi + 2]
    share, allshare = float((nb > 0).mean()), float((Z > 0).mean())
    need = _p(p, "pass_share")
    st = "pass" if share >= need else ("watch" if share >= need - 0.2 else "fail")
    return dict(verdict=verdict(st, f"{share:.0%} of settings around band {ctx.band} / veto {ctx.veto} are profitable; {allshare:.0%} of the whole grid."),
                tables=[table("Net profit (USD): rows = tail veto, columns = no-trade band", ["Veto \\ band", *[str(b) for b in bands]],
                              [[str(v), *[float(x) for x in Z[i]]] for i, v in enumerate(vetoes)])],
                charts=[chart("heatmap", "Net profit by band and veto (USD)", x=[str(b) for b in bands], y=[str(v) for v in vetoes],
                              z=[[round(float(x), 0) for x in row] for row in Z], xlabel="No-trade band (%)", ylabel="Tail veto")])


def sc_walk_forward(ctx, p):
    is_m, oos_m, anchored, obj = _p(p, "is_months"), _p(p, "oos_months"), _p(p, "anchored"), str(_p(p, "objective")).lower()
    bands, vetoes, holds = _num_list(_p(p, "bands")), _num_list(_p(p, "vetoes")), _num_list(_p(p, "holds"), int)
    evs = {hd: ctx.events(hold=hd) for hd in holds}

    def score(t, yrs):
        m = trade_metrics(t, yrs)
        return (m["sharpe_t"] if obj == "sharpe" else m["net"]) if m["trades"] >= 3 else -np.inf

    rows, oos_frames, is_rates, oos_rates = [], [], [], []
    w0 = ctx.start
    while True:
        is_start = ctx.start if anchored else w0
        is_end = w0 + pd.DateOffset(months=is_m); oos_end = min(is_end + pd.DateOffset(months=oos_m), ctx.end + pd.Timedelta(days=1))
        if is_end >= ctx.end:
            break
        best, best_s = None, -np.inf
        for hd in holds:
            e = evs[hd]
            ei = e[(e["base_date"] >= is_start) & (e["base_date"] < is_end) & (e["exit"] < is_end)]
            for b in bands:
                for v in vetoes:
                    sc = score(ctx.trades(ei, ctx.signals(ei, band=b, veto=v)), (is_end - is_start).days / 365.25)
                    if sc is not None and sc > best_s:
                        best, best_s = (hd, b, v), sc
        if best is not None:
            hd, b, v = best; e = evs[hd]
            ei = e[(e["base_date"] >= is_start) & (e["base_date"] < is_end) & (e["exit"] < is_end)]
            eo = e[(e["base_date"] >= is_end) & (e["base_date"] < oos_end)]
            ti, to = ctx.trades(ei, ctx.signals(ei, band=b, veto=v)), ctx.trades(eo, ctx.signals(eo, band=b, veto=v))
            yi, yo = (is_end - is_start).days / 365.25, max((oos_end - is_end).days / 365.25, 1 / 365)
            mi, mo = trade_metrics(ti, yi), trade_metrics(to, yo)
            is_rates.append(mi["net"] / yi); oos_rates.append(mo["net"] / yo)
            rows.append([f"{is_start.date()} → {is_end.date()}", f"{is_end.date()} → {(oos_end - pd.Timedelta(days=1)).date()}",
                         hd, b, v, mi["trades"], mi["net"], mo["trades"], mo["net"]])
            if len(to):
                oos_frames.append(to)
        w0 = w0 + pd.DateOffset(months=oos_m)
    if not rows:
        return dict(verdict=verdict("fail", f"The period is shorter than one in-sample window ({is_m} months), or no window had 3+ trades."), tables=[], charts=[])
    oos = pd.concat(oos_frames).sort_values("exit") if oos_frames else pd.DataFrame(columns=["exit", "net", "pct"])
    wfe = np.mean(oos_rates) / np.mean(is_rates) if np.mean(is_rates) > 0 else np.nan
    _, _, base_t = ctx.base()
    m_oos = trade_metrics(oos, ctx.years); need = _p(p, "pass_wfe")
    st = "pass" if m_oos["net"] > 0 and wfe >= need else ("watch" if m_oos["net"] > 0 else "fail")
    curves = [{"name": n, "x": [d.strftime("%Y-%m-%d") for d in t.sort_values("exit")["exit"]],
               "y": [round(float(v), 0) for v in t.sort_values("exit")["net"].cumsum()]}
              for n, t in (("Walk-forward (out-of-sample only)", oos), ("Fixed settings (baseline)", base_t)) if len(t)]
    return dict(verdict=verdict(st, f"Out-of-sample net ${m_oos['net']:,.0f} over {m_oos['trades']} trades; walk-forward efficiency "
                                    f"{wfe:.0%} (out-of-sample vs in-sample profit rate; pass at {need:.0%})."),
                tables=[table("Walk-forward windows (chosen in-sample, traded out-of-sample)",
                              ["In-sample", "Out-of-sample", "Hold", "Band", "Veto", "IS trades", "IS net $", "OOS trades", "OOS net $"], rows)],
                charts=[chart("lines", "Cumulative net profit (USD)", curves=curves, yprefix="$")])


def _paths(nets, n, sims, start, rng):
    d = rng.choice(nets, size=(sims, n), replace=True)
    return np.concatenate([np.full((sims, 1), start), start + np.cumsum(d, axis=1)], axis=1)


def sc_monte_carlo(ctx, p):
    sims, years, quit_pct = _p(p, "sims"), _p(p, "years"), _p(p, "quit_pct")
    _, _, t = ctx.base()
    if len(t) < 10:
        return dict(verdict=verdict("fail", "Fewer than 10 trades: too few to resample."), tables=[], charts=[])
    n = max(int(round(len(t) / ctx.years * years)), 1)
    paths = _paths(t["net"].values, n, sims, ctx.capital, np.random.default_rng(SEED))
    ruin = (paths <= ctx.capital * quit_pct / 100).any(axis=1)
    pk = np.maximum.accumulate(paths, axis=1)
    dd = ((paths - pk) / pk).min(axis=1) * 100
    ret = (paths[:, -1] / ctx.capital - 1) * 100
    rdd = np.divide(ret, np.abs(dd), out=np.full_like(ret, np.nan), where=dd < 0)
    ror, mrdd = ruin.mean() * 100, float(np.nanmedian(rdd)) if np.isfinite(rdd).any() else float("inf")
    ok_r, ok_d = ror < _p(p, "max_ruin"), mrdd > _p(p, "min_ret_dd")
    st = "pass" if ok_r and ok_d else ("watch" if ok_r or ok_d else "fail")
    pc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    return dict(verdict=verdict(st, f"Risk of ruin {ror:.1f}% (pass below {_p(p, 'max_ruin')}%), median return/drawdown {mrdd:.2f} "
                                    f"(pass above {_p(p, 'min_ret_dd')}), {n} trades per simulated year."),
                tables=[table(f"{sims} simulations of {years} year(s), start ${ctx.capital:,.0f}, notional ${ctx.notional:,.0f}/trade",
                              ["Measure", "Value"],
                              [["Risk of ruin %", ror], ["Median max drawdown %", float(np.median(dd))], ["Median return %", float(np.median(ret))],
                               ["Median return / drawdown", mrdd], ["Probability of profit %", float((ret > 0).mean() * 100)],
                               ["5th pct return %", float(np.percentile(ret, 5))], ["95th pct return %", float(np.percentile(ret, 95))],
                               ["Worst 5% drawdown %", float(np.percentile(dd, 5))]])],
                charts=[chart("fan", "Simulated equity (USD): median, 25–75% and 5–95% bands", x=list(range(n + 1)), p5=pc[0].round(0).tolist(),
                              p25=pc[1].round(0).tolist(), p50=pc[2].round(0).tolist(), p75=pc[3].round(0).tolist(), p95=pc[4].round(0).tolist(),
                              xlabel="trade number", yprefix="$"),
                        chart("hist", "Max drawdown per simulation (%)", values=[round(v, 2) for v in dd], xsuffix="%")])


def sc_incubation(ctx, p):
    sims = _p(p, "sims")
    _, _, t = ctx.base()
    t = t.sort_values("entry").reset_index(drop=True)
    if len(t) < 12:
        return dict(verdict=verdict("fail", "Too few trades to split."), tables=[], charts=[])
    sd = str(_p(p, "split_date")).strip()
    split = pd.Timestamp(sd) if sd else t["entry"].iloc[int(len(t) * 0.75)]
    pre, post = t[t["exit"] < split], t[t["entry"] >= split]
    if len(pre) < 8 or len(post) < 2:
        return dict(verdict=verdict("fail", f"Split {split.date()} leaves {len(pre)} trades before and {len(post)} after."), tables=[], charts=[])
    k = len(post)
    paths = _paths(pre["net"].values, k, sims, 0.0, np.random.default_rng(SEED))
    lo, med, hi = np.percentile(paths, [5, 50, 95], axis=0)
    act = np.r_[0.0, post["net"].cumsum().values]
    inside = float(((act >= lo) & (act <= hi)).mean() * 100); fp = float((paths[:, -1] < act[-1]).mean() * 100)
    st = "pass" if fp >= 5 and inside >= 80 else ("watch" if fp >= 5 else "fail")
    note = (" It ran above the expected range: better than the earlier trades suggested, so check the conditions changed for a "
            "reason you understand." if fp > 95 else " It fell below the expected range." if fp < 5 else "")
    return dict(verdict=verdict(st, f"Incubation from {split.date()}: {k} trades, net ${act[-1]:,.0f}, at the {fp:.0f}th percentile of what the "
                                    f"earlier trades predicted; {inside:.0f}% of the path inside the 5–95% band." + note),
                tables=[table("Before vs after the split", ["Period", "Trades", "Net $", "Avg net/trade $", "Win %"],
                              [[f"Before {split.date()}", len(pre), float(pre["net"].sum()), float(pre["net"].mean()), float((pre["net"] > 0).mean() * 100)],
                               [f"From {split.date()}", k, float(post["net"].sum()), float(post["net"].mean()), float((post["net"] > 0).mean() * 100)]])],
                charts=[chart("fan", "Incubation trades vs the expected range (cumulative USD)", x=list(range(k + 1)), p5=lo.round(0).tolist(),
                              p50=med.round(0).tolist(), p95=hi.round(0).tolist(), actual=act.round(0).tolist(), xlabel="incubation trade number", yprefix="$")])


def sc_position_sizing(ctx, p):
    fr, lev = _num_list(_p(p, "fractions")), _p(p, "max_leverage")
    _, _, t = ctx.base()
    if len(t) == 0:
        return dict(verdict=verdict("fail", "No trades."), tables=[], charts=[])
    pct = t.sort_values("entry")["pct"].values
    rows, curves = [], []
    for name, f in [(f"Fixed notional ${ctx.notional:,.0f}", None)] + [(f"Fixed fractional {x:g}%", x / 100) for x in fr]:
        eq, path, worst, sizes = ctx.capital, [ctx.capital], 5.0, []
        for v in pct:
            nn = ctx.notional if f is None else min(max(eq, 0) * f / (worst / 100), max(eq, 0) * lev)
            sizes.append(nn); eq += nn * v / 100; path.append(eq)
            worst = max(worst, -v)                       # risk unit learns only from closed trades
        a = np.array(path); pk = np.maximum.accumulate(a)
        rows.append([name, len(pct), float(a[-1] - ctx.capital), float((a[-1] / ctx.capital - 1) * 100), float(((a - pk) / pk).min() * 100),
                     float(np.mean(sizes)), float(np.max(sizes))])
        curves.append({"name": name, "x": list(range(len(a))), "y": [round(float(x), 0) for x in a]})
    if all(r[2] <= 0 for r in rows):
        st, txt = "fail", "No sizing plan makes money: sizing scales an edge, it cannot create one."
    else:
        best = max(rows, key=lambda r: r[3] / abs(r[4]) if r[4] < 0 else r[3])
        st, txt = "info", f"Best return per unit of drawdown: {best[0]} ({best[3]:.0f}% return, {best[4]:.1f}% max drawdown)."
    return dict(verdict=verdict(st, txt + " The first risk unit is a 5% loss until a larger one has been seen."),
                tables=[table("Same trades, different sizing", ["Plan", "Trades", "Net $", "Return %", "Max DD %", "Avg notional $", "Max notional $"], rows)],
                charts=[chart("lines", "Equity by sizing plan (USD)", curves=curves, xlabel="trade number", yprefix="$")])


def sc_cost_stress(ctx, p):
    costs = _num_list(_p(p, "costs"))
    ev, s, _ = ctx.base()
    rows, nets = [], []
    for c in costs:
        m = trade_metrics(ctx.trades(ev, s, cost_bps=c), ctx.years)
        rows.append(_row(f"{c:g} bps", m)); nets.append(m["net"])
    gross = trade_metrics(ctx.trades(ev, s, cost_bps=0), ctx.years)
    be = gross["pct"] / gross["trades"] * 100 if gross["trades"] else 0.0   # bps per trade that wipes out the gross edge
    at20 = next((n for c, n in zip(costs, nets) if c == 20), None)
    st = "pass" if be >= 20 else ("watch" if be >= ctx.cost_bps else "fail")
    return dict(verdict=verdict(st, f"Break-even cost ≈ {be:.0f} bps per trade (the panel uses {ctx.cost_bps:g})." +
                                (f" At 20 bps: ${at20:,.0f}." if at20 is not None else "")),
                tables=[table("Net profit as costs rise", COLS, rows)],
                charts=[chart("bar", "Net profit by cost (USD)", x=[f"{c:g}" for c in costs], series=[{"name": "net", "y": [round(n, 0) for n in nets]}],
                              xlabel="cost per trade (bps)", yprefix="$")])


def sc_entry_lag(ctx, p):
    lags = _num_list(_p(p, "lags"), int)
    rows, nets = [], []
    for lg in lags:
        ev = ctx.events(lag=lg)
        m = trade_metrics(ctx.trades(ev, ctx.signals(ev)), ctx.years)
        rows.append(_row(f"Lag {lg}", m)); nets.append(m["net"])
    st = "pass" if all(n > 0 for n in nets[:3]) else ("watch" if nets and nets[-1] > 0 else "fail")
    return dict(verdict=verdict(st, "Net at each lag: " + ", ".join(f"{l}: ${n:,.0f}" for l, n in zip(lags, nets)) +
                                ". Lag 0 uses the event-date return, as the Validation panel does."),
                tables=[table("Entry delay (trading days after the event)", COLS, rows)],
                charts=[chart("bar", "Net profit by entry lag (USD)", x=[str(l) for l in lags], series=[{"name": "net", "y": [round(n, 0) for n in nets]}], yprefix="$")])


def sc_consistency(ctx, p):
    _, _, t = ctx.base()
    if len(t) == 0:
        return dict(verdict=verdict("fail", "No trades."), tables=[], charts=[])
    y = t.groupby(t["exit"].dt.year)["net"].agg(["count", "sum"])
    q = t.groupby(t["exit"].dt.to_period("Q").astype(str))["net"].agg(["count", "sum"])
    share, need = float((q["sum"] > 0).mean()), _p(p, "pass_share")
    tot = t["net"].sum()
    top = float(t["net"].nlargest(max(1, len(t) // 10)).sum() / tot * 100) if tot > 0 else None
    st = "pass" if share >= need else ("watch" if share >= need - 0.15 else "fail")
    return dict(verdict=verdict(st, f"{share:.0%} of quarters profitable (pass at {need:.0%})" +
                                (f"; the best 10% of trades make {top:.0f}% of the total." if top is not None else ".")),
                tables=[table("By year", ["Year", "Trades", "Net $"], [[str(i), int(r["count"]), float(r["sum"])] for i, r in y.iterrows()]),
                        table("By quarter", ["Quarter", "Trades", "Net $"], [[str(i), int(r["count"]), float(r["sum"])] for i, r in q.iterrows()])],
                charts=[chart("bar", "Net profit by quarter (USD)", x=list(q.index), series=[{"name": "net", "y": [round(float(v), 0) for v in q["sum"]]}], yprefix="$")])


def sc_diversification(ctx, p):
    monthly = {}
    for hz in (7, 15, 30):
        try:
            ev = ctx.events(h=hz, hold=hz)
        except FileNotFoundError:
            continue
        t = ctx.trades(ev, ctx.signals(ev))
        if len(t):
            monthly[f"{hz}-day"] = t.groupby(t["exit"].dt.to_period("M"))["net"].sum()
    if len(monthly) < 2:
        return dict(verdict=verdict("info", "Needs KDE results for at least two horizons (7, 15, 30 days)."), tables=[], charts=[])
    pn = pd.DataFrame(monthly).fillna(0.0).sort_index()
    combo = pn.mean(axis=1)

    def stats(m):
        eq = ctx.capital + m.cumsum(); pk = eq.cummax()
        return [float(m.sum()), float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if m.std(ddof=1) > 0 else None,
                float(((eq - pk) / pk).min() * 100), float((m > 0).mean() * 100)]

    rows = [[c, *stats(pn[c])] for c in pn.columns] + [["Equal-capital combination", *stats(combo)]]
    corr = pn.corr()
    off = corr.values[np.triu_indices(len(pn.columns), 1)]
    avg_dd, combo_dd, combo_net = np.mean([r[3] for r in rows[:-1]]), rows[-1][3], rows[-1][1]
    st = "fail" if combo_net <= 0 else ("pass" if combo_dd >= avg_dd and np.mean(off) < 0.7 else "watch")
    return dict(verdict=verdict(st, f"Average correlation {np.mean(off):.2f}; combined max drawdown {combo_dd:.1f}% vs {avg_dd:.1f}% average; "
                                    f"combined net ${combo_net:,.0f}."),
                tables=[table("Monthly P&L by forecast horizon", ["Account", "Net $", "Sharpe (monthly)", "Max DD %", "Positive months %"], rows)],
                charts=[chart("heatmap", "Correlation of monthly P&L", x=list(pn.columns), y=list(pn.columns), z=corr.round(2).values.tolist(), zmin=-1, zmax=1)])


RUNNERS = {"baseline": sc_baseline, "entry_test": sc_entry_test, "monkey": sc_monkey, "reverse": sc_reverse, "sensitivity": sc_sensitivity,
           "walk_forward": sc_walk_forward, "monte_carlo": sc_monte_carlo, "incubation": sc_incubation, "position_sizing": sc_position_sizing,
           "cost_stress": sc_cost_stress, "entry_lag": sc_entry_lag, "consistency": sc_consistency, "diversification": sc_diversification}


def run_scenarios(cfg: MarketConfig, bt: dict, scenarios, prices: Optional[PriceData] = None, kde_frames: Optional[dict] = None, progress=None):
    say = progress or (lambda *_: None)
    if scenarios in ("all", ["all"]):
        scenarios = [c["id"] for c in CATALOG]
    items = [s if isinstance(s, dict) else {"id": s, "params": {}} for s in scenarios]
    bad = [s["id"] for s in items if s["id"] not in RUNNERS]
    if bad:
        raise ValueError(f"unknown scenario(s): {bad}")
    ctx = Ctx(cfg, bt, prices, kde_frames)
    out = []
    for i, s in enumerate(items):
        meta = CATALOG_BY_ID[s["id"]]
        say(f"scenario {i + 1}/{len(items)}: {meta['title']}")
        params = {**(s.get("params") or {}), "_id": s["id"]}
        try:
            r = RUNNERS[s["id"]](ctx, params)
        except Exception as e:  # one failing scenario must not hide the others
            r = dict(verdict=verdict("error", f"{type(e).__name__}: {e}"), tables=[], charts=[])
        out.append({"id": s["id"], "title": meta["title"], "source": meta["source"], "what": meta["what"],
                    "params": {q["key"]: _p(params, q["key"]) for q in meta["params"]}, **r})
    return {"market": cfg["display_name"], "focus": f"{ctx.h}-day forecast · band {ctx.band} · veto {ctx.veto} · {ctx.cost_bps:g} bps",
            "period": [str(ctx.start.date()), str(ctx.end.date())], "position_limit": "none (every event is its own trade)",
            "scenarios": out, "provenance": _provenance(ctx)}


def _provenance(ctx):
    import hashlib, json
    files = {}
    for hz in (7, 15, 30):
        f = ctx.cfg.path("kde_dir") / f"kde_v3_results_h{hz}.csv"
        if f.exists():
            files[f"kde_h{hz}"] = hashlib.sha256(f.read_bytes()).hexdigest()
    pf = ctx.cfg.path("prices")
    if pf is not None and pf.exists():
        files["prices"] = hashlib.sha256(pf.read_bytes()).hexdigest()
    return {"sha256": files, "config_sha256": hashlib.sha256(json.dumps(ctx.bt, sort_keys=True, default=str).encode()).hexdigest(),
            "seed": SEED}
