"""Streamlit front-end for redflag_kit (oil market): the visual layer the kit's README says is missing.
Calls the retrieval / KDE / scoring / backtest modules directly in-process — no FastAPI server needed."""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from demo.generate_demo_data import build_demo_market
from redflag_kit import kde as kde_mod
from redflag_kit import scoring
from redflag_kit.backtest import run_backtest
from redflag_kit.config import MarketConfig
from redflag_kit.prices import PriceData
from redflag_kit.retrieval import Retriever

MARKET_JSON = BASE_DIR / "markets" / "oil.json"
DEMO_ROOT = BASE_DIR / "demo" / "data_root"

st.set_page_config(layout="wide", page_title="Red Flag — Oil Intelligence")
st.title("🛢️ Red Flag — Oil Market Intelligence Dashboard")
st.markdown(
    "### Event Analyzer  |  Price & Curve  |  Backtest  |  Corpus Explorer  |  About the Pipeline"
)
st.caption("Retrieval → KDE → Scoring → Backtest, the same `redflag_kit` pipeline, now with charts.")
st.divider()

with st.sidebar:
    st.header("⚙️ Data Source")
    use_demo = st.toggle("Use synthetic demo data", value=True)
    if use_demo:
        raw_paths = MarketConfig.load(MARKET_JSON).d["paths"]
        build_demo_market(DEMO_ROOT, raw_paths)
        root_str = str(DEMO_ROOT)
        st.caption(
            "Random-walk prices + invented headlines built from oil.json's own keyword lists — "
            "for exercising the pipeline only, not a forecast of anything real."
        )
    else:
        root_str = st.text_input(
            "Market root folder",
            value=str(BASE_DIR),
            help="Folder containing the real files at the paths markets/oil.json expects "
                 "(train_corpus, test_corpus, prices, vol_index).",
        )
    st.caption("Config: `markets/oil.json` (same file either way — only the data root changes).")


@st.cache_resource(show_spinner="Loading market data and building the retriever...")
def load_engines(root_str: str):
    cfg = MarketConfig.load(MARKET_JSON, root=Path(root_str))
    problems = cfg.validate()
    if problems:
        raise ValueError("Config problems: " + "; ".join(problems))
    retr = Retriever.from_config(cfg)
    pxdata = PriceData(cfg)
    return cfg, retr, pxdata


try:
    cfg, retr, pxdata = load_engines(root_str)
except Exception as exc:
    st.error(f"Could not load market data from `{root_str}`: {exc}")
    st.stop()

test_events = sorted(retr.test, key=lambda r: r["date"], reverse=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📰 Event Analyzer", "📈 Price & Curve", "🧮 Backtest", "🧬 Corpus Explorer", "ℹ️ About the Pipeline"]
)

# ============================================================
# EVENT ANALYZER — retrieval -> KDE -> scoring for one event
# ============================================================
with tab1:
    st.subheader("📰 Select a Test-Corpus Event")
    labels = [f"{e['date'].date()} — {e['text']}" for e in test_events]
    sel = st.selectbox("Event", range(len(labels)), format_func=lambda i: labels[i], key="ev_sel")
    event = test_events[sel]

    cl = scoring.classify_headline(cfg, event["text"])
    sev_label = cfg["severity_labels"][str(cl["severity"])]
    colA, colB, colC = st.columns(3)
    colA.metric("Event Type", cl["event_type"])
    colB.metric("Severity", f"{cl['severity']} — {sev_label}")
    colC.metric("Date", str(event["date"].date()))

    horizon = st.select_slider("Forecast horizon (days)", options=cfg["kde"]["horizons"],
                                value=cfg["kde"]["horizons"][1], key="ev_horizon")

    retrieval_res = retr.retrieve_adaptive(event, cfg["retrieval"]["tau"], cfg["kde"]["min_analogues"])
    st.caption(
        f"Retrieved **{retrieval_res['n_analogues']}** analogues (τ used: {retrieval_res['tau_used']:.2f}) via "
        "cosine similarity on L2-normalised embeddings, restricted to dates before this event — "
        "`redflag_kit.retrieval.Retriever.retrieve_adaptive`."
    )

    kde_result, det, grid = kde_mod.run_event(
        cfg, event["event_id"], event["date"], retrieval_res["analogues"], pxdata, horizon
    )

    if "error" in kde_result:
        st.warning(f"Not enough analogues for a KDE posterior at this horizon ({kde_result.get('n_analogues', 0)}).")
    else:
        x, dens = grid
        stats = {k[4:]: v for k, v in kde_result.items() if k.startswith("kde_")}

        fig_kde = go.Figure()
        fig_kde.add_trace(go.Scatter(x=x, y=dens, mode="lines", fill="tozeroy",
                                      line=dict(color="goldenrod"), name="Posterior density"))
        for q, color in [("p5", "crimson"), ("median", "white"), ("p95", "lime")]:
            fig_kde.add_vline(x=stats[q], line=dict(color=color, dash="dot"),
                               annotation_text=f"{q}={stats[q]:+.1f}%", annotation_font_color=color)
        fig_kde.update_layout(
            title=f"KDE Posterior of {horizon}-Day Forward Return — Bayesian Mixture Over {kde_result['n_analogues']} Analogues",
            template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
            font=dict(color="white"), xaxis_title="Forward Return (%)", yaxis_title="Density",
        )
        st.plotly_chart(fig_kde, use_container_width=True)

        colD, colE, colF, colG = st.columns(4)
        colD.metric("Median Return", f"{stats['median']:+.2f}%")
        colE.metric("p5 / p95", f"{stats['p5']:+.1f}% / {stats['p95']:+.1f}%")
        colF.metric("P(down >10%)", f"{stats['p_down10'] * 100:.1f}%")
        colG.metric("Red Flag", "🚩 YES" if kde_result["red_flag"] else "No")

        st.markdown(
            f"**Signal:** `{kde_result['signal']}`  ·  **Bimodal:** {'Yes' if stats['is_bimodal'] else 'No'}  ·  "
            f"**τ effective:** {kde_result['tau_effective']}  ·  "
            f"**Covers 90% band (realised):** {kde_result.get('covers_90', '—')}"
        )

        st.divider()
        st.subheader("🔗 Retrieved Analogues")
        an_df = pd.DataFrame(retrieval_res["analogues"])
        rcol = f"r{horizon}" if f"r{horizon}" in an_df.columns else "r15"
        fig_an = px.scatter(
            an_df, x="similarity", y=rcol, color="event_type", size="recency_weight",
            hover_data=["date", "event_id", "quantile_cat"], template="plotly_dark",
            title=f"Analogue Similarity vs Realised {horizon}-Day Forward Return",
        )
        st.plotly_chart(fig_an, use_container_width=True)

        with st.expander(f"📋 {len(an_df)} analogues (table)"):
            st.dataframe(
                an_df[["event_id", "date", "event_type", "similarity", "recency_weight", "quantile_cat",
                       "r7", "r15", "r30"]].sort_values("similarity", ascending=False),
                use_container_width=True,
            )

        st.divider()
        st.subheader("🎯 Dashboard v6 Score (ported 1:1 to `scoring.py`)")
        kde_fields = {k: v for k, v in kde_result.items() if k.startswith("kde_") or k.startswith("base_")}
        score = scoring.score_event(cfg, event["text"], retrieval_res, kde_fields)

        fig_score = go.Figure(go.Bar(
            x=["Informational", "Behavioural", "Structural"],
            y=[score["informational"]["total"], score["behavioural"]["total"], score["structural"]["total"]],
            marker_color=["#4c78a8", "#f58518", "#54a24b"], text=[score["informational"]["total"],
            score["behavioural"]["total"], score["structural"]["total"]], textposition="outside",
        ))
        fig_score.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                                 font=dict(color="white"), title="Score Breakdown (0-100)", yaxis_range=[0, 100])
        st.plotly_chart(fig_score, use_container_width=True)

        colH, colI = st.columns(2)
        colH.metric("Composite Score", f"{score['composite']['score']} ({score['composite']['label']})")
        colI.metric("Conviction", score["composite"]["conviction"])
        st.markdown(
            f"### 🎫 Ticket: **{score['action']}**  ·  Behaviour mode: **{score['behavioural']['mode']}** "
            f"(over-react {score['behavioural']['over']:.0%} / under-react {score['behavioural']['under']:.0%})"
        )

# ============================================================
# PRICE & CURVE
# ============================================================
with tab2:
    st.subheader(f"📈 {cfg['display_name']} — Price, Curve & Classified Events")
    df = pxdata.df.reset_index()

    ev_rows = []
    for e in retr.test:
        c = scoring.classify_headline(cfg, e["text"])
        ev_rows.append({"Date": e["date"], "Price": pxdata.px_at(e["date"]), "Type": c["event_type"],
                         "Severity": c["severity"], "Headline": e["text"]})
    ev_df = pd.DataFrame(ev_rows).dropna(subset=["Price"])

    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=df["Date"], y=df["PX"], mode="lines",
                                 line=dict(color="goldenrod", width=1.5), name=cfg["ticker_label"]))
    fig_px.add_trace(go.Scatter(
        x=ev_df["Date"], y=ev_df["Price"], mode="markers",
        marker=dict(size=8, color=ev_df["Severity"], colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Severity"), line=dict(width=1, color="white")),
        text=ev_df["Headline"], name="Classified Events",
    ))
    fig_px.update_layout(
        template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
        title=f"{cfg['ticker_label']} ({cfg['price_unit']}) with Classified News Events", hovermode="x unified",
    )
    st.plotly_chart(fig_px, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "SPREAD" in df.columns:
            fig_sp = px.line(df, x="Date", y="SPREAD", template="plotly_dark", title="Curve Spread (Structural Regime)")
            fig_sp.add_hrect(y0=-cfg["dashboard"]["structural"]["curve_flat_band"],
                              y1=cfg["dashboard"]["structural"]["curve_flat_band"],
                              fillcolor="gray", opacity=0.2, line_width=0, annotation_text="Flat band")
            st.plotly_chart(fig_sp, use_container_width=True)
    with col2:
        if "VOL" in df.columns:
            fig_vol = px.line(df, x="Date", y="VOL", template="plotly_dark",
                               title=f"{cfg['dashboard']['vol_label']} Volatility Index")
            fig_vol.add_hline(y=cfg["dashboard"]["structural"]["vol_lo"], line=dict(color="red", dash="dot"),
                               annotation_text="structural vol floor")
            st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("📊 Event Type Mix")
    st.plotly_chart(
        px.histogram(ev_df, x="Type", color="Type", template="plotly_dark", title="Classified Test Events by Type"),
        use_container_width=True,
    )

# ============================================================
# BACKTEST
# ============================================================
with tab3:
    st.subheader("🧮 Backtest — KDE Median Strategy (`redflag_kit.backtest.run_backtest`)")
    horizon_bt = st.select_slider("Horizon (days)", options=cfg["kde"]["horizons"],
                                   value=cfg["backtest"]["horizon"], key="bt_horizon")
    st.caption(
        f"Rule: trade the KDE median with a ±{cfg['backtest']['no_trade_band']}% dead-band, veto if "
        f"P(down>10%) > {cfg['backtest']['tail_veto']}, {cfg['backtest']['cost_bps']} bps flat cost per trade."
    )

    if st.button("▶ Run Backtest Across Test Corpus", key="run_bt"):
        rows = []
        prog = st.progress(0.0, text="Scoring test events...")
        test_sorted = sorted(retr.test, key=lambda r: r["date"])
        for i, e in enumerate(test_sorted):
            r_res = retr.retrieve_adaptive(e, cfg["retrieval"]["tau"], cfg["kde"]["min_analogues"])
            res, _, _ = kde_mod.run_event(cfg, e["event_id"], e["date"], r_res["analogues"], pxdata, horizon_bt)
            rows.append(res)
            prog.progress((i + 1) / len(test_sorted))
        prog.empty()
        kdf = pd.DataFrame(rows)
        bt_cfg = dict(cfg["backtest"])
        bt_cfg["horizon"] = horizon_bt
        valid_dates = pd.to_datetime(kdf["base_date"], errors="coerce").dropna()
        bt_cfg["start_date"] = str(valid_dates.min().date()) if len(valid_dates) else "2024-01-01"
        bt_cfg["end_date"] = str(valid_dates.max().date()) if len(valid_dates) else "2026-04-28"
        st.session_state["bt_result"] = run_backtest(kdf, bt_cfg)
        st.session_state["bt_n_scored"] = len(kdf)

    result = st.session_state.get("bt_result")
    if result is None:
        st.info("Run the backtest to see results (loops the KDE pipeline over every test-corpus event).")
    else:
        m = result["metrics"]
        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Events Scored", st.session_state.get("bt_n_scored", "—"))
        colB.metric("Trades", m["direction"]["n_trades"])
        colC.metric("Win Rate", f"{m['direction']['win_rate']:.1f}%")
        colD.metric("Sharpe", m["trading"]["sharpe_ratio"])
        colE.metric("90% Coverage", f"{m['calibration']['coverage_90pct']:.1f}%")

        fig_eq = go.Figure(go.Scatter(x=result["dates"], y=result["equity_curve"], mode="lines",
                                       line=dict(color="lime", width=2), fill="tozeroy", name="Cumulative Return"))
        fig_eq.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                              font=dict(color="white"), title="Equity Curve (Cumulative Return, %)")
        st.plotly_chart(fig_eq, use_container_width=True)

        colF, colG, colH = st.columns(3)
        colF.metric("Max Drawdown", f"{m['trading']['max_drawdown_pct']:.2f}%")
        colG.metric("Flagged Mean Return", f"{m['red_flag_utility']['flagged_mean_r']:.2f}%")
        colH.metric("Unflagged Mean Return", f"{m['red_flag_utility']['unflagged_mean_r']:.2f}%")

        with st.expander("📋 Full metrics JSON"):
            st.json(m)

# ============================================================
# CORPUS EXPLORER — embeddings, clusters, temporal density
# ============================================================
with tab4:
    st.subheader("🧬 Train-Corpus Embeddings")
    train = retr.train
    embeddings = np.vstack([r["embedding"] for r in train])
    reduced = PCA(n_components=3).fit_transform(embeddings)
    corpus_df = pd.DataFrame({
        "x": reduced[:, 0], "y": reduced[:, 1], "z": reduced[:, 2],
        "event_type": [r["event_type"] for r in train],
        "date": [r["date"] for r in train],
        "text": [r["text"] for r in train],
        "quantile_cat": [r["quantile_cat"] for r in train],
    })
    fig_3d = px.scatter_3d(
        corpus_df, x="x", y="y", z="z", color="event_type", hover_name="text",
        hover_data={"date": True, "quantile_cat": True}, template="plotly_dark",
        title=f"{len(train)} Train-Corpus Events — 768-d BERT Embeddings (PCA to 3D)",
    )
    fig_3d.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
                          paper_bgcolor="black", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        type_counts = corpus_df["event_type"].value_counts().reset_index()
        type_counts.columns = ["Event Type", "Count"]
        st.plotly_chart(
            px.bar(type_counts, x="Event Type", y="Count", color="Count", template="plotly_dark",
                   title="Event Type Mix (Train Corpus)", color_continuous_scale="sunset"),
            use_container_width=True,
        )
    with col2:
        st.subheader("⏳ Event Density Over Time (KDE)")
        day_nums = corpus_df["date"].values.astype("datetime64[D]").astype(float)
        kernel = gaussian_kde(day_nums)
        grid_x = np.linspace(day_nums.min(), day_nums.max(), 500)
        dens = kernel(grid_x)
        fig_dens = go.Figure(go.Scatter(x=pd.to_datetime(grid_x, unit="D"), y=dens, mode="lines",
                                         fill="tozeroy", line=dict(color="goldenrod")))
        fig_dens.update_layout(template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
                                font=dict(color="white"), title="Event Density Over Time")
        st.plotly_chart(fig_dens, use_container_width=True)

# ============================================================
# ABOUT THE PIPELINE
# ============================================================
with tab5:
    st.subheader("ℹ️ How This Dashboard Works")
    st.markdown(
        f"""
        This is the visual layer `redflag_kit`'s README says is missing — everything below calls the kit's
        own modules directly (no FastAPI server, no intermediate files):

        1. **Retrieval** (`redflag_kit.retrieval.Retriever`) — cosine similarity on L2-normalised 768-d
           BERT embeddings, restricted to analogues strictly before the event date, τ-ladder relaxation
           until enough candidates exist.
        2. **KDE** (`redflag_kit.kde`) — per-analogue Gaussian prior (mean = realised forward return,
           width = path volatility), a 3-factor weight (similarity^0.5 × quantile-match^0.2 × DTW-match^0.3),
           mixed into a posterior over a {cfg['kde']['n_grid']}-point grid. Red-flagged if
           P(down >{cfg['kde']['tail_threshold_pct']}%) > {cfg['kde']['red_flag_p_down']}.
        3. **Scoring** (`redflag_kit.scoring`) — a direct Python port of the dashboard v6 JS scoring
           (informational / behavioural / structural → composite → trade ticket), parity-tested against
           the original JS in the kit's test suite.
        4. **Backtest** (`redflag_kit.backtest`) — trades the KDE median with a dead-band and tail veto,
           reports Sharpe, win rate, drawdown, and 90%/50% interval coverage.

        **Market config:** `markets/oil.json` — every threshold above (τ, horizons, score weights, backtest
        rules) lives there, not in this dashboard's code.

        **Data:** the toggle in the sidebar switches between synthetic demo data (random-walk prices +
        invented headlines, generated by `demo/generate_demo_data.py`) and a real data root. To use your
        own oil corpus, point the root at a folder containing, relative to it:
        """
    )
    st.code(
        "\n".join(f"{k}: {v}" for k, v in MarketConfig.load(MARKET_JSON).d["paths"].items()),
        language="yaml",
    )
    st.markdown(
        "Known limitations carried over from the kit (see `REDFLAG_SCHEMA.md` section 12): the oil "
        "defaults are calibrated from the original v5/v6 system, tested here only on synthetic data; "
        "`dtw_mode: legacy_full_path` reproduces the original engine's outcome leakage (use "
        "`pre_event_only` for live forecasting) and the backtest's Sharpe assumes independent events."
    )

st.divider()
st.markdown(f"© Red Flag — {cfg['display_name']} Market Intelligence (demo dashboard over `redflag_kit`)")
