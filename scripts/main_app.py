import ast
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

from utils import add_indicators, load_futures

BASE_DIR = os.path.dirname(__file__)
EVENT_CSV = os.path.join(BASE_DIR, "causal_gold_articles_with_topics_bert.csv")

st.set_page_config(layout="wide", page_title="Commodity Event Intelligence")

st.title("🌐 Commodity Event Intelligence Dashboard")
st.markdown(
    "### Scenario-1: Event Memory  |  Scenario-2: Quant Dashboard  |  "
    "Clusters & Retrieval  |  Data Pipeline"
)
st.divider()

# ============================================================
# SHARED CACHED LOADERS (module scope — usable from any tab)
# ============================================================
@st.cache_data
def load_event_data():
    df = pd.read_csv(EVENT_CSV)
    df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    reduced = PCA(n_components=3).fit_transform(np.vstack(df["embedding"].values))
    df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


@st.cache_data
def load_event_csv_cached():
    return pd.read_csv(EVENT_CSV) if os.path.exists(EVENT_CSV) else pd.DataFrame()


@st.cache_data
def load_futures_cached(ticker="GC=F", start="2000-01-01"):
    return add_indicators(load_futures(ticker, start))


@st.cache_data
def compute_embedding_clusters(k):
    """KMeans over the full 384-d embeddings (not the 3D PCA projection)."""
    df = load_event_data()
    embeddings = np.vstack(df["embedding"].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


@st.cache_data
def compute_temporal_density():
    """KDE over event dates — same gaussian_kde used for the returns distribution
    in scenario2_quant_dashboard.py, applied here to event timestamps to surface
    bursts ('temporal clusters') of news activity."""
    df = load_event_data()
    day_nums = df["Date"].values.astype("datetime64[D]").astype(float)
    kde = gaussian_kde(day_nums)
    grid = np.linspace(day_nums.min(), day_nums.max(), 1500)
    density = kde(grid)
    grid_dates = pd.to_datetime(grid, unit="D")
    peak_idx, _ = find_peaks(density, distance=15, prominence=density.max() * 0.08)
    return grid_dates, density, grid_dates[peak_idx]


def retrieve_similar_events(df, idx, top_k=10):
    """Cosine-similarity retrieval — same approach already used for prototype/topic
    matching in gold_relvant_data_extraction.ipynb and gold_bert_topics.ipynb
    (`cosine_similarity(article_embeddings, prototype_embeddings)`), here applied
    event-to-event instead of event-to-prototype."""
    embeddings = np.vstack(df["embedding"].values)
    sims = cosine_similarity(embeddings[idx : idx + 1], embeddings)[0]
    order = [i for i in np.argsort(-sims) if i != idx][:top_k]
    out = df.iloc[order].copy()
    out["similarity"] = sims[order]
    return out


def forecast_price(dates, prices, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            prices.values, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
    mean_fc = model.forecast(horizon)
    resid_std = float(np.std(model.resid))
    band = resid_std * np.sqrt(np.arange(1, horizon + 1))
    future_dates = pd.bdate_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    return future_dates, mean_fc, mean_fc - 1.96 * band, mean_fc + 1.96 * band


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🧠 Scenario 1 — Event Memory",
        "🏆 Scenario 2 — Quant Dashboard",
        "🧬 Clusters & Retrieval",
        "🗂 Data Pipeline",
    ]
)

# ============================================================
# SCENARIO 1 — 3D EVENT EMBEDDINGS + ANALYSIS
# ============================================================
with tab1:
    st.subheader("🪙 3D Visualization of Gold News Articles Over Time")

    df_events = load_event_data()

    min_d, max_d = df_events["Date"].min().date(), df_events["Date"].max().date()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d, key="s1_start")
    end_date = col2.date_input("End Date", max_d, min_value=min_d, max_value=max_d, key="s1_end")

    mask = (df_events["Date"].dt.date >= start_date) & (df_events["Date"].dt.date <= end_date)
    filtered_events = df_events[mask]

    fig1 = px.scatter_3d(
        filtered_events,
        x="x", y="y", z="z",
        color="assigned_topic_bert",
        hover_name="Headline",
        hover_data={
            "Doc_ID": True,
            "gold_relevance_score": True,
            "topic_similarity": True,
            "assigned_topic_bert": True,
            "Date": True,
        },
        title=f"3D Projection of News Articles ({start_date} → {end_date})",
    )
    fig1.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark",
        paper_bgcolor="black",
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": True})
    st.markdown(f"**{len(filtered_events)}** articles displayed for the selected range.")

    if st.button("🔍 Analyse Event Window"):
        st.subheader("📊 Event Type Frequency (Topic-BERT)")
        topic_counts = filtered_events["assigned_topic_bert"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Frequency"]
        fig_topics = px.bar(
            topic_counts, x="Topic", y="Frequency", color="Frequency", text="Frequency",
            title="Event Frequency by Topic (BERT)", color_continuous_scale="sunset",
            template="plotly_dark",
        )
        fig_topics.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_topics, use_container_width=True)

        st.subheader("🔥 Major Causes (Keyword Frequency)")
        causes_text = " ".join(filtered_events["gold_cause"].dropna().astype(str).tolist())
        words = re.findall(r"[A-Za-z']+", causes_text)
        words = [w.lower() for w in words if len(w) > 3]
        common_words = Counter(words).most_common(20)

        if common_words:
            cause_df = pd.DataFrame(common_words, columns=["Cause Keyword", "Frequency"])
            fig_causes = px.bar(
                cause_df, x="Frequency", y="Cause Keyword", orientation="h", text="Frequency",
                title="Most Frequent Causes in Selected Period", color="Frequency",
                color_continuous_scale="thermal", template="plotly_dark",
            )
            st.plotly_chart(fig_causes, use_container_width=True)

            st.subheader("☁️ Word Cloud of Causes")
            wc = WordCloud(width=800, height=400, background_color="black", colormap="autumn")
            wc = wc.generate_from_frequencies(dict(common_words))
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("Not enough cause text in this window to summarize.")

        with st.expander("🔎 View Raw Event Data"):
            st.dataframe(
                filtered_events[
                    ["Date", "Headline", "assigned_topic_bert", "gold_cause", "gold_effect", "gold_relevance_score"]
                ].sort_values("Date")
            )

# ============================================================
# SCENARIO 2 — FUTURES QUANT DASHBOARD
# ============================================================
with tab2:
    st.subheader("🏆 Gold Futures — Technical & Event-Overlay Analysis")

    df_fut = load_futures_cached()
    event_df = load_event_csv_cached()

    min_d2, max_d2 = df_fut["Date"].min().date(), df_fut["Date"].max().date()
    col3, col4 = st.columns(2)
    start_d = col3.date_input("Start Date (Futures)", min_d2, min_value=min_d2, max_value=max_d2, key="s2_start")
    end_d = col4.date_input("End Date (Futures)", max_d2, min_value=min_d2, max_value=max_d2, key="s2_end")

    mask2 = (df_fut["Date"].dt.date >= start_d) & (df_fut["Date"].dt.date <= end_d)
    filt_df = df_fut[mask2]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_fut["Date"], y=df_fut["Price"], mode="lines",
                               line=dict(color="gray", width=1), name="Full History", opacity=0.2))
    fig2.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["Price"], mode="lines",
                               line=dict(color="gold", width=2.5), name="Selected Window"))
    fig2.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MA_7"], mode="lines",
                               line=dict(color="orange"), name="MA 7"))
    fig2.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MA_15"], mode="lines",
                               line=dict(color="deepskyblue"), name="MA 15"))
    fig2.add_trace(go.Scatter(x=filt_df.loc[filt_df["is_peak"], "Date"],
                               y=filt_df.loc[filt_df["is_peak"], "Price"],
                               mode="markers", marker=dict(size=8, color="crimson"), name="Peaks"))
    fig2.add_trace(go.Scatter(x=filt_df.loc[filt_df["is_trough"], "Date"],
                               y=filt_df.loc[filt_df["is_trough"], "Price"],
                               mode="markers", marker=dict(size=8, color="lime"), name="Troughs"))

    if not event_df.empty and "Date" in event_df.columns and not filt_df.empty:
        event_df["Date"] = pd.to_datetime(event_df["Date"], errors="coerce")
        mask_ev = (event_df["Date"].dt.date >= start_d) & (event_df["Date"].dt.date <= end_d)
        for _, ev in event_df[mask_ev].iterrows():
            fig2.add_trace(go.Scatter(
                x=[ev["Date"]], y=[filt_df["Price"].iloc[-1]],
                mode="markers+text",
                text=[ev.get("assigned_topic_bert", "Event")],
                textposition="top center",
                marker=dict(size=9, color="orange", symbol="diamond"),
                name="Event", showlegend=False,
            ))

    fig2.update_layout(
        title=f"Gold Futures ({start_d} → {end_d}) — MA, Peaks, Troughs & Events",
        hovermode="x unified", template="plotly_dark",
        plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True})

    st.markdown("### 📊 Summary for Selected Window")
    summary = filt_df.tail(30)
    colA, colB, colC = st.columns(3)
    colA.metric("Latest Price", f"{summary['Price'].iloc[-1]:,.2f}" if len(summary) else "—")
    colB.metric("30-Day Mean", f"{summary['Price'].mean():,.2f}" if len(summary) else "—")
    colC.metric("Volatility (%)", f"{summary['Pct_Change'].std():.2f}" if len(summary) else "—")
    st.markdown(
        f"<p style='text-align:center;color:gold;'>Highlighted window: <b>{start_d} → {end_d}</b></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("📉 Rolling Correlation (MA7 vs MA15)")
    st.plotly_chart(
        px.line(filt_df, x="Date", y="MA_corr", title="30-Day Rolling Correlation Between MA7 and MA15",
                 template="plotly_dark"),
        use_container_width=True,
    )

    st.subheader("⚡ RSI (14-day Momentum)")
    fig_rsi = px.line(filt_df, x="Date", y="RSI_14", title="RSI 14-Day Indicator", template="plotly_dark")
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, line_width=0)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.2, line_width=0)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("📊 MACD Indicator")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MACD"], mode="lines",
                                   name="MACD", line=dict(color="gold")))
    fig_macd.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["Signal"], mode="lines",
                                   name="Signal", line=dict(color="skyblue")))
    fig_macd.update_layout(template="plotly_dark", title="MACD vs Signal Line")
    st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader("📈 KDE of Daily % Returns")
    returns = filt_df["Pct_Change"].dropna()
    if len(returns) > 5:
        kde = gaussian_kde(returns)
        xs = np.linspace(returns.min(), returns.max(), 200)
        fig_kde = px.area(x=xs, y=kde(xs), template="plotly_dark",
                           labels={"x": "% Change", "y": "Density"},
                           title="Distribution of Daily % Returns")
        st.plotly_chart(fig_kde, use_container_width=True)
    else:
        st.info("Not enough data for KDE estimation.")

    if not event_df.empty and "Date" in event_df.columns:
        st.subheader("🌋 Volatility vs Event Count Overlay")
        mask_ev = (event_df["Date"].dt.date >= start_d) & (event_df["Date"].dt.date <= end_d)
        counts = event_df[mask_ev].groupby(event_df["Date"].dt.date).size()
        vol = df_fut.groupby(df_fut["Date"].dt.date)["Volatility_30"].mean()
        overlay = pd.DataFrame({"Volatility": vol, "Events": counts}).fillna(0)

        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["Volatility"],
                                          mode="lines", name="Volatility (σ%)", yaxis="y1"))
        fig_overlay.add_trace(go.Bar(x=overlay.index, y=overlay["Events"],
                                      name="Event Count", yaxis="y2", opacity=0.5))
        fig_overlay.update_layout(
            title="Volatility vs Event Count", template="plotly_dark",
            yaxis=dict(title="Volatility (%)", side="left"),
            yaxis2=dict(title="Event Count", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

    st.divider()
    snapshot = {
        "start_date": str(start_d),
        "end_date": str(end_d),
        "mean_price": float(filt_df["Price"].mean()) if len(filt_df) else None,
        "volatility": float(filt_df["Volatility_30"].mean()) if len(filt_df) else None,
        "num_peaks": int(filt_df["is_peak"].sum()),
        "num_troughs": int(filt_df["is_trough"].sum()),
    }
    st.download_button("📥 Export Snapshot (JSON)", json.dumps(snapshot, indent=2),
                        file_name="window_snapshot.json", mime="application/json")

    st.divider()
    st.subheader("🔮 Price Forecast (Holt's Damped-Trend Exponential Smoothing)")
    horizon = st.slider("Forecast horizon (business days)", 5, 90, 30, key="fc_horizon")
    train = filt_df.dropna(subset=["Price"])
    if len(train) < 30:
        st.info("Select a window with at least 30 trading days to fit a forecast.")
    else:
        future_dates, mean_fc, lower, upper = forecast_price(train["Date"], train["Price"], horizon)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=train["Date"], y=train["Price"], mode="lines",
                                     line=dict(color="gold", width=2), name="Historical (Selected Window)"))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(0,191,255,0.15)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% Interval",
        ))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_fc, mode="lines",
                                     line=dict(color="deepskyblue", width=2.5, dash="dash"), name="Forecast"))
        fig_fc.update_layout(
            title=f"Gold Futures Forecast — Next {horizon} Business Days",
            template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
            font=dict(color="white"), hovermode="x unified", margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        last_price = train["Price"].iloc[-1]
        end_price = mean_fc[-1]
        colF1, colF2, colF3 = st.columns(3)
        colF1.metric("Last Actual Price", f"{last_price:,.2f}")
        colF2.metric(f"Forecast (+{horizon}d)", f"{end_price:,.2f}", f"{(end_price / last_price - 1) * 100:+.2f}%")
        colF3.metric("95% Interval Width (end)", f"±{(upper[-1] - lower[-1]) / 2:,.2f}")

# ============================================================
# CLUSTERS & RETRIEVAL — embeddings, KMeans clusters, temporal
# bursts (KDE), per-cluster price windows, similarity retrieval
# ============================================================
with tab3:
    # reset_index(drop=True) so the positional row order here exactly matches
    # np.vstack(df_all["embedding"].values) — load_event_data()'s dropna() leaves
    # gaps in the original index, and several sections below rely on positional
    # (0..N-1) row lookups into the embeddings array.
    df_all = load_event_data().reset_index(drop=True)

    st.subheader("🧬 Embedding Clusters")
    st.caption(
        f"{len(df_all)} events · {np.vstack(df_all['embedding'].values).shape[1]}-dim embeddings "
        "(same `gold_general_embedding` vectors used for the 3D PCA view)."
    )
    k = st.slider("Number of clusters (KMeans)", 2, 12, 6, key="cluster_k")
    cluster_labels = compute_embedding_clusters(k)
    df_all["cluster"] = [f"Cluster {c}" for c in cluster_labels]

    fig_clusters = px.scatter_3d(
        df_all, x="x", y="y", z="z", color="cluster",
        hover_name="Headline",
        hover_data={"Date": True, "assigned_topic_bert": True, "cluster": True},
        title=f"KMeans Clusters (k={k}) over Event Embeddings",
    )
    fig_clusters.update_layout(
        scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
        margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark", paper_bgcolor="black",
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        size_df = df_all["cluster"].value_counts().reset_index()
        size_df.columns = ["Cluster", "Events"]
        st.plotly_chart(
            px.bar(size_df, x="Cluster", y="Events", color="Events", template="plotly_dark",
                   title="Cluster Sizes", color_continuous_scale="sunset"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.histogram(df_all, x="Date", color="cluster", nbins=60, template="plotly_dark",
                         title="Cluster Activity Over Time — Respective Time Series"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("⏳ Temporal Clusters — Event Bursts via KDE")
    st.caption(
        "Density estimated with `scipy.stats.gaussian_kde`, the same KDE routine already used for "
        "Daily % Returns in Scenario-2 — applied here to event dates to find bursts of news activity."
    )
    window_days = st.slider("Burst window (± days around each peak)", 3, 30, 10, key="burst_window")
    grid_dates, density, peak_dates = compute_temporal_density()

    fig_density = go.Figure()
    fig_density.add_trace(go.Scatter(x=grid_dates, y=density, mode="lines", fill="tozeroy",
                                      line=dict(color="gold"), name="Event Density"))
    fig_density.add_trace(go.Scatter(
        x=peak_dates, y=np.interp(peak_dates.astype(np.int64), grid_dates.astype(np.int64), density),
        mode="markers", marker=dict(size=9, color="crimson", line=dict(width=1, color="white")),
        name="Detected Bursts",
    ))
    fig_density.update_layout(
        title="Event Density Over Time (KDE) with Detected Bursts",
        template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(fig_density, use_container_width=True)

    cluster_rows = []
    for i, pk in enumerate(peak_dates):
        w_start, w_end = pk - pd.Timedelta(days=window_days), pk + pd.Timedelta(days=window_days)
        count = int(((df_all["Date"] >= w_start) & (df_all["Date"] <= w_end)).sum())
        cluster_rows.append({
            "Cluster": i, "Peak Date": pk.date(), "Window Start": w_start.date(),
            "Window End": w_end.date(), "Events": count,
        })
    temporal_df = pd.DataFrame(cluster_rows)
    st.dataframe(temporal_df, use_container_width=True)

    st.divider()
    st.subheader("📉 Respective Time-Series Data for a Temporal Cluster")
    if temporal_df.empty:
        st.info("No bursts detected at the current sensitivity.")
    else:
        choice = st.selectbox(
            "Choose a temporal cluster",
            temporal_df.index,
            format_func=lambda i: f"Cluster {i} — peak {temporal_df.loc[i, 'Peak Date']} "
                                   f"({temporal_df.loc[i, 'Events']} events)",
            key="temporal_choice",
        )
        row = temporal_df.loc[choice]
        df_fut_all = load_futures_cached()
        pad = pd.Timedelta(days=5)
        win_mask = (df_fut_all["Date"] >= pd.Timestamp(row["Window Start"]) - pad) & \
                   (df_fut_all["Date"] <= pd.Timestamp(row["Window End"]) + pad)
        win_price = df_fut_all[win_mask]

        fig_win = go.Figure()
        fig_win.add_trace(go.Scatter(x=win_price["Date"], y=win_price["Price"], mode="lines",
                                      line=dict(color="gold", width=2.5), name="Price"))
        fig_win.add_vrect(x0=row["Window Start"], x1=row["Window End"],
                           fillcolor="rgba(255,99,71,0.2)", line_width=0, annotation_text="Burst Window")
        fig_win.update_layout(
            title=f"Gold Futures Around Temporal Cluster {choice} ({row['Window Start']} → {row['Window End']})",
            template="plotly_dark", plot_bgcolor="black", paper_bgcolor="black",
            font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_win, use_container_width=True)

        win_events = df_all[(df_all["Date"] >= pd.Timestamp(row["Window Start"])) &
                             (df_all["Date"] <= pd.Timestamp(row["Window End"]))]
        with st.expander(f"📰 {len(win_events)} events in this burst window"):
            st.dataframe(
                win_events[["Date", "Headline", "assigned_topic_bert", "gold_cause", "cluster"]]
                .sort_values("Date"),
                use_container_width=True,
            )

    st.divider()
    st.subheader("🔍 Select an Event → Retrieve Similar Events")
    st.caption(
        "Retrieval reuses the `cosine_similarity(embeddings, prototype_embeddings)` pattern already "
        "in `gold_relvant_data_extraction.ipynb` / `gold_bert_topics.ipynb`, applied event-to-event."
    )
    df_sorted = df_all.sort_values("Date", ascending=False)
    labels = [f"{r.Date.date()} — {r.Headline[:90]}" for r in df_sorted.itertuples()]
    sel = st.selectbox("Event", range(len(labels)), format_func=lambda i: labels[i], key="retrieval_event")
    top_k = st.slider("How many similar events to retrieve", 5, 25, 10, key="retrieval_k")

    selected_row = df_sorted.iloc[sel]
    selected_idx = df_sorted.index[sel]  # position in df_all / embeddings (df_all has a clean 0..N-1 index)
    st.markdown(
        f"**Selected:** {selected_row['Headline']}  \n"
        f"*{selected_row['Date'].date()} — {selected_row['assigned_topic_bert']}*  \n"
        f"Cause: {selected_row.get('gold_cause', '—')}"
    )

    similar = retrieve_similar_events(df_all, selected_idx, top_k)
    st.dataframe(
        similar[["Date", "Headline", "assigned_topic_bert", "gold_cause", "similarity"]]
        .assign(similarity=lambda d: d["similarity"].round(3))
        .sort_values("similarity", ascending=False),
        use_container_width=True,
    )

# ============================================================
# DATA PIPELINE — DOCUMENTATION / INGESTION STATUS
# ============================================================
with tab4:
    st.subheader("🗂 How the Event Data Gets Here")
    st.markdown(
        """
        1. **`pdf to text.py`** — extracts raw text per page from the source WSJ/Factiva PDFs.
        2. **`text_to_csv.py`** — splits each text dump into articles via the table-of-contents
           headings, pulls out title/date/source, and writes `all_clean_articles.csv`.
        3. **`text to csv using llama.py`** — alternative path: prompts a local Ollama model
           (`llama3.1`) per article to extract the same fields as structured JSON/JSONL.
        4. Downstream notebooks (`gold_relvant_data_extraction.ipynb`, `gold_bert_topics.ipynb`)
           score gold-relevance, summarize cause/effect, embed, and assign topics — producing
           `causal_gold_articles_with_topics.csv` / `..._bert.csv`, which every tab above reads.
        """
    )

    txt_dir = os.path.join(BASE_DIR, "pdf_to_text data")
    data_dir = os.path.join(os.path.dirname(BASE_DIR), "data")

    colX, colY = st.columns(2)
    with colX:
        n_txt = len(
            [f for f in os.listdir(txt_dir) if f.lower().endswith(".txt")]
        ) if os.path.isdir(txt_dir) else 0
        st.metric("Extracted article text files", n_txt)
    with colY:
        st.metric("Gold events in current dashboard CSV", len(load_event_data()))

    if os.path.isdir(data_dir):
        with st.expander("📁 Files in data/"):
            for f in sorted(os.listdir(data_dir)):
                st.write(f"- {f}")

    if os.path.isdir(txt_dir):
        sample_files = sorted(f for f in os.listdir(txt_dir) if f.lower().endswith(".txt"))[:5]
        if sample_files:
            with st.expander("📄 Preview a sample extracted text file"):
                chosen = st.selectbox("File", sample_files, key="pipeline_preview")
                with open(os.path.join(txt_dir, chosen), encoding="utf-8", errors="ignore") as fh:
                    st.text(fh.read()[:3000])

st.divider()
st.markdown("© Brahmanda SaaS — Event-Driven Market Intelligence")
