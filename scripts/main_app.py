import ast
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from wordcloud import WordCloud

from utils import add_indicators, load_futures

BASE_DIR = os.path.dirname(__file__)
EVENT_CSV = os.path.join(BASE_DIR, "causal_gold_articles_with_topics_bert.csv")

st.set_page_config(layout="wide", page_title="Commodity Event Intelligence")

st.title("🌐 Commodity Event Intelligence Dashboard")
st.markdown(
    "### Scenario-1: Event Memory Exploration  |  Scenario-2: Futures Quant Dashboard  |  Data Pipeline"
)
st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🧠 Scenario 1 — Event Memory", "🏆 Scenario 2 — Quant Dashboard", "🗂 Data Pipeline"]
)

# ============================================================
# SCENARIO 1 — 3D EVENT EMBEDDINGS + ANALYSIS
# ============================================================
with tab1:
    st.subheader("🪙 3D Visualization of Gold News Articles Over Time")

    @st.cache_data
    def load_event_data():
        df = pd.read_csv(EVENT_CSV)
        df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
        reduced = PCA(n_components=3).fit_transform(np.vstack(df["embedding"].values))
        df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"])

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

    @st.cache_data
    def load_futures_cached(ticker="GC=F", start="2000-01-01"):
        return add_indicators(load_futures(ticker, start))

    @st.cache_data
    def load_event_csv_cached():
        return pd.read_csv(EVENT_CSV) if os.path.exists(EVENT_CSV) else pd.DataFrame()

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

# ============================================================
# DATA PIPELINE — DOCUMENTATION / INGESTION STATUS
# ============================================================
with tab3:
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
           `causal_gold_articles_with_topics.csv` / `..._bert.csv`, which both Scenario tabs read.
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
