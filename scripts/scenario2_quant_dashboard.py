import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
import os, json
from utils import load_futures, add_indicators
import yfinance as yf
  # Clears all locally cached Yahoo data


st.set_page_config(layout="wide", page_title="Scenario-2 Quant Dashboard")
st.title("📊 Scenario-2: Quantitative Analysis of Gold Futures")

# ---------- LOAD FUTURES ----------
df = load_futures("GC=F", start="2000-01-01")
df = add_indicators(df)

# ---------- SELECT DATE WINDOW ----------
min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
st.sidebar.header("🕒 Select Time Window")
date_window = st.sidebar.slider(
    "Select Range:",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM-DD"
)
start_d, end_d = date_window


mask = (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
filt_df = df[mask]

# ---------- LOAD EVENT CSV ----------
event_path = os.path.join(os.path.dirname(__file__), "causal_gold_articles_with_topics_bert.csv")
event_df = pd.read_csv(event_path) if os.path.exists(event_path) else pd.DataFrame()

# ---------- MAIN FUTURES PLOT ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode="lines",
                         line=dict(color="gray", width=1), name="Full History", opacity=0.2))
fig.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["Price"],
                         mode="lines", line=dict(color="gold", width=2.5), name="Selected Window"))
fig.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MA_7"],
                         mode="lines", line=dict(color="orange"), name="MA 7"))
fig.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MA_15"],
                         mode="lines", line=dict(color="deepskyblue"), name="MA 15"))
fig.add_trace(go.Scatter(x=filt_df.loc[filt_df["is_peak"], "Date"],
                         y=filt_df.loc[filt_df["is_peak"], "Price"],
                         mode="markers", marker=dict(size=8, color="crimson"), name="Peaks"))
fig.add_trace(go.Scatter(x=filt_df.loc[filt_df["is_trough"], "Date"],
                         y=filt_df.loc[filt_df["is_trough"], "Price"],
                         mode="markers", marker=dict(size=8, color="lime"), name="Troughs"))

# ---- Event markers (from CSV) ----
if not event_df.empty and "Date" in event_df.columns:
    event_df["Date"] = pd.to_datetime(event_df["Date"], errors="coerce")
    mask_ev = (event_df["Date"].dt.date >= start_d) & (event_df["Date"].dt.date <= end_d)
    for _, ev in event_df[mask_ev].iterrows():
        fig.add_trace(go.Scatter(
            x=[ev["Date"]], y=[filt_df["Price"].iloc[-1]],
            mode="markers+text",
            text=[ev.get("assigned_topic_bert", "Event")],
            textposition="top center",
            marker=dict(size=9, color="orange", symbol="diamond"),
            name="Event"
        ))

fig.update_layout(
    title=f"Gold Futures ({start_d} → {end_d}) with MA, Peaks, Troughs, and Events",
    hovermode="x unified", template="plotly_dark",
    plot_bgcolor="black", paper_bgcolor="black",
    font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0)
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# ---------- ROLLING CORRELATION ----------
st.subheader("📉 Rolling Correlation (MA7 vs MA15)")
fig_corr = px.line(filt_df, x="Date", y="MA_corr",
                   title="30-Day Rolling Correlation Between MA7 and MA15",
                   template="plotly_dark")
st.plotly_chart(fig_corr, use_container_width=True)

# ---------- RSI ----------
st.subheader("⚡ RSI (14-day Momentum)")
fig_rsi = px.line(filt_df, x="Date", y="RSI_14",
                  title="RSI 14-Day Indicator", template="plotly_dark")
fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, line_width=0)
fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.2, line_width=0)
st.plotly_chart(fig_rsi, use_container_width=True)

# ---------- MACD ----------
st.subheader("📊 MACD Indicator")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["MACD"], mode="lines",
                              name="MACD", line=dict(color="gold")))
fig_macd.add_trace(go.Scatter(x=filt_df["Date"], y=filt_df["Signal"], mode="lines",
                              name="Signal", line=dict(color="skyblue")))
fig_macd.update_layout(template="plotly_dark", title="MACD vs Signal Line")
st.plotly_chart(fig_macd, use_container_width=True)

# ---------- KDE OF RETURNS ----------
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

# ---------- VOLATILITY vs EVENT COUNT ----------
if not event_df.empty and "Date" in event_df.columns:
    st.subheader("🌋 Volatility vs Event Count Overlay")
    event_df["Date"] = pd.to_datetime(event_df["Date"], errors="coerce")
    mask_ev = (event_df["Date"].dt.date >= start_d) & (event_df["Date"].dt.date <= end_d)
    counts = event_df[mask_ev].groupby(event_df["Date"].dt.date).size()
    vol = df.groupby(df["Date"].dt.date)["Volatility_30"].mean()
    overlay = pd.DataFrame({"Volatility": vol, "Events": counts}).fillna(0)

    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Scatter(x=overlay.index, y=overlay["Volatility"],
                                     mode="lines", name="Volatility (σ%)", yaxis="y1"))
    fig_overlay.add_trace(go.Bar(x=overlay.index, y=overlay["Events"],
                                 name="Event Count", yaxis="y2", opacity=0.5))
    fig_overlay.update_layout(
        title="Volatility vs Event Count",
        template="plotly_dark",
        yaxis=dict(title="Volatility (%)", side="left"),
        yaxis2=dict(title="Event Count", overlaying="y", side="right")
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

# ---------- EXPORT SNAPSHOT ----------
snapshot = {
    "start_date": str(start_d),
    "end_date": str(end_d),
    "mean_price": float(filt_df["Price"].mean()),
    "volatility": float(filt_df["Volatility_30"].mean()),
    "num_peaks": int(filt_df["is_peak"].sum()),
    "num_troughs": int(filt_df["is_trough"].sum()),
}
st.download_button("📥 Export Snapshot (JSON)",
                   json.dumps(snapshot, indent=2),
                   file_name="window_snapshot.json",
                   mime="application/json")
