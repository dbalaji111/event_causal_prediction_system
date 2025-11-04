import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

st.set_page_config(layout="wide", page_title="Gold Futures Interactive Dashboard")
st.title("🏆 Gold Futures Quantile & Peaks with Interactive Time Window")

# ---- Load Data ----
@st.cache_data
def load_gold_futures():
    df = yf.download("GC=F", start="2000-01-01", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "Price"}, inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    return df

df = load_gold_futures()

# ---- Compute Quantiles ----
df["Q25"] = df["Price"].rolling(30).quantile(0.25)
df["Q50"] = df["Price"].rolling(30).quantile(0.5)
df["Q75"] = df["Price"].rolling(30).quantile(0.75)
df["Pct_Change"] = df["Price"].pct_change() * 100

# ---- Detect Peaks ----
peaks, _ = find_peaks(df["Price"].to_numpy().flatten(), distance=5, prominence=5)
df["is_peak"] = False
df.loc[df.index[peaks], "is_peak"] = True

# ---- Date Range Selector ----
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
st.sidebar.header("⏱ Select Time Window")
date_window = st.sidebar.slider(
    "Choose range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

mask = (df["Date"].dt.date >= date_window[0]) & (df["Date"].dt.date <= date_window[1])
filtered_df = df[mask]

# ---- Plot ----
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Price"],
    mode="lines", line=dict(color="gray", width=1),
    name="Full History", opacity=0.2
))

# Highlight selected window in gold
fig.add_trace(go.Scatter(
    x=filtered_df["Date"], y=filtered_df["Price"],
    mode="lines", line=dict(color="gold", width=2.5),
    name="Selected Window"
))

# Quantile shading
fig.add_trace(go.Scatter(
    x=pd.concat([filtered_df["Date"], filtered_df["Date"][::-1]]),
    y=pd.concat([filtered_df["Q75"], filtered_df["Q25"][::-1]]),
    fill="toself", fillcolor="rgba(255,215,0,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip", name="30-Day Quantile Range"
))

# Peaks in the window
window_peaks = filtered_df[filtered_df["is_peak"]]
fig.add_trace(go.Scatter(
    x=window_peaks["Date"], y=window_peaks["Price"],
    mode="markers", marker=dict(size=8, color="crimson", line=dict(width=1, color="white")),
    name="Detected Peaks"
))

fig.update_layout(
    title=f"Gold Futures Price ({date_window[0]} → {date_window[1]})",
    hovermode="x unified",
    xaxis_title="Date", yaxis_title="Price (USD/oz)",
    template="plotly_dark",
    plot_bgcolor="black", paper_bgcolor="black",
    font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ---- Summary for Selected Window ----
st.markdown("### 📊 Summary for Selected Window")
summary = filtered_df.tail(30)  # last 30 observations inside window
col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", f"{summary['Price'].iloc[-1]:,.2f}")
col2.metric("30-Day Mean", f"{summary['Price'].mean():,.2f}")
col3.metric("Volatility (%)", f"{summary['Pct_Change'].std():.2f}")

st.markdown(
    f"<p style='text-align:center;color:gold;'>Highlighted window: "
    f"<b>{date_window[0]} → {date_window[1]}</b></p>",
    unsafe_allow_html=True
)
