import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import ast, os

st.set_page_config(layout="wide", page_title="Commodity Event Intelligence")

st.title("🌐 Commodity Event Intelligence Dashboard")
st.markdown("### Scenario-1: 3D Event Memories  |  Scenario-2: Futures Time-Series Quantile View")

st.divider()

# ============ SCENARIO 1 : 3D EVENT EMBEDDINGS ============
st.subheader("🪙 Scenario-1 — 3D Visualization of Gold News Articles Over Time")

@st.cache_data
def load_event_data():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "causal_gold_articles_with_topics_bert.csv")
    df = pd.read_csv(path)
    df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    embeddings = np.vstack(df["embedding"].values)
    reduced = PCA(n_components=3).fit_transform(embeddings)
    df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])

df_events = load_event_data()

# ---- Date selection ----
min_d, max_d = df_events["Date"].min().date(), df_events["Date"].max().date()
col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end_date = col2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)

mask = (df_events["Date"].dt.date >= start_date) & (df_events["Date"].dt.date <= end_date)
filtered_events = df_events[mask]

fig1 = px.scatter_3d(
    filtered_events, x="x", y="y", z="z",
    color="assigned_topic",
    hover_name="Headline",
    hover_data={
        "Doc_ID": True,
        "gold_relevance_score": True,
        "topic_similarity": True,
        "assigned_topic": True,
        "Date": True,
    },
    title=f"3D Projection of News Articles ({start_date} → {end_date})"
)
fig1.update_layout(scene=dict(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3"),
                   margin=dict(l=0, r=0, b=0, t=40))

st.plotly_chart(fig1, use_container_width=True, height=700)
st.markdown(f"**{len(filtered_events)}** articles displayed for the selected range.")
st.divider()

# ============ SCENARIO 2 : TIME SERIES ============
st.subheader("🏆 Scenario-2 — Gold Futures Quantile & Peaks with Interactive Window")

@st.cache_data
def load_gold_futures():
    df = yf.download("GC=F", start="2000-01-01", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "Price"}, inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Price"])

df_fut = load_gold_futures()
df_fut["Q25"] = df_fut["Price"].rolling(30).quantile(0.25)
df_fut["Q50"] = df_fut["Price"].rolling(30).quantile(0.5)
df_fut["Q75"] = df_fut["Price"].rolling(30).quantile(0.75)
df_fut["Pct_Change"] = df_fut["Price"].pct_change() * 100
peaks, _ = find_peaks(df_fut["Price"].to_numpy().flatten(), distance=5, prominence=5)
df_fut["is_peak"] = False
df_fut.loc[df_fut.index[peaks], "is_peak"] = True

# Date range controls
min_d2, max_d2 = df_fut["Date"].min().date(), df_fut["Date"].max().date()
col3, col4 = st.columns(2)
start2 = col3.date_input("Start Date (Futures)", min_d2, min_value=min_d2, max_value=max_d2)
end2 = col4.date_input("End Date (Futures)", max_d2, min_value=min_d2, max_value=max_d2)

mask2 = (df_fut["Date"].dt.date >= start2) & (df_fut["Date"].dt.date <= end2)
filt_fut = df_fut[mask2]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_fut["Date"], y=df_fut["Price"],
                          mode="lines", line=dict(color="gray", width=1),
                          name="Full History", opacity=0.2))
fig2.add_trace(go.Scatter(x=filt_fut["Date"], y=filt_fut["Price"],
                          mode="lines", line=dict(color="gold", width=2.5),
                          name="Selected Window"))
fig2.add_trace(go.Scatter(
    x=pd.concat([filt_fut["Date"], filt_fut["Date"][::-1]]),
    y=pd.concat([filt_fut["Q75"], filt_fut["Q25"][::-1]]),
    fill="toself", fillcolor="rgba(255,215,0,0.15)",
    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
    name="30-Day Quantile Range"))
window_peaks = filt_fut[filt_fut["is_peak"]]
fig2.add_trace(go.Scatter(
    x=window_peaks["Date"], y=window_peaks["Price"],
    mode="markers", marker=dict(size=8, color="crimson", line=dict(width=1, color="white")),
    name="Detected Peaks"))
fig2.update_layout(
    title=f"Gold Futures Price ({start2} → {end2})",
    hovermode="x unified",
    xaxis_title="Date", yaxis_title="Price (USD/oz)",
    template="plotly_dark",
    plot_bgcolor="black", paper_bgcolor="black",
    font=dict(color="white"), margin=dict(l=0, r=0, t=60, b=0)
)
st.plotly_chart(fig2, use_container_width=True, height=700)

# Summary
st.markdown("### 📊 Summary for Selected Window")
summary = filt_fut.tail(30)
colA, colB, colC = st.columns(3)
colA.metric("Latest Price", f"{summary['Price'].iloc[-1]:,.2f}")
colB.metric("30-Day Mean", f"{summary['Price'].mean():,.2f}")
colC.metric("Volatility (%)", f"{summary['Pct_Change'].std():.2f}")

st.markdown(
    f"<p style='text-align:center;color:gold;'>Highlighted window: "
    f"<b>{start2} → {end2}</b></p>", unsafe_allow_html=True)

st.divider()
st.markdown("© Brahmanda SaaS — Event-Driven Market Intelligence")
