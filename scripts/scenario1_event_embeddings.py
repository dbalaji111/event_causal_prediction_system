# Scenario 1 – Event Memory Exploration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import ast

st.set_page_config(layout="wide", page_title="Scenario 1 - Event Memories")
st.title("🪙 3D Visualization of Gold News Articles Over Time")

@st.cache_data
def load_event_data():
    df = pd.read_csv("causal_gold_articles_with_topics_bert.csv")
    df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    embeddings = np.vstack(df["embedding"].values)
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)
    df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])

df = load_event_data()

# --- Date range filter ---
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)
mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
filtered_df = df[mask]

# --- 3D scatter plot ---
fig = px.scatter_3d(
    filtered_df,
    x="x", y="y", z="z",
    color="assigned_topic",
    hover_name="Headline",
    hover_data={
        "Doc_ID": True,
        "gold_relevance_score": True,
        "topic_similarity": True,
        "assigned_topic": True,
        "Date": True,
    },
    title=f"3D Projection of News Articles ({date_range[0]} to {date_range[1]})"
)

fig.update_layout(
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"Showing **{len(filtered_df)}** events between "
    f"**{date_range[0]}** and **{date_range[1]}**."
)
