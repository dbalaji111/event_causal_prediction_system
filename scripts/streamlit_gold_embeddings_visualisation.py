import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import ast

st.set_page_config(layout="wide")
st.title("🪙 3D Visualization of Gold News Articles Over Time")
df_with_embeddings = pd.read_csv(r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\scripts\causal_gold_articles_with_topics_bert.csv")

# Load the CSV file
df = df_with_embeddings.copy()

# Convert 'gold_general_embedding' from string to list of floats
df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

# Stack embeddings into a 2D array
embeddings = np.vstack(df["embedding"].values)

# Reduce to 3D using PCA
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)
df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]

# Convert Date to datetime and drop malformed rows
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna(subset=["Date"])

# Create date range slider using native Python datetime.date
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter data based on date range
mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
filtered_df = df[mask]

# 3D Scatter plot using Plotly
fig = px.scatter_3d(
    filtered_df,
    x="x",
    y="y",
    z="z",
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
