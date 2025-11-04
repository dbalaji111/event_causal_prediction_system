import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from collections import Counter
import ast, os, re
from wordcloud import WordCloud

st.set_page_config(layout="wide", page_title="Scenario 1 – Event Memory Analysis")
st.title("🧠 Scenario-1: Event Memory Exploration & Analysis")

# ============ LOAD DATA ============
@st.cache_data
def load_event_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "causal_gold_articles_with_topics_bert.csv")
    df = pd.read_csv(csv_path)
    df["embedding"] = df["gold_general_embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    reduced = PCA(n_components=3).fit_transform(np.vstack(df["embedding"].values))
    df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])

df = load_event_data()

# ============ DATE SELECTION ============
min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end_date = col2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)

mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
filtered_df = df[mask]

st.markdown(f"**{len(filtered_df)}** events found between {start_date} and {end_date}.")

# ============ 3D SCATTER ============
fig_3d = px.scatter_3d(
    filtered_df, x="x", y="y", z="z",
    color="assigned_topic_bert",
    hover_name="Headline",
    hover_data={"Date": True, "gold_relevance_score": True, "assigned_topic_bert": True},
    title=f"3D Event Map ({start_date} → {end_date})"
)
fig_3d.update_layout(
    scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3"),
    margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": True})

# ============ ANALYSIS BUTTON ============
if st.button("🔍 Analyse Event Window"):
    st.subheader("📊 Event Type Frequency (Topic-BERT)")

    # --- Topic Frequency ---
    topic_counts = filtered_df["assigned_topic_bert"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Frequency"]

    fig_topics = px.bar(
        topic_counts,
        x="Topic",
        y="Frequency",
        color="Frequency",
        text="Frequency",
        title="Event Frequency by Topic (BERT)",
        color_continuous_scale="sunset",  # ✅ valid colorscale
    )
    fig_topics.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_topics, use_container_width=True, config={"displayModeBar": True})

    # --- Major Causes ---
    st.subheader("🔥 Major Causes (Keyword Frequency)")
    causes_text = " ".join(filtered_df["gold_cause"].dropna().astype(str).tolist())
    words = re.findall(r"[A-Za-z']+", causes_text)
    words = [w.lower() for w in words if len(w) > 3]
    common_words = Counter(words).most_common(20)
    cause_df = pd.DataFrame(common_words, columns=["Cause Keyword", "Frequency"])

    fig_causes = px.bar(
        cause_df,
        x="Frequency",
        y="Cause Keyword",
        orientation="h",
        text="Frequency",
        title="Most Frequent Causes in Selected Period",
        color="Frequency",
        color_continuous_scale="thermal",  # ✅ valid colorscale
    )
    st.plotly_chart(fig_causes, use_container_width=True, config={"displayModeBar": True})

    # --- Word Cloud ---
    st.subheader("☁️ Word Cloud of Causes")
    wc = WordCloud(width=800, height=400, background_color="black", colormap="autumn").generate_from_frequencies(dict(common_words))
    st.image(wc.to_array(), use_container_width=True)

    # --- Raw Data Table ---
    with st.expander("🔎 View Raw Event Data"):
        st.dataframe(
            filtered_df[
                ["Date", "Headline", "assigned_topic_bert", "gold_cause", "gold_effect", "gold_relevance_score"]
            ].sort_values("Date")
        )
