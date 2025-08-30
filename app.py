import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
pip install matplotlib seaborn

st.set_page_config(page_title="Assessment 4 Dashboard", layout="wide")
st.title("📊 Dataset Exploration Dashboard")
st.caption("Assessment 4 – Task One")

# ---------- Load data ----------
def safe_read_csv(name):
    p = Path(name)
    if not p.exists():
        st.error(f"Missing required file: {name}. Please export it from your notebook first.")
        st.stop()
    return pd.read_csv(p)

df = safe_read_csv("df.csv")

# If numeric_df was exported, try to load it
numeric_df = None
try:
    numeric_df = safe_read_csv("numeric_df.csv")
except Exception:
    st.warning("`numeric_df.csv` not found – continuing with df only.")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📈 Data & Insights", "📊 Correlations"])

with tab1:
    st.subheader("Quick Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{len(df)}")
    col2.metric("Columns", f"{df.shape[1]}")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Column Breakdown")
    st.write(df.describe(include="all"))

    if "age" in df.columns:
        st.markdown("### Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["age"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("Correlation Heatmap")

    if numeric_df is not None:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric_df.csv provided — skipping correlation heatmap.")
