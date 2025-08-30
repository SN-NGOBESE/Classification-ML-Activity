import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up the page
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("📊 Healthcare Demand and Model Results Dashboard")
st.caption("Assessment 4 – Task One")

# ---------- Load data ----------
def safe_read_csv(name):
    p = Path(name)
    if not p.exists():
        st.error(f"Missing required file: {name}. Please export it from your notebook first.")
        st.stop()
    return pd.read_csv(p)

# Load all the exported CSVs
df = safe_read_csv("df.csv")
daily = safe_read_csv("daily.csv")
results_df = safe_read_csv("results_df.csv")
gender_counts = safe_read_csv("gender_counts.csv")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📈 Data & Insights", "🤖 Model Results"])

# ------------- Tab 1: Data & Insights -------------
with tab1:
    st.subheader("Quick Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{len(df)}")
    col2.metric("Columns", f"{df.shape[1]}")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Daily Hospital Demand")
    st.dataframe(daily.head())

    st.markdown("### Gender-Based Admissions")
    st.dataframe(gender_counts)

    # Plotting Daily Demand Over Time
    st.markdown("### Daily Demand Over Time")
    fig, ax = plt.subplots()
    ax.plot(daily["Date of Admission"], daily["y_hosp"])
    ax.set_xlabel("Date of Admission")
    ax.set_ylabel("Hospital Demand")
    ax.set_title("Daily Hospital Demand")
    st.pyplot(fig)

    # Plotting Admissions by Gender
    st.markdown("### Admissions by Gender")
    fig2, ax2 = plt.subplots()
    ax2.bar(gender_counts["Gender"], gender_counts["Admissions"])
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Admissions")
    ax2.set_title("Admissions by Gender")
    st.pyplot(fig2)

# ------------- Tab 2: Model Results -------------
with tab2:
    st.subheader("Model Comparison")

    # Show the model accuracy results
    st.markdown("### Model Accuracy Comparison")
    st.dataframe(results_df.sort_values("Accuracy", ascending=False), use_container_width=True)

    st.markdown("### Logistic Regression vs Random Forest")

    # Plot model comparison
    fig3, ax3 = plt.subplots()
    ax3.bar(results_df["Model"], results_df["Accuracy"])
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Model Accuracy Comparison")
    st.pyplot(fig3)
