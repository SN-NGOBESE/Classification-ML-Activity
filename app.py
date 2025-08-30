import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

df = safe_read_csv("df.csv")   # ✅ only load df.csv, since that's what you exported
