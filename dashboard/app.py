import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_PATH, REPORTS_DIR, DATE_COLUMN, TARGET_COLUMN


st.set_page_config(
    page_title="Drift-Aware Credit Risk Dashboard",
    layout="wide",
)

st.title("Drift-Aware Credit Risk Monitoring Dashboard")


@st.cache_data
def load_sample_data(path, chunksize=50_000, max_rows=50_000):
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=True):
        needed = max_rows - total_rows
        selected = chunk.head(needed)
        chunks.append(selected)
        total_rows += len(selected)

        if total_rows >= max_rows:
            break

    return pd.concat(chunks, ignore_index=True)


df = load_sample_data(PROCESSED_DATA_PATH)

st.success("Dashboard loaded using a memory-safe sample of the processed dataset.")

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Sample Rows Loaded", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", f"{df.shape[1]:,}")

with col3:
    if TARGET_COLUMN in df.columns:
        default_rate = df[TARGET_COLUMN].mean()
        st.metric("Sample Default Rate", f"{default_rate:.2%}")
    else:
        st.metric("Sample Default Rate", "N/A")


st.subheader("Loan Issue Date Distribution")

if DATE_COLUMN in df.columns:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    monthly_counts = df.groupby(df[DATE_COLUMN].dt.to_period("M")).size()
    monthly_counts.index = monthly_counts.index.astype(str)
    st.line_chart(monthly_counts)
else:
    st.warning(f"Date column `{DATE_COLUMN}` not found.")


st.subheader("Target Distribution")

if TARGET_COLUMN in df.columns:
    target_counts = df[TARGET_COLUMN].value_counts().rename(
        index={0: "Good Loan", 1: "Bad Loan"}
    )
    st.bar_chart(target_counts)
else:
    st.warning(f"Target column `{TARGET_COLUMN}` not found.")


st.subheader("Drift Report")

report_path = REPORTS_DIR / "data_drift_report.html"

if report_path.exists():
    st.success("Drift report found.")

    with open(report_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    components.html(html_content, height=900, scrolling=True)
else:
    st.warning(
        "Drift report not found. Run `python -m src.monitoring.drift_report` first."
    )