from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dedup.pipeline import run_pipeline

st.set_page_config(page_title="Student Dedup Analyzer", page_icon="📊", layout="wide")

st.title("Student Dataset Duplicate Analyzer")
st.write("Upload a CSV file, detect duplicate entries, and download cleaned output files.")

uploaded = st.file_uploader("Upload student CSV", type=["csv"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "uploaded_students.csv"
        input_path.write_bytes(uploaded.getvalue())

        output_dir = tmp_path / "reports"
        metrics = run_pipeline(input_path, output_dir)

        st.subheader("Run Metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        duplicates_path = output_dir / "duplicate_pairs.csv"
        cleaned_path = output_dir / "students_cleaned.csv"

        dup_df = pd.read_csv(duplicates_path)
        clean_df = pd.read_csv(cleaned_path)

        left, right = st.columns(2)
        with left:
            st.subheader("Detected Duplicate Pairs")
            st.dataframe(dup_df, use_container_width=True)
        with right:
            st.subheader("Cleaned Dataset")
            st.dataframe(clean_df, use_container_width=True)

        st.download_button(
            label="Download Cleaned CSV",
            data=cleaned_path.read_bytes(),
            file_name="students_cleaned.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Duplicate Pairs CSV",
            data=duplicates_path.read_bytes(),
            file_name="duplicate_pairs.csv",
            mime="text/csv",
        )
else:
    st.info("Use the sample file in data/students_raw.csv to test quickly.")
