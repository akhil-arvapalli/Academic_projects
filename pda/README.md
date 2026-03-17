# Student Dataset Duplicate Elimination

This project removes duplicate student records using a data analytics workflow suitable for Principles of Data Analytics lab work.

## Features
- Data preprocessing for name, email, phone, and date normalization
- Rule-based duplicate detection:
  - exact email match
  - exact roll number match
  - exact phone match
  - same normalized name plus DOB
- Fuzzy duplicate detection with name similarity using RapidFuzz
- Canonical record selection and duplicate removal
- Output reports for cleaned data, duplicate pairs, and run metrics
- Optional Streamlit app for demo UI

## Project Structure
- data: raw input CSV samples
- src: deduplication logic and CLI
- reports: generated output files
- app: optional Streamlit app

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (CLI)

```bash
python -m src.main --input data/students_raw.csv --out reports
```

Generated outputs:
- reports/students_normalized.csv
- reports/duplicate_pairs.csv
- reports/students_cleaned.csv
- reports/removed_duplicates.csv
- reports/metrics.csv

## Run (Optional Streamlit)

```bash
streamlit run app/streamlit_app.py
```

## Evaluation Checklist (for lab report)
- Compare row count before and after cleanup
- Inspect duplicate categories in duplicate_pairs.csv
- Validate random duplicate groups manually
- Document threshold used for fuzzy matching
