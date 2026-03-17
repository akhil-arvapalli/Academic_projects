from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_n"] = out["full_name"].astype(str).str.lower().str.split().str.join(" ")
    out["email_n"] = out["email"].astype(str).str.lower().str.strip()
    out["phone_n"] = out["phone"].astype(str).str.replace(r"\D", "", regex=True).str[-10:]
    out["roll_n"] = out["roll_no"].astype(str).str.lower().str.strip()
    out["dob_n"] = pd.to_datetime(out["dob"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def mark_duplicates(df: pd.DataFrame) -> pd.Series:
    by_email = df.duplicated("email_n", keep=False) & df["email_n"].ne("")
    by_roll = df.duplicated("roll_n", keep=False) & df["roll_n"].ne("")
    by_phone = df.duplicated("phone_n", keep=False) & df["phone_n"].ne("")
    by_name_dob = df.duplicated(["name_n", "dob_n"], keep=False)
    return by_email | by_roll | by_phone | by_name_dob


def run(input_csv: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    required = {"student_id", "roll_no", "full_name", "email", "phone", "dob", "class_section"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = normalize(df)
    is_dup = mark_duplicates(df)

    # Compact key priority: email -> roll -> phone -> (name+dob).
    name_dob = df["name_n"] + "|" + df["dob_n"].fillna("")
    bucket = df["email_n"].copy()
    bucket = bucket.mask(bucket.eq(""), df["roll_n"])
    bucket = bucket.mask(bucket.eq(""), df["phone_n"])
    bucket = bucket.mask(bucket.eq(""), name_dob)
    bucket = bucket.mask(bucket.eq("|"), "row_" + df.index.astype(str))
    keep_idx = df.sort_values("student_id").groupby(bucket, dropna=False).head(1).index

    cleaned = df.loc[keep_idx].sort_values("student_id")
    removed = df[~df.index.isin(keep_idx)]
    duplicates = df[is_dup].sort_values("student_id")

    cleaned.to_csv(out_dir / "students_cleaned_simple.csv", index=False)
    removed.to_csv(out_dir / "removed_duplicates_simple.csv", index=False)
    duplicates.to_csv(out_dir / "duplicates_flagged_simple.csv", index=False)

    metrics = {
        "raw_rows": int(len(df)),
        "duplicates_flagged": int(is_dup.sum()),
        "removed_rows": int(len(removed)),
        "clean_rows": int(len(cleaned)),
    }
    print(metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tiny student dedup script")
    p.add_argument("--input", default="data/students_raw.csv")
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    run(Path(a.input), Path(a.out))
