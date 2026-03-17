from pathlib import Path
import pandas as pd


def run(input_csv: str = "data/students_raw.csv", out_dir: str = "reports") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Basic normalization for matching.
    df["email_n"] = df["email"].astype(str).str.lower().str.strip()
    df["roll_n"] = df["roll_no"].astype(str).str.lower().str.strip()
    df["phone_n"] = df["phone"].astype(str).str.replace(r"\D", "", regex=True).str[-10:]
    df["name_n"] = df["full_name"].astype(str).str.lower().str.split().str.join(" ")
    df["dob_n"] = pd.to_datetime(df["dob"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Duplicate if any key repeats.
    dup = (
        (df.duplicated("email_n", keep=False) & df["email_n"].ne(""))
        | (df.duplicated("roll_n", keep=False) & df["roll_n"].ne(""))
        | (df.duplicated("phone_n", keep=False) & df["phone_n"].ne(""))
        | df.duplicated(["name_n", "dob_n"], keep=False)
    )

    cleaned = df[~dup].copy()
    duplicates = df[dup].copy()

    cleaned.to_csv(out_path / "students_cleaned_min.csv", index=False)
    duplicates.to_csv(out_path / "duplicates_min.csv", index=False)

    print({
        "raw_rows": len(df),
        "duplicates": int(dup.sum()),
        "clean_rows": len(cleaned),
    })


if __name__ == "__main__":
    run()
