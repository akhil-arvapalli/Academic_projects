import re
import pandas as pd


REQUIRED_COLUMNS = [
    "student_id",
    "roll_no",
    "full_name",
    "email",
    "phone",
    "dob",
    "class_section",
]


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split()).strip()


def normalize_name(value: str) -> str:
    if pd.isna(value):
        return ""
    return _normalize_whitespace(str(value)).lower()


def normalize_email(value: str) -> str:
    if pd.isna(value):
        return ""
    return _normalize_whitespace(str(value)).lower()


def normalize_phone(value: str, keep_digits: int = 10) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) > keep_digits:
        digits = digits[-keep_digits:]
    return digits


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_students(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    out = df.copy()
    out["full_name_norm"] = out["full_name"].apply(normalize_name)
    out["email_norm"] = out["email"].apply(normalize_email)
    out["phone_norm"] = out["phone"].apply(normalize_phone)
    out["dob_norm"] = pd.to_datetime(out["dob"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["roll_no_norm"] = out["roll_no"].astype(str).str.strip().str.lower()
    return out
