from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import DEFAULT_CONFIG, DedupConfig
from .detector import choose_canonical_records, detect_duplicates
from .preprocess import preprocess_students


def run_pipeline(input_csv: Path, output_dir: Path, config: DedupConfig = DEFAULT_CONFIG) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_csv)
    prep_df = preprocess_students(raw_df)
    pair_df = detect_duplicates(prep_df, config)
    clean_df, removed_df = choose_canonical_records(prep_df, pair_df)

    normalized_out = output_dir / "students_normalized.csv"
    duplicates_out = output_dir / "duplicate_pairs.csv"
    cleaned_out = output_dir / "students_cleaned.csv"
    removed_out = output_dir / "removed_duplicates.csv"

    prep_df.to_csv(normalized_out, index=False)
    pair_df.to_csv(duplicates_out, index=False)
    clean_df.to_csv(cleaned_out, index=False)
    removed_df.to_csv(removed_out, index=False)

    metrics = {
        "raw_rows": int(len(raw_df)),
        "duplicate_pairs": int(len(pair_df)),
        "removed_rows": int(len(removed_df)),
        "clean_rows": int(len(clean_df)),
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    return metrics
