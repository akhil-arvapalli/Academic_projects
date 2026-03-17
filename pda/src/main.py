from __future__ import annotations

import argparse
from pathlib import Path

from src.dedup.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student dataset deduplication pipeline")
    parser.add_argument(
        "--input",
        default="data/students_raw.csv",
        help="Path to raw student CSV",
    )
    parser.add_argument(
        "--out",
        default="reports",
        help="Directory where cleaned outputs will be generated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_pipeline(Path(args.input), Path(args.out))
    print("Deduplication completed.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
