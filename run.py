"""
Entry point.

Examples
--------
    python run.py
    python run.py --data data/my_dataset.csv
    python run.py --steps eda
    python run.py --steps eda,train,evaluate,optimize,causal
"""
from __future__ import annotations

import argparse

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="User pricing EDA & modelling pipeline")
    parser.add_argument(
        "--data", default=None,
        help="Path to .xlsx/.csv dataset (default: data/pricing_data.xlsx)",
    )
    parser.add_argument(
        "--steps", default="eda,train,evaluate,optimize",
        help="Comma-separated subset of: eda,train,evaluate,optimize,causal",
    )
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    run_pipeline(args.data, steps)


if __name__ == "__main__":
    main()
