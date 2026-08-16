"""
Load the dataset and run basic integrity checks.

Keeps I/O and validation in one place so the rest of the pipeline can assume a
clean, typed DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as C


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the dataset from .xlsx or .csv."""
    path = Path(path) if path is not None else C.DEFAULT_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Put your file in the data/ folder "
            f"(default name '{C.DEFAULT_DATA_FILE.name}') or pass --data <path>."
        )
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .xlsx or .csv.")
    return df


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Coerce key dtypes and return (clean_df, integrity_report)."""
    df = df.copy()
    report: dict = {"n_rows": len(df), "n_cols": df.shape[1]}

    # Parse date
    if C.DATE_COL in df.columns:
        df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL], errors="coerce")
        report["date_min"] = str(df[C.DATE_COL].min())
        report["date_max"] = str(df[C.DATE_COL].max())

    # Grain check: one row per (user, date)
    if C.ID_COL in df.columns:
        report["n_users"] = int(df[C.ID_COL].nunique())
    if C.ID_COL in df.columns and C.DATE_COL in df.columns:
        report["duplicate_user_date_rows"] = int(
            df.duplicated([C.ID_COL, C.DATE_COL]).sum()
        )

    # Target
    if C.TARGET in df.columns:
        df[C.TARGET] = (
            pd.to_numeric(df[C.TARGET], errors="coerce").fillna(0).astype(int)
        )
        report["booked_rate"] = round(float(df[C.TARGET].mean()), 4)

    # Missing core columns are fatal for later steps — surface them early.
    report["missing_core_columns"] = [
        c for c in (C.TARGET, C.PRICE_COL, C.ID_COL) if c not in df.columns
    ]
    return df, report


def print_report(report: dict) -> None:
    print("=== Data integrity ===")
    for k, v in report.items():
        print(f"  {k:28s}: {v}")
    if report.get("duplicate_user_date_rows", 0):
        print("  ! duplicate (user, date) rows found — check the grain.")
    if report.get("missing_core_columns"):
        print("  ! missing core columns — rename in config.py to match your data.")
