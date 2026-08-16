"""
Train/test splitting.

Default is a TIME-based split (predict the next session from the past). A
grouped-by-user split is also provided for testing generalisation to new users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest TIME_SPLIT_QUANTILE of dates -> train, remainder -> test."""
    if C.DATE_COL in df.columns and df[C.DATE_COL].notna().any():
        cutoff = df[C.DATE_COL].quantile(C.TIME_SPLIT_QUANTILE)
        train = df[df[C.DATE_COL] <= cutoff]
        test = df[df[C.DATE_COL] > cutoff]
        if len(train) and len(test):
            return train.copy(), test.copy()
    # Fallback: random split (e.g. dates missing or degenerate)
    rng = np.random.default_rng(C.RANDOM_SEED)
    mask = rng.random(len(df)) < C.TIME_SPLIT_QUANTILE
    return df[mask].copy(), df[~mask].copy()


def grouped_folds(df: pd.DataFrame, n_splits: int = 5):
    """Yield (train_idx, test_idx) with no user shared across folds."""
    groups = df[C.ID_COL] if C.ID_COL in df.columns else np.arange(len(df))
    n_splits = min(n_splits, df[C.ID_COL].nunique()) if C.ID_COL in df.columns else n_splits
    n_splits = max(2, n_splits)
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, groups=groups)
