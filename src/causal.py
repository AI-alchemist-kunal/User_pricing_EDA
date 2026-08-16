"""
Tier 4 (optional): per-user price elasticity via Double ML.

Requires EconML (`pip install econml`). If it isn't installed the pipeline
skips this cleanly. The estimated per-user effect is what you segment on for the
price-sensitivity bonus question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def estimate_elasticity(df: pd.DataFrame, features: list[str], confounders: list[str] | None = None):
    """
    Fit CausalForestDML with:
      Y = booked_flag, T = effective_price,
      X = effect modifiers (heterogeneity), W = confounders (controls).
    Returns df with a 'price_effect' column (dP(book)/dPrice per user), or None
    if EconML is unavailable / the fit fails.
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    except Exception:
        print("  ! EconML not installed — skipping Tier 4 (pip install econml).")
        return None

    needed = {C.TARGET, C.PRICE_COL}
    if not needed.issubset(df.columns):
        print("  ! causal step needs target + price columns — skipping.")
        return None

    modifiers = [f for f in features if f not in C.PRICE_LEVERS]
    confounders = confounders or modifiers
    d = df.dropna(subset=[C.TARGET, C.PRICE_COL]).copy()
    X = d[modifiers].fillna(d[modifiers].median(numeric_only=True))
    W = d[confounders].fillna(d[confounders].median(numeric_only=True))
    Y = d[C.TARGET].values
    T = d[C.PRICE_COL].values

    try:
        est = CausalForestDML(
            model_y=RandomForestClassifier(n_estimators=200, random_state=C.RANDOM_SEED),
            model_t=RandomForestRegressor(n_estimators=200, random_state=C.RANDOM_SEED),
            discrete_treatment=False,
            random_state=C.RANDOM_SEED,
        )
        est.fit(Y, T, X=X, W=W)
        d["price_effect"] = est.effect(X)
        print("  Tier 4: per-user price_effect estimated "
              f"(mean={d['price_effect'].mean():.5f}).")
        return d[[C.ID_COL, "price_effect"]] if C.ID_COL in d else d[["price_effect"]]
    except Exception as exc:
        print(f"  ! causal fit failed: {exc}")
        return None
