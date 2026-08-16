"""
Feature engineering.

The important part: the demand model's price channel is a set of *relative*
price levers (price vs the user's own reference prices), NOT raw effective_price
— because raw price is confounded. The optimizer later recomputes exactly these
levers for each candidate price, so training and scoring stay consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _safe_ratio(num, den):
    """Elementwise num/den, returning NaN where den is missing or 0."""
    den = pd.Series(den).replace(0, np.nan)
    return pd.Series(num).values / den.values


def price_levers_scalar(price: float, lb, mb, ls, bo) -> dict:
    """Compute the four price levers for a single candidate price.

    lb=last_booked_price, mb=max_effective_price_ever_booked,
    ls=last_seen_price, bo=avg_effective_price_when_bounced.
    """
    def ratio(a, b):
        return a / b if (b is not None and pd.notna(b) and b != 0) else np.nan

    return {
        "price_vs_lastbooked": ratio(price, lb),
        "price_vs_maxbooked": ratio(price, mb),
        "price_minus_lastseen": price - ls if pd.notna(ls) else np.nan,
        "price_in_bounce_band": ratio(price, bo),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative-price levers, new-user + missingness flags, encoded categoricals."""
    df = df.copy()
    p = df[C.PRICE_COL]

    if "last_booked_price" in df:
        df["price_vs_lastbooked"] = _safe_ratio(p, df["last_booked_price"])
    if "max_effective_price_ever_booked" in df:
        df["price_vs_maxbooked"] = _safe_ratio(p, df["max_effective_price_ever_booked"])
    if "last_seen_price" in df:
        df["price_minus_lastseen"] = p - df["last_seen_price"]
    if "avg_effective_price_when_bounced" in df:
        df["price_in_bounce_band"] = _safe_ratio(p, df["avg_effective_price_when_bounced"])

    # New / thin-history user flag
    if C.LOYALTY_COL in df:
        df["is_new_user"] = (df[C.LOYALTY_COL].fillna(0) == 0).astype(int)

    # Informative-missingness flags for the price-history block
    for col in list(C.ANCHORS.values()) + ["days_since_last_booking",
                                            "avg_inter_booking_gap"]:
        if col in df:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Ordinal-encode categoricals (-1 for NaN); keeps a stable numeric matrix
    for col in C.CATEGORICALS:
        if col in df:
            df[f"{col}_code"] = df[col].astype("category").cat.codes

    return df


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Select model input columns. Price levers come first (for the monotone vector)."""
    exclude = set(
        [C.TARGET, C.ID_COL, C.DATE_COL, C.PRICE_COL]
        + C.POST_TREATMENT
        + C.LEAKAGE_SUSPECTS
        + C.CATEGORICALS
    )
    feats: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        keep_flag = (
            col in C.PRICE_LEVERS
            or col == "is_new_user"
            or col.endswith("_missing")
            or col.endswith("_code")
        )
        if keep_flag or pd.api.types.is_numeric_dtype(df[col]):
            feats.append(col)

    # De-dup, put levers first
    feats = list(dict.fromkeys(feats))
    levers = [c for c in C.PRICE_LEVERS if c in feats]
    rest = [c for c in feats if c not in levers]
    return levers + rest


def monotone_vector(features: list[str]) -> list[int]:
    """-1 (decreasing) for each price lever, 0 otherwise."""
    return [-1 if f in C.PRICE_LEVERS else 0 for f in features]
