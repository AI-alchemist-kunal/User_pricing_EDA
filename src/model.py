"""
The model ladder.

Tier 1  logistic regression  -> interpretable baseline
Tier 2  gradient-boosted trees, MONOTONE-decreasing in the price levers
Tier 3  probability calibration (sigmoid)

LightGBM is used if available; otherwise we fall back to scikit-learn's
HistGradientBoostingClassifier, which also supports monotone constraints and
native missing-value handling — so this module runs with only scikit-learn.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config as C
from src.features import monotone_vector


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


BACKEND = "lightgbm" if _lightgbm_available() else "sklearn-histgbm"


def build_gbm(monotone: list[int] | None):
    """Return an unfitted GBM classifier honouring the monotone constraints."""
    if BACKEND == "lightgbm":
        from lightgbm import LGBMClassifier
        kw = dict(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=C.RANDOM_SEED, verbose=-1,
        )
        if monotone is not None:
            kw["monotone_constraints"] = monotone
        return LGBMClassifier(**kw)

    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_iter=400, learning_rate=0.05, random_state=C.RANDOM_SEED)
    if monotone is not None:
        kw["monotonic_cst"] = monotone
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kw)
    except TypeError:  # older sklearn without class_weight
        return HistGradientBoostingClassifier(**kw)


# --------------------------------------------------------------------------- #
# Tier 1 — logistic baseline
# --------------------------------------------------------------------------- #
def train_logit(train: pd.DataFrame, features: list[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(train[features], train[C.TARGET])
    return pipe


# --------------------------------------------------------------------------- #
# Tier 2 + 3 — monotone GBM, optionally calibrated
# --------------------------------------------------------------------------- #
def _can_calibrate(y: pd.Series) -> bool:
    return y.nunique() == 2 and y.value_counts().min() >= 8


def train_demand_model(train: pd.DataFrame, features: list[str], calibrate: bool = True):
    """Fit the monotone GBM demand model; wrap with calibration when feasible."""
    X, y = train[features], train[C.TARGET]
    if y.nunique() < 2:
        raise ValueError("Training split has a single target class — widen the split.")

    mono = monotone_vector(features)

    def _fit(use_mono, Xf, yf):
        try:
            est = build_gbm(mono if use_mono else None)
            est.fit(Xf, yf)
            return est
        except Exception as exc:
            if use_mono:
                print(f"  ! monotone fit failed ({exc}); retrying unconstrained")
                return _fit(False, Xf, yf)
            raise

    base = _fit(True, X, y)

    if calibrate and _can_calibrate(y):
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import train_test_split
            Xtr, Xca, ytr, yca = train_test_split(
                X, y, test_size=0.25, random_state=C.RANDOM_SEED, stratify=y
            )
            prefit = _fit(True, Xtr, ytr)
            try:  # sklearn >= 1.6
                from sklearn.frozen import FrozenEstimator
                cal = CalibratedClassifierCV(FrozenEstimator(prefit), method="sigmoid")
            except Exception:  # older sklearn
                cal = CalibratedClassifierCV(prefit, cv="prefit", method="sigmoid")
            cal.fit(Xca, yca)
            return cal, BACKEND + "+calibrated"
        except Exception as exc:
            print(f"  ! calibration skipped ({exc}); using uncalibrated model")

    return base, BACKEND


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """P(book=1) as a 1-D array."""
    return model.predict_proba(X)[:, 1]


def save_model(model, name: str = "demand_model.joblib"):
    path = C.MODEL_DIR / name
    joblib.dump(model, path)
    return path


def load_model(name: str = "demand_model.joblib"):
    return joblib.load(C.MODEL_DIR / name)
