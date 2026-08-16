"""
Evaluation.

Demand-model quality (calibration first, because we multiply prob x price),
plus a directional revenue simulation of the recommended policy vs the current
one. Revenue numbers are model-based estimates, not causal ground truth.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

import config as C
from src.model import predict_proba


def evaluate_demand_model(model, test: pd.DataFrame, features: list[str]) -> dict:
    y = test[C.TARGET].values
    p = predict_proba(model, test[features])
    res: dict = {"n_test": int(len(y))}

    if len(np.unique(y)) == 2:
        res["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        res["pr_auc"] = round(float(average_precision_score(y, p)), 4)
        res["brier"] = round(float(brier_score_loss(y, p)), 4)
        _plot_calibration(y, p)
    else:
        res["note"] = "test split single-class — metrics skipped (widen split / more data)"
    return res


def _plot_calibration(y, p):
    try:
        from sklearn.calibration import calibration_curve
        if len(y) < 20:
            return
        n_bins = max(3, min(10, len(y) // 5))
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(mean_pred, frac_pos, marker="o", label="model")
        ax.set_xlabel("predicted P(book)")
        ax.set_ylabel("observed frequency")
        ax.set_title("Calibration (reliability curve)")
        ax.legend()
        fig.savefig(C.FIG_DIR / "calibration.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  ! calibration plot failed: {exc}")


def revenue_simulation(recs: pd.DataFrame) -> dict:
    """Directional: total expected revenue under recommended vs current prices."""
    cur = recs["exp_rev_current"].sum(skipna=True)
    rec = recs["exp_rev_rec"].sum(skipna=True)
    up = (recs["price_change_pct"] > 0.5).mean()
    down = (recs["price_change_pct"] < -0.5).mean()
    return {
        "exp_revenue_current": round(float(cur), 1),
        "exp_revenue_optimized": round(float(rec), 1),
        "revenue_uplift_pct": round(float((rec / cur - 1) * 100), 2) if cur > 0 else None,
        "share_users_price_up": round(float(up), 3),
        "share_users_price_down": round(float(down), 3),
        "mean_conv_current": round(float(recs["p_book_at_current"].mean()), 4),
        "mean_conv_optimized": round(float(recs["p_book_at_rec"].mean()), 4),
    }
