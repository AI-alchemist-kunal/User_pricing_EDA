"""
Price optimizer (not ML).

For every row we sweep a grid of candidate prices, recompute the relative-price
levers for each candidate (so the demand curve is self-consistent), predict
P(book), and pick the price that maximises expected revenue = price x P(book),
subject to the guardrails in config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from src.model import predict_proba


def _multipliers() -> np.ndarray:
    """Grid of price multipliers, clipped by guardrails, always including 1.0 (current)."""
    lo = max(C.PRICE_GRID_LOW, 1 - C.MAX_PRICE_DROP)
    hi = min(C.PRICE_GRID_HIGH, 1 + C.MAX_PRICE_UPLIFT)
    grid = np.linspace(lo, hi, C.PRICE_GRID_STEPS)
    return np.unique(np.concatenate([grid, [1.0]]))


def _anchor(d: pd.DataFrame, col: str, n: int, reps: int) -> np.ndarray:
    vals = d[col].values if col in d.columns else np.full(n, np.nan)
    return np.repeat(vals, reps).astype(float)


def optimize_dataframe(model, dff: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return one recommendation row per input row."""
    d = dff.reset_index(drop=True)
    n = len(d)
    cur = pd.to_numeric(d[C.PRICE_COL], errors="coerce").values.astype(float)
    valid = np.isfinite(cur) & (cur > 0)

    mult = _multipliers()
    G = len(mult)
    cur_col = int(np.argmin(np.abs(mult - 1.0)))

    # Default output = keep current price (used for invalid rows)
    rec_price = cur.copy()
    p_cur = np.full(n, np.nan)
    p_rec = np.full(n, np.nan)

    if valid.any():
        dv = d[valid].reset_index(drop=True)
        m = len(dv)
        grid = dv[C.PRICE_COL].values.astype(float)[:, None] * mult[None, :]  # (m, G)
        flat_price = grid.reshape(-1)

        big = dv[features].loc[dv.index.repeat(G)].reset_index(drop=True).astype(float)
        lb = _anchor(dv, "last_booked_price", m, G)
        mb = _anchor(dv, "max_effective_price_ever_booked", m, G)
        ls = _anchor(dv, "last_seen_price", m, G)
        bo = _anchor(dv, "avg_effective_price_when_bounced", m, G)

        with np.errstate(divide="ignore", invalid="ignore"):
            if "price_vs_lastbooked" in big:
                big["price_vs_lastbooked"] = np.where(lb > 0, flat_price / lb, np.nan)
            if "price_vs_maxbooked" in big:
                big["price_vs_maxbooked"] = np.where(mb > 0, flat_price / mb, np.nan)
            if "price_minus_lastseen" in big:
                big["price_minus_lastseen"] = flat_price - ls
            if "price_in_bounce_band" in big:
                big["price_in_bounce_band"] = np.where(bo > 0, flat_price / bo, np.nan)

        probs = predict_proba(model, big[features]).reshape(m, G)
        revenue = grid * probs
        best = np.argmax(revenue, axis=1)

        idx = np.arange(m)
        rec_price[valid] = grid[idx, best]
        p_rec[valid] = probs[idx, best]
        p_cur[valid] = probs[:, cur_col]

    exp_rev_current = cur * p_cur
    exp_rev_rec = rec_price * p_rec
    out = pd.DataFrame({
        C.ID_COL: d[C.ID_COL] if C.ID_COL in d else np.arange(n),
        "current_price": cur,
        "recommended_price": np.round(rec_price, 2),
        "price_change_pct": np.round((rec_price / cur - 1) * 100, 2),
        "p_book_at_current": np.round(p_cur, 4),
        "p_book_at_rec": np.round(p_rec, 4),
        "exp_rev_current": exp_rev_current,
        "exp_rev_rec": exp_rev_rec,
    })
    return out
