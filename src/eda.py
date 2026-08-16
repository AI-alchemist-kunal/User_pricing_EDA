"""
EDA: the plots and tables that matter for a pricing problem.

Nothing here draws conclusions for you — it produces the figures and the
leakage table so YOU can read them. Figures are written to outputs/figures/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C


def _save(fig, name: str):
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def funnel_summary(df: pd.DataFrame) -> dict:
    out = {"sessions": len(df)}
    if "add_to_cart_flag" in df:
        out["add_to_cart"] = int(df["add_to_cart_flag"].sum())
    if C.TARGET in df:
        out["booked"] = int(df[C.TARGET].sum())
    if "add_to_cart_flag" in df and C.TARGET in df:
        out["cart_abandoners"] = int(((df["add_to_cart_flag"] == 1) & (df[C.TARGET] == 0)).sum())
        out["browsers"] = int(((df["add_to_cart_flag"] == 0) & (df[C.TARGET] == 0)).sum())
    return out


def plot_missingness(df: pd.DataFrame):
    frac = df.isna().mean().sort_values(ascending=False)
    frac = frac[frac > 0]
    if frac.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(frac))))
    frac.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("missing fraction")
    ax.set_title("Missingness by column")
    return _save(fig, "missingness.png")


def plot_confounding(df: pd.DataFrame):
    """The signature plot: conversion vs price, overall vs within loyalty tiers."""
    if C.PRICE_COL not in df or C.TARGET not in df:
        return None
    d = df[[C.PRICE_COL, C.TARGET]].dropna()
    if len(d) < 20 or d[C.PRICE_COL].nunique() < 5:
        return None
    nb = min(5, d[C.PRICE_COL].nunique())
    fig, ax = plt.subplots(figsize=(7, 4.5))

    d = d.assign(bin=pd.qcut(d[C.PRICE_COL], q=nb, duplicates="drop"))
    overall = d.groupby("bin", observed=True)[C.TARGET].mean()
    ax.plot(range(len(overall)), overall.values, marker="o", lw=2.5, label="overall")

    if C.LOYALTY_COL in df:
        dd = df[[C.PRICE_COL, C.TARGET, C.LOYALTY_COL]].dropna()
        try:
            dd = dd.assign(
                loy=pd.qcut(dd[C.LOYALTY_COL], q=3, duplicates="drop", labels=False),
                bin=pd.qcut(dd[C.PRICE_COL], q=nb, duplicates="drop"),
            )
            for lv, g in dd.groupby("loy", observed=True):
                s = g.groupby("bin", observed=True)[C.TARGET].mean()
                ax.plot(range(len(s)), s.values, marker=".", alpha=0.75,
                        label=f"loyalty tier {int(lv)}")
        except Exception:
            pass

    ax.set_xlabel("price bin (low → high)")
    ax.set_ylabel("conversion rate")
    ax.set_title("Conversion vs price: overall vs within loyalty tiers")
    ax.legend(fontsize=8)
    return _save(fig, "confounding_price_curve.png")


def plot_price_policy(df: pd.DataFrame):
    """Reveals how the current system assigns price (price vs loyalty)."""
    if C.PRICE_COL not in df or C.LOYALTY_COL not in df:
        return None
    d = df[[C.PRICE_COL, C.LOYALTY_COL]].dropna()
    if len(d) < 20:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d[C.LOYALTY_COL], d[C.PRICE_COL], alpha=0.5, s=15)
    ax.set_xlabel(C.LOYALTY_COL)
    ax.set_ylabel(C.PRICE_COL)
    ax.set_title("Current pricing policy: price vs loyalty")
    return _save(fig, "price_policy.png")


def correlation_heatmap(df: pd.DataFrame, features: list[str]):
    num = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    if len(num) < 2:
        return None
    corr = df[num].corr()
    side = min(14, 0.45 * len(num) + 2)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)))
    ax.set_xticklabels(num, rotation=90, fontsize=6)
    ax.set_yticks(range(len(num)))
    ax.set_yticklabels(num, fontsize=6)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation")
    return _save(fig, "correlation_heatmap.png")


def leakage_scan(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Univariate AUC of each feature vs target. Values near 1.0 => inspect for leakage."""
    if C.TARGET not in df or df[C.TARGET].nunique() < 2:
        return pd.DataFrame()
    y = df[C.TARGET]
    rows = []
    for f in features:
        s = df[f]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        mask = s.notna()
        if mask.sum() < 10 or y[mask].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y[mask], s[mask])
            rows.append((f, round(max(auc, 1 - auc), 3)))
        except Exception:
            continue
    return (
        pd.DataFrame(rows, columns=["feature", "univariate_auc"])
        .sort_values("univariate_auc", ascending=False)
        .reset_index(drop=True)
    )


def run_all(df: pd.DataFrame, features: list[str]) -> dict:
    print("=== EDA ===")
    funnel = funnel_summary(df)
    print("  funnel:", funnel)

    figs = []
    for fn in (plot_missingness, plot_confounding, plot_price_policy):
        try:
            p = fn(df)
            if p:
                figs.append(p.name)
        except Exception as e:
            print(f"  ! {fn.__name__} failed: {e}")
    try:
        p = correlation_heatmap(df, features)
        if p:
            figs.append(p.name)
    except Exception as e:
        print(f"  ! correlation_heatmap failed: {e}")

    try:
        leak = leakage_scan(df, features)
        if not leak.empty:
            leak.to_csv(C.OUTPUT_DIR / "leakage_scan.csv", index=False)
            print("  leakage scan (top univariate AUCs — inspect anything > 0.90):")
            for _, r in leak.head(5).iterrows():
                flag = "  <-- INSPECT" if r["univariate_auc"] > 0.90 else ""
                print(f"     {r['feature']:34s} {r['univariate_auc']:.3f}{flag}")
    except Exception as e:
        print(f"  ! leakage_scan failed: {e}")

    print(f"  figures -> {C.FIG_DIR}")
    return {"funnel": funnel, "figures": figs}
