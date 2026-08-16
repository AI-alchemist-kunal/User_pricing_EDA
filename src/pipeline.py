"""
Orchestration: wire the modules into an end-to-end run.

    load -> validate -> features -> [eda] -> split -> train -> [evaluate] ->
    [optimize] -> [causal] -> report
"""
from __future__ import annotations

import config as C
from src import eda
from src.causal import estimate_elasticity
from src.data_load import load_data, print_report, validate
from src.evaluate import evaluate_demand_model, revenue_simulation
from src.features import build_features, get_model_features
from src.model import save_model, train_demand_model, train_logit
from src.optimize import optimize_dataframe
from src.split import temporal_split

DEFAULT_STEPS = ["eda", "train", "evaluate", "optimize"]


def run_pipeline(data_path=None, steps=None) -> dict:
    steps = set(steps or DEFAULT_STEPS)

    # --- Load & validate ---
    df = load_data(data_path)
    df, report = validate(df)
    print_report(report)

    # --- Features ---
    dff = build_features(df)
    feats = get_model_features(dff)
    print("\n=== Features ===")
    print(f"  {len(feats)} model features")
    print("  price levers:", [f for f in feats if f in C.PRICE_LEVERS])
    summary = {"integrity": report, "n_model_features": len(feats), "model_features": feats}

    # --- EDA ---
    if "eda" in steps:
        summary["eda"] = eda.run_all(dff, feats)

    # --- Train ---
    model, test = None, None
    if steps & {"train", "evaluate", "optimize"}:
        train, test = temporal_split(dff)
        print("\n=== Model ===")
        print(f"  split: train={len(train)}  test={len(test)}")
        try:
            logit = train_logit(train, feats)
            summary["logit_baseline"] = evaluate_demand_model(logit, test, feats)
            print("  logit baseline:", summary["logit_baseline"])
        except Exception as exc:
            print(f"  ! logit baseline failed: {exc}")
        model, backend = train_demand_model(train, feats)
        print(f"  demand model backend: {backend}")
        save_model(model)

    # --- Evaluate ---
    if "evaluate" in steps and model is not None and test is not None:
        summary["demand_model"] = evaluate_demand_model(model, test, feats)
        print("  demand metrics:", summary["demand_model"])

    # --- Optimize ---
    if "optimize" in steps and model is not None:
        recs = optimize_dataframe(model, dff, feats)
        out = C.OUTPUT_DIR / "price_recommendations.csv"
        recs.to_csv(out, index=False)
        summary["revenue_sim"] = revenue_simulation(recs)
        print("  revenue simulation:", summary["revenue_sim"])
        print(f"  recommendations -> {out}")

    # --- Causal (optional) ---
    if "causal" in steps:
        eff = estimate_elasticity(dff, feats)
        if eff is not None:
            eff.to_csv(C.OUTPUT_DIR / "user_price_effects.csv", index=False)

    _write_report(summary)
    return summary


def _write_report(summary: dict) -> None:
    path = C.OUTPUT_DIR / "report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("USER PRICING — RUN REPORT\n" + "=" * 40 + "\n\n")
        for k, v in summary.items():
            if k == "model_features":
                f.write(f"{k} ({len(v)}):\n  " + ", ".join(v) + "\n\n")
                continue
            f.write(f"{k}:\n")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"  {v}\n")
            f.write("\n")
    print(f"\nReport -> {path}")
