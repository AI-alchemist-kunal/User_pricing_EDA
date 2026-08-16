# User Pricing — EDA & Modelling Pipeline

A small, modular pipeline that learns each user's **demand curve**
`P(book | price, context)` and recommends the next-session price that
maximises expected revenue `price × P(book | price)` under business guardrails.

> **Core idea.** Absolute price is *confounded* — higher prices are shown to
> loyal, high-intent users, so raw `effective_price` looks like it *helps*
> conversion. The real levers are **price relative to the user's own reference
> price** (last booked, max ever booked, last seen, bounce band), with loyalty
> and supply controlled. The demand model is therefore constrained to be
> **monotonically decreasing** in those relative-price levers.

## 1. Setup

```bash
# from the project folder, in a terminal / VS Code terminal
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Add your data

Put the full dataset in `data/` and either name it `pricing_data.xlsx`
or pass its path with `--data`. `.xlsx` and `.csv` are both supported.

## 3. Run

```bash
python run.py                       # runs everything on data/pricing_data.xlsx
python run.py --data data/my.csv    # custom file
python run.py --steps eda           # just the EDA
python run.py --steps eda,train,evaluate,optimize
```

Outputs land in `outputs/`:

- `outputs/figures/` — EDA plots (missingness, funnel, the confounding plot, calibration, …)
- `outputs/models/` — the trained demand model
- `outputs/price_recommendations.csv` — recommended next-session price per row
- `outputs/report.txt` — a plain-text run summary (metrics + headline numbers)

## 4. Project structure

```
user_pricing/
├── run.py              # CLI entry point
├── config.py           # ALL paths / columns / params live here — edit this first
├── requirements.txt
├── data/               # <- put your dataset here (git-ignored)
├── outputs/            # <- generated figures, models, csv (git-ignored)
└── src/
    ├── data_load.py    # load + integrity checks
    ├── features.py     # relative-price levers, flags, encoding, feature selection
    ├── eda.py          # funnel, missingness, the confounding plot, leakage scan
    ├── split.py        # temporal split (+ optional grouped CV)
    ├── model.py        # Tier 1 logit → Tier 2 monotone GBM → Tier 3 calibration
    ├── causal.py       # Tier 4 (optional) EconML elasticity — needs `econml`
    ├── optimize.py     # price sweep + guardrails → recommended price
    ├── evaluate.py     # calibration, AUC/PR, revenue simulation
    └── pipeline.py     # orchestrates the steps
```

## 5. The model ladder

| Tier | Model | Why |
|------|-------|-----|
| 1 | Logistic regression (log price) | Interpretable elasticity + sanity baseline |
| 2 | LightGBM / HistGBM, **monotone ↓ in price** | Captures price × loyalty × supply interactions; sensible demand curve |
| 3 | Probability calibration (sigmoid) | You multiply prob × price, so calibration > raw AUC |
| 4 | EconML DML / causal forest *(optional)* | Per-user causal elasticity for segmentation |

If `lightgbm` is not installed, Tier 2 automatically falls back to
scikit-learn's `HistGradientBoostingClassifier`, which also supports monotone
constraints and native missing-value handling — so the pipeline still runs.

## 6. Design notes (worth reading)

- **Leakage first.** `add_to_cart_flag` is an *outcome* and is never a model
  input. The `*_when_booked` / `*_when_bounced` columns are excluded by default
  (`config.LEAKAGE_SUSPECTS`) until you confirm they are computed strictly
  *before* `date`. Audit, then re-include if clean.
- **Missingness is informative.** History features are missing for new/thin
  users; we add `*_missing` flags instead of blindly imputing price anchors.
- **Split by time, not at random**, because you predict the *next* session.
- **The optimizer recomputes the price levers** for every candidate price, so
  the demand curve it sweeps is internally consistent.

## 7. Caveats

This is an **observational** pipeline: the price effect is identified only under
an unconfoundedness assumption (given the controls). The gold-standard
validation — and the only clean source of price variation — is an online
**A/B price test**. Treat the revenue-simulation numbers as directional.
