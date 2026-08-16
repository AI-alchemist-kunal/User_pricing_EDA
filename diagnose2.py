import sys
sys.path.insert(0, ".")
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import config as C
from src.data_load import load_data, validate
from src.features import build_features, get_model_features, price_levers_scalar
from src.split import temporal_split
from src.model import train_demand_model, predict_proba

df, _ = validate(load_data("data/pricing_data.xlsx"))
dff = build_features(df)
feats = get_model_features(dff)
train, test = temporal_split(dff)
model, _ = train_demand_model(train, feats)

# TEST 1 — permutation importance on the held-out test set (real contribution)
Xte, yte = test[feats], test[C.TARGET].values
print("Test AUC =", round(roc_auc_score(yte, predict_proba(model, Xte)), 4))
r = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
print("\nPermutation importance (AUC drop when the feature is shuffled):")
print(pd.Series(r.importances_mean, index=feats).sort_values(ascending=False).head(12).round(4))

# TEST 2 — does P(book) actually fall as price rises, for one user?
row = test.iloc[0]
anchors = (row.get("last_booked_price"), row.get("max_effective_price_ever_booked"),
           row.get("last_seen_price"), row.get("avg_effective_price_when_bounced"))
base = row[feats].to_dict()
print("\nPrice sweep for one user (P(book) should fall as price rises):")
for mult in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
    price = row[C.PRICE_COL] * mult
    f = dict(base)
    for k, v in price_levers_scalar(price, *anchors).items():
        if k in f: f[k] = v
    p = predict_proba(model, pd.DataFrame([f])[feats])[0]
    print(f"  x{mult:.1f}  price={price:7.1f}  P(book)={p:.3f}  revenue={price*p:7.1f}")