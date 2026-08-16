import sys
sys.path.insert(0, ".")
import pandas as pd
from lightgbm import LGBMClassifier
from src.data_load import load_data, validate
from src.features import build_features, get_model_features
import config as C

df, _ = validate(load_data("data/pricing_data.xlsx"))
dff = build_features(df)
feats = get_model_features(dff)
model = LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
model.fit(dff[feats], dff[C.TARGET])

importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print(importances.head(15))