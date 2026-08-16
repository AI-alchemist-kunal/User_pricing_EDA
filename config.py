"""
Central configuration for the user-pricing pipeline.

Every module imports from this file, so this is the single place to edit
paths, column names, split settings and optimizer guardrails.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (DATA_DIR, OUTPUT_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Drop your dataset here, or pass --data on the command line.
DEFAULT_DATA_FILE = DATA_DIR / "pricing_data.xlsx"

# --------------------------------------------------------------------------- #
# Core columns
# --------------------------------------------------------------------------- #
TARGET = "booked_flag"
PRICE_COL = "effective_price"
ID_COL = "user_id"
DATE_COL = "date"

# Outcome / post-treatment columns: never allowed as model inputs.
POST_TREATMENT = ["add_to_cart_flag"]

# Historical price anchors used to build the relative-price levers.
# (These are decision-time references — the user's OWN past prices.)
ANCHORS = {
    "last_booked_price": "last_booked_price",
    "last_seen_price": "last_seen_price",
    "max_booked": "max_effective_price_ever_booked",
    "bounce": "avg_effective_price_when_bounced",
}

# Engineered relative-price levers (created in features.py).
# The demand model is constrained to be DECREASING in each of these.
PRICE_LEVERS = [
    "price_vs_lastbooked",
    "price_vs_maxbooked",
    "price_minus_lastseen",
    "price_in_bounce_band",
]

# Columns to AUDIT for leakage (derived from booking/bounce outcomes).
# Excluded from the model by default — audit them, then decide.
LEAKAGE_SUSPECTS = [
    "coupon_discount",
    "avg_effective_price_when_booked",
    "avg_discount_when_booked",
    "avg_effective_price_when_bounced",
    "avg_discount_when_bounced",
    "max_effective_price_when_bounced",
    "last_3_booking_discount_avg",
]

CATEGORICALS = ["city", "client_type", "country"]

# Loyalty proxy used for the "confounding" EDA plot (quartile strata).
LOYALTY_COL = "bookings_last_90d"

# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
TIME_SPLIT_QUANTILE = 0.80   # train on the earliest 80% of dates

# --------------------------------------------------------------------------- #
# Optimizer / guardrails
# --------------------------------------------------------------------------- #
PRICE_GRID_LOW = 0.80        # sweep candidate prices from 0.8x ...
PRICE_GRID_HIGH = 1.30       # ... to 1.3x the current price
PRICE_GRID_STEPS = 26
MAX_PRICE_UPLIFT = 0.30      # guardrail: never raise more than +30%
MAX_PRICE_DROP = 0.20        # guardrail: never drop more than -20%

RANDOM_SEED = 42
