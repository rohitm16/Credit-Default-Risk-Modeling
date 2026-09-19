import pandas as pd
import numpy as np
import joblib


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/logistic_regression.pkl"

WOE_IV_PATH = "outputs/final_woe_iv_all.csv"

OUTPUT_PATH = "outputs/final_scorecard.csv"


# ============================================================
# SCORECARD SETTINGS
# ============================================================

# Standard scorecard scaling:
# Score = BASE_SCORE + PDO * log2(odds / BASE_ODDS)

BASE_SCORE = 600
PDO = 20

# Odds = Good / Bad
BASE_ODDS = 4


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({
    "feature": model.feature_names_in_,
    "coefficient": model.coef_[0]
})


# ============================================================
# SCORE SCALING
# ============================================================

# Logistic model:
#
# logit(PD) = intercept + Σ(beta_i * WoE_i)
#
# Score should decrease as default risk increases.
#
# Therefore:
#
# Score = BASE_SCORE
#         - PDO / ln(2)
#         * (logit(PD) - logit(BASE_PD))
#
# where BASE_ODDS = (1-PD)/PD


base_log_odds = np.log(
    BASE_ODDS
)

factor = PDO / np.log(2)

offset = (
    BASE_SCORE
    + factor * base_log_odds
)


# ============================================================
# FEATURE SCORE CONTRIBUTION
# ============================================================

coefficients["factor"] = factor

coefficients["score_per_woe"] = (
    -factor
    * coefficients["coefficient"]
)


# ============================================================
# SCORECARD SUMMARY
# ============================================================

scorecard = coefficients[
    [
        "feature",
        "coefficient",
        "score_per_woe"
    ]
].copy()


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("FINAL CREDIT SCORECARD")
print("=" * 80)

print(
    f"Base score       : {BASE_SCORE}"
)

print(
    f"Points to double : {PDO}"
)

print(
    f"Base odds        : {BASE_ODDS}:1"
)

print(
    f"Factor           : {factor:.6f}"
)

print(
    f"Offset           : {offset:.6f}"
)


print("\n" + "=" * 80)
print("FEATURE SCORE CONTRIBUTIONS")
print("=" * 80)

print(
    scorecard
    .sort_values(
        "score_per_woe",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

scorecard.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)