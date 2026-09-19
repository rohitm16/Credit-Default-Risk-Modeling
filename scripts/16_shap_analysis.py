import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/xgboost_challenger.pkl"

OOT_PATH = "outputs/oot_woe.csv"

IMPORTANCE_OUTPUT = "outputs/xgboost_shap_importance.csv"
IMAGE_OUTPUT = "images/xgboost_shap_summary.png"


# ============================================================
# LOAD
# ============================================================

model = joblib.load(
    MODEL_PATH
)

oot = pd.read_csv(
    OOT_PATH,
    low_memory=False
)


# ============================================================
# FEATURES
# ============================================================

TARGET = "default_flag"

features = [
    col
    for col in oot.columns
    if col != TARGET
]

X = oot[features]


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 80)
print("SHAP ANALYSIS")
print("=" * 80)

print(f"Observations : {len(X):,}")
print(f"Features     : {len(features)}")


explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X
)


# ============================================================
# SHAP IMPORTANCE
# ============================================================

mean_abs_shap = (
    abs(shap_values)
    .mean(axis=0)
)

importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": mean_abs_shap
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE IMPORTANCE
# ============================================================

importance.to_csv(
    IMPORTANCE_OUTPUT,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 SHAP FEATURES")
print("=" * 80)

print(
    importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

shap.summary_plot(
    shap_values,
    X,
    show=False,
    max_display=20
)

plt.title(
    "XGBoost SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    IMAGE_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(IMPORTANCE_OUTPUT)
print(IMAGE_OUTPUT)