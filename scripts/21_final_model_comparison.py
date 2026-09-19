import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

LOGISTIC_PATH = "outputs/model_metrics.csv"
XGBOOST_PATH = "outputs/xgboost_metrics.csv"

OUTPUT_PATH = "outputs/final_model_comparison.csv"
IMAGE_PATH = "images/final_model_performance.png"


# ============================================================
# LOAD RESULTS
# ============================================================

logistic = pd.read_csv(LOGISTIC_PATH)
xgboost = pd.read_csv(XGBOOST_PATH)


# ============================================================
# CLEAN DATASET NAMES
# ============================================================

logistic["model"] = "Logistic Regression"
xgboost["model"] = "XGBoost Challenger"


# ============================================================
# COMBINE
# ============================================================

comparison = pd.concat(
    [logistic, xgboost],
    ignore_index=True
)


# ============================================================
# ORDER
# ============================================================

dataset_order = [
    "TRAIN",
    "VALIDATION",
    "OOT"
]

model_order = [
    "Logistic Regression",
    "XGBoost Challenger"
]

comparison["dataset"] = pd.Categorical(
    comparison["dataset"],
    categories=dataset_order,
    ordered=True
)

comparison["model"] = pd.Categorical(
    comparison["model"],
    categories=model_order,
    ordered=True
)

comparison = comparison.sort_values(
    ["dataset", "model"]
)


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 80)

print(
    comparison[
        [
            "model",
            "dataset",
            "n_observations",
            "default_rate",
            "mean_predicted_pd",
            "auc",
            "gini",
            "ks"
        ]
    ].to_string(index=False)
)


# ============================================================
# LOGISTIC TRAIN → VALIDATION → OOT
# ============================================================

logistic_only = comparison[
    comparison["model"] == "Logistic Regression"
].copy()

logistic_only = logistic_only.sort_values(
    "dataset"
)


print("\n" + "=" * 80)
print("LOGISTIC REGRESSION: TRAIN → VALIDATION → OOT")
print("=" * 80)

print(
    logistic_only[
        [
            "dataset",
            "default_rate",
            "mean_predicted_pd",
            "auc",
            "gini",
            "ks"
        ]
    ].to_string(index=False)
)


# ============================================================
# OOT DEGRADATION
# ============================================================

train_auc = float(
    logistic_only.loc[
        logistic_only["dataset"] == "TRAIN",
        "auc"
    ].iloc[0]
)

validation_auc = float(
    logistic_only.loc[
        logistic_only["dataset"] == "VALIDATION",
        "auc"
    ].iloc[0]
)

oot_auc = float(
    logistic_only.loc[
        logistic_only["dataset"] == "OOT",
        "auc"
    ].iloc[0]
)

train_gini = float(
    logistic_only.loc[
        logistic_only["dataset"] == "TRAIN",
        "gini"
    ].iloc[0]
)

oot_gini = float(
    logistic_only.loc[
        logistic_only["dataset"] == "OOT",
        "gini"
    ].iloc[0]
)

train_ks = float(
    logistic_only.loc[
        logistic_only["dataset"] == "TRAIN",
        "ks"
    ].iloc[0]
)

oot_ks = float(
    logistic_only.loc[
        logistic_only["dataset"] == "OOT",
        "ks"
    ].iloc[0]
)


auc_drop = train_auc - oot_auc
gini_drop = train_gini - oot_gini
ks_drop = train_ks - oot_ks

auc_drop_pct = (
    auc_drop / train_auc * 100
)

gini_drop_pct = (
    gini_drop / train_gini * 100
)

ks_drop_pct = (
    ks_drop / train_ks * 100
)


print("\n" + "=" * 80)
print("OOT PERFORMANCE CHANGE")
print("=" * 80)

print(
    f"AUC  : {train_auc:.4f} → {oot_auc:.4f} "
    f"(drop = {auc_drop:.4f}, {auc_drop_pct:.2f}%)"
)

print(
    f"Gini : {train_gini:.4f} → {oot_gini:.4f} "
    f"(drop = {gini_drop:.4f}, {gini_drop_pct:.2f}%)"
)

print(
    f"KS   : {train_ks:.4f} → {oot_ks:.4f} "
    f"(drop = {ks_drop:.4f}, {ks_drop_pct:.2f}%)"
)


# ============================================================
# XGBOOST VS LOGISTIC ON VALIDATION / OOT
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC VS XGBOOST")
print("=" * 80)

for dataset in ["VALIDATION", "OOT"]:

    logistic_row = comparison[
        (comparison["model"] == "Logistic Regression")
        & (comparison["dataset"] == dataset)
    ].iloc[0]

    xgb_row = comparison[
        (comparison["model"] == "XGBoost Challenger")
        & (comparison["dataset"] == dataset)
    ].iloc[0]

    print(f"\n{dataset}")

    print(
        f"Logistic AUC : {logistic_row['auc']:.4f}"
    )

    print(
        f"XGBoost AUC  : {xgb_row['auc']:.4f}"
    )

    print(
        f"Difference   : "
        f"{xgb_row['auc'] - logistic_row['auc']:+.4f}"
    )


# ============================================================
# SAVE TABLE
# ============================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PERFORMANCE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

for model in model_order:

    model_data = comparison[
        comparison["model"] == model
    ].sort_values("dataset")

    plt.plot(
        model_data["dataset"].astype(str),
        model_data["auc"],
        marker="o",
        label=model
    )


plt.xlabel("Dataset")
plt.ylabel("ROC-AUC")
plt.title("Model Performance: Train vs Validation vs OOT")

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    IMAGE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(IMAGE_PATH)