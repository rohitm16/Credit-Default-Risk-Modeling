import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/logistic_regression.pkl"

VALIDATION_PATH = "outputs/validation_woe.csv"
OOT_PATH = "outputs/oot_woe.csv"

CALIBRATION_MODEL_PATH = "models/calibration_model.pkl"

OUTPUT_PATH = "outputs/recalibration_results.csv"

IMAGE_PATH = "images/recalibration_curve.png"


# ============================================================
# LOAD
# ============================================================

model = joblib.load(
    MODEL_PATH
)

validation = pd.read_csv(
    VALIDATION_PATH,
    low_memory=False
)

oot = pd.read_csv(
    OOT_PATH,
    low_memory=False
)


# ============================================================
# ORIGINAL PREDICTIONS
# ============================================================

X_validation = validation[
    model.feature_names_in_
]

X_oot = oot[
    model.feature_names_in_
]

y_validation = validation[
    "default_flag"
]

y_oot = oot[
    "default_flag"
]


validation_pd = model.predict_proba(
    X_validation
)[:, 1]

oot_pd = model.predict_proba(
    X_oot
)[:, 1]


# ============================================================
# LOGIT TRANSFORMATION
# ============================================================

def logit(p):

    p = np.clip(
        p,
        1e-6,
        1 - 1e-6
    )

    return np.log(
        p / (1 - p)
    )


validation_logit = logit(
    validation_pd
)

oot_logit = logit(
    oot_pd
)


# ============================================================
# FIT CALIBRATION MODEL
# VALIDATION ONLY
# ============================================================

calibration_model = LogisticRegression(
    C=1e6,
    max_iter=1000
)

calibration_model.fit(
    validation_logit.reshape(-1, 1),
    y_validation
)


# ============================================================
# APPLY CALIBRATION
# ============================================================

validation_recalibrated_pd = (
    calibration_model.predict_proba(
        validation_logit.reshape(-1, 1)
    )[:, 1]
)

oot_recalibrated_pd = (
    calibration_model.predict_proba(
        oot_logit.reshape(-1, 1)
    )[:, 1]
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    y,
    predicted_pd
):

    auc = roc_auc_score(
        y,
        predicted_pd
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        predicted_pd
    )

    ks = max(
        tpr - fpr
    )

    gini = (
        2 * auc - 1
    )

    return {
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "actual_default_rate": y.mean(),
        "mean_predicted_pd": predicted_pd.mean()
    }


# ============================================================
# RESULTS
# ============================================================

results = []


# ------------------------------------------------------------
# VALIDATION - BEFORE
# ------------------------------------------------------------

metrics = calculate_metrics(
    y_validation,
    validation_pd
)

metrics["dataset"] = "VALIDATION"
metrics["stage"] = "BEFORE"

results.append(
    metrics
)


# ------------------------------------------------------------
# VALIDATION - AFTER
# ------------------------------------------------------------

metrics = calculate_metrics(
    y_validation,
    validation_recalibrated_pd
)

metrics["dataset"] = "VALIDATION"
metrics["stage"] = "AFTER"

results.append(
    metrics
)


# ------------------------------------------------------------
# OOT - BEFORE
# ------------------------------------------------------------

metrics = calculate_metrics(
    y_oot,
    oot_pd
)

metrics["dataset"] = "OOT"
metrics["stage"] = "BEFORE"

results.append(
    metrics
)


# ------------------------------------------------------------
# OOT - AFTER
# ------------------------------------------------------------

metrics = calculate_metrics(
    y_oot,
    oot_recalibrated_pd
)

metrics["dataset"] = "OOT"
metrics["stage"] = "AFTER"

results.append(
    metrics
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df[
    [
        "dataset",
        "stage",
        "actual_default_rate",
        "mean_predicted_pd",
        "auc",
        "gini",
        "ks"
    ]
]

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION MODEL")
print("=" * 80)

print(
    f"Calibration intercept : "
    f"{calibration_model.intercept_[0]:.6f}"
)

print(
    f"Calibration slope     : "
    f"{calibration_model.coef_[0][0]:.6f}"
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CALIBRATION MODEL
# ============================================================

joblib.dump(
    calibration_model,
    CALIBRATION_MODEL_PATH
)


# ============================================================
# CALIBRATION CURVE
# ============================================================

plt.figure(
    figsize=(8, 7)
)

# Validation
val_bins = pd.DataFrame({
    "predicted": validation_recalibrated_pd,
    "actual": y_validation
})

val_bins["bin"] = pd.qcut(
    val_bins["predicted"],
    10,
    labels=False,
    duplicates="drop"
)

val_curve = (
    val_bins
    .groupby("bin")
    .agg(
        predicted=("predicted", "mean"),
        actual=("actual", "mean")
    )
)

plt.plot(
    val_curve["predicted"],
    val_curve["actual"],
    marker="o",
    label="Validation"
)


# OOT
oot_bins = pd.DataFrame({
    "predicted": oot_recalibrated_pd,
    "actual": y_oot
})

oot_bins["bin"] = pd.qcut(
    oot_bins["predicted"],
    10,
    labels=False,
    duplicates="drop"
)

oot_curve = (
    oot_bins
    .groupby("bin")
    .agg(
        predicted=("predicted", "mean"),
        actual=("actual", "mean")
    )
)

plt.plot(
    oot_curve["predicted"],
    oot_curve["actual"],
    marker="o",
    label="OOT"
)


# Perfect calibration
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Default Rate"
)

plt.title(
    "Calibration After Recalibration"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    IMAGE_PATH,
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

print(OUTPUT_PATH)
print(CALIBRATION_MODEL_PATH)
print(IMAGE_PATH)