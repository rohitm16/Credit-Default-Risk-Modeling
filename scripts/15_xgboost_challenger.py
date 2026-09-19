import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "outputs/train_woe.csv"
VALIDATION_PATH = "outputs/validation_woe.csv"
OOT_PATH = "outputs/oot_woe.csv"

MODEL_PATH = "models/xgboost_challenger.pkl"
OUTPUT_PATH = "outputs/xgboost_metrics.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
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
# FEATURES / TARGET
# ============================================================

TARGET = "default_flag"

features = [
    col for col in train.columns
    if col != TARGET
]

X_train = train[features]
y_train = train[TARGET]

X_validation = validation[features]
y_validation = validation[TARGET]

X_oot = oot[features]
y_oot = oot[TARGET]


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST CHALLENGER")
print("=" * 80)

print(f"Training rows : {len(X_train):,}")
print(f"Features      : {len(features)}")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_validation, y_validation)
    ],
    verbose=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    X,
    y,
    name
):

    predicted_pd = model.predict_proba(
        X
    )[:, 1]

    auc = roc_auc_score(
        y,
        predicted_pd
    )

    gini = (
        2 * auc
        - 1
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        predicted_pd
    )

    ks = max(
        tpr - fpr
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Observations      : {len(y):,}"
    )

    print(
        f"Default rate      : {y.mean():.6f}"
    )

    print(
        f"Mean predicted PD : {predicted_pd.mean():.6f}"
    )

    print(
        f"ROC-AUC           : {auc:.6f}"
    )

    print(
        f"Gini              : {gini:.6f}"
    )

    print(
        f"KS                : {ks:.6f}"
    )

    return {
        "dataset": name,
        "n_observations": len(y),
        "default_rate": y.mean(),
        "mean_predicted_pd": predicted_pd.mean(),
        "auc": auc,
        "gini": gini,
        "ks": ks
    }


# ============================================================
# RUN EVALUATION
# ============================================================

results = []

results.append(
    evaluate_model(
        X_train,
        y_train,
        "TRAIN"
    )
)

results.append(
    evaluate_model(
        X_validation,
        y_validation,
        "VALIDATION"
    )
)

results.append(
    evaluate_model(
        X_oot,
        y_oot,
        "OOT"
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    "outputs/xgboost_feature_importance.csv",
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("TOP 15 FEATURES")
print("=" * 80)

print(
    importance.head(15).to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(MODEL_PATH)
print(OUTPUT_PATH)
print("outputs/xgboost_feature_importance.csv")