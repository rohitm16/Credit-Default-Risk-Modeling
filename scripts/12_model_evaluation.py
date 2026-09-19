import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/logistic_regression.pkl"

TRAIN_PATH = "outputs/train_woe.csv"
VALIDATION_PATH = "outputs/validation_woe.csv"
OOT_PATH = "outputs/oot_woe.csv"

OUTPUT_PATH = "outputs/model_metrics.csv"


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================
# IMPORTANT:
# F1, accuracy, precision, recall, etc. depend on this threshold.
#
# 0.5 is used here as a starting point.
# For credit risk, you should later select an appropriate
# threshold using the VALIDATION dataset and evaluate it on OOT.

THRESHOLD = 0.35


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(df, name, threshold=0.50):

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = df["default_flag"]

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------
    # Use exactly the features the model was trained on.

    X = df[
        model.feature_names_in_
    ]

    # --------------------------------------------------------
    # PREDICTED PROBABILITY OF DEFAULT
    # --------------------------------------------------------

    predicted_pd = model.predict_proba(
        X
    )[:, 1]

    # ========================================================
    # DISCRIMINATION METRICS
    # ========================================================

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y,
        predicted_pd
    )

    # --------------------------------------------------------
    # GINI
    # --------------------------------------------------------

    gini = (
        2 * auc
        - 1
    )

    # --------------------------------------------------------
    # KS
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y,
        predicted_pd
    )

    ks = max(
        tpr - fpr
    )

    # ========================================================
    # CLASSIFICATION PREDICTIONS
    # ========================================================

    predicted_default = (
        predicted_pd >= threshold
    ).astype(int)

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predicted_default
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y,
        predicted_default,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall / Sensitivity
    # --------------------------------------------------------

    recall = recall_score(
        y,
        predicted_default,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y,
        predicted_default,
        zero_division=0
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted_default,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # ========================================================
    # APPROVAL / DEFAULT RATES
    # ========================================================

    # predicted_default = 1 means predicted default
    # predicted_default = 0 means predicted non-default

    predicted_default_rate = (
        predicted_default.mean()
    )

    predicted_non_default_rate = (
        1 - predicted_default_rate
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Observations             : {len(df):,}"
    )

    print(
        f"Actual default rate      : {y.mean():.6f}"
    )

    print(
        f"Mean predicted PD        : {predicted_pd.mean():.6f}"
    )

    print(
        f"Classification threshold : {threshold:.2f}"
    )

    print("\n--- Discrimination ---")

    print(
        f"ROC-AUC                  : {auc:.6f}"
    )

    print(
        f"Gini                     : {gini:.6f}"
    )

    print(
        f"KS                       : {ks:.6f}"
    )

    print("\n--- Classification ---")

    print(
        f"Accuracy                 : {accuracy:.6f}"
    )

    print(
        f"Precision                : {precision:.6f}"
    )

    print(
        f"Recall                   : {recall:.6f}"
    )

    print(
        f"F1 Score                 : {f1:.6f}"
    )

    print(
        f"Specificity              : {specificity:.6f}"
    )

    print("\n--- Confusion Matrix ---")

    print(
        f"True Negatives (TN)      : {tn:,}"
    )

    print(
        f"False Positives (FP)     : {fp:,}"
    )

    print(
        f"False Negatives (FN)     : {fn:,}"
    )

    print(
        f"True Positives (TP)      : {tp:,}"
    )

    print("\n--- Classification Rates ---")

    print(
        f"Predicted Default Rate   : {predicted_default_rate:.6f}"
    )

    print(
        f"Predicted Non-Default    : {predicted_non_default_rate:.6f}"
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "dataset": name,

        "n_observations": len(df),

        "threshold": threshold,

        # Basic rates
        "actual_default_rate": y.mean(),
        "mean_predicted_pd": predicted_pd.mean(),

        # Discrimination
        "auc": auc,
        "gini": gini,
        "ks": ks,

        # Classification
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,

        # Confusion matrix
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,

        # Classification rates
        "predicted_default_rate": predicted_default_rate,
        "predicted_non_default_rate": predicted_non_default_rate
    }


# ============================================================
# EVALUATE
# ============================================================

results = []


results.append(
    evaluate_model(
        train,
        "TRAIN",
        threshold=THRESHOLD
    )
)


results.append(
    evaluate_model(
        validation,
        "VALIDATION",
        threshold=THRESHOLD
    )
)


results.append(
    evaluate_model(
        oot,
        "OOT",
        threshold=THRESHOLD
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(
    OUTPUT_PATH
)