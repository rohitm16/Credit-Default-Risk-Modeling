import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

TRAIN_WOE_PATH = "outputs/train_woe.csv"
MODEL_PATH = "models/logistic_regression.pkl"


# ============================================================
# LOAD
# ============================================================

train = pd.read_csv(
    TRAIN_WOE_PATH,
    low_memory=False
)


# ============================================================
# TARGET / FEATURES
# ============================================================

TARGET = "default_flag"

X = train.drop(
    columns=[TARGET]
)

y = train[TARGET]


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION")
print("=" * 80)

print(
    f"Training rows : {X.shape[0]}"
)

print(
    f"Features      : {X.shape[1]}"
)

print(
    f"Intercept     : {model.intercept_[0]:.6f}"
)

print(
    f"Iterations    : {model.n_iter_[0]}"
)


print("\n" + "=" * 80)
print("COEFFICIENTS")
print("=" * 80)

coef_table = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})

print(
    coef_table
    .sort_values(
        "coefficient",
        ascending=False
    )
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(MODEL_PATH)