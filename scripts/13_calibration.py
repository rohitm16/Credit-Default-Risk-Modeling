import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/logistic_regression.pkl"

TRAIN_PATH = "outputs/train_woe.csv"
VALIDATION_PATH = "outputs/validation_woe.csv"
OOT_PATH = "outputs/oot_woe.csv"

OUTPUT_PATH = "outputs/calibration_summary.csv"

IMAGE_PATH = "images/calibration_curve.png"


# ============================================================
# LOAD
# ============================================================

model = joblib.load(MODEL_PATH)

datasets = {
    "TRAIN": pd.read_csv(
        TRAIN_PATH,
        low_memory=False
    ),
    "VALIDATION": pd.read_csv(
        VALIDATION_PATH,
        low_memory=False
    ),
    "OOT": pd.read_csv(
        OOT_PATH,
        low_memory=False
    )
}


# ============================================================
# CALIBRATION TABLE
# ============================================================

def calibration_table(df, name):

    y = df["default_flag"]

    X = df[
        model.feature_names_in_
    ]

    predicted_pd = model.predict_proba(
        X
    )[:, 1]

    temp = pd.DataFrame({
        "actual": y,
        "predicted_pd": predicted_pd
    })

    temp["decile"] = pd.qcut(
        temp["predicted_pd"],
        q=10,
        labels=False,
        duplicates="drop"
    ) + 1

    result = (
        temp
        .groupby("decile")
        .agg(
            observations=("actual", "count"),
            actual_default_rate=("actual", "mean"),
            average_predicted_pd=("predicted_pd", "mean")
        )
        .reset_index()
    )

    result["difference"] = (
        result["average_predicted_pd"]
        - result["actual_default_rate"]
    )

    result.insert(
        0,
        "dataset",
        name
    )

    return result, predicted_pd


# ============================================================
# CALCULATE
# ============================================================

results = []
plot_data = {}

for name, df in datasets.items():

    table, predicted_pd = calibration_table(
        df,
        name
    )

    results.append(table)

    plot_data[name] = (
        df["default_flag"].values,
        predicted_pd
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# SAVE CALIBRATION TABLE
# ============================================================

calibration = pd.concat(
    results,
    ignore_index=True
)

calibration.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# CALIBRATION PLOT
# ============================================================

plt.figure(figsize=(8, 7))

for name, (y, predicted_pd) in plot_data.items():

    fraction_of_positives, mean_predicted_value = (
        calibration_curve(
            y,
            predicted_pd,
            n_bins=10,
            strategy="quantile"
        )
    )

    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label=name
    )


# Perfect calibration line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Default Rate")

plt.title("Calibration Curve")

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
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(IMAGE_PATH)