import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "outputs/train_selected.csv"
VALIDATION_PATH = "outputs/validation.csv"
OOT_PATH = "outputs/oot_test.csv"

OUTPUT_PATH = "outputs/psi_summary.csv"
IMAGE_PATH = "images/psi_summary.png"


# ============================================================
# SETTINGS
# ============================================================

TARGET = "default_flag"
N_BINS = 10


# ============================================================
# LOAD
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
# FEATURES
# ============================================================

features = [
    col for col in train.columns
    if col != TARGET
]


# ============================================================
# PSI FUNCTION
# ============================================================

def calculate_psi(
    expected,
    actual
):

    expected = np.asarray(
        expected,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    # Avoid division by zero
    expected = np.clip(
        expected,
        0.0001,
        None
    )

    actual = np.clip(
        actual,
        0.0001,
        None
    )

    psi = np.sum(
        (actual - expected)
        *
        np.log(actual / expected)
    )

    return psi


# ============================================================
# NUMERICAL PSI
# ============================================================

def numerical_psi(
    train_series,
    comparison_series
):

    train_series = train_series.dropna()
    comparison_series = comparison_series.dropna()

    if train_series.nunique() < 2:
        return 0.0

    # Quantile bins learned from TRAIN
    quantiles = np.linspace(
        0,
        1,
        N_BINS + 1
    )

    breakpoints = (
        train_series
        .quantile(quantiles)
        .drop_duplicates()
        .values
    )

    if len(breakpoints) < 2:
        return 0.0

    # Make boundaries infinite
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    train_bins = pd.cut(
        train_series,
        bins=breakpoints,
        include_lowest=True
    )

    comparison_bins = pd.cut(
        comparison_series,
        bins=breakpoints,
        include_lowest=True
    )

    train_dist = (
        train_bins
        .value_counts(
            normalize=True,
            sort=False
        )
    )

    comparison_dist = (
        comparison_bins
        .value_counts(
            normalize=True,
            sort=False
        )
    )

    return calculate_psi(
        train_dist.values,
        comparison_dist.values
    )


# ============================================================
# CATEGORICAL PSI
# ============================================================

def categorical_psi(
    train_series,
    comparison_series
):

    train_series = (
        train_series
        .astype("object")
        .fillna("Missing")
        .astype(str)
    )

    comparison_series = (
        comparison_series
        .astype("object")
        .fillna("Missing")
        .astype(str)
    )

    # Categories learned from TRAIN
    categories = sorted(
        set(train_series.unique())
        |
        set(comparison_series.unique())
    )

    train_dist = (
        train_series
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
    )

    comparison_dist = (
        comparison_series
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
    )

    return calculate_psi(
        train_dist.values,
        comparison_dist.values
    )


# ============================================================
# CALCULATE PSI
# ============================================================

results = []

for feature in features:

    train_series = train[feature]
    validation_series = validation[feature]
    oot_series = oot[feature]

    if pd.api.types.is_numeric_dtype(
        train_series
    ):

        validation_psi = numerical_psi(
            train_series,
            validation_series
        )

        oot_psi = numerical_psi(
            train_series,
            oot_series
        )

    else:

        validation_psi = categorical_psi(
            train_series,
            validation_series
        )

        oot_psi = categorical_psi(
            train_series,
            oot_series
        )

    results.append({
        "feature": feature,
        "psi_train_validation": validation_psi,
        "psi_train_oot": oot_psi
    })


# ============================================================
# SUMMARY
# ============================================================

psi = pd.DataFrame(
    results
)

psi = psi.sort_values(
    "psi_train_oot",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# CLASSIFICATION
# ============================================================

def psi_level(value):

    if value < 0.10:
        return "Stable"

    elif value < 0.25:
        return "Moderate shift"

    else:
        return "Significant shift"


psi["oot_shift"] = (
    psi["psi_train_oot"]
    .apply(psi_level)
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("POPULATION STABILITY INDEX")
print("=" * 80)

print(
    psi.to_string(
        index=False
    )
)


# ============================================================
# SAVE CSV
# ============================================================

psi.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PLOT TOP 15 OOT PSI
# ============================================================

top_15 = (
    psi
    .head(15)
    .sort_values(
        "psi_train_oot"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_15["feature"],
    top_15["psi_train_oot"]
)

plt.axvline(
    0.10,
    linestyle="--",
    label="0.10"
)

plt.axvline(
    0.25,
    linestyle="--",
    label="0.25"
)

plt.xlabel(
    "PSI (Train → OOT)"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Population Stability Index — Top 15 Features"
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
# DONE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(IMAGE_PATH)