import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "outputs/train_selected.csv"
VALIDATION_PATH = "outputs/validation.csv"
OOT_PATH = "outputs/oot_test.csv"

BOUNDARIES_PATH = "outputs/final_bin_boundaries.csv"

TRAIN_WOE_OUTPUT = "outputs/train_woe.csv"
VALIDATION_WOE_OUTPUT = "outputs/validation_woe.csv"
OOT_WOE_OUTPUT = "outputs/oot_woe.csv"

WOE_IV_OUTPUT = "outputs/final_woe_iv_all.csv"


# ============================================================
# SETTINGS
# ============================================================

TARGET = "default_flag"


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

boundaries = pd.read_csv(
    BOUNDARIES_PATH
)


# ============================================================
# CONVERT BOUNDARIES
# ============================================================

def convert_bound(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value == "-inf":
        return -np.inf

    if value == "inf":
        return np.inf

    return float(value)


boundaries["lower_bound"] = (
    boundaries["lower_bound"]
    .apply(convert_bound)
)

boundaries["upper_bound"] = (
    boundaries["upper_bound"]
    .apply(convert_bound)
)


# ============================================================
# FEATURES
# ============================================================

features = [
    col
    for col in train.columns
    if col != TARGET
]

boundary_features = set(
    boundaries["feature"].unique()
)

numerical_features = [
    feature
    for feature in features
    if feature in boundary_features
]

categorical_features = [
    feature
    for feature in features
    if feature not in boundary_features
]


print("\n" + "=" * 80)
print("FEATURES")
print("=" * 80)

print(f"Total predictors : {len(features)}")
print(f"Numerical        : {len(numerical_features)}")
print(f"Categorical      : {len(categorical_features)}")

print("\nCategorical features:")

for feature in categorical_features:
    print(f"  - {feature}")


# ============================================================
# NUMERICAL BIN ASSIGNMENT
# ============================================================

def assign_numerical_bins(
    df,
    feature,
    feature_boundaries
):

    bins = pd.Series(
        np.nan,
        index=df.index,
        dtype="object"
    )

    feature_boundaries = (
        feature_boundaries
        .sort_values("bin_number")
    )

    for _, row in feature_boundaries.iterrows():

        lower = row["lower_bound"]
        upper = row["upper_bound"]

        bin_number = int(
            row["bin_number"]
        )

        mask = (
            df[feature].notna()
            & (df[feature] > lower)
            & (df[feature] <= upper)
        )

        bins.loc[mask] = (
            f"Bin_{bin_number}"
        )

    # Missing values
    bins.loc[
        df[feature].isna()
    ] = "Missing"

    return bins


# ============================================================
# CALCULATE WOE MAPPING FROM TRAIN ONLY
# ============================================================

def calculate_woe_mapping(
    bins,
    target
):

    temp = pd.DataFrame({
        "bin": bins,
        "target": target
    })

    summary = (
        temp
        .groupby(
            "bin",
            observed=False
        )["target"]
        .agg(
            total="count",
            bad="sum"
        )
    )

    summary["good"] = (
        summary["total"]
        - summary["bad"]
    )

    summary["default_rate"] = (
        summary["bad"]
        / summary["total"]
    )

    total_good = summary["good"].sum()
    total_bad = summary["bad"].sum()

    n_bins = len(summary)

    # Smoothing prevents log(0)
    summary["dist_good"] = (
        (summary["good"] + 0.5)
        /
        (total_good + 0.5 * n_bins)
    )

    summary["dist_bad"] = (
        (summary["bad"] + 0.5)
        /
        (total_bad + 0.5 * n_bins)
    )

    summary["woe"] = np.log(
        summary["dist_good"]
        /
        summary["dist_bad"]
    )

    summary["iv"] = (
        (
            summary["dist_good"]
            -
            summary["dist_bad"]
        )
        *
        summary["woe"]
    )

    mapping = (
        summary["woe"]
        .to_dict()
    )

    return mapping, summary


# ============================================================
# TRANSFORM ONE DATASET
# ============================================================

def transform_dataset(
    df,
    mappings,
    numerical_boundaries
):

    result = pd.DataFrame(
        index=df.index
    )

    result[TARGET] = df[TARGET]

    for feature in features:

        # ----------------------------------------------------
        # Numerical
        # ----------------------------------------------------

        if feature in numerical_features:

            bins = assign_numerical_bins(
                df,
                feature,
                numerical_boundaries[
                    numerical_boundaries["feature"]
                    == feature
                ]
            )

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        else:

            bins = (
                df[feature]
                .astype("object")
                .fillna("Missing")
                .astype(str)
            )

        # ----------------------------------------------------
        # Apply TRAIN WoE mapping
        # ----------------------------------------------------

        result[feature] = (
            bins
            .map(
                mappings[feature]
            )
            .fillna(0.0)
        )

    return result


# ============================================================
# BUILD TRAIN WOE MAPPINGS
# ============================================================

print("\n" + "=" * 80)
print("LEARNING WOE MAPPINGS FROM TRAIN")
print("=" * 80)

woe_mappings = {}
all_tables = []


# ============================================================
# NUMERICAL FEATURES
# ============================================================

for feature in numerical_features:

    bins = assign_numerical_bins(
        train,
        feature,
        boundaries[
            boundaries["feature"] == feature
        ]
    )

    mapping, summary = calculate_woe_mapping(
        bins,
        train[TARGET]
    )

    woe_mappings[feature] = mapping

    summary.insert(
        0,
        "feature",
        feature
    )

    all_tables.append(
        summary.reset_index()
    )

    print(
        f"{feature:35s} numerical"
    )


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

for feature in categorical_features:

    bins = (
        train[feature]
        .astype("object")
        .fillna("Missing")
        .astype(str)
    )

    mapping, summary = calculate_woe_mapping(
        bins,
        train[TARGET]
    )

    woe_mappings[feature] = mapping

    summary.insert(
        0,
        "feature",
        feature
    )

    all_tables.append(
        summary.reset_index()
    )

    print(
        f"{feature:35s} categorical"
    )


# ============================================================
# TRANSFORM ALL DATASETS
# ============================================================

print("\n" + "=" * 80)
print("APPLYING TRAIN WOE MAPPINGS")
print("=" * 80)

train_woe = transform_dataset(
    train,
    woe_mappings,
    boundaries
)

validation_woe = transform_dataset(
    validation,
    woe_mappings,
    boundaries
)

oot_woe = transform_dataset(
    oot,
    woe_mappings,
    boundaries
)


# ============================================================
# SAVE WOE DATASETS
# ============================================================

train_woe.to_csv(
    TRAIN_WOE_OUTPUT,
    index=False
)

validation_woe.to_csv(
    VALIDATION_WOE_OUTPUT,
    index=False
)

oot_woe.to_csv(
    OOT_WOE_OUTPUT,
    index=False
)


# ============================================================
# SAVE WOE / IV TABLE
# ============================================================

final_woe_iv = pd.concat(
    all_tables,
    ignore_index=True
)

final_woe_iv.to_csv(
    WOE_IV_OUTPUT,
    index=False
)


# ============================================================
# FINAL IV SUMMARY
# ============================================================

iv_summary = (
    final_woe_iv
    .groupby(
        "feature",
        as_index=False
    )["iv"]
    .sum()
    .sort_values(
        "iv",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL WOE TRANSFORMATION")
print("=" * 80)

print(
    f"Train shape      : {train_woe.shape}"
)

print(
    f"Validation shape : {validation_woe.shape}"
)

print(
    f"OOT shape        : {oot_woe.shape}"
)

print(
    f"Predictors       : {len(features)}"
)

print(
    f"Numerical        : {len(numerical_features)}"
)

print(
    f"Categorical      : {len(categorical_features)}"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING WOE VALUES")
print("=" * 80)

print(
    "Train      :",
    train_woe.isna().sum().sum()
)

print(
    "Validation :",
    validation_woe.isna().sum().sum()
)

print(
    "OOT        :",
    oot_woe.isna().sum().sum()
)


# ============================================================
# IV SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL IV SUMMARY")
print("=" * 80)

print(
    iv_summary.to_string(
        index=False
    )
)


# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(TRAIN_WOE_OUTPUT)
print(VALIDATION_WOE_OUTPUT)
print(OOT_WOE_OUTPUT)
print(WOE_IV_OUTPUT)