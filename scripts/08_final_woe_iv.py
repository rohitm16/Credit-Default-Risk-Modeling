import pandas as pd
import numpy as np


TRAIN_PATH = "outputs/train.csv"
BOUNDARIES_PATH = "outputs/final_bin_boundaries.csv"

BIN_OUTPUT_PATH = "outputs/final_woe_iv_bins.csv"
IV_OUTPUT_PATH = "outputs/final_iv_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

boundaries = pd.read_csv(
    BOUNDARIES_PATH
)


# ============================================================
# CONVERT BOUNDARY VALUES
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


boundaries["lower_bound"] = boundaries["lower_bound"].apply(
    convert_bound
)

boundaries["upper_bound"] = boundaries["upper_bound"].apply(
    convert_bound
)


# ============================================================
# ASSIGN FINAL BINS
# ============================================================

def assign_bins(series, feature_boundaries):

    result = pd.Series(
        index=series.index,
        dtype="object"
    )

    # Numerical bins
    numerical = feature_boundaries[
        feature_boundaries["lower_bound"].notna()
        & feature_boundaries["upper_bound"].notna()
    ].copy()

    numerical = numerical.sort_values("bin_number")

    for _, row in numerical.iterrows():

        lower = row["lower_bound"]
        upper = row["upper_bound"]
        bin_number = int(row["bin_number"])

        mask = (
            series.notna()
            & (series > lower)
            & (series <= upper)
        )

        result.loc[mask] = f"Bin_{bin_number}"

    # Missing values
    result.loc[series.isna()] = "Missing"

    return result


# ============================================================
# CALCULATE WOE / IV
# ============================================================

def calculate_woe_iv(df, bins, target="default_flag"):

    temp = pd.DataFrame({
        "bin": bins,
        "target": df[target]
    })

    summary = (
        temp
        .groupby("bin", observed=False)["target"]
        .agg(
            total="count",
            bad="sum"
        )
    )

    summary["good"] = (
        summary["total"] - summary["bad"]
    )

    summary["default_rate"] = (
        summary["bad"] /
        summary["total"]
    )

    # Total Good / Bad
    total_good = summary["good"].sum()
    total_bad = summary["bad"].sum()

    # Distribution
    summary["dist_good"] = (
        summary["good"] / total_good
    )

    summary["dist_bad"] = (
        summary["bad"] / total_bad
    )

    # --------------------------------------------------------
    # 0.5 adjustment
    # --------------------------------------------------------

    n_bins = len(summary)

    summary["dist_good_adj"] = (
        (summary["good"] + 0.5) /
        (total_good + 0.5 * n_bins)
    )

    summary["dist_bad_adj"] = (
        (summary["bad"] + 0.5) /
        (total_bad + 0.5 * n_bins)
    )

    # --------------------------------------------------------
    # WoE
    # --------------------------------------------------------

    summary["woe"] = np.log(
        summary["dist_good_adj"] /
        summary["dist_bad_adj"]
    )

    # --------------------------------------------------------
    # IV contribution
    # --------------------------------------------------------

    summary["iv"] = (
        (
            summary["dist_good_adj"]
            - summary["dist_bad_adj"]
        )
        * summary["woe"]
    )

    return summary


# ============================================================
# PROCESS EVERY FEATURE
# ============================================================

all_woe = []
iv_summary = []


features = boundaries["feature"].unique()


for feature in features:

    if feature not in train.columns:
        print(
            f"WARNING: {feature} not found in train.csv"
        )
        continue

    feature_boundaries = boundaries[
        boundaries["feature"] == feature
    ].copy()

    # Assign observations to final bins
    bins = assign_bins(
        train[feature],
        feature_boundaries
    )

    # Calculate WoE / IV
    table = calculate_woe_iv(
        train,
        bins,
        target="default_flag"
    )

    # --------------------------------------------------------
    # Sort bins numerically
    # --------------------------------------------------------

    def sort_key(x):

        if x == "Missing":
            return float("inf")

        return int(x.split("_")[1])


    table = table.sort_index(
        key=lambda index: index.map(sort_key)
    )

    # Add feature name
    table.insert(
        0,
        "feature",
        feature
    )

    # Add bin name
    table.insert(
        1,
        "bin",
        table.index
    )

    table = table.reset_index(drop=True)

    all_woe.append(table)

    # Total IV
    total_iv = table["iv"].sum()

    iv_summary.append({
        "feature": feature,
        "iv": total_iv,
        "n_bins": len(table)
    })


# ============================================================
# COMBINE RESULTS
# ============================================================

final_woe = pd.concat(
    all_woe,
    ignore_index=True
)

iv_summary = pd.DataFrame(
    iv_summary
)


# ============================================================
# IV STRENGTH
# ============================================================

def iv_strength(iv):

    if iv < 0.02:
        return "Unpredictive"

    elif iv < 0.10:
        return "Weak"

    elif iv < 0.30:
        return "Medium"

    elif iv < 0.50:
        return "Strong"

    else:
        return "Suspicious"


iv_summary["strength"] = (
    iv_summary["iv"]
    .apply(iv_strength)
)


# Sort strongest first
iv_summary = iv_summary.sort_values(
    "iv",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

final_woe.to_csv(
    BIN_OUTPUT_PATH,
    index=False
)

iv_summary.to_csv(
    IV_OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL WOE / IV CALCULATION")
print("=" * 80)

print(
    f"\nSaved bin-level WoE/IV to:"
    f"\n{BIN_OUTPUT_PATH}"
)

print(
    f"\nSaved feature-level IV summary to:"
    f"\n{IV_OUTPUT_PATH}"
)

print("\n" + "=" * 80)
print("FINAL IV SUMMARY")
print("=" * 80)

print(iv_summary.to_string(index=False))