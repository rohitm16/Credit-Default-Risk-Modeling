import pandas as pd
import numpy as np


INPUT_PATH = "outputs/train.csv"


def make_quantile_bins(df, feature, n_bins=10):

    x = df[feature]
    non_missing = x.dropna()

    bins = np.unique(
        non_missing.quantile(
            np.linspace(0, 1, n_bins + 1)
        ).values
    )

    # Create bins for non-missing values
    non_missing_bins = pd.cut(
        x.loc[x.notna()],
        bins=bins,
        include_lowest=True,
        duplicates="drop"
    )

    # Use object dtype so we can later add "Missing"
    binned = pd.Series(
        index=df.index,
        dtype="object"
    )

    binned.loc[x.notna()] = non_missing_bins.astype(object)

    # Missing values are their own bin
    binned.loc[x.isna()] = "Missing"

    return binned

def calculate_woe_iv(df, bins, target="default_flag"):
    """
    Calculate Good/Bad distributions, WoE and IV
    for each bin.
    """

    temp = pd.DataFrame({
        "bin": bins,
        "target": df[target]
    })

    # Good = 0, Bad = 1
    summary = (
        temp
        .groupby("bin", observed=False)["target"]
        .agg(
            total="count",
            bad="sum"
        )
    )

    summary["good"] = summary["total"] - summary["bad"]

    # Total Good and Bad
    total_good = summary["good"].sum()
    total_bad = summary["bad"].sum()

    # Distributions
    summary["dist_good"] = summary["good"] / total_good
    summary["dist_bad"] = summary["bad"] / total_bad

    # Avoid log(0)
    # 0.5 adjustment for sparse bins
    summary["dist_good_adj"] = (
        (summary["good"] + 0.5) /
        (total_good + 0.5 * len(summary))
    )

    summary["dist_bad_adj"] = (
        (summary["bad"] + 0.5) /
        (total_bad + 0.5 * len(summary))
    )

    # WoE
    summary["woe"] = np.log(
        summary["dist_good_adj"] /
        summary["dist_bad_adj"]
    )

    # IV contribution
    summary["iv"] = (
        (summary["dist_good_adj"] -
         summary["dist_bad_adj"])
        * summary["woe"]
    )

    return summary

if __name__ == "__main__":

    train = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    target = "default_flag"

    # --------------------------------------------------
    # Identify numerical features
    # --------------------------------------------------

    numerical_features = train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Remove target from predictors
    numerical_features.remove(target)

    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES")
    print("=" * 80)

    print(f"Number of numerical features: {len(numerical_features)}")

    # --------------------------------------------------
    # Calculate WoE / IV for every numerical feature
    # --------------------------------------------------

    iv_results = []

    all_woe_tables = {}

    for feature in numerical_features:

        # Skip completely missing columns
        if train[feature].notna().sum() == 0:
            continue

        # Initial quantile bins
        bins = make_quantile_bins(
            train,
            feature,
            n_bins=10
        )

        # WoE / IV
        woe_table = calculate_woe_iv(
            train,
            bins,
            target=target
        )

        total_iv = woe_table["iv"].sum()

        iv_results.append({
            "feature": feature,
            "iv": total_iv,
            "non_missing": train[feature].notna().sum(),
            "missing": train[feature].isna().sum(),
            "n_bins": len(woe_table)
        })

        all_woe_tables[feature] = woe_table

    # --------------------------------------------------
    # IV summary
    # --------------------------------------------------

    iv_summary = pd.DataFrame(iv_results)

    iv_summary = iv_summary.sort_values(
        "iv",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("IV SUMMARY — INITIAL QUANTILE BINS")
    print("=" * 80)

    print(iv_summary.to_string(index=False))

    # --------------------------------------------------
    # IV interpretation
    # --------------------------------------------------

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

    iv_summary["strength"] = iv_summary["iv"].apply(
        iv_strength
    )

    print("\n" + "=" * 80)
    print("IV SUMMARY WITH STRENGTH")
    print("=" * 80)

    print(iv_summary.to_string(index=False))

    # --------------------------------------------------
    # Save IV summary
    # --------------------------------------------------

    OUTPUT_PATH = "outputs/iv_summary_initial.csv"

    iv_summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT_PATH)