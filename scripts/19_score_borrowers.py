import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "outputs/validation.csv"
SCORECARD_PATH = "outputs/final_bin_scorecard.csv"

OUTPUT_PATH = "outputs/validation_scored.csv"


# ============================================================
# LOAD
# ============================================================

data = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

scorecard = pd.read_csv(
    SCORECARD_PATH
)


# ============================================================
# START SCORE
# ============================================================

BASE_SCORE = 600

data["credit_score"] = BASE_SCORE


# ============================================================
# FEATURES
# ============================================================

features = scorecard["feature"].unique()


# ============================================================
# SCORE EACH FEATURE
# ============================================================

for feature in features:

    if feature not in data.columns:
        print(f"WARNING: {feature} not found. Skipping.")
        continue

    feature_card = scorecard[
        scorecard["feature"] == feature
    ].copy()

    def calculate_points(value):

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        if pd.isna(value):

            missing = feature_card[
                feature_card["bin"].astype(str).str.lower()
                == "missing"
            ]

            if len(missing) > 0:
                return float(
                    missing.iloc[0]["points"]
                )

            return 0.0

        # ----------------------------------------------------
        # Numerical bins
        # ----------------------------------------------------

        numeric_bins = feature_card[
            feature_card["lower_bound"].notna()
            & feature_card["upper_bound"].notna()
        ]

        if len(numeric_bins) > 0:

            for _, row in numeric_bins.iterrows():

                lower = row["lower_bound"]
                upper = row["upper_bound"]

                if value >= lower and value < upper:
                    return float(row["points"])

            # Handle +inf upper boundary
            for _, row in numeric_bins.iterrows():

                lower = row["lower_bound"]
                upper = row["upper_bound"]

                if np.isneginf(lower) and value < upper:
                    return float(row["points"])

                if np.isposinf(upper) and value >= lower:
                    return float(row["points"])

        # ----------------------------------------------------
        # Categorical bins
        # ----------------------------------------------------

        categorical_bins = feature_card[
            feature_card["lower_bound"].isna()
            & feature_card["upper_bound"].isna()
            & (
                feature_card["bin"].astype(str).str.lower()
                != "missing"
            )
        ]

        for _, row in categorical_bins.iterrows():

            if str(value) == str(row["bin"]):
                return float(row["points"])

        return 0.0

    points_column = f"{feature}_points"

    data[points_column] = (
        data[feature]
        .apply(calculate_points)
    )

    data["credit_score"] += (
        data[points_column]
    )


# ============================================================
# ROUND SCORE
# ============================================================

data["credit_score"] = (
    data["credit_score"]
    .round()
    .astype(int)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BORROWER SCORECARD")
print("=" * 80)

print(
    f"Observations : {len(data):,}"
)

print(
    f"Features scored : {len(features)}"
)

print(
    f"Mean score : {data['credit_score'].mean():.2f}"
)

print(
    f"Minimum score : {data['credit_score'].min()}"
)

print(
    f"Maximum score : {data['credit_score'].max()}"
)

print(
    f"Default rate : {data['default_flag'].mean():.6f}"
)


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(f"Shape: {data.shape}")