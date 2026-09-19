import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

SCORECARD_PATH = "outputs/final_scorecard.csv"
WOE_BINS_PATH = "outputs/final_woe_iv_bins.csv"
BOUNDARIES_PATH = "outputs/final_bin_boundaries.csv"

OUTPUT_PATH = "outputs/final_bin_scorecard.csv"


# ============================================================
# LOAD
# ============================================================

scorecard = pd.read_csv(
    SCORECARD_PATH
)

woe_bins = pd.read_csv(
    WOE_BINS_PATH
)

boundaries = pd.read_csv(
    BOUNDARIES_PATH
)


# ============================================================
# PRINT INPUT COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("INPUT COLUMNS")
print("=" * 80)

print("\nScorecard:")
print(scorecard.columns.tolist())

print("\nWoE bins:")
print(woe_bins.columns.tolist())

print("\nBoundaries:")
print(boundaries.columns.tolist())


# ============================================================
# MERGE LOGISTIC COEFFICIENTS
# ============================================================

df = woe_bins.merge(
    scorecard[
        [
            "feature",
            "coefficient",
            "score_per_woe"
        ]
    ],
    on="feature",
    how="inner"
)


# ============================================================
# CALCULATE POINTS
# ============================================================

df["points"] = (
    df["woe"]
    * df["score_per_woe"]
)


# ============================================================
# ADD NUMERICAL BIN BOUNDARIES
# ============================================================

# Expected columns:
# feature, bin_number, lower_bound, upper_bound

if "bin_number" in boundaries.columns:

    boundaries["bin_number"] = pd.to_numeric(
        boundaries["bin_number"],
        errors="coerce"
    )

    df["bin_number"] = (
        df["bin"]
        .astype(str)
        .str.extract(r"(\d+)")
        [0]
    )

    df["bin_number"] = pd.to_numeric(
        df["bin_number"],
        errors="coerce"
    )

    df = df.merge(
        boundaries[
            [
                "feature",
                "bin_number",
                "lower_bound",
                "upper_bound"
            ]
        ],
        on=[
            "feature",
            "bin_number"
        ],
        how="left"
    )

else:

    print(
        "\nWARNING: bin_number not found in boundaries file."
    )


# ============================================================
# ROUND
# ============================================================

df["woe"] = df["woe"].round(6)

df["coefficient"] = (
    df["coefficient"]
    .round(6)
)

df["score_per_woe"] = (
    df["score_per_woe"]
    .round(6)
)

df["points"] = (
    df["points"]
    .round(2)
)


# ============================================================
# REMOVE HELPER COLUMN
# ============================================================

if "bin_number" in df.columns:
    df = df.drop(
        columns=["bin_number"]
    )


# ============================================================
# ORDER COLUMNS
# ============================================================

preferred_columns = [
    "feature",
    "bin",
    "lower_bound",
    "upper_bound",
    "woe",
    "coefficient",
    "score_per_woe",
    "points"
]

columns = [
    col
    for col in preferred_columns
    if col in df.columns
]

df = df[columns]


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "feature",
        "bin"
    ]
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL BIN-LEVEL SCORECARD")
print("=" * 80)

print(
    f"Features : {df['feature'].nunique()}"
)

print(
    f"Rows     : {len(df):,}"
)


# ============================================================
# CHECK BOUNDARIES
# ============================================================

numeric_boundary_count = (
    df["lower_bound"].notna().sum()
    if "lower_bound" in df.columns
    else 0
)

print(
    f"Rows with boundaries : {numeric_boundary_count:,}"
)


# ============================================================
# SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE")
print("=" * 80)

print(
    df.head(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_PATH)

print(
    f"Shape: {df.shape}"
)