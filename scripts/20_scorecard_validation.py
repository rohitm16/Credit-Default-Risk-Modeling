import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "outputs/oot_scored.csv"
OUTPUT_PATH = "outputs/oot_score_band_validation.csv"
IMAGE_PATH = "images/oot_score_band_default_rate.png"

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)


# ============================================================
# CHECK
# ============================================================

required = [
    "credit_score",
    "default_flag"
]

for col in required:
    if col not in df.columns:
        raise ValueError(
            f"{col} not found in validation_scored.csv"
        )


# ============================================================
# SCORE BANDS
# ============================================================

bins = [
    -np.inf,
    550,
    600,
    650,
    700,
    750,
    np.inf
]

labels = [
    "<550",
    "550-600",
    "600-650",
    "650-700",
    "700-750",
    "750+"
]

df["score_band"] = pd.cut(
    df["credit_score"],
    bins=bins,
    labels=labels,
    right=False
)


# ============================================================
# BAND SUMMARY
# ============================================================

summary = (
    df.groupby(
        "score_band",
        observed=False
    )
    .agg(
        observations=("default_flag", "size"),
        defaults=("default_flag", "sum"),
        default_rate=("default_flag", "mean"),
        mean_score=("credit_score", "mean")
    )
    .reset_index()
)


summary["default_rate"] = (
    summary["default_rate"] * 100
).round(2)

summary["mean_score"] = (
    summary["mean_score"]
    .round(2)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SCORECARD VALIDATION")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# MONOTONICITY CHECK
# ============================================================

rates = summary["default_rate"].dropna().values

is_monotonic = all(
    rates[i] >= rates[i + 1]
    for i in range(len(rates) - 1)
)

print("\n" + "=" * 80)
print("MONOTONICITY CHECK")
print("=" * 80)

if is_monotonic:
    print(
        "PASS: Default rate decreases as score increases."
    )
else:
    print(
        "WARNING: Default rate is not perfectly monotonic."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    summary["score_band"].astype(str),
    summary["default_rate"],
    marker="o"
)

plt.xlabel("Credit Score Band")
plt.ylabel("Actual Default Rate (%)")
plt.title("Default Rate by Credit Score Band")

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
# SAVE MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(IMAGE_PATH)