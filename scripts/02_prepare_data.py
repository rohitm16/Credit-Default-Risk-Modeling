import pandas as pd
print("\n" )

DATA_PATH = "data/raw/accepted_200k.csv"
OUTPUT_PATH = "outputs/processed_loans.csv"

print("\n" )

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Raw rows: {len(df):,}")
print("\n" )

good_statuses = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]

bad_statuses = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
]

resolved_statuses = good_statuses + bad_statuses

df = df[df["loan_status"].isin(resolved_statuses)].copy()

df["default_flag"] = df["loan_status"].isin(bad_statuses).astype(int)


# ============================================================
# Basic target checks
# ============================================================

print("\n" + "=" * 80)
print("RESOLVED LOANS")
print("=" * 80)

print(f"Rows: {len(df):,}")

print("\nLoan status:")
print(df["loan_status"].value_counts())

print("\nTarget distribution:")
print(df["default_flag"].value_counts())

print("\nTarget proportion:")
print(df["default_flag"].value_counts(normalize=True))


# ============================================================
# Save processed dataset
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(f"Shape: {df.shape}")