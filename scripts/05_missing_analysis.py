import pandas as pd

INPUT_PATH = "outputs/train.csv"

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\n" + "=" * 80)
print("TRAINING DATA")
print("=" * 80)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": df.isna().mean() * 100,
})

missing = missing.sort_values(
    "missing_pct",
    ascending=False
)

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(missing.to_string())


print("\n" + "=" * 80)
print("UNIQUE VALUES")
print("=" * 80)

unique_counts = df.nunique(dropna=False).sort_values()

print(unique_counts.to_string())


print("\n" + "=" * 80)
print("TARGET")
print("=" * 80)

print(df["default_flag"].value_counts())
print()
print(df["default_flag"].value_counts(normalize=True))

missing.to_csv(
    "outputs/missingness_report.csv"
)

print("\nSaved:")
print("outputs/missingness_report.csv")