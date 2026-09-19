import pandas as pd
DATA_PATH = "data/raw/accepted_200k.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n" )
print("SHAPE")
print("\n" )
print(df.shape)
print("\n" )

print("COLUMNS")

for i, col in enumerate(df.columns):
    print(f"{i:3d}  {col}")

print("\n" )
print("LOAN STATUS")
# print("=" * 80)
print(df["loan_status"].value_counts(dropna=False))
print("\n" )

print("LOAN STATUS PROPORTION")
print(df["loan_status"].value_counts(normalize=True, dropna=False))
print("\n" )

print("MISSING VALUES - TOP 30")
print(df.isna().sum().sort_values(ascending=False).head(30))
print("\n" )

print("DATA TYPES")
print(df.dtypes)
print("\n" )

print("FIRST 5 ROWS")
print(df.head())