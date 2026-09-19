import pandas as pd


INPUT_PATH = "outputs/modeling_data.csv"

TRAIN_OUTPUT = "outputs/train.csv"
VALID_OUTPUT = "outputs/validation.csv"
OOT_OUTPUT = "outputs/oot_test.csv"

df = pd.read_csv(INPUT_PATH, low_memory=False)
df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

train = df[df["issue_d"].dt.year <= 2016].copy()
validation = df[df["issue_d"].dt.year == 2017].copy()
oot_test = df[df["issue_d"].dt.year == 2018].copy()

print("\n" + "=" * 80)
print("TIME-BASED SPLIT")
print("=" * 80)

print(f"Training    : {len(train):,}")
print(f"Validation  : {len(validation):,}")
print(f"OOT Test    : {len(oot_test):,}")

print("\n" + "=" * 80)
print("DEFAULT RATE")
print("=" * 80)

print(
    f"Training    : {train['default_flag'].mean():.4%}"
)

print(
    f"Validation  : {validation['default_flag'].mean():.4%}"
)

print(
    f"OOT Test    : {oot_test['default_flag'].mean():.4%}"
)


print("\n" + "=" * 80)
print("DATE RANGES")
print("=" * 80)

print(
    f"Training    : {train['issue_d'].min().date()} "
    f"→ {train['issue_d'].max().date()}"
)

print(
    f"Validation  : {validation['issue_d'].min().date()} "
    f"→ {validation['issue_d'].max().date()}"
)

print(
    f"OOT Test    : {oot_test['issue_d'].min().date()} "
    f"→ {oot_test['issue_d'].max().date()}"
)

train.to_csv(TRAIN_OUTPUT, index=False)
validation.to_csv(VALID_OUTPUT, index=False)
oot_test.to_csv(OOT_OUTPUT, index=False)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(TRAIN_OUTPUT)
print(VALID_OUTPUT)
print(OOT_OUTPUT)