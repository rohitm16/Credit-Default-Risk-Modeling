import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "outputs/train.csv"
IV_PATH = "outputs/final_iv_summary.csv"

OUTPUT_PATH = "outputs/train_selected.csv"

LOW_IV_THRESHOLD = 0.02


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

iv = pd.read_csv(
    IV_PATH
)


# ============================================================
# EXPLICITLY DROP FEATURES
# ============================================================

explicit_drop_features = [
    # Categorical / manually selected drops
    "purpose",
    "addr_state",
    "emp_length",
    "disbursement_method",
    "application_type",
    "verification_status_joint",
    "initial_list_status",
    "pymnt_plan",
    "emp_title",
    "sub_grade",
    "issue_d",
    "earliest_cr_line",
    "title",
    "zip_code",

    # Highly correlated raw features (|correlation| > 0.90)
    "fico_range_high",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
    "num_rev_tl_bal_gt_0",
    "tot_cur_bal",

    # Categorical IV = 0
    "policy_code",

    # Highly correlated after WoE transformation (|correlation| > 0.90)
    "grade",
    "open_il_24m"
]

# Only keep columns that actually exist
explicit_drop_features = [
    col
    for col in explicit_drop_features
    if col in train.columns
]


# ============================================================
# DROP FEATURES WITH IV < 0.02
# ============================================================

low_iv_features = iv.loc[
    iv["iv"] < LOW_IV_THRESHOLD,
    "feature"
].tolist()


# Only drop features actually present in train
low_iv_features = [
    col
    for col in low_iv_features
    if col in train.columns
]


# ============================================================
# COMBINE DROP LIST
# ============================================================

drop_features = list(
    dict.fromkeys(
        explicit_drop_features
        + low_iv_features
    )
)


# Never drop target
drop_features = [
    col
    for col in drop_features
    if col != "default_flag"
]


# ============================================================
# DROP
# ============================================================

train_selected = train.drop(
    columns=drop_features,
    errors="ignore"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

print(f"Original columns : {train.shape[1]}")
print(f"Dropped columns  : {len(drop_features)}")
print(f"Remaining columns: {train_selected.shape[1]}")


# ============================================================
# EXPLICIT DROPS
# ============================================================

print("\n" + "=" * 80)
print("EXPLICITLY DROPPED FEATURES")
print("=" * 80)

for feature in explicit_drop_features:
    print(feature)


# ============================================================
# LOW-IV DROPS
# ============================================================

print("\n" + "=" * 80)
print("DROPPED FEATURES: IV < 0.02")
print("=" * 80)

for feature in low_iv_features:
    print(feature)


# ============================================================
# REMAINING FEATURES
# ============================================================

print("\n" + "=" * 80)
print("REMAINING FEATURES")
print("=" * 80)

for feature in train_selected.columns:
    print(feature)


# ============================================================
# SAVE
# ============================================================

train_selected.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(f"Shape: {train_selected.shape}")