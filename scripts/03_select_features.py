import pandas as pd

INPUT_PATH = "outputs/processed_loans.csv"
OUTPUT_PATH = "outputs/modeling_data.csv"
df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"Input shape: {df.shape}")

identifier_columns = [
    "id",
    "member_id",
    "url",
    "desc",
]

post_origination_columns = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",

    # Payments
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",

    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",

    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",

    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]


target_column = "default_flag"
target_source_column = "loan_status"


columns_to_drop = (
    identifier_columns
    + post_origination_columns
    + [target_source_column]
)

columns_to_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

df_model = df.drop(columns=columns_to_drop)

feature_columns = [
    col for col in df_model.columns
    if col != target_column
]

df_model = df_model[feature_columns + [target_column]]


print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

print(f"Original columns : {len(df.columns)}")
print(f"Removed columns  : {len(columns_to_drop)}")
print(f"Model columns    : {len(df_model.columns)}")

print("\nRemoved columns:")

for col in columns_to_drop:
    print(f"  - {col}")

print("\nRemaining columns:")

for col in df_model.columns:
    print(f"  - {col}")

df_model.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)
print(f"Shape: {df_model.shape}")