"""
STAGE 2 — Feature Engineering
Goal: Transform the cleaned data from Stage 1 into a rich
feature set ready for ML modeling.

We create:
  - Engineered numeric features (ratios, flags, interactions)
  - Encoded categorical features
  - A final feature matrix saved for Stage 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# SETUP: PATHS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data/processed/stage1_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/stage2_features.csv")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs/plots")

print("=" * 55)
print("STAGE 2: FEATURE ENGINEERING")
print("=" * 55)


# STEP 1: LOAD CLEANED DATA FROM STAGE 1

# We read the CSV saved by data_pipeline.py.
# All cleaning is already done — we only add here.

print("\nLoading stage1_cleaned.csv ...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns available:\n{list(df.columns)}")


# STEP 2: ENGINEERED NUMERIC FEATURES

# These are the features that add real intelligence.
# Each one represents a financial risk concept.

print("\n" + "=" * 55)
print("STEP 2: ENGINEERED NUMERIC FEATURES")
print("=" * 55)

# --- 2A: Loan to Income Ratio ---
# How large is the loan relative to annual income?
# A person earning 4L taking a 10L loan = high risk.
# Formula: loan_amount / annual_income
df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).round(4)

# Cap extreme outliers at 99th percentile to avoid skewing the model
cap_val = df["loan_to_income"].quantile(0.99)
df["loan_to_income"] = df["loan_to_income"].clip(upper=cap_val)
print("loan_to_income created")



# --- 2K: FICO Score (midpoint) ---
# Average the two bounds into one clean credit score feature.
# This is the most important credit risk signal in the dataset.
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)
    print("fico_score created (midpoint of range)")

# --- 2B: Installment to Income Ratio ---
# Monthly payment burden vs monthly income.
# More practical than annual DTI for default prediction.
# Formula: monthly_installment / (annual_income / 12)
df["installment_to_income"] = (
    df["installment"] / (df["annual_inc"] / 12)
).round(4)
cap_val = df["installment_to_income"].quantile(0.99)
df["installment_to_income"] = df["installment_to_income"].clip(upper=cap_val)
print("installment_to_income created")

# --- 2C: Revolving Utilization (already exists, but clean it) ---
# revol_util is credit utilization % — how much of credit limit is used.
# High utilization (>80%) = financial stress signal.
# It already exists from Stage 1 but may need capping.
if "revol_util" in df.columns:
    df["revol_util"] = df["revol_util"].clip(0, 100)
    print("revol_util capped between 0–100")

# --- 2D: Interest Rate as Risk Signal ---
# LendingClub assigns higher rates to riskier borrowers.
# int_rate is already numeric from Stage 1.
# We bucket it into risk tiers for interpretability.
df["int_rate_tier"] = pd.cut(
    df["int_rate"],
    bins=[0, 8, 13, 18, 25, 100],
    labels=[0, 1, 2, 3, 4]        # 0 = lowest risk tier
).astype(float)
print("int_rate_tier created (0=lowest, 4=highest)")

# --- 2E: Credit Account Utilization ---
# How many open accounts out of total ever opened?
# Very few open accounts relative to total = possible closures due to default.
df["open_acc_ratio"] = (
    df["open_acc"] / df["total_acc"].replace(0, np.nan)
).round(4)
df["open_acc_ratio"].fillna(df["open_acc_ratio"].median(), inplace=True)
print(" open_acc_ratio created")

# --- 2F: Delinquency Flag ---
# Binary: has this borrower ever been delinquent in the last 2 years?
# Past behavior is one of the strongest predictors of future default.
df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
print("has_delinquency flag created")

# --- 2G: Public Record Flag ---
# Any bankruptcies, tax liens, or judgements on record?
if "pub_rec" in df.columns:
    df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)
    print("✅ has_pub_rec flag created")

# --- 2H: High Inquiry Flag ---
# More than 2 credit inquiries in 6 months = borrower is actively
# seeking credit elsewhere = financial distress signal.
if "inq_last_6mths" in df.columns:
    df["high_inq_flag"] = (df["inq_last_6mths"] > 2).astype(int)
    print("high_inq_flag created")

# --- 2I: Loan Amount Tier ---
# Bucket loan size — small/medium/large/very large
df["loan_amnt_tier"] = pd.cut(
    df["loan_amnt"],
    bins=[0, 5000, 15000, 25000, 40000],
    labels=[0, 1, 2, 3]
).astype(float)
df["loan_amnt_tier"].fillna(1, inplace=True)
print("loan_amnt_tier created")

# --- 2J: Short Term Flag ---
# 36-month loans are less risky than 60-month loans.
# Borrowers who need 60 months to repay are more stretched.
df["is_short_term"] = (df["term"] == 36).astype(int)
print("is_short_term flag created")


# STEP 3: ENCODE CATEGORICAL FEATURES

# ML models need numbers, not text.
# We convert text columns to numeric here.

print("\n" + "=" * 55)
print("STEP 3: ENCODING CATEGORICAL FEATURES")
print("=" * 55)

# --- 3A: Grade → Ordinal (A=0, B=1, ..., G=6) ---
# Grade has a natural order (A is best, G is worst).
# Ordinal encoding preserves this ordering.
if "grade" in df.columns:
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    df["grade_encoded"] = df["grade"].map(grade_map)
    # Fill any unmapped values with median
    df["grade_encoded"].fillna(3, inplace=True)
    df.drop(columns=["grade"], inplace=True)
    print("grade → grade_encoded (ordinal 0–6)")

# --- 3B: Home Ownership → One-Hot Encoding ---
# No natural order here — RENT is not "more than" OWN.
# We create separate binary columns for each category.
if "home_ownership" in df.columns:
    # Consolidate rare categories
    df["home_ownership"] = df["home_ownership"].replace(
        {"ANY": "OTHER", "NONE": "OTHER"}
    )
    home_dummies = pd.get_dummies(
        df["home_ownership"], prefix="home", drop_first=True
    )
    df = pd.concat([df, home_dummies], axis=1)
    df.drop(columns=["home_ownership"], inplace=True)
    print(f"home_ownership → one-hot: {list(home_dummies.columns)}")

# --- 3C: Purpose → One-Hot Encoding ---
# Loan purpose matters — debt_consolidation has different risk
# profile than small_business or medical.
if "purpose" in df.columns:
    # Keep top 8 purposes, group rest as "other"
    top_purposes = df["purpose"].value_counts().nlargest(8).index
    df["purpose"] = df["purpose"].where(
        df["purpose"].isin(top_purposes), other="other"
    )
    purpose_dummies = pd.get_dummies(
        df["purpose"], prefix="purpose", drop_first=True
    )
    df = pd.concat([df, purpose_dummies], axis=1)
    df.drop(columns=["purpose"], inplace=True)
    print(f"purpose → one-hot: {df.shape[1]} total columns now")



# STEP 4: DROP COLUMNS WE NO LONGER NEED
# Some columns were only useful for creating features.
# Keeping them would cause data leakage or redundancy.

print("\n" + "=" * 55)
print("STEP 4: DROPPING REDUNDANT COLUMNS")
print("=" * 55)

cols_to_drop = [
    "funded_amnt",      # Almost identical to loan_amnt
    "installment",      # Captured in installment_to_income
    "open_acc",         # Captured in open_acc_ratio
    "total_acc",        # Captured in open_acc_ratio
    "delinq_2yrs",      # Captured in has_delinquency
    "pub_rec",          # Captured in has_pub_rec
    "inq_last_6mths",   # Captured in high_inq_flag
    "term",             # Captured in is_short_term
]

existing_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=existing_drop, inplace=True)
print(f"Dropped {len(existing_drop)} redundant columns")
print(f"Remaining columns: {df.shape[1]}")

# STEP 5: FINAL FEATURE SUMMARY

print("\n" + "=" * 55)
print("STEP 5: FINAL FEATURE SUMMARY")
print("=" * 55)

feature_cols = [c for c in df.columns if c != "target"]
print(f"\nTotal features for modeling: {len(feature_cols)}")
print("\nAll features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2}. {col}")

# Check for any remaining nulls
nulls = df.isnull().sum().sum()
print(f"\n🔍 Total null values remaining: {nulls}")
if nulls > 0:
    print("   Filling remaining nulls with median...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(" Done")



# STEP 6: CORRELATION ANALYSIS PLOT
# Which features correlate most with default?
# This is a quick sanity check before modeling.

print("\n" + "=" * 55)
print("STEP 6: FEATURE CORRELATION WITH TARGET")
print("=" * 55)

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["target"].drop("target").sort_values()

#plotting graphs
# Plot top 15 positive and negative correlations
top_corr = pd.concat([correlations.head(10), correlations.tail(10)])

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_corr.values]
bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Feature Correlation with Default (Target)", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
for bar, val in zip(bars, top_corr.values):
    ax.text(
        val + (0.002 if val >= 0 else -0.002),
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center",
        ha="left" if val >= 0 else "right", fontsize=8
    )
plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "05_feature_correlations.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f"✅ Saved: 05_feature_correlations.png")


# STEP 7: ENGINEERED FEATURE DISTRIBUTIONS
# Visual check: do our new features separate defaulters
# from non-defaulters? They should show different distributions.

key_features = ["loan_to_income", "installment_to_income", "int_rate", "open_acc_ratio"]
existing_key = [f for f in key_features if f in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(existing_key):
    ax = axes[i]
    df[df["target"] == 0][feat].clip(
        df[feat].quantile(0.01), df[feat].quantile(0.99)
    ).hist(bins=40, alpha=0.6, label="No Default", color="#2ecc71", ax=ax)
    df[df["target"] == 1][feat].clip(
        df[feat].quantile(0.01), df[feat].quantile(0.99)
    ).hist(bins=40, alpha=0.6, label="Default", color="#e74c3c", ax=ax)
    ax.set_title(f"{feat}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.suptitle("Engineered Feature Distributions by Default Status",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "06_engineered_feature_distributions.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: 06_engineered_feature_distributions.png")

# STEP 8: SAVE FINAL FEATURE DATASET

print("\n" + "=" * 55)
print("STEP 8: SAVING FEATURE DATASET")
print("=" * 55)

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved: data/processed/stage2_features.csv")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Feature columns: {df.shape[1] - 1}")
print(f"   Target column: 'target'")

print("\n" + "=" * 55)
print("✅ STAGE 2 COMPLETE")
print("=" * 55)
print("\nNext Step → Run train.py")
