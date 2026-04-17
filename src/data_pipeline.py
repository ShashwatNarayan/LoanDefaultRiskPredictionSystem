"""
STAGE 1 — Data Loading, Exploration & Initial Cleaning
Load the LendingClub dataset, understand its structure,
and produce a clean, analysis-ready DataFrame
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# STEP 1: PROJECT FOLDER STRUCTURE SETUP
# Creates all necessary folders so your project is organized
# from day one. Run this once.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders = [
    "data/raw",         # Original downloaded CSV — never modify this
    "data/processed",   # Cleaned data saved here
    "notebooks",        # For any Jupyter exploration
    "outputs/plots",    # All charts saved here
    "outputs/models",   # Trained models saved here
    "outputs/reports",  # SHAP and evaluation reports
]

for folder in folders:
    path = os.path.join(BASE_DIR, folder)
    os.makedirs(path, exist_ok=True)

print("✅ Project folder structure created.\n")

# STEP 2: LOAD THE DATASET
# INSTRUCTIONS TO GET THE DATA:
#   1. Go to: https://www.kaggle.com/datasets/wordsforthewise/lending-club
#   2. Download "accepted_2007_to_2018Q4.csv.gz"
#   3. Place it in your  data/raw/  folder
#   4. Update RAW_DATA_PATH below if needed

RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/accepted_2007_to_2018Q4.csv")

print("Loading dataset...")

try:
    df = pd.read_csv(
        RAW_DATA_PATH,
        nrows=150_000,
        low_memory=False
    )
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except FileNotFoundError:
    print("File not found. Please download the dataset from Kaggle.")
    print(f"   Expected path: {RAW_DATA_PATH}")
    raise

# STEP 3: INITIAL EXPLORATION

# Before cleaning anything, we look at what we have.
# This is EDA — Exploratory Data Analysis.

print("=" * 55)
print("STEP 3: INITIAL EXPLORATION")
print("=" * 55)

print(f"\nShape: {df.shape}")
print(f"\nColumn names (first 30):\n{list(df.columns[:30])}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nMissing values (top 20 columns):")
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].head(20).to_string())


# STEP 4: SELECT RELEVANT COLUMNS
# The raw dataset has 150+ columns. We keep only
# what's meaningful for credit risk modeling.

print("\n" + "=" * 55)
print("STEP 4: COLUMN SELECTION")
print("=" * 55)

SELECTED_COLUMNS = [
    # Target variable
    "loan_status",

    # Loan characteristics
    "loan_amnt",          # Requested loan amount
    "funded_amnt",        # Actual funded amount
    "term",               # 36 or 60 months
    "int_rate",           # Interest rate
    "installment",        # Monthly payment
    "grade",              # LC assigned grade (A–G)
    "purpose",            # Loan purpose

    # Borrower financials
    "annual_inc",         # Annual income
    "dti",                # Debt-to-income ratio
    "emp_length",         # Employment length

    # Credit history
    "home_ownership",     # RENT / OWN / MORTGAGE
    "open_acc",           # Open credit accounts
    "total_acc",          # Total credit accounts
    "revol_bal",          # Revolving balance
    "revol_util",         # Credit utilization (%)
    "delinq_2yrs",        # Delinquencies in 2 years
    "inq_last_6mths",     # Credit inquiries last 6 months
    "pub_rec",            # Public derogatory records
    "mort_acc",           # Mortgage accounts
    "earliest_cr_line",   # First credit line date
    "fico_range_low",     # FICO score lower bound
    "fico_range_high",    # FICO score upper bound
]

# Keep only columns that exist in the dataset
available_cols = [c for c in SELECTED_COLUMNS if c in df.columns]
df = df[available_cols].copy()

print(f"Kept {len(available_cols)} columns out of {len(SELECTED_COLUMNS)} requested.")
if len(available_cols) < len(SELECTED_COLUMNS):
    missing_cols = set(SELECTED_COLUMNS) - set(available_cols)
    print(f"   Missing columns (not in dataset): {missing_cols}")


# STEP 5: CREATE THE TARGET VARIABLE
# loan_status has many categories. We convert it
# to binary: 1 = Default, 0 = No Default.

print("\n" + "=" * 55)
print("STEP 5: TARGET VARIABLE")
print("=" * 55)

print("\n📋 Raw loan_status distribution:")
print(df["loan_status"].value_counts().to_string())

# Define which statuses count as "default"
DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
]

df["target"] = df["loan_status"].apply(
    lambda x: 1 if x in DEFAULT_STATUSES else 0
)

# Drop rows where loan is still Current (ambiguous — not defaulted yet,
# but we don't know the final outcome)
df = df[df["loan_status"] != "Current"].copy()
df.drop(columns=["loan_status"], inplace=True)

default_rate = df["target"].mean()
print(f"\nTarget created.")
print(f"   Default rate: {default_rate:.1%}")
print(f"   Class distribution:\n{df['target'].value_counts().to_string()}")

# STEP 6: CLEAN MESSY COLUMNS

print("\n" + "=" * 55)
print("STEP 6: CLEANING MESSY COLUMNS")
print("=" * 55)

# --- int_rate: "13.56%" → 13.56 ---
if df["int_rate"].dtype == object:
    df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)
    print("int_rate: stripped '%' and converted to float")

# --- revol_util: "45.6%" → 45.6 ---
if "revol_util" in df.columns and df["revol_util"].dtype == object:
    df["revol_util"] = df["revol_util"].str.replace("%", "").astype(float)
    print("revol_util: stripped '%' and converted to float")

# --- term: " 36 months" → 36 ---
if df["term"].dtype == object:
    df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
    print("term: extracted numeric value")

# --- emp_length: "10+ years" → 10, "< 1 year" → 0 ---
def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    val = str(val)
    if "10+" in val:
        return 10.0
    if "< 1" in val:
        return 0.0
    digits = "".join(filter(str.isdigit, val))
    return float(digits) if digits else np.nan

df["emp_length"] = df["emp_length"].apply(parse_emp_length)
print("emp_length: parsed to numeric years")

# --- earliest_cr_line: "Jan-2005" → credit history in years ---
if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    reference_date = pd.Timestamp("2018-01-01")
    df["credit_history_years"] = (
        (reference_date - df["earliest_cr_line"]).dt.days / 365.25
    ).round(1)
    df.drop(columns=["earliest_cr_line"], inplace=True)
    print("earliest_cr_line: converted to credit_history_years")

# STEP 7: HANDLE MISSING VALUES

print("\n" + "=" * 55)
print("STEP 7: MISSING VALUE TREATMENT")
print("=" * 55)

print("\nMissing values before treatment:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Numeric columns — fill with median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "target"]

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# Categorical columns — fill with mode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

print("\nMissing values after treatment:")
remaining = df.isnull().sum().sum()
print(f"   Total remaining nulls: {remaining}")


# STEP 8: EXPLORATORY PLOTS

print("\n" + "=" * 55)
print("STEP 8: GENERATING EXPLORATORY PLOTS")
print("=" * 55)

PLOTS_DIR = os.path.join(BASE_DIR, "outputs/plots")

# Plot 1: Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["target"].value_counts()
ax.bar(["No Default (0)", "Default (1)"], counts.values,
       color=["#2ecc71", "#e74c3c"], edgecolor="black", width=0.5)
ax.set_title("Class Distribution: Default vs No Default", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 100, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=150)
plt.close()
print("Saved: 01_class_distribution.png")

# Plot 2: Default rate by loan grade
if "grade" in df.columns:
    grade_default = df.groupby("grade")["target"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grade_default.index, grade_default.values * 100,
                  color=sns.color_palette("RdYlGn_r", len(grade_default)),
                  edgecolor="black")
    ax.set_title("Default Rate by Loan Grade", fontsize=13, fontweight="bold")
    ax.set_xlabel("Loan Grade (A = Best, G = Worst)")
    ax.set_ylabel("Default Rate (%)")
    for bar, val in zip(bars, grade_default.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{val:.1%}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_default_by_grade.png"), dpi=150)
    plt.close()
    print("Saved: 02_default_by_grade.png")

# Plot 3: Loan amount distribution by target
fig, ax = plt.subplots(figsize=(8, 4))
df[df["target"] == 0]["loan_amnt"].hist(
    bins=40, alpha=0.6, label="No Default", color="#2ecc71", ax=ax)
df[df["target"] == 1]["loan_amnt"].hist(
    bins=40, alpha=0.6, label="Default", color="#e74c3c", ax=ax)
ax.set_title("Loan Amount Distribution by Default Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Loan Amount ($)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_loan_amount_distribution.png"), dpi=150)
plt.close()
print("Saved: 03_loan_amount_distribution.png")

# Plot 4: DTI distribution
if "dti" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    df[df["target"] == 0]["dti"].clip(0, 60).hist(
        bins=40, alpha=0.6, label="No Default", color="#2ecc71", ax=ax)
    df[df["target"] == 1]["dti"].clip(0, 60).hist(
        bins=40, alpha=0.6, label="Default", color="#e74c3c", ax=ax)
    ax.set_title("Debt-to-Income Ratio by Default Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("DTI Ratio")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_dti_distribution.png"), dpi=150)
    plt.close()
    print("Saved: 04_dti_distribution.png")



# STEP 9: SAVE CLEANED DATA

print("\n" + "=" * 55)
print("STEP 9: SAVING CLEANED DATA")
print("=" * 55)

PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/stage1_cleaned.csv")
df.to_csv(PROCESSED_PATH, index=False)

print(f"Cleaned data saved to: data/processed/stage1_cleaned.csv")
print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Target column: 'target'  (1 = Default, 0 = No Default)")

print("\n" + "=" * 55)
print("STAGE 1 COMPLETE")
print("=" * 55)
print("\nNext Step → Run stage2_feature_engineering.py")