"""
=============================================================
STAGE 4 — SHAP Explainability
=============================================================
Goal: Explain WHY the XGBoost model makes each prediction.

SHAP (SHapley Additive exPlanations) answers:
  - Which features drive default risk globally?
  - Why did THIS specific borrower get flagged?
  - How does each feature push risk up or down?

This is the most resume-differentiating stage.
Banks legally must justify credit decisions — SHAP is how.

Run AFTER train.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data/processed/stage2_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "outputs/models")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs/plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs/reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 55)
print("STAGE 4: SHAP EXPLAINABILITY")
print("=" * 55)


# ─────────────────────────────────────────
# STEP 1: LOAD MODEL & DATA
# ─────────────────────────────────────────
# We load the saved XGBoost model and a sample of the
# feature data. SHAP on 133k rows is slow, so we use
# a representative 2,000-row sample for global plots.

print("\n📂 Loading model and data...")

with open(os.path.join(MODELS_DIR, "xgboost_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

with open(os.path.join(MODELS_DIR, "optimal_threshold.pkl"), "rb") as f:
    optimal_threshold = pickle.load(f)

df = pd.read_csv(DATA_PATH)
X  = df[feature_cols]
y  = df["target"]

# Sample for SHAP global analysis — 2000 rows is enough
# for stable SHAP values and runs in ~60 seconds
np.random.seed(42)
sample_idx  = np.random.choice(len(X), size=2000, replace=False)
X_sample    = X.iloc[sample_idx].reset_index(drop=True)
y_sample    = y.iloc[sample_idx].reset_index(drop=True)

print(f"✅ Model loaded")
print(f"✅ Data loaded: {X.shape[0]:,} rows, using 2,000-row sample for SHAP")
print(f"   Optimal threshold: {optimal_threshold:.3f}")


# ─────────────────────────────────────────
# STEP 2: COMPUTE SHAP VALUES
# ─────────────────────────────────────────
# TreeExplainer is optimised for tree-based models.
# It computes exact SHAP values (not approximations).
#
# shap_values[i][j] = how much feature j pushed
# the prediction for row i away from the base rate.
# Positive = pushed toward default
# Negative = pushed away from default

print("\n" + "=" * 55)
print("STEP 2: COMPUTING SHAP VALUES")
print("=" * 55)
print("   (This takes ~60 seconds...)")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification XGBoost returns a single array
# shap_values shape: (n_samples, n_features)
print(f"✅ SHAP values computed")
print(f"   Shape: {shap_values.shape}")
print(f"   Base value (avg model output): {explainer.expected_value:.4f}")


# ─────────────────────────────────────────
# STEP 3: GLOBAL SUMMARY PLOT
# ─────────────────────────────────────────
# The beeswarm plot is the most information-dense SHAP chart.
# Each dot = one borrower.
# X position = SHAP value (impact on prediction)
# Color = feature value (red=high, blue=low)
#
# Reading it:
#   int_rate_tier red dots on right = high interest rate
#   strongly increases default probability (makes sense)
#   is_short_term blue dots on right = long term (60mo)
#   increases default probability (also makes sense)

print("\n" + "=" * 55)
print("STEP 3: GLOBAL SUMMARY PLOT (Beeswarm)")
print("=" * 55)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_sample,
    max_display=20,
    show=False,
    plot_size=None
)
plt.title("SHAP Summary — Feature Impact on Default Probability",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_shap_summary_beeswarm.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 11_shap_summary_beeswarm.png")


# ─────────────────────────────────────────
# STEP 4: SHAP BAR PLOT (Mean Absolute Impact)
# ─────────────────────────────────────────
# Simpler than beeswarm — shows average magnitude of
# each feature's impact across all borrowers.
# This is the clearest chart for a README or presentation.

print("\n" + "=" * 55)
print("STEP 4: GLOBAL BAR PLOT (Mean |SHAP|)")
print("=" * 55)

mean_shap = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(mean_shap["feature"], mean_shap["mean_abs_shap"],
               color="#e74c3c", edgecolor="black", alpha=0.85)
ax.set_title("Mean |SHAP Value| — Global Feature Importance",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
for bar, val in zip(bars, mean_shap["mean_abs_shap"]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "12_shap_bar_global.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 12_shap_bar_global.png")


# ─────────────────────────────────────────
# STEP 5: SHAP DEPENDENCE PLOTS
# ─────────────────────────────────────────
# Shows HOW a feature affects default risk as its
# value changes. Captures non-linear effects that
# correlation analysis misses.
# Color = interaction with a second feature.

print("\n" + "=" * 55)
print("STEP 5: DEPENDENCE PLOTS (Top 3 Features)")
print("=" * 55)

# Get top 3 features by mean |SHAP|
top3 = mean_shap.tail(3)["feature"].tolist()[::-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feat in enumerate(top3):
    feat_idx = list(feature_cols).index(feat)
    shap.dependence_plot(
        feat_idx, shap_values, X_sample,
        ax=axes[i], show=False,
        alpha=0.4
    )
    axes[i].set_title(f"SHAP Dependence: {feat}",
                      fontsize=10, fontweight="bold")

plt.suptitle("How Top Features Affect Default Probability",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "13_shap_dependence_plots.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: 13_shap_dependence_plots.png  (features: {top3})")


# ─────────────────────────────────────────
# STEP 6: INDIVIDUAL BORROWER EXPLANATIONS
# ─────────────────────────────────────────
# This is the most powerful part for interviews.
# We explain individual predictions — this is exactly
# what a bank would show a loan officer.
#
# We pick 3 borrowers:
#   - A clear low-risk borrower
#   - A clear high-risk borrower
#   - A borderline borrower (close to threshold)

print("\n" + "=" * 55)
print("STEP 6: INDIVIDUAL BORROWER EXPLANATIONS")
print("=" * 55)

# Get predicted probabilities for the sample
probs = model.predict_proba(X_sample)[:, 1]

# Find representative borrowers
low_risk_idx    = np.argmin(probs)                          # lowest risk
high_risk_idx   = np.argmax(probs)                          # highest risk
border_idx      = np.argmin(np.abs(probs - optimal_threshold))  # borderline

borrowers = {
    "LOW_RISK":   low_risk_idx,
    "HIGH_RISK":  high_risk_idx,
    "BORDERLINE": border_idx
}

for label, idx in borrowers.items():
    prob      = probs[idx]
    actual    = y_sample.iloc[idx]
    risk_cat  = "Low" if prob < 0.3 else ("High" if prob > 0.6 else "Medium")
    decision  = "Approve" if prob < optimal_threshold else "Review/Reject"

    print(f"\n  [{label}]")
    print(f"   Default probability : {prob:.3f}")
    print(f"   Risk category       : {risk_cat}")
    print(f"   Decision            : {decision}")
    print(f"   Actual outcome      : {'Default' if actual == 1 else 'No Default'}")

    # Top 5 contributing features for this borrower
    borrower_shap = shap_values[idx]
    contrib = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": borrower_shap,
        "feature_value": X_sample.iloc[idx].values
    }).sort_values("shap_value", key=abs, ascending=False).head(5)

    print(f"   Top 5 risk drivers:")
    for _, row in contrib.iterrows():
        direction = "↑ INCREASES" if row["shap_value"] > 0 else "↓ DECREASES"
        print(f"     {direction} risk | {row['feature']:<30} = {row['feature_value']:.3f}  (SHAP: {row['shap_value']:+.4f})")


# ─────────────────────────────────────────
# STEP 7: WATERFALL PLOTS FOR EACH BORROWER
# ─────────────────────────────────────────
# Waterfall shows how each feature pushes the prediction
# from the base rate to the final score.
# Red bars = increase default risk
# Blue bars = decrease default risk

print("\n" + "=" * 55)
print("STEP 7: WATERFALL PLOTS")
print("=" * 55)

for label, idx in borrowers.items():
    prob = probs[idx]

    # Build waterfall data manually for clean plot
    borrower_shap  = shap_values[idx]
    base_val       = explainer.expected_value

    contrib = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": borrower_shap,
        "feature_value": X_sample.iloc[idx].values
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    # Reverse for waterfall (bottom to top)
    contrib = contrib.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib["shap_value"]]
    bars   = ax.barh(
        [f"{r['feature']} = {r['feature_value']:.2f}" for _, r in contrib.iterrows()],
        contrib["shap_value"],
        color=colors, edgecolor="black", alpha=0.85
    )
    ax.axvline(0, color="black", linewidth=1)

    risk_label = "LOW RISK" if prob < 0.3 else ("HIGH RISK" if prob > 0.6 else "BORDERLINE")
    ax.set_title(
        f"SHAP Waterfall — {label} Borrower\n"
        f"Default Probability: {prob:.3f}  |  {risk_label}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("SHAP Value (impact on default probability)")

    red_patch   = mpatches.Patch(color="#e74c3c", label="Increases default risk")
    green_patch = mpatches.Patch(color="#2ecc71", label="Decreases default risk")
    ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)

    for bar, val in zip(bars, contrib["shap_value"]):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8
        )

    plt.tight_layout()
    fname = f"14_waterfall_{label.lower()}.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {fname}")


# ─────────────────────────────────────────
# STEP 8: SHAP INSIGHTS REPORT
# ─────────────────────────────────────────
# This is the "What the model learned" section
# mentioned earlier — the part that gets you
# remembered in interviews.

print("\n" + "=" * 55)
print("STEP 8: GENERATING INSIGHTS REPORT")
print("=" * 55)

# Compute mean SHAP for defaulters vs non-defaulters
default_mask     = y_sample == 1
non_default_mask = y_sample == 0

mean_shap_default     = shap_values[default_mask].mean(axis=0)
mean_shap_nondefault  = shap_values[non_default_mask].mean(axis=0)

insights_df = pd.DataFrame({
    "feature":             feature_cols,
    "mean_abs_shap_all":   np.abs(shap_values).mean(axis=0),
    "mean_shap_defaulters":     mean_shap_default,
    "mean_shap_nondefaulters":  mean_shap_nondefault,
    "shap_difference":     mean_shap_default - mean_shap_nondefault
}).sort_values("mean_abs_shap_all", ascending=False)

report_path = os.path.join(REPORTS_DIR, "shap_feature_insights.csv")
insights_df.to_csv(report_path, index=False)
print(f"✅ Saved: outputs/reports/shap_feature_insights.csv")

# Print human-readable summary
print("\n📋 TOP RISK INSIGHTS (for your README):")
print("─" * 55)
top5 = insights_df.head(5)
for _, row in top5.iterrows():
    direction = "higher" if row["shap_difference"] > 0 else "lower"
    print(f"  • {row['feature']:<28} → {direction} values increase default risk")
    print(f"    Defaulters avg SHAP:     {row['mean_shap_defaulters']:+.4f}")
    print(f"    Non-defaulters avg SHAP: {row['mean_shap_nondefaulters']:+.4f}")
    print()


# ─────────────────────────────────────────
# STEP 9: DECISION SIMULATOR
# ─────────────────────────────────────────
# Simulates the full prediction pipeline on a
# hypothetical new borrower. This demonstrates
# the end-to-end system in your README/interviews.

print("=" * 55)
print("STEP 9: PREDICTION SIMULATOR (Sample Borrower)")
print("=" * 55)

# Create a sample borrower using median values from dataset
sample_borrower = pd.DataFrame([X.median()], columns=feature_cols)

# Override a few fields to make it an interesting case
# (mid-range borrower — the kind that needs manual review)
if "int_rate" in feature_cols:
    sample_borrower["int_rate"] = 15.0
if "int_rate_tier" in feature_cols:
    sample_borrower["int_rate_tier"] = 2.0
if "loan_to_income" in feature_cols:
    sample_borrower["loan_to_income"] = 0.35
if "is_short_term" in feature_cols:
    sample_borrower["is_short_term"] = 0       # 60-month loan
if "grade_encoded" in feature_cols:
    sample_borrower["grade_encoded"] = 3.0     # Grade D

prob      = model.predict_proba(sample_borrower)[0][1]
risk_cat  = "🟢 Low" if prob < 0.3 else ("🔴 High" if prob > 0.6 else "🟡 Medium")
decision  = "✅ APPROVE" if prob < optimal_threshold else "⚠️  REVIEW / REJECT"

# SHAP explanation for this borrower
sb_shap = explainer.shap_values(sample_borrower)[0]
contrib  = pd.DataFrame({
    "feature":       feature_cols,
    "shap_value":    sb_shap,
    "feature_value": sample_borrower.iloc[0].values
}).sort_values("shap_value", key=abs, ascending=False).head(5)

print(f"""
╔══════════════════════════════════════════════╗
║         LOAN DEFAULT RISK ASSESSMENT         ║
╠══════════════════════════════════════════════╣
║  Default Probability : {prob:.3f}                 ║
║  Risk Category       : {risk_cat:<22}  ║
║  Decision            : {decision:<22}  ║
╠══════════════════════════════════════════════╣
║  TOP RISK DRIVERS (SHAP Explanation)         ║
╚══════════════════════════════════════════════╝""")

for _, row in contrib.iterrows():
    arrow = "▲" if row["shap_value"] > 0 else "▼"
    print(f"  {arrow} {row['feature']:<28} val={row['feature_value']:.2f}  SHAP={row['shap_value']:+.4f}")


print("\n" + "=" * 55)
print("✅ STAGE 4 COMPLETE")
print("=" * 55)
print("""
Outputs saved:
  plots/  → 11_shap_summary_beeswarm.png
            12_shap_bar_global.png
            13_shap_dependence_plots.png
            14_waterfall_low_risk.png
            14_waterfall_high_risk.png
            14_waterfall_borderline.png
  reports/ → shap_feature_insights.csv

Next Step → Write your README using the insights above.
""")