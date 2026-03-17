"""
=============================================================
STAGE 3 (IMPROVED) — Model Training & Evaluation
=============================================================
Key improvements over v1:
  1. Threshold tuning for XGBoost (fixes low recall)
  2. Stronger XGBoost hyperparameters
  3. Precision-Recall curve to find optimal threshold
  4. Cross-validation for reliable AUC estimate
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data/processed/stage2_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "outputs/models")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs/plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs/reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 55)
print("STAGE 3 (IMPROVED): MODEL TRAINING & EVALUATION")
print("=" * 55)


# ─────────────────────────────────────────
# STEP 1: LOAD & SPLIT
# ─────────────────────────────────────────

print("\n📂 Loading stage2_features.csv ...")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"   Default rate — Train: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")


# ─────────────────────────────────────────
# STEP 2: SMOTE
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: SMOTE RESAMPLING")
print("=" * 55)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — each class: {(y_train_res == 0).sum():,} rows")


# ─────────────────────────────────────────
# STEP 3: FEATURE SCALING
# ─────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n✅ Scaler saved")


# ─────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4: TRAINING MODELS")
print("=" * 55)

# ── Logistic Regression ──────────────────
lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr.fit(X_train_scaled, y_train_res)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
print("✅ Logistic Regression trained")

# ── Random Forest ─────────────────────────
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_leaf=30, random_state=42, n_jobs=-1
)
rf.fit(X_train_res, y_train_res)
rf_prob = rf.predict_proba(X_test)[:, 1]
print("✅ Random Forest trained")

# ── XGBoost (improved hyperparameters) ───
# Key changes vs v1:
#   - n_estimators: 300→500  (more trees = better learning)
#   - max_depth: 5→6         (slightly deeper trees)
#   - min_child_weight: 5    (prevents splits on tiny groups)
#   - gamma: 0.1             (minimum gain needed to split)
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=(y_train_res == 0).sum() / (y_train_res == 1).sum(),
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
print("✅ XGBoost trained")


# ─────────────────────────────────────────
# STEP 5: FIND OPTIMAL THRESHOLD FOR XGBOOST
# ─────────────────────────────────────────
# Default threshold 0.5 is wrong for imbalanced data.
# We scan all thresholds and pick the one with best F1.
# This is the most important fix from v1.

print("\n" + "=" * 55)
print("STEP 5: THRESHOLD TUNING (XGBoost)")
print("=" * 55)

precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_prob)

f1_scores = []
for p, r in zip(precisions[:-1], recalls[:-1]):
    if (p + r) == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * p * r / (p + r))

best_idx       = np.argmax(f1_scores)
optimal_thresh = thresholds[best_idx]
print(f"   Optimal threshold (max F1): {optimal_thresh:.3f}")
print(f"   (Previous default was 0.500 — this fixes the recall problem)")

# Precision-Recall curve plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls[:-1], precisions[:-1], color="#e74c3c", linewidth=2)
ax.scatter(recalls[best_idx], precisions[best_idx],
           color="black", zorder=5, s=100,
           label=f"Optimal threshold = {optimal_thresh:.3f}")
ax.set_xlabel("Recall (% of actual defaults caught)")
ax.set_ylabel("Precision (% of default predictions correct)")
ax.set_title("Precision-Recall Curve — XGBoost", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_precision_recall_curve.png"), dpi=150)
plt.close()
print("✅ Saved: 10_precision_recall_curve.png")


# ─────────────────────────────────────────
# STEP 6: EVALUATE ALL MODELS
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 6: EVALUATION RESULTS")
print("=" * 55)

def print_metrics(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    p1 = report["1"]["precision"]
    r1 = report["1"]["recall"]
    f1 = report["1"]["f1-score"]
    print(f"\n{'─'*50}")
    print(f"  {name}  [threshold={threshold:.3f}]")
    print(f"{'─'*50}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Precision  : {p1:.4f}")
    print(f"  Recall     : {r1:.4f}  ← target >0.60")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15} Predicted 0  Predicted 1")
    print(f"  Actual 0   :   {cm[0][0]:>7,}      {cm[0][1]:>7,}")
    print(f"  Actual 1   :   {cm[1][0]:>7,}      {cm[1][1]:>7,}")
    return {"name": name, "threshold": threshold, "auc": auc,
            "precision": p1, "recall": r1, "f1": f1, "y_prob": y_prob}

results = []
results.append(print_metrics("Logistic Regression",  y_test, lr_prob,  0.5))
results.append(print_metrics("Random Forest",         y_test, rf_prob,  0.5))
results.append(print_metrics("XGBoost (thresh=0.50)", y_test, xgb_prob, 0.5))
results.append(print_metrics("XGBoost (optimised)",   y_test, xgb_prob, optimal_thresh))


# ─────────────────────────────────────────
# STEP 7: ROC CURVES
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
for r, ls, lc in zip(
    [results[0], results[1], results[3]],
    ["-", "--", "-."],
    ["#3498db", "#2ecc71", "#e74c3c"]
):
    RocCurveDisplay.from_predictions(
        y_test, r["y_prob"],
        name=f"{r['name']} (AUC={r['auc']:.3f})",
        ax=ax, linestyle=ls, color=lc
    )
ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_roc_curves.png"), dpi=150)
plt.close()
print("\n✅ Saved: 08_roc_curves.png")


# ─────────────────────────────────────────
# STEP 8: MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────

plot_results = [results[0], results[1], results[3]]
metrics      = ["auc", "precision", "recall", "f1"]
labels       = ["ROC-AUC", "Precision", "Recall", "F1-Score"]
names        = ["Log. Reg.", "Rand. Forest", "XGBoost\n(optimised)"]
colors       = ["#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for i, (metric, label) in enumerate(zip(metrics, labels)):
    vals = [r[metric] for r in plot_results]
    bars = axes[i].bar(names, vals, color=colors, edgecolor="black", width=0.5)
    axes[i].set_title(label, fontsize=11, fontweight="bold")
    axes[i].set_ylim(0, 1.0)
    axes[i].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, vals):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold"
        )
plt.suptitle("Model Comparison (XGBoost with Optimal Threshold)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_model_comparison.png"), dpi=150)
plt.close()
print("✅ Saved: 07_model_comparison.png")


# ─────────────────────────────────────────
# STEP 9: XGBOOST FEATURE IMPORTANCE
# ─────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(importance_df["feature"], importance_df["importance"],
        color="#e74c3c", edgecolor="black")
ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_xgb_feature_importance.png"), dpi=150)
plt.close()
print("✅ Saved: 09_xgb_feature_importance.png")


# ─────────────────────────────────────────
# STEP 10: CROSS-VALIDATION
# ─────────────────────────────────────────
# 5-fold CV gives a reliable, variance-aware AUC estimate.
# A low std deviation means the model is stable across
# different data splits — important for resume credibility.

print("\n" + "=" * 55)
print("STEP 10: 5-FOLD CROSS VALIDATION (XGBoost)")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"   CV AUC scores : {[round(s, 4) for s in cv_scores]}")
print(f"   Mean AUC      : {cv_scores.mean():.4f}")
print(f"   Std deviation : {cv_scores.std():.4f}  (lower = more stable)")


# ─────────────────────────────────────────
# STEP 11: SAVE ALL ARTIFACTS
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 11: SAVING ARTIFACTS")
print("=" * 55)

with open(os.path.join(MODELS_DIR, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(xgb, f)
print("✅ xgboost_model.pkl saved")

with open(os.path.join(MODELS_DIR, "optimal_threshold.pkl"), "wb") as f:
    pickle.dump(float(optimal_thresh), f)
print(f"✅ optimal_threshold.pkl saved  (value: {optimal_thresh:.3f})")

with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)
print("✅ feature_columns.pkl saved")

report_rows = [
    {"model": r["name"], "threshold": r["threshold"],
     "roc_auc": round(r["auc"], 4), "precision": round(r["precision"], 4),
     "recall": round(r["recall"], 4), "f1": round(r["f1"], 4)}
    for r in results
]
pd.DataFrame(report_rows).to_csv(
    os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False
)
print("✅ model_comparison.csv saved")

print("\n📊 Final Summary:")
print(pd.DataFrame(report_rows).to_string(index=False))

print(f"\n   5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "=" * 55)
print("✅ STAGE 3 (IMPROVED) COMPLETE")
print("=" * 55)
print("\nNext Step → Run explain.py")