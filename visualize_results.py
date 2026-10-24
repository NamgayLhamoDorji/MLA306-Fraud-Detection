"""
FRAUD DETECTION - EVALUATION GRAPHS (final model, unseen month 7)

Regenerates:
    graphs/confusion_matrix.png
    graphs/precision_recall_curve.png
    graphs/threshold_comparison.png

Uses the FINAL DNN on the held-out test month (month 7) from processed/.
The alert threshold is chosen on validation month 6 (1% false-positive rate),
never on the test data.

Before running: check the model file names with   ls models
and edit MODEL_PATH / SCALER_PATH below if they differ.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from tensorflow import keras

# ---------------------------------------------------------------- settings
MODEL_PATH = "models/dnn_final.keras"        # <-- check with: ls models
SCALER_PATH = "models/dnn_final_scaler.pkl"  # <-- check with: ls models
TARGET_FPR = 0.01                            # 1% of legitimate applications flagged
OUT_DIR = "graphs"

print("=" * 55)
print("FRAUD DETECTION - EVALUATION GRAPHS (final DNN, month 7)")
print("=" * 55)


# ----------------------------------------------------------------- helpers
def load_xy(name):
    X = pd.read_csv(f"processed/X_{name}.csv")
    y = pd.read_csv(f"processed/y_{name}.csv").iloc[:, 0].astype(int).to_numpy()
    return X, y


def align_features(X, scaler):
    """Make X match the columns the scaler/model were trained on."""
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return X.reindex(columns=list(names), fill_value=0)
    n = getattr(scaler, "n_features_in_", X.shape[1])
    if X.shape[1] == n + 1 and "month" in X.columns:
        X = X.drop(columns=["month"])
    if X.shape[1] != n:
        raise SystemExit(
            f"Feature mismatch: data has {X.shape[1]} columns, model expects {n}."
        )
    return X


def score(model, scaler, X):
    X = align_features(X, scaler)
    return model.predict(scaler.transform(X), batch_size=4096, verbose=0).ravel()


def metrics_at(y, s, thr):
    flagged = s >= thr
    tp = int(((flagged) & (y == 1)).sum())
    fp = int(((flagged) & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# -------------------------------------------------------------------- load
print("\n1. Loading processed data (validation = month 6, test = month 7)...")
X_val, y_val = load_xy("val")
X_test, y_test = load_xy("test")
print(f"   Validation rows: {len(y_val)}   Test rows: {len(y_test)}")
print(f"   Test fraud cases: {int(y_test.sum())}")

print("2. Loading final model and scaler...")
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
if isinstance(scaler, dict):
    scaler = scaler.get("scaler", scaler)

print("3. Scoring validation and test data...")
s_val = score(model, scaler, X_val)
s_test = score(model, scaler, X_test)

# -------------------------------------------- threshold from validation only
thr = float(np.quantile(s_val[y_val == 0], 1 - TARGET_FPR))
precision_op, recall_op, f1_op = metrics_at(y_test, s_test, thr)
ap = average_precision_score(y_test, s_test)
auc = roc_auc_score(y_test, s_test)

print("\nSUMMARY (check these against your report)")
print(f"   Threshold at {TARGET_FPR:.0%} FPR on validation: {thr:.4f}")
print(f"   Test PR-AUC: {ap:.4f}   Test ROC-AUC: {auc:.4f}")
print(f"   At that threshold: recall {recall_op:.3f}, precision {precision_op:.3f}")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------- confusion matrix
print("\n4. Creating confusion matrix...")
pred = (s_test >= thr).astype(int)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=True)
ax.set_title(
    f"DNN final: Confusion Matrix, month 7\n"
    f"(threshold {thr:.3f}, set for {TARGET_FPR:.0%} false-positive rate on month 6)"
)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close(fig)

# ------------------------------------------------- precision-recall curve
print("5. Creating precision-recall curve...")
prec, rec, _ = precision_recall_curve(y_test, s_test)
prevalence = y_test.mean()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, label="DNN final")
ax.axhline(prevalence, color="grey", linestyle="--",
           label=f"Random model (fraud rate {prevalence:.3f})")
ax.scatter([recall_op], [precision_op], color="red", zorder=5,
           label=f"Operating point ({TARGET_FPR:.0%} FPR)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve, month 7 (AP = {ap:.3f})")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/precision_recall_curve.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------ threshold comparison
print("6. Creating threshold comparison chart...")
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
rows = [(t, *metrics_at(y_test, s_test, t)) for t in grid]
sweep = pd.DataFrame(rows, columns=["Threshold", "Precision", "Recall", "F1-score"])
sweep.to_csv("results/threshold_sweep_month7.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sweep["Threshold"], sweep["Recall"], marker="o", label="Recall")
ax.plot(sweep["Threshold"], sweep["Precision"], marker="s", label="Precision")
ax.plot(sweep["Threshold"], sweep["F1-score"], marker="^", label="F1-score")
ax.axvline(thr, color="black", linestyle="--",
           label=f"{TARGET_FPR:.0%} FPR threshold ({thr:.2f})")
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Score")
ax.set_title("DNN final: performance at different thresholds (month 7)")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/threshold_comparison.png", dpi=300)
plt.close(fig)

print("\nAll graphs saved:")
print(f"   {OUT_DIR}/confusion_matrix.png")
print(f"   {OUT_DIR}/precision_recall_curve.png")
print(f"   {OUT_DIR}/threshold_comparison.png")
print("   results/threshold_sweep_month7.csv")
print("\nVISUALIZATION COMPLETED!")