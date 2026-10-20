
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 65)
print("FINAL DNN EVALUATION - TEST MONTH 7")
print("=" * 65)

# 1. Load the selected model
model_path = "models/dnn_leaky_relu.keras"

# The activation-comparison script did not save its scaler.
# The comparison used a StandardScaler fitted on training data.
# Refit that scaler using training data only, never the test data.
X_train_df = pd.read_csv("processed/X_train.csv")
scaler = joblib.load("models/dnn_improved_scaler.pkl")

model = tf.keras.models.load_model(model_path)

# 2. Load the untouched test month
X_test_df = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv").iloc[:, 0].to_numpy()

print("Test data shape:", X_test_df.shape)
print("Test fraud cases:", int(np.sum(y_test == 1)))
print("Test legitimate cases:", int(np.sum(y_test == 0)))

# 3. Transform using a scaler fitted on training data only
X_test = scaler.transform(X_test_df).astype("float32")

# 4. Predict probabilities
y_prob = model.predict(
    X_test,
    batch_size=1024,
    verbose=1
).ravel()

# Keep threshold fixed at 0.5 for this final evaluation
threshold = 0.5
y_pred = (y_prob >= threshold).astype("int32")

# 5. Calculate final metrics
ap = average_precision_score(y_test, y_prob)
roc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nFINAL TEST RESULTS")
print("-" * 65)
print("PR-AUC / Average Precision:", round(ap, 4))
print("ROC-AUC:", round(roc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# 6. Save final metrics
metrics = {
    "model": "dnn_leaky_relu.keras",
    "test_month": 7,
    "threshold": threshold,
    "pr_auc": float(ap),
    "roc_auc": float(roc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "test_rows": int(len(y_test)),
    "test_fraud_cases": int(np.sum(y_test == 1)),
    "test_legitimate_cases": int(np.sum(y_test == 0))
}

os.makedirs("results", exist_ok=True)

with open(
    "results/final_test_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nSaved final metrics:")
print("results/final_test_metrics.json")
print("\nFINAL TEST EVALUATION COMPLETED!")
