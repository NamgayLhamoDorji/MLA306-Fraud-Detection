
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

print("=" * 60)
print("DNN FRAUD DETECTION - VALIDATION THRESHOLD COMPARISON")
print("=" * 60)

# 1. Load validation data only
print("\n1. Loading validation data...")

X_val = pd.read_csv("processed/X_val.csv")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

print("Validation shape:", X_val.shape)
print("Actual fraud cases:", int(y_val.sum()))

# 2. Load the saved DNN and scaler
print("\n2. Loading saved DNN and scaler...")

model = tf.keras.models.load_model("models/dnn_baseline.keras")
scaler = joblib.load("models/dnn_scaler.pkl")

# 3. Scale validation data using the fitted training scaler
X_val_scaled = scaler.transform(X_val).astype("float32")

# 4. Get predicted fraud probabilities
print("\n3. Predicting fraud probabilities...")

probabilities = model.predict(
    X_val_scaled,
    batch_size=1024,
    verbose=1
).ravel()

print(
    "Validation Average Precision:",
    round(average_precision_score(y_val, probabilities), 4)
)
print(
    "Validation ROC-AUC:",
    round(roc_auc_score(y_val, probabilities), 4)
)

# 5. Compare thresholds
print("\n4. Comparing thresholds...")

thresholds = [
    0.01, 0.02, 0.03, 0.05, 0.075,
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50
]

results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4),
        "True_Negatives": int(tn),
        "False_Alarms": int(fp),
        "Fraud_Missed": int(fn),
        "Fraud_Detected": int(tp)
    })

results_df = pd.DataFrame(results)

print("\n", results_df.to_string(index=False))

# 6. Save results
os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/dnn_threshold_comparison.csv",
    index=False
)

print("\nSaved: results/dnn_threshold_comparison.csv")
print("\nTHRESHOLD COMPARISON COMPLETED!")
