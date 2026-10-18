
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)

print("=" * 50)
print("FRAUD DETECTION - VISUALIZATION")
print("=" * 50)

# 1. Load dataset
print("\n1. Loading dataset...")

df = pd.read_csv("data/Base.csv", low_memory=False)

X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# 2. Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Encode categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

X_train = pd.get_dummies(
    X_train, columns=categorical_columns, dtype=int
)

X_test = pd.get_dummies(
    X_test, columns=categorical_columns, dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns, fill_value=0
)

# 4. Load trained model
print("2. Loading trained model...")

saved = joblib.load("models/fraud_detection_model.pkl")
model = saved["model"]

# 5. Predict probabilities
print("3. Generating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.4).astype(int)

# Create output folder
import os
os.makedirs("graphs", exist_ok=True)

# 6. Confusion matrix
print("4. Creating confusion matrix...")

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

display.plot(values_format="d")
plt.title("Fraud Detection - Confusion Matrix (Threshold 0.4)")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300)
plt.close()

# 7. Precision-recall curve
print("5. Creating precision-recall curve...")

precision, recall, thresholds = precision_recall_curve(
    y_test, probabilities
)

ap = average_precision_score(y_test, probabilities)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (AP = {ap:.4f})")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/precision_recall_curve.png", dpi=300)
plt.close()

# 8. Threshold comparison chart
print("6. Creating threshold comparison chart...")

results = pd.read_csv("processed/threshold_comparison.csv")

plt.figure(figsize=(9, 6))
plt.plot(
    results["Threshold"],
    results["Recall"],
    marker="o",
    label="Recall"
)
plt.plot(
    results["Threshold"],
    results["Precision"],
    marker="s",
    label="Precision"
)
plt.plot(
    results["Threshold"],
    results["F1-score"],
    marker="^",
    label="F1-score"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Performance at Different Thresholds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/threshold_comparison.png", dpi=300)
plt.close()

print("\nAll graphs saved successfully:")
print("graphs/confusion_matrix.png")
print("graphs/precision_recall_curve.png")
print("graphs/threshold_comparison.png")

print("\nVISUALIZATION COMPLETED!")
