
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

print("=" * 55)
print("FRAUD DETECTION - THRESHOLD IMPROVEMENT")
print("=" * 55)

# 1. Load the original dataset
print("\n1. Loading dataset...")

df = pd.read_csv("data/Base.csv", low_memory=False)

X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# 2. Use the same train-test split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Encode categorical features
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

# 4. Load the trained model
print("2. Loading trained Random Forest model...")

saved = joblib.load("models/fraud_detection_model.pkl")
model = saved["model"]

# 5. Get fraud probabilities
print("3. Predicting fraud probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

print("\nAverage Precision:",
      round(average_precision_score(y_test, probabilities), 4))

# 6. Compare different thresholds
print("\n4. Comparing thresholds")

results = []

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "False Alarms": int(fp),
        "Fraud Detected": int(tp),
        "Fraud Missed": int(fn)
    })

results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))

# 7. Save the comparison
results_df.to_csv(
    "processed/threshold_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("processed/threshold_comparison.csv")

print("\nTHRESHOLD COMPARISON COMPLETED!")
