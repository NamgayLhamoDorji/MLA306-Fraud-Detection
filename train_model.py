
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("=" * 50)
print("FRAUD DETECTION - MODEL TRAINING")
print("=" * 50)

# Load the original dataset
print("\n1. Loading dataset...")

df = pd.read_csv("data/Base.csv", low_memory=False)

X = df.drop(columns=["fraud_bool"])
y = df["fraud_bool"]

# Split before fitting preprocessing steps
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Identify categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Convert categorical columns to dummy variables
X_train = pd.get_dummies(
    X_train, columns=categorical_columns, dtype=int
)

X_test = pd.get_dummies(
    X_test, columns=categorical_columns, dtype=int
)

# Make sure train and test columns match
X_test = X_test.reindex(
    columns=X_train.columns, fill_value=0
)

print("Number of features:", X_train.shape[1])

# Build a pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
print("\n2. Training Random Forest...")
model.fit(X_train, y_train)
print("Training completed!")

# Make predictions
print("\n3. Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(
    "\nAverage Precision:",
    round(average_precision_score(y_test, y_prob), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4)
)

# Save model and feature names
os.makedirs("models", exist_ok=True)

joblib.dump(
    {"model": model, "feature_names": X_train.columns.tolist()},
    "models/fraud_detection_model.pkl"
)

print("\nModel saved to models/fraud_detection_model.pkl")
print("MODEL TRAINING COMPLETED!")
