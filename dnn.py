
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

print("=" * 55)
print("FRAUD DETECTION - DNN BASELINE")
print("=" * 55)

# 1. Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load training and validation data only
print("\n1. Loading training and validation data...")

X_train = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0]

X_val = pd.read_csv("processed/X_val.csv")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 3. Scale using training data only
print("\n2. Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype("float32")
X_val = scaler.transform(X_val).astype("float32")

y_train = y_train.to_numpy().astype("int32")
y_val = y_val.to_numpy().astype("int32")

# 4. Calculate class weights from training labels only
print("\n3. Calculating class weights...")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("Class weights:", class_weights)

# 5. Build baseline DNN
print("\n4. Building DNN...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR")
    ]
)

model.summary()

# 6. Train using month 6 for validation
print("\n5. Training DNN...")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/dnn_baseline"
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate using validation data only
# Month 7 is deliberately not loaded or evaluated here.
print("\n6. Evaluating on validation month 6...")

y_val_prob = model.predict(X_val, verbose=0).ravel()
y_val_pred = (y_val_prob >= 0.5).astype("int32")

print("\nValidation Classification Report:")
print(classification_report(
    y_val, y_val_pred, zero_division=0
))

cm = confusion_matrix(y_val, y_val_pred)

print("Validation Confusion Matrix:")
print(cm)

ap = average_precision_score(y_val, y_val_prob)
roc_auc = roc_auc_score(y_val, y_val_prob)
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)
f1 = f1_score(y_val, y_val_pred, zero_division=0)

print("Validation Average Precision / PR-AUC:", round(ap, 4))
print("Validation ROC-AUC:", round(roc_auc, 4))
print("Validation Precision:", round(precision, 4))
print("Validation Recall:", round(recall, 4))
print("Validation F1-score:", round(f1, 4))

# 8. Save baseline model, scaler and validation metrics
model.save("models/dnn_baseline.keras")
joblib.dump(scaler, "models/dnn_scaler.pkl")

results = {
    "model": "DNN baseline",
    "validation_month": 6,
    "validation_average_precision": float(ap),
    "validation_roc_auc": float(roc_auc),
    "validation_precision": float(precision),
    "validation_recall": float(recall),
    "validation_f1": float(f1),
    "validation_confusion_matrix": cm.tolist()
}

with open("results/dnn_baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved model: models/dnn_baseline.keras")
print("Saved scaler: models/dnn_scaler.pkl")
print("Saved metrics: results/dnn_baseline_metrics.json")
print("\nDNN BASELINE COMPLETED!")
