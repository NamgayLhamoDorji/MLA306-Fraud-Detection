import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 60)
print("FRAUD DETECTION - IMPROVED DNN")
print("=" * 60)

# 1. Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load training and validation data
print("\n1. Loading data...")

X_train_df = pd.read_csv("processed/X_train.csv").drop(columns=["month"])
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()

X_val_df = pd.read_csv("processed/X_val.csv").drop(columns=["month"])
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

print("Training shape:", X_train_df.shape)
print("Validation shape:", X_val_df.shape)

# 3. Scale features using training data only
print("\n2. Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_df).astype("float32")
X_val = scaler.transform(X_val_df).astype("float32")

y_train = y_train.astype("int32")
y_val = y_val.astype("int32")

# 4. Handle class imbalance
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

# 5. Build improved DNN
print("\n4. Building improved DNN...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(128, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(64, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR")
    ]
)

model.summary()

# 6. Train with month 6 as validation
print("\n5. Training improved DNN...")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/dnn_improved"
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

# 7. Evaluate on validation month 6 only
print("\n6. Evaluating on validation month 6...")

y_prob = model.predict(
    X_val,
    batch_size=1024,
    verbose=0
).ravel()

threshold = 0.5
y_pred = (y_prob >= threshold).astype("int32")

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred, zero_division=0
))

cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

ap = average_precision_score(y_val, y_prob)
roc = roc_auc_score(y_val, y_prob)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

print("\nValidation Average Precision / PR-AUC:", round(ap, 4))
print("Validation ROC-AUC:", round(roc, 4))
print("Validation Precision:", round(precision, 4))
print("Validation Recall:", round(recall, 4))
print("Validation F1-score:", round(f1, 4))

# 8. Save under new names so the baseline remains unchanged
model.save("models/dnn_improved.keras")
joblib.dump(scaler, "models/dnn_improved_scaler.pkl")

results = {
    "model": "DNN with Batch Normalization and Dropout",
    "validation_month": 6,
    "threshold": threshold,
    "average_precision": float(ap),
    "roc_auc": float(roc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "epochs_trained": len(history.history["loss"])
}

with open("results/dnn_improved_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved model: models/dnn_improved.keras")
print("Saved scaler: models/dnn_improved_scaler.pkl")
print("Saved metrics: results/dnn_improved_metrics.json")
print("\nIMPROVED DNN COMPLETED!")