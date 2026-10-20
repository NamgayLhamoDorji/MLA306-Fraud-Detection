
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

print("=" * 60)
print("ADAM VS ADAMW OPTIMIZER COMPARISON")
print("=" * 60)

# Load training and validation data only
X_train_df = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()

X_val_df = pd.read_csv("processed/X_val.csv")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

# Fit scaler on training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df).astype("float32")
X_val = scaler.transform(X_val_df).astype("float32")

y_train = y_train.astype("int32")
y_val = y_val.astype("int32")

# Handle class imbalance
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

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

optimizers = ["Adam", "AdamW"]

for optimizer_name in optimizers:

    print("\n" + "=" * 60)
    print("Training with:", optimizer_name)
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # Same architecture for a fair comparison
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(128, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001
        )
    else:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate using the same validation threshold
    y_prob = model.predict(
        X_val,
        batch_size=1024,
        verbose=0
    ).ravel()

    y_pred = (y_prob >= 0.5).astype("int32")

    ap = average_precision_score(y_val, y_prob)
    roc = roc_auc_score(y_val, y_prob)
    precision = precision_score(
        y_val, y_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    print("\nOptimizer:", optimizer_name)
    print("PR-AUC:", round(ap, 4))
    print("ROC-AUC:", round(roc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion matrix:\n", cm)

    results.append({
        "optimizer": optimizer_name,
        "pr_auc": float(ap),
        "roc_auc": float(roc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
        "epochs_trained": len(history.history["loss"])
    })

    model.save(f"models/dnn_{optimizer_name.lower()}.keras")

# Save comparison results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="pr_auc",
    ascending=False
)

results_df.to_csv(
    "results/optimizer_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL OPTIMIZER COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nSaved to results/optimizer_comparison.csv")
print("OPTIMIZER COMPARISON COMPLETED!")
