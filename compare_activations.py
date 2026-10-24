import os
import json
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

# Make results more reproducible
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ACTIVATION FUNCTION COMPARISON")
print("=" * 60)

# 1. Load training and validation data only
X_train_df = pd.read_csv("processed/X_train.csv").drop(columns=["month"])
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()

X_val_df = pd.read_csv("processed/X_val.csv").drop(columns=["month"])
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

print("Training data:", X_train_df.shape)
print("Validation data:", X_val_df.shape)

# 2. Scale features using training data only
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_df).astype("float32")
X_val = scaler.transform(X_val_df).astype("float32")

y_train = y_train.astype("int32")
y_val = y_val.astype("int32")

# 3. Calculate class weights using training labels only
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

# 4. Define the three activation functions
activations = ["relu", "leaky_relu", "gelu"]

results = []

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

for activation in activations:

    print("\n" + "=" * 60)
    print("Training activation:", activation)
    print("=" * 60)

    # Reset model initialization between experiments
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # 5. Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(128, use_bias=False),
        tf.keras.layers.BatchNormalization(),

        # Select activation
        tf.keras.layers.Activation(activation)
        if activation != "leaky_relu"
        else tf.keras.layers.LeakyReLU(negative_slope=0.01),

        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation(activation)
        if activation != "leaky_relu"
        else tf.keras.layers.LeakyReLU(negative_slope=0.01),

        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    # 6. Stop when validation PR-AUC stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    )

    # 7. Train
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

    # 8. Evaluate on validation month 6
    y_prob = model.predict(
        X_val,
        batch_size=1024,
        verbose=0
    ).ravel()

    # Keep threshold fixed at 0.5 for a fair comparison
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

    print("\nActivation:", activation)
    print("PR-AUC / Average Precision:", round(ap, 4))
    print("ROC-AUC:", round(roc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion matrix:\n", cm)

    results.append({
        "activation": activation,
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

    # Save each model separately
    model.save(f"models/dnn_{activation}.keras")

# 9. Save and display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="pr_auc",
    ascending=False
)

results_df.to_csv(
    "results/activation_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL ACTIVATION COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nSaved comparison to:")
print("results/activation_comparison.csv")
print("\nACTIVATION COMPARISON COMPLETED!")