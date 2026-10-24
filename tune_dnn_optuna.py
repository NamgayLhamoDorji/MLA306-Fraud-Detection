"""Optuna search for the final DNN (hyperparameters + simple architecture search).

Run from the project root:
    python tune_dnn_optuna.py
Quick test:
    DNN_TRIALS=3 python tune_dnn_optuna.py

Searches: number of hidden layers (1-3), width of each layer, dropout of each layer,
activation (relu / leaky_relu / gelu), optimizer (Adam / AdamW), learning rate, batch size.
The "month" column is dropped: it is a time label, not a customer feature.

Saves
    models/dnn_final.keras, models/dnn_final_scaler.pkl   (used by src/hybrid.py)
    models/dnn_optuna_best.keras, models/dnn_optuna_scaler.pkl
    results/optuna_trials.csv, optuna_best_params.json, dnn_optuna_best_metrics.json
    logs/optuna_dnn/trial_N  (TensorBoard)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SEED = 42
N_TRIALS = int(os.getenv("DNN_TRIALS", 25))
DROP_COLUMNS = ["month"]

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("=" * 60)
print("OPTUNA DNN TUNING (HYPERPARAMETERS + DEPTH/WIDTH)")
print("=" * 60)

# 1. Load training and validation data ONLY (month 7 is never touched here)
X_train_df = pd.read_csv("processed/X_train.csv").drop(columns=DROP_COLUMNS, errors="ignore")
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()
X_val_df = pd.read_csv("processed/X_val.csv").drop(columns=DROP_COLUMNS, errors="ignore")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

print("Training shape:", X_train_df.shape)
print("Validation shape:", X_val_df.shape)

# 2. Scale using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df).astype("float32")
X_val = scaler.transform(X_val_df).astype("float32")
y_train = y_train.astype("int32")
y_val = y_val.astype("int32")

# 3. Class weights from training labels only
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

for folder in ("models", "results", "logs"):
    os.makedirs(folder, exist_ok=True)


def activation_layer(name):
    if name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.01)
    return tf.keras.layers.Activation(name)


def suggest_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    params = {"n_layers": n_layers}
    for i in range(1, n_layers + 1):
        params[f"units_{i}"] = trial.suggest_categorical(f"units_{i}", [32, 64, 128, 256])
        params[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.1, 0.4)
    params["activation"] = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
    params["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    return params


def build_from_params(p, n_features):
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for i in range(1, p["n_layers"] + 1):
        layers += [
            tf.keras.layers.Dense(p[f"units_{i}"], use_bias=False),
            tf.keras.layers.BatchNormalization(),
            activation_layer(p["activation"]),
            tf.keras.layers.Dropout(p[f"dropout_{i}"]),
        ]
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)

    if p["optimizer"] == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=p["learning_rate"])
    else:
        opt = tf.keras.optimizers.AdamW(learning_rate=p["learning_rate"], weight_decay=1e-4)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def callbacks_for(log_dir):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]


def objective(trial):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + trial.number)

    params = suggest_params(trial)
    model = build_from_params(params, X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=params["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks_for(f"logs/optuna_dnn/trial_{trial.number}"),
        verbose=0,
    )

    y_prob = model.predict(X_val, batch_size=4096, verbose=0).ravel()
    ap = average_precision_score(y_val, y_prob)
    trial.set_user_attr("epochs_trained", len(history.history["loss"]))
    print(f"Trial {trial.number}: PR-AUC={ap:.5f}  params={trial.params}")
    return ap


study = optuna.create_study(
    study_name="fraud_detection_dnn",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "=" * 60)
print("OPTUNA TUNING COMPLETED")
print("=" * 60)
print("Best validation PR-AUC:", study.best_value)
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
with open("results/optuna_best_params.json", "w", encoding="utf-8") as f:
    json.dump(
        {"n_trials": N_TRIALS,
         "best_validation_pr_auc": study.best_value,
         "best_parameters": study.best_params},
        f, indent=4,
    )

# 9. Retrain the best configuration once and save it as the final DNN
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)

best_params = dict(study.best_params)
best_model = build_from_params(best_params, X_train.shape[1])
best_model.summary()

history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=best_params["batch_size"],
    class_weight=class_weights,
    callbacks=callbacks_for("logs/dnn_final"),
    verbose=1,
)

y_prob = best_model.predict(X_val, batch_size=4096, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype("int32")

metrics = {
    "validation_month": 6,
    "threshold": 0.5,
    "pr_auc": float(average_precision_score(y_val, y_prob)),
    "roc_auc": float(roc_auc_score(y_val, y_prob)),
    "precision": float(precision_score(y_val, y_pred, zero_division=0)),
    "recall": float(recall_score(y_val, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist(),
    "epochs_trained": len(history.history["loss"]),
    "features": X_train.shape[1],
    "best_parameters": best_params,
}

best_model.save("models/dnn_optuna_best.keras")
best_model.save("models/dnn_final.keras")
joblib.dump(scaler, "models/dnn_optuna_scaler.pkl")
joblib.dump(scaler, "models/dnn_final_scaler.pkl")

with open("results/dnn_optuna_best_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nTUNED MODEL VALIDATION RESULTS")
print(json.dumps(metrics, indent=4))
print("\nSaved: models/dnn_final.keras, models/dnn_final_scaler.pkl")
print("Saved: results/optuna_trials.csv, optuna_best_params.json, dnn_optuna_best_metrics.json")