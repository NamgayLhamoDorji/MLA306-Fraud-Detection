
import os
import json
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
    f1_score
)

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

print("=" * 60)
print("OPTUNA DNN HYPERPARAMETER TUNING")
print("=" * 60)

# 1. Load training and validation data ONLY
X_train_df = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()

X_val_df = pd.read_csv("processed/X_val.csv")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

print("Training shape:", X_train_df.shape)
print("Validation shape:", X_val_df.shape)

# 2. Scale using training data only
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

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 4. Build a model using Optuna's suggested parameters
def build_model(trial):

    units1 = trial.suggest_categorical(
        "units1", [64, 128, 256]
    )
    units2 = trial.suggest_categorical(
        "units2", [32, 64, 128]
    )
    dropout1 = trial.suggest_float(
        "dropout1", 0.1, 0.4
    )
    dropout2 = trial.suggest_float(
        "dropout2", 0.1, 0.3
    )
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [256, 512]
    )
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW"]
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(units1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.Dropout(dropout1),

        tf.keras.layers.Dense(units2, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.Dropout(dropout2),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4
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

    return model, batch_size


# 5. Define the Optuna objective
def objective(trial):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42 + trial.number)

    model, batch_size = build_model(trial)

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
        epochs=15,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )

    # Use validation probabilities for Average Precision
    y_prob = model.predict(
        X_val,
        batch_size=1024,
        verbose=0
    ).ravel()

    ap = average_precision_score(y_val, y_prob)

    trial.set_user_attr(
        "epochs_trained",
        len(history.history["loss"])
    )

    # Save only the current trial's validation score
    print(
        f"Trial {trial.number}: "
        f"PR-AUC={ap:.5f}, "
        f"parameters={trial.params}"
    )

    return ap


# 6. Run the search
study = optuna.create_study(
    study_name="fraud_detection_dnn",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    objective,
    n_trials=10,
    show_progress_bar=True
)

# 7. Print the best result
print("\n" + "=" * 60)
print("OPTUNA TUNING COMPLETED")
print("=" * 60)

print("Best validation PR-AUC:", study.best_value)
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# 8. Save all trial results
trials_df = study.trials_dataframe()
trials_df.to_csv(
    "results/optuna_trials.csv",
    index=False
)

with open(
    "results/optuna_best_params.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "best_validation_pr_auc": study.best_value,
            "best_parameters": study.best_params
        },
        f,
        indent=4
    )

# 9. Train a final candidate using the best parameters
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

best_trial = study.best_trial
best_model, best_batch_size = build_model(best_trial)

final_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

best_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=best_batch_size,
    class_weight=class_weights,
    callbacks=[final_early_stopping],
    verbose=1
)

# 10. Evaluate the tuned candidate on validation month 6
y_prob = best_model.predict(
    X_val,
    batch_size=1024,
    verbose=0
).ravel()

y_pred = (y_prob >= 0.5).astype("int32")

metrics = {
    "validation_month": 6,
    "threshold": 0.5,
    "pr_auc": float(average_precision_score(y_val, y_prob)),
    "roc_auc": float(roc_auc_score(y_val, y_prob)),
    "precision": float(precision_score(
        y_val, y_pred, zero_division=0
    )),
    "recall": float(recall_score(
        y_val, y_pred, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_val, y_pred, zero_division=0
    )),
    "confusion_matrix": tf.math.confusion_matrix(
        y_val, y_pred, num_classes=2
    ).numpy().tolist(),
    "best_parameters": study.best_params
}

best_model.save("models/dnn_optuna_best.keras")
pd.to_pickle(scaler, "models/dnn_optuna_scaler.pkl")

with open(
    "results/dnn_optuna_best_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nTUNED MODEL VALIDATION RESULTS")
print(json.dumps(metrics, indent=4))
print("\nSaved model: models/dnn_optuna_best.keras")
print("Saved scaler: models/dnn_optuna_scaler.pkl")
print("Saved metrics: results/dnn_optuna_best_metrics.json")
