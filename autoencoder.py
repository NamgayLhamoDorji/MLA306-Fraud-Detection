"""
MEMBER 3 - AUTOENCODER FOR FRAUD DETECTION (MLA306)

Idea in one line:
    Train a network to copy NORMAL applications only. A fraud application looks
    unusual, so the network copies it badly -> big "reconstruction error" -> flag.

Steps in this file:
    1. Load train + validation data (month 7 test data is NEVER loaded here)
    2. Scale the features (scaler fitted on normal training rows only)
    2B. Compare feature sets: all features vs the strongest 8 / 12
    3. Baseline autoencoder (64 -> 32 -> 16 -> 32 -> 64)
    4. Compare bottleneck sizes
    5. Plot the error distribution (normal vs fraud)
    6. Optuna study
    7. Train the final autoencoder with the best parameters
    8. Choose the alert threshold
    9. Save model, scaler, threshold and results

Run:   python autoencoder.py
Quick test on a laptop:   AE_MAX_TRAIN_ROWS=100000 AE_TRIALS=5 python autoencoder.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up window
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ----------------------------------------------------------------------
# Settings (change here, not inside the code)
# ----------------------------------------------------------------------
SEED = 42
DROP_COLUMNS = ["month"]      # month is a time label, not a customer feature
CLIP_VALUE = 5.0              # stop rare one-hot columns from dominating the error
FEATURE_OPTIONS = [None, 8, 12]   # None = all features; 8 = the 8 strongest features, etc.
BOTTLENECK_SIZES = [4, 8, 16, 32]
N_TRIALS = int(os.getenv("AE_TRIALS", 10))
MAX_TRAIN_ROWS = int(os.getenv("AE_MAX_TRAIN_ROWS", 0)) or None   # None = use all
THRESHOLD_PERCENTILES = [90, 95, 97, 99, 99.5]
CHOSEN_PERCENTILE = None      # None = pick the percentile with the best validation F1
                              # or type a number such as 95 to force your own choice

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

for folder in ["models", "results", "graphs", "logs"]:
    os.makedirs(folder, exist_ok=True)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def prepare_features(df, scaler, keep_idx=None):
    """Drop unused columns, scale, clip, keep the chosen columns.
    Member 4 / 5 reuse this for new data (keep_idx is saved in ae_threshold.json)."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    x = scaler.transform(df)
    x = np.clip(x, -CLIP_VALUE, CLIP_VALUE)
    if keep_idx is not None:
        x = x[:, keep_idx]
    return x.astype("float32")


def build_autoencoder(n_features, units1=64, units2=32, bottleneck=16,
                      learning_rate=0.001, dropout=0.0):
    """Encoder -> bottleneck -> decoder. Output is linear (no activation)
    because the scaled inputs can be negative."""
    layers = [
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(units1, activation="relu"),
        tf.keras.layers.Dense(units2, activation="relu"),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers += [
        tf.keras.layers.Dense(bottleneck, activation="relu"),   # bottleneck
        tf.keras.layers.Dense(units2, activation="relu"),
        tf.keras.layers.Dense(units1, activation="relu"),
        tf.keras.layers.Dense(n_features, activation=None),     # rebuild the input
    ]
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def train_autoencoder(model, x_train_normal, x_val_normal,
                      epochs=30, batch_size=512, log_name=None, verbose=0):
    """Input = output, because the goal is to copy the application.
    Early stopping watches the loss on NORMAL validation rows only."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-3, patience=3,
            restore_best_weights=True
        )
    ]
    if log_name:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=f"logs/{log_name}")
        )
    history = model.fit(
        x_train_normal, x_train_normal,
        validation_data=(x_val_normal, x_val_normal),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


def reconstruction_error(model, x):
    """Average squared difference between the input and its rebuild, per row."""
    rebuilt = model.predict(x, batch_size=2048, verbose=0)
    return np.mean(np.square(x - rebuilt), axis=1)


def score_errors(y_true, errors):
    """How well does the error separate fraud from normal? (no threshold needed)"""
    return {
        "pr_auc": float(average_precision_score(y_true, errors)),
        "roc_auc": float(roc_auc_score(y_true, errors)),
    }


# ----------------------------------------------------------------------
print("=" * 60)
print("FRAUD DETECTION - AUTOENCODER (MEMBER 3)")
print("=" * 60)

# ----------------------------------------------------------------------
# STEP 1. Load training and validation data only
# ----------------------------------------------------------------------
print("\nSTEP 1. Loading data (train + validation only)...")

X_train_df = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv").iloc[:, 0].to_numpy()
X_val_df = pd.read_csv("processed/X_val.csv")
y_val = pd.read_csv("processed/y_val.csv").iloc[:, 0].to_numpy()

if MAX_TRAIN_ROWS and len(X_train_df) > MAX_TRAIN_ROWS:
    keep = np.random.RandomState(SEED).choice(
        len(X_train_df), MAX_TRAIN_ROWS, replace=False
    )
    keep.sort()
    X_train_df = X_train_df.iloc[keep].reset_index(drop=True)
    y_train = y_train[keep]
    print(f"Using a {MAX_TRAIN_ROWS:,}-row sample of the training data")

print("Train:", X_train_df.shape, "| fraud:", int(y_train.sum()))
print("Val  :", X_val_df.shape, "| fraud:", int(y_val.sum()))

# ----------------------------------------------------------------------
# STEP 2. Keep only NORMAL rows for training, then scale
# ----------------------------------------------------------------------
print("\nSTEP 2. Selecting normal rows and scaling...")

X_train_df = X_train_df.drop(columns=[c for c in DROP_COLUMNS if c in X_train_df.columns])
feature_columns = X_train_df.columns.tolist()

normal_train_df = X_train_df[y_train == 0]            # fraud rows removed
scaler = StandardScaler().fit(normal_train_df)        # fitted on normal training rows only

x_train_normal = prepare_features(normal_train_df, scaler)
x_val_all = prepare_features(X_val_df[feature_columns], scaler)
x_val_normal = x_val_all[y_val == 0]

n_features = x_train_normal.shape[1]
print("Features used:", n_features, "(dropped:", DROP_COLUMNS, ")")
print("Normal training rows :", len(x_train_normal))
print("Normal validation rows:", len(x_val_normal))

# ----------------------------------------------------------------------
# STEP 2B. Which features should the autoencoder look at?
#   Fraud in this dataset is a SUBTLE shift, not a strange outlier. With all
#   features the useful signal is drowned by noisy columns. We rank features
#   using TRAINING labels only (never validation) and test a few feature sets.
# ----------------------------------------------------------------------
print("\nSTEP 2B. Comparing feature sets...")

train_auc = pd.Series({
    col: abs(roc_auc_score(y_train, X_train_df[col]) - 0.5)
    for col in feature_columns
}).sort_values(ascending=False)
print("Strongest features (training data):", train_auc.index[:8].tolist())

x_train_all = prepare_features(normal_train_df, scaler)   # all columns, scaled
x_val_full = x_val_all                                     # all columns, scaled
col_position = {c: i for i, c in enumerate(feature_columns)}

feature_rows = []
feature_sets = {}
for k in FEATURE_OPTIONS:
    cols = feature_columns if k is None else train_auc.index[:k].tolist()
    idx = [col_position[c] for c in cols]
    feature_sets[k] = (cols, idx)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    xt = x_train_all[:, idx]
    xv = x_val_full[:, idx]
    model = build_autoencoder(len(idx), units1=32, units2=32, bottleneck=8)
    train_autoencoder(model, xt, xv[y_val == 0])
    scores = score_errors(y_val, reconstruction_error(model, xv))
    feature_rows.append({
        "features": "all" if k is None else f"top_{k}",
        "n_features": len(idx),
        "pr_auc": scores["pr_auc"],
        "roc_auc": scores["roc_auc"],
    })
    print(f"  {feature_rows[-1]['features']:>7}  n={len(idx):>2}  "
          f"PR-AUC={scores['pr_auc']:.4f}  ROC-AUC={scores['roc_auc']:.4f}")

feature_df = pd.DataFrame(feature_rows)
feature_df.to_csv("results/ae_feature_set_comparison.csv", index=False)

best_row = int(feature_df["pr_auc"].idxmax())
best_k = FEATURE_OPTIONS[best_row]
selected_columns, keep_idx = feature_sets[best_k]
print("Chosen feature set:", feature_df.loc[best_row, "features"],
      f"({len(keep_idx)} features)")

# From here on, everything uses only the chosen columns
x_train_normal = x_train_all[:, keep_idx]
x_val_all = x_val_full[:, keep_idx]
x_val_normal = x_val_all[y_val == 0]
n_features = x_train_normal.shape[1]

# ----------------------------------------------------------------------
# STEP 3. Baseline autoencoder
# ----------------------------------------------------------------------
print("\nSTEP 3. Baseline autoencoder (64-32-16-32-64)...")

baseline = build_autoencoder(n_features)
baseline.summary()

train_autoencoder(
    baseline, x_train_normal, x_val_normal,
    log_name="ae_baseline", verbose=1
)

val_err = reconstruction_error(baseline, x_val_all)
baseline_scores = score_errors(y_val, val_err)
print("Baseline validation scores:", baseline_scores)

baseline.save("models/autoencoder_baseline.keras")
with open("results/ae_baseline_metrics.json", "w") as f:
    json.dump({"model": "Autoencoder baseline", "validation_month": 6,
               "bottleneck": 16, **baseline_scores}, f, indent=4)

# ----------------------------------------------------------------------
# STEP 4. Compare bottleneck sizes
# ----------------------------------------------------------------------
print("\nSTEP 4. Comparing bottleneck sizes:", BOTTLENECK_SIZES)

rows = []
for size in BOTTLENECK_SIZES:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_autoencoder(n_features, bottleneck=size)
    history = train_autoencoder(model, x_train_normal, x_val_normal)
    err = reconstruction_error(model, x_val_all)
    scores = score_errors(y_val, err)

    rows.append({
        "bottleneck": size,
        "pr_auc": scores["pr_auc"],
        "roc_auc": scores["roc_auc"],
        "mean_error_normal": float(err[y_val == 0].mean()),
        "mean_error_fraud": float(err[y_val == 1].mean()),
        "epochs_trained": len(history.history["loss"]),
    })
    print(f"  bottleneck={size:>2}  PR-AUC={scores['pr_auc']:.4f}  "
          f"ROC-AUC={scores['roc_auc']:.4f}")

bottleneck_df = pd.DataFrame(rows)
bottleneck_df.to_csv("results/ae_bottleneck_comparison.csv", index=False)
best_bottleneck = int(bottleneck_df.sort_values("pr_auc", ascending=False).iloc[0]["bottleneck"])
print("Best bottleneck by PR-AUC:", best_bottleneck)

# ----------------------------------------------------------------------
# STEP 5. Error distribution (normal vs fraud) for the baseline
# ----------------------------------------------------------------------
def plot_error_distribution(errors, labels, title, path, threshold=None):
    plt.figure(figsize=(8, 5))
    upper = np.percentile(errors, 99.9)
    bins = np.linspace(0, upper, 80)
    plt.hist(errors[labels == 0], bins=bins, alpha=0.6, density=True,
             label="Normal", color="tab:blue")
    plt.hist(errors[labels == 1], bins=bins, alpha=0.6, density=True,
             label="Fraud", color="tab:red")
    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--",
                    label=f"Threshold = {threshold:.3f}")
    plt.yscale("log")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Density (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


print("\nSTEP 5. Plotting error distribution...")
plot_error_distribution(
    val_err, y_val,
    "Autoencoder baseline: reconstruction error (validation month 6)",
    "graphs/ae_error_distribution_baseline.png"
)

# ----------------------------------------------------------------------
# STEP 6. Optuna study
# ----------------------------------------------------------------------
print(f"\nSTEP 6. Optuna study ({N_TRIALS} trials)...")


def objective(trial):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + trial.number)

    params = {
        "units1": trial.suggest_categorical("units1", [32, 64, 128]),
        "units2": trial.suggest_categorical("units2", [16, 32, 64]),
        "bottleneck": trial.suggest_categorical("bottleneck", [4, 8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    model = build_autoencoder(n_features, **params)
    history = train_autoencoder(
        model, x_train_normal, x_val_normal, epochs=15, batch_size=batch_size
    )
    err = reconstruction_error(model, x_val_all)
    pr_auc = average_precision_score(y_val, err)

    trial.set_user_attr("epochs_trained", len(history.history["loss"]))
    print(f"Trial {trial.number}: PR-AUC={pr_auc:.5f}  {trial.params}")
    return pr_auc


study = optuna.create_study(
    study_name="fraud_detection_autoencoder",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=N_TRIALS)

print("Best validation PR-AUC:", study.best_value)
print("Best parameters:", study.best_params)

study.trials_dataframe().to_csv("results/ae_optuna_trials.csv", index=False)
with open("results/ae_optuna_best_params.json", "w") as f:
    json.dump({"best_validation_pr_auc": study.best_value,
               "best_parameters": study.best_params}, f, indent=4)

# ----------------------------------------------------------------------
# STEP 7. Train the final autoencoder with the best parameters
# ----------------------------------------------------------------------
print("\nSTEP 7. Training final autoencoder...")

best = dict(study.best_params)
best_batch_size = best.pop("batch_size")

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)

final_model = build_autoencoder(n_features, **best)
final_history = train_autoencoder(
    final_model, x_train_normal, x_val_normal,
    epochs=30, batch_size=best_batch_size, log_name="ae_final", verbose=1
)

val_err = reconstruction_error(final_model, x_val_all)
final_scores = score_errors(y_val, val_err)
print("Final validation scores:", final_scores)

# ----------------------------------------------------------------------
# STEP 8. Choose the alert threshold
#   threshold = a high percentile of the errors of NORMAL validation rows
#   (meaning: "flag anything that is more unusual than X% of normal applications")
# ----------------------------------------------------------------------
print("\nSTEP 8. Choosing the threshold...")

normal_val_err = val_err[y_val == 0]
threshold_rows = []

for pct in THRESHOLD_PERCENTILES:
    thr = float(np.percentile(normal_val_err, pct))
    pred = (val_err >= thr).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    threshold_rows.append({
        "percentile": pct,
        "threshold": thr,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1_score": f1_score(y_val, pred, zero_division=0),
        "flag_rate": float(pred.mean()),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv("results/ae_threshold_comparison.csv", index=False)
print(threshold_df[["percentile", "threshold", "precision", "recall",
                    "f1_score", "flag_rate"]].round(4).to_string(index=False))

if CHOSEN_PERCENTILE is None:
    chosen = threshold_df.sort_values("f1_score", ascending=False).iloc[0]
else:
    chosen = threshold_df[threshold_df["percentile"] == CHOSEN_PERCENTILE].iloc[0]

threshold = float(chosen["threshold"])
print(f"\nChosen: {chosen['percentile']}th percentile -> threshold {threshold:.5f}")

plot_error_distribution(
    val_err, y_val,
    "Final autoencoder: reconstruction error (validation month 6)",
    "graphs/ae_error_distribution_final.png",
    threshold=threshold
)

# ----------------------------------------------------------------------
# STEP 9. Save everything Member 4 needs
# ----------------------------------------------------------------------
print("\nSTEP 9. Saving...")

final_model.save("models/autoencoder_best.keras")
joblib.dump(scaler, "models/ae_scaler.pkl")

with open("models/ae_threshold.json", "w") as f:
    json.dump({
        "threshold": threshold,
        "percentile_of_normal_val_errors": float(chosen["percentile"]),
        "clip_value": CLIP_VALUE,
        "dropped_columns": DROP_COLUMNS,
        "feature_columns": feature_columns,
        "selected_columns": selected_columns,
        "keep_idx": keep_idx,
        "best_parameters": study.best_params,
    }, f, indent=4)

metrics = {
    "model": "Autoencoder (trained on non-fraud only)",
    "validation_month": 6,
    "features_used": selected_columns,
    "pr_auc": final_scores["pr_auc"],
    "roc_auc": final_scores["roc_auc"],
    "threshold": threshold,
    "percentile": float(chosen["percentile"]),
    "precision": float(chosen["precision"]),
    "recall": float(chosen["recall"]),
    "f1_score": float(chosen["f1_score"]),
    "flag_rate": float(chosen["flag_rate"]),
    "confusion_matrix": [
        [int(chosen["true_negatives"]), int(chosen["false_positives"])],
        [int(chosen["false_negatives"]), int(chosen["true_positives"])],
    ],
    "epochs_trained": len(final_history.history["loss"]),
    "best_parameters": study.best_params,
}
with open("results/ae_final_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved: models/autoencoder_best.keras")
print("Saved: models/ae_scaler.pkl")
print("Saved: models/ae_threshold.json")
print("Saved: results/ae_*.csv / ae_*.json, graphs/ae_*.png")
print("\nAUTOENCODER COMPLETED! (Month 7 test data was not touched.)")
