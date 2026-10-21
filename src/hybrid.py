"""Hybrid fraud scoring utilities for MLA306 Member 4.

This module supports evaluation and SHAP explanations. If the testing-only
stand-in autoencoder is loaded, its results must not be presented as final.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"

TARGET = "fraud_bool"
MONTH = "month"
TARGET_FPR = 0.01


def load_processed():
    """Return concatenated processed features and labels (train, val, test)."""
    x_frames, y_frames = [], []
    for split in ("train", "val", "test"):
        x_path = PROCESSED_DIR / f"X_{split}.csv"
        y_path = PROCESSED_DIR / f"y_{split}.csv"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {x_path} or {y_path}. Run preprocessing first."
            )
        x_frames.append(pd.read_csv(x_path))
        y_frames.append(pd.read_csv(y_path).iloc[:, 0].rename(TARGET))
    X = pd.concat(x_frames, ignore_index=True)
    y = pd.concat(y_frames, ignore_index=True).to_frame()
    return X, y


def load_data(path=None):
    """Load the raw dataset for explanations/fairness checks."""
    data_path = Path(path) if path else DATA_DIR / "Base.csv"
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def load_artifacts():
    """Load the trained DNN, feature scaler, and available autoencoder."""
    dnn_path = MODELS_DIR / "dnn_improved.keras"
    scaler_path = MODELS_DIR / "dnn_improved_scaler.pkl"
    ae_path = MODELS_DIR / "autoencoder.keras"
    standin_path = MODELS_DIR / "autoencoder_standin.keras"

    if not dnn_path.exists():
        raise FileNotFoundError(f"Missing DNN model: {dnn_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing DNN scaler: {scaler_path}")

    dnn = load_model(dnn_path)
    scaler = joblib.load(scaler_path)

    if ae_path.exists():
        ae = load_model(ae_path)
        ae_name = "Member 3 autoencoder"
    elif standin_path.exists():
        warnings.warn(
            "Using TESTING-ONLY autoencoder stand-in. Do not use these "
            "results as final project results.",
            RuntimeWarning,
        )
        ae = load_model(standin_path)
        ae_name = "TESTING-ONLY stand-in"
    else:
        raise FileNotFoundError(
            "No autoencoder found. Expected models/autoencoder.keras "
            "from Member 3."
        )

    return {
        "dnn": dnn,
        "ae": ae,
        "prep": scaler,
        "scaler": scaler,
        "ae_name": ae_name,
    }


def feature_names(prep, n_features=None):
    """Get feature names from the fitted scaler where available."""
    names = getattr(prep, "feature_names_in_", None)
    if names is not None:
        names = list(names)
    else:
        count = n_features if n_features is not None else len(prep.mean_)
        names = [f"feature_{i}" for i in range(count)]
    if n_features is not None:
        names = names[:n_features]
    return names


def _prepare_features(artifacts, frame):
    """Align raw/processed columns with the DNN scaler and scale them."""
    scaler = artifacts.get("scaler", artifacts.get("prep"))
    expected = feature_names(scaler)
    data = frame.drop(columns=[TARGET], errors="ignore").copy()
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    if not set(expected).issubset(data.columns):
        data = pd.get_dummies(data, drop_first=True, dtype=int)

    data = data.reindex(columns=expected, fill_value=0)
    data = data.replace([np.inf, -np.inf], np.nan)

    # The scaler was fitted on the training features; use its training means
    # for any unexpected missing values before applying the same transform.
    means = getattr(scaler, "mean_", np.zeros(len(expected)))
    for i, col in enumerate(expected):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(float(means[i]))

    return scaler.transform(data)


def predict_components(artifacts, frame):
    """Return scaled X, DNN fraud probabilities, and AE reconstruction errors."""
    X = _prepare_features(artifacts, frame)
    probabilities = np.asarray(
        artifacts["dnn"].predict(X, verbose=0)
    ).reshape(-1)
    reconstructed = artifacts["ae"].predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)
    return X, probabilities, errors


def threshold_at_fpr(y_true, scores, target_fpr=TARGET_FPR):
    """Select a score threshold based on the negative examples."""
    y = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    negatives = scores[y == 0]
    if negatives.size == 0:
        raise ValueError("Cannot choose an FPR threshold without negative cases.")
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    return float(np.quantile(negatives, q, method="higher"))


def tpr_at_fpr(y_true, scores, target_fpr=TARGET_FPR):
    """Return recall/TPR at the requested false-positive-rate target."""
    y = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    threshold = threshold_at_fpr(y, scores, target_fpr)
    positives = y == 1
    if not positives.any():
        return float("nan")
    return float((scores[positives] >= threshold).mean())


class HybridScorer:
    """Blend DNN probability with a normalized AE reconstruction-anomaly score."""

    def __init__(self, dnn=None, ae=None, weight=0.5, target_fpr=TARGET_FPR):
        self.dnn = dnn
        self.ae = ae
        self.weight = float(weight)  # weight assigned to AE score
        self.w = self.weight
        self.threshold = None
        self.target_fpr = float(target_fpr)
        self.sorted_ae = None
        self.threshold_ = None
        self.sweep_ = pd.DataFrame()

    def _normalize_ae(self, errors):
        errors = np.asarray(errors, dtype=float).reshape(-1)
        if self.sorted_ae is None or len(self.sorted_ae) == 0:
            raise ValueError("Hybrid scorer has not been fitted/calibrated.")
        ranks = np.searchsorted(self.sorted_ae, errors, side="right")
        return ranks / len(self.sorted_ae)

    def score_components(self, probabilities, errors):
        p = np.asarray(probabilities, dtype=float).reshape(-1)
        ae_score = self._normalize_ae(errors)
        return (1.0 - self.weight) * p + self.weight * ae_score


    def combine(self, probabilities, errors):
        """Combine DNN probabilities and autoencoder anomaly errors."""
        return self.score_components(probabilities, errors)

    def fit_components(self, probabilities, errors, y_true):
        p = np.asarray(probabilities, dtype=float).reshape(-1)
        e = np.asarray(errors, dtype=float).reshape(-1)
        y = np.asarray(y_true).reshape(-1).astype(int)
        if not (len(p) == len(e) == len(y)):
            raise ValueError("Probabilities, errors, and labels must have equal lengths.")

        self.sorted_ae = np.sort(e)
        rows = []
        best_tpr, best_weight, best_threshold = -1.0, 0.0, None

        for weight in np.linspace(0.0, 1.0, 21):
            score = (1.0 - weight) * p + weight * self._normalize_ae(e)
            threshold = threshold_at_fpr(y, score, self.target_fpr)
            recall = tpr_at_fpr(y, score, self.target_fpr)
            negatives = y == 0
            measured_fpr = (
                float((score[negatives] >= threshold).mean())
                if negatives.any() else float("nan")
            )
            rows.append({
                "ae_weight": float(weight),
                "dnn_weight": float(1.0 - weight),
                "tpr_at_target_fpr": recall,
                "threshold": threshold,
                "measured_fpr": measured_fpr,
            })
            comparable = np.nan_to_num(recall, nan=-1.0)
            if comparable > best_tpr:
                best_tpr = comparable
                best_weight = float(weight)
                best_threshold = threshold

        self.sweep_ = pd.DataFrame(rows)
        self.weight = best_weight
        self.w = self.weight
        self.threshold_ = float(best_threshold)
        self.threshold = self.threshold_
        return self

    def score(self, X):
        """Score already-scaled feature rows; used by SHAP and the app."""
        if self.dnn is None or self.ae is None:
            raise ValueError("DNN and autoencoder models are required for score(X).")
        X = np.asarray(X)
        p = np.asarray(self.dnn.predict(X, verbose=0)).reshape(-1)
        reconstructed = self.ae.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructed), axis=1)
        return self.score_components(p, errors)

    def predict(self, probabilities, errors):
        if self.threshold_ is None:
            raise ValueError("Fit the hybrid scorer before predicting.")
        return (self.score_components(probabilities, errors) >= self.threshold_).astype(int)

    def save(self, path=None):
        path = Path(path) if path else MODELS_DIR / "hybrid_calibration.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.sorted_ae is None or self.threshold_ is None:
            raise ValueError("Fit the hybrid scorer before saving it.")
        np.savez(
            path,
            sorted_ae=self.sorted_ae,
            weight=self.weight,
            threshold=self.threshold_,
            target_fpr=self.target_fpr,
        )
        return path

    @classmethod
    def load(cls, dnn=None, ae=None, path=None):
        path = Path(path) if path else MODELS_DIR / "hybrid_calibration.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Hybrid calibration not found: {path}. Run src.evaluate first."
            )
        with np.load(path) as saved:
            scorer = cls(
                dnn=dnn,
                ae=ae,
                weight=float(saved["weight"]),
                target_fpr=float(saved["target_fpr"]),
            )
            scorer.sorted_ae = saved["sorted_ae"].copy()
            scorer.threshold_ = float(saved["threshold"])
        return scorer
