"""Hybrid fraud scoring utilities for MLA306 Member 4.

This module supports evaluation and SHAP explanations. If the testing-only
stand-in autoencoder is loaded, its results must not be presented as final.

The autoencoder (Member 3) was trained on 8 selected features, using its own
StandardScaler (fitted on 46 columns, "month" dropped) and a clip of +/-5.
`load_artifacts` rebuilds that exact pipeline from models/ae_threshold.json and
models/ae_scaler.pkl and exposes it as `artifacts["ae_input"]`, which maps the
DNN-scaled feature matrix to the autoencoder's input.
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"

TARGET = "fraud_bool"
MONTH = "month"
TARGET_FPR = 0.01

# Keys that may hold the autoencoder's selected feature list in ae_threshold.json.
_AE_FEATURE_KEYS = (
    "selected_columns", "features", "feature_names", "feature_list",
    "top_features", "top_8", "ae_features", "selected_features", "feature_set",
)


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


def _ae_n_inputs(ae):
    """Number of input features the autoencoder expects."""
    shape = getattr(ae, "input_shape", None)
    if shape is not None:
        if isinstance(shape, list):
            shape = shape[0]
        return int(shape[-1])
    return int(ae.layers[0].kernel.shape[0])


def _find_feature_list(cfg):
    """Look for a list of feature names in a loaded JSON config."""
    if not isinstance(cfg, dict):
        return None
    for key in _AE_FEATURE_KEYS:
        value = cfg.get(key)
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return list(value)
    for value in cfg.values():  # one level of nesting
        if isinstance(value, dict):
            found = _find_feature_list(value)
            if found:
                return found
    return None


def _load_ae_config(ae, expected):
    """Read which features, scaler and clip the autoencoder was trained with."""
    n_in = _ae_n_inputs(ae)

    cfg = {}
    cfg_path = MODELS_DIR / "ae_threshold.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}

    features = _find_feature_list(cfg)
    if features is None and n_in == len(expected):
        features = list(expected)  # AE used every feature

    if features is None:
        raise ValueError(
            f"Autoencoder expects {n_in} inputs but no feature list was found in "
            f"{cfg_path.name}. Paste its contents so the key can be mapped."
        )

    missing = [f for f in features if f not in expected]
    if missing:
        raise ValueError(f"AE features not in DNN feature set: {missing}")
    if len(features) != n_in:
        raise ValueError(
            f"AE expects {n_in} inputs but feature list has {len(features)}."
        )

    ae_scaler = None
    ae_scaler_path = MODELS_DIR / "ae_scaler.pkl"
    if ae_scaler_path.exists():
        ae_scaler = joblib.load(ae_scaler_path)

    # Columns the AE scaler was fitted on (46 columns, "month" dropped).
    scaler_cols = None
    if ae_scaler is not None and getattr(ae_scaler, "feature_names_in_", None) is not None:
        scaler_cols = list(ae_scaler.feature_names_in_)
    elif cfg.get("feature_columns"):
        scaler_cols = list(cfg["feature_columns"])

    if ae_scaler is not None:
        if scaler_cols is None:
            raise ValueError(
                "ae_scaler.pkl has no feature names and ae_threshold.json has no "
                "'feature_columns'; cannot map the AE scaler to columns."
            )
        bad = [f for f in features if f not in scaler_cols]
        if bad:
            raise ValueError(f"AE features not in AE scaler columns: {bad}")
        bad = [c for c in scaler_cols if c not in expected]
        if bad:
            raise ValueError(f"AE scaler columns not in DNN feature set: {bad}")

    return {
        "features": features,
        "ae_scaler": ae_scaler,
        "scaler_cols": scaler_cols,
        "clip": cfg.get("clip_value"),
    }


def make_ae_input(artifacts):
    """Return f(X_dnn_scaled) -> array in exactly the form the autoencoder expects.

    Mirrors Member 3's prepare_features: raw values -> AE scaler -> clip ->
    keep the selected columns (by name, in the saved order).
    """
    dnn_scaler = artifacts["scaler"]
    expected = feature_names(dnn_scaler)
    features = artifacts["ae_features"]
    ae_scaler = artifacts.get("ae_scaler")
    scaler_cols = artifacts.get("ae_scaler_cols")
    clip = artifacts.get("ae_clip")

    if ae_scaler is None:
        idx = [expected.index(f) for f in features]

        def ae_input(X):
            return np.asarray(X, dtype=np.float32)[:, idx]  # DNN-scaled values

        return ae_input

    sel = [scaler_cols.index(f) for f in features]

    def ae_input(X):
        raw = dnn_scaler.inverse_transform(np.asarray(X, dtype=float))  # unscaled
        frame = pd.DataFrame(raw, columns=expected)[scaler_cols]
        z = ae_scaler.transform(frame)
        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))
        return z[:, sel].astype("float32")

    return ae_input


def load_artifacts():
    """Load the trained DNN, feature scaler, and available autoencoder."""
    dnn_path = MODELS_DIR / "dnn_improved.keras"
    scaler_path = MODELS_DIR / "dnn_improved_scaler.pkl"
    ae_best_path = MODELS_DIR / "autoencoder_best.keras"  # saved by autoencoder.py
    ae_path = MODELS_DIR / "autoencoder.keras"
    standin_path = MODELS_DIR / "autoencoder_standin.keras"

    if not dnn_path.exists():
        raise FileNotFoundError(f"Missing DNN model: {dnn_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing DNN scaler: {scaler_path}")

    dnn = load_model(dnn_path)
    scaler = joblib.load(scaler_path)

    if ae_best_path.exists():
        ae = load_model(ae_best_path)
        ae_name = "Member 3 autoencoder"
    elif ae_path.exists():
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
            "No autoencoder found. Expected models/autoencoder_best.keras "
            "from Member 3."
        )

    expected = feature_names(scaler)
    ae_cfg = _load_ae_config(ae, expected)

    artifacts = {
        "dnn": dnn,
        "ae": ae,
        "prep": scaler,
        "scaler": scaler,
        "ae_name": ae_name,
        "ae_features": ae_cfg["features"],
        "ae_scaler": ae_cfg["ae_scaler"],
        "ae_scaler_cols": ae_cfg["scaler_cols"],
        "ae_clip": ae_cfg["clip"],
    }
    artifacts["ae_input"] = make_ae_input(artifacts)
    return artifacts


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
    A = artifacts["ae_input"](X)
    reconstructed = artifacts["ae"].predict(A, verbose=0)
    errors = np.mean(np.square(A - reconstructed), axis=1)
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

    def __init__(self, dnn=None, ae=None, weight=0.5, target_fpr=TARGET_FPR,
                 ae_input=None):
        self.dnn = dnn
        self.ae = ae
        self.ae_input = ae_input  # maps DNN-scaled X to the AE's input
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
        A = self.ae_input(X) if self.ae_input is not None else X
        reconstructed = self.ae.predict(A, verbose=0)
        errors = np.mean(np.square(A - reconstructed), axis=1)
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
    def load(cls, dnn=None, ae=None, path=None, ae_input=None):
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
                ae_input=ae_input,
            )
            scorer.sorted_ae = saved["sorted_ae"].copy()
            scorer.threshold_ = float(saved["threshold"])
            scorer.threshold = scorer.threshold_
        return scorer