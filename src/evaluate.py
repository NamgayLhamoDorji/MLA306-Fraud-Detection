"""Final evaluation of all candidate models (MLA306 Member 4).

Run from the project root:
    python -m src.evaluate                  # score everything, then evaluate
    python -m src.evaluate --reuse-scores   # reuse cached DNN/AE scores (only if models are unchanged)

IMPORTANT: delete results/score_cache.npz after retraining any model.

What it does
    1. Scores every row with the final DNN and the autoencoder.
    2. Calibrates the hybrid blend and ALL alert thresholds on the VALIDATION
       split only (the stratified validation file).
    3. Reports results on the untouched TEST split (stratified test file) for:
         DNN baseline, DNN improved, DNN final, autoencoder, hybrid blend,
         hybrid OR rule (flag if either model passes its own threshold).
       Same split, same metrics, same 1% false-positive-rate rule for all.
    4. Adds 95% bootstrap intervals for PR-AUC and recall on the test split.
    5. Fairness table on the test split by age, employment status and income band.
    6. Test-split metrics per month (stability check; the split is random, not by month).

Outputs (results/): final_unseen_month_results.csv, fairness_month7.csv,
monthly_metrics.csv, hybrid_weight_sweep.csv, evaluation_summary.json,
and models/hybrid_calibration.npz.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras.models import load_model  # type: ignore

from src.hybrid import (
    PROCESSED_DIR,
    MODELS_DIR,
    MONTH,
    RESULTS_DIR,
    TARGET_FPR,
    HybridScorer,
    feature_names,
    load_artifacts,
    load_processed,
    predict_components,
    threshold_at_fpr,
    tpr_at_fpr,
)

# Stratified (random) split: train / val / test come from processed/X_*.csv.
# load_processed() stacks them in this order, so row ranges identify the split.
SPLIT_NAMES = ("train", "val", "test")
SCORE_CACHE = RESULTS_DIR / "score_cache.npz"
N_BOOT = 200

# Earlier DNN candidates, scored on the same split with their own scalers.
EXTRA_DNNS = {
    "DNN baseline": ("dnn_baseline.keras", "dnn_scaler.pkl"),
    "DNN improved": ("dnn_improved.keras", "dnn_improved_scaler.pkl"),
}


def split_labels():
    """Array of 0 (train), 1 (val), 2 (test) matching the rows of load_processed()."""
    parts = []
    for code, name in enumerate(SPLIT_NAMES):
        n = len(pd.read_csv(PROCESSED_DIR / f"y_{name}.csv"))
        parts.append(np.full(n, code, dtype=int))
    return np.concatenate(parts)


def metrics_at(y, scores, thr):
    """Threshold-free and thresholded metrics for one set of rows."""
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    pred = scores >= thr
    pos = y == 1
    neg = ~pos

    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) and not np.isnan(recall) else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    both = pos.any() and neg.any()
    return {
        "rows": int(len(y)),
        "fraud_cases": int(pos.sum()),
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "recall_at_1pct_fpr": tpr_at_fpr(y, scores, 0.01) if both else float("nan"),
        "recall_at_5pct_fpr": tpr_at_fpr(y, scores, 0.05) if both else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(y, scores, thr, n_boot=N_BOOT, seed=42):
    """95% bootstrap intervals for PR-AUC and recall at the fixed threshold."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(y)
    prs, recs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb, sb = y[i], scores[i]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        prs.append(average_precision_score(yb, sb))
        recs.append((sb[yb == 1] >= thr).mean())
    pr_lo, pr_hi = np.percentile(prs, [2.5, 97.5])
    rc_lo, rc_hi = np.percentile(recs, [2.5, 97.5])
    return {"pr_auc_ci_low": float(pr_lo), "pr_auc_ci_high": float(pr_hi),
            "recall_ci_low": float(rc_lo), "recall_ci_high": float(rc_hi)}


def score_saved_dnn(model_file, scaler_file, X):
    """Score rows with an earlier DNN using its own saved scaler."""
    mp, sp = MODELS_DIR / model_file, MODELS_DIR / scaler_file
    if not (mp.exists() and sp.exists()):
        return None
    model = load_model(mp)
    scaler = joblib.load(sp)
    cols = feature_names(scaler)
    Z = scaler.transform(X.reindex(columns=cols, fill_value=0)).astype("float32")
    return np.asarray(model.predict(Z, batch_size=4096, verbose=0)).reshape(-1)


def fairness_table(X, y, s, thr, model):
    """Alert rate, recall and false-positive rate per group (test split)."""
    emp_cols = [c for c in X.columns if c.startswith("employment_status_")]
    if emp_cols:
        emp = X[emp_cols].idxmax(axis=1).where(X[emp_cols].sum(axis=1) > 0,
                                               "employment_status_CA (baseline)")
        emp = emp.to_numpy()
    else:
        emp = np.array(["unknown"] * len(X))
    groups = {
        "age": np.where(X["customer_age"].to_numpy() >= 50, "50+", "under 50"),
        "employment": emp,
        "income": pd.cut(X["income"], [0, 0.3, 0.6, 1.0],
                         labels=["<=0.3", "0.3-0.6", ">0.6"]).astype(str).to_numpy(),
    }
    flag = np.asarray(s) >= thr
    y = np.asarray(y)
    rows = []
    for attr, g in groups.items():
        for val in np.unique(g):
            m = g == val
            pos, neg = m & (y == 1), m & (y == 0)
            rows.append({
                "model": model, "attribute": attr, "group": val,
                "rows": int(m.sum()), "fraud": int(pos.sum()),
                "alert_rate": float(flag[m].mean()),
                "recall": float(flag[pos].mean()) if pos.any() else float("nan"),
                "fpr": float(flag[neg].mean()) if neg.any() else float("nan"),
            })
    return pd.DataFrame(rows)


def show(df, columns):
    print(df[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-scores", action="store_true",
        help="reuse cached DNN/AE scores from results/score_cache.npz",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    Xdf, ydf = load_processed()
    y = ydf.iloc[:, 0].to_numpy().astype(int).reshape(-1)
    months = Xdf[MONTH].to_numpy().astype(int)
    split = split_labels()
    if len(split) != len(y):
        raise ValueError("Split sizes do not match the data. Re-run preprocessing.")
    val = split == 1
    test = split == 2
    print(f"[split] stratified split: train={int((split == 0).sum()):,}  "
          f"val={int(val.sum()):,}  test={int(test.sum()):,}  "
          f"fraud rate val={y[val].mean():.3%} test={y[test].mean():.3%}")
    if not val.any() or not test.any():
        raise ValueError("Validation or test split is empty. Re-run preprocessing.")

    art = load_artifacts()
    print(f"[models] DNN: {art['dnn_name']}  |  AE: {art['ae_name']}")
    if art["ae_name"].startswith("TESTING-ONLY"):
        print("\n*** WARNING: testing-only stand-in autoencoder in use. "
              "Do NOT report these results as final. ***\n")

    # ---- score every row with the final DNN and the autoencoder ---------
    if args.reuse_scores and SCORE_CACHE.exists():
        with np.load(SCORE_CACHE) as saved:
            p, e = saved["p"], saved["e"]
        if len(p) != len(y):
            raise ValueError("Cached scores do not match the data. Rerun without --reuse-scores.")
        print(f"[score] reused cached scores from {SCORE_CACHE.name}")
    else:
        print("[score] running DNN + autoencoder over the whole dataset ...")
        _, p, e = predict_components(art, Xdf)
        np.savez(SCORE_CACHE, p=p, e=e)

    # keep the validation/test rows (unscaled) for the extra models and fairness
    X_val = Xdf[val].reset_index(drop=True)
    X_test = Xdf[test].reset_index(drop=True)
    del Xdf

    # ---- calibrate on validation only -----------------------------------
    scorer = HybridScorer(
        dnn=art["dnn"], ae=art["ae"], target_fpr=TARGET_FPR,
        ae_input=art["ae_input"],
    )
    scorer.fit_components(p[val], e[val], y[val])
    scorer.save()
    h = scorer.score_components(p, e)
    print(f"[hybrid] AE weight={scorer.weight:.2f}  DNN weight={1 - scorer.weight:.2f}  "
          f"threshold={scorer.threshold_:.4f}  (chosen on the validation split)")
    scorer.sweep_.to_csv(RESULTS_DIR / "hybrid_weight_sweep.csv", index=False)

    # OR rule from the proposal: flag if EITHER model passes its own threshold.
    # Each model gets half the FPR budget so the combined FPR stays near 1%.
    thr_p = max(threshold_at_fpr(y[val], p[val], TARGET_FPR / 2), 1e-12)
    thr_e = max(threshold_at_fpr(y[val], e[val], TARGET_FPR / 2), 1e-12)
    or_score = np.maximum(p / thr_p, e / thr_e)   # >= 1.0  <=>  either model flags

    scores = {
        "DNN final": p,
        "Autoencoder only": e,
        "Hybrid blend": h,
        "Hybrid OR rule": or_score,
    }
    thresholds = {
        "DNN final": threshold_at_fpr(y[val], p[val], TARGET_FPR),
        "Autoencoder only": threshold_at_fpr(y[val], e[val], TARGET_FPR),
        "Hybrid blend": scorer.threshold_,
        "Hybrid OR rule": 1.0,
    }

    # ---- final unseen-test results -------------------------------------
    rows = []

    def add_row(name, s_test, thr):
        rows.append({"model": name, "threshold": thr,
                     **metrics_at(y[test], s_test, thr),
                     **bootstrap_ci(y[test], s_test, thr)})

    for name in ("DNN baseline", "DNN improved"):
        mf, sf = EXTRA_DNNS[name]
        sv = score_saved_dnn(mf, sf, X_val)
        if sv is None:
            print(f"[skip] {name}: {mf} or {sf} not found in models/")
            continue
        st = score_saved_dnn(mf, sf, X_test)
        add_row(name, st, threshold_at_fpr(y[val], sv, TARGET_FPR))
    for name in scores:
        add_row(name, scores[name][test], thresholds[name])

    final_df = pd.DataFrame(rows)
    final_df.to_csv(RESULTS_DIR / "final_unseen_month_results.csv", index=False)

    print(f"\n=== FINAL UNSEEN TEST-SPLIT RESULTS (thresholds set at "
          f"{TARGET_FPR:.0%} FPR on validation) ===")
    show(final_df, ["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc",
                    "precision", "recall", "recall_ci_low", "recall_ci_high", "f1",
                    "fpr", "recall_at_1pct_fpr"])

    # ---- fairness on the test split --------------------------------------------
    fair = pd.concat([
        fairness_table(X_test, y[test], scores[n][test], thresholds[n], n)
        for n in scores
    ], ignore_index=True)
    fair.to_csv(RESULTS_DIR / "fairness_month7.csv", index=False)
    print("\n=== FAIRNESS (test split, hybrid blend) ===")
    show(fair[fair["model"] == "Hybrid blend"],
         ["attribute", "group", "rows", "fraud", "alert_rate", "recall", "fpr"])

    # ---- stability across months, test rows only --------------------------
    monthly = []
    for mo in sorted(np.unique(months[test])):
        mk = test & (months == mo)
        for name in scores:
            monthly.append({"month": int(mo), "period": "test (unseen)",
                            "model": name,
                            **metrics_at(y[mk], scores[name][mk], thresholds[name])})
    monthly_df = pd.DataFrame(monthly)
    monthly_df.to_csv(RESULTS_DIR / "monthly_metrics.csv", index=False)

    print("\n=== TEST-SPLIT METRICS PER MONTH (rows are unseen; split is random) ===")
    show(monthly_df, ["month", "period", "model", "fraud_cases", "pr_auc",
                      "roc_auc", "recall", "fpr", "recall_at_1pct_fpr"])

    # ---- summary ----------------------------------------------------------
    summary = {
        "dnn": art["dnn_name"],
        "autoencoder": art["ae_name"],
        "ae_features": art["ae_features"],
        "split": "stratified random 70/15/15 (preprocess_balanced.py, seed 42)",
        "val_rows": int(val.sum()),
        "val_fraud_cases": int(y[val].sum()),
        "target_fpr": TARGET_FPR,
        "hybrid_ae_weight": scorer.weight,
        "hybrid_dnn_weight": 1.0 - scorer.weight,
        "or_rule_thresholds": {"dnn": thr_p, "autoencoder": thr_e, "each_fpr": TARGET_FPR / 2},
        "thresholds": thresholds,
        "bootstrap_resamples": N_BOOT,
        "test_rows": int(test.sum()),
        "test_fraud_cases": int(y[test].sum()),
    }
    with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved: results/final_unseen_month_results.csv, fairness_month7.csv, "
          "monthly_metrics.csv, hybrid_weight_sweep.csv, evaluation_summary.json")
    print("Saved: models/hybrid_calibration.npz")
    print("EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()