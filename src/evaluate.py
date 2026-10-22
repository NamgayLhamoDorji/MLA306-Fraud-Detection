"""Final evaluation of DNN, autoencoder and hybrid scorer (MLA306 Member 4).

Run from the project root:
    python -m src.evaluate                  # score everything, then evaluate
    python -m src.evaluate --reuse-scores   # reuse cached DNN/AE scores (fast)

What it does
    1. Scores every row with the DNN and the autoencoder.
    2. Calibrates the hybrid blend and all alert thresholds on the VALIDATION
       month only (month 6).
    3. Reports results on the untouched TEST month (month 7).
    4. Reports per-month metrics (months 0-7) to show stability over time.
       Months 0-5 were used to train the DNN/AE, so their numbers are
       optimistic and are labelled as such.

Outputs (results/): final_unseen_month_results.csv, monthly_metrics.csv,
hybrid_weight_sweep.csv, evaluation_summary.json, and models/hybrid_calibration.npz.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.hybrid import (
    MONTH,
    RESULTS_DIR,
    TARGET_FPR,
    HybridScorer,
    load_artifacts,
    load_processed,
    predict_components,
    threshold_at_fpr,
    tpr_at_fpr,
)

TRAIN_MONTHS = list(range(0, 6))
VAL_MONTHS = [6]
TEST_MONTHS = [7]  # make sure this matches preprocess.py
SCORE_CACHE = RESULTS_DIR / "score_cache.npz"


def period_of(month):
    if month in TRAIN_MONTHS:
        return "train (seen by models)"
    if month in VAL_MONTHS:
        return "validation"
    if month in TEST_MONTHS:
        return "test (unseen)"
    return "other"


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
    y = ydf.iloc[:, 0].to_numpy().astype(int).reshape(-1)  # flat 1-D labels
    months = Xdf[MONTH].to_numpy().astype(int)
    val = np.isin(months, VAL_MONTHS)
    test = np.isin(months, TEST_MONTHS)
    print(f"[split] val months={VAL_MONTHS}  test months={TEST_MONTHS}  "
          "(make sure this matches preprocess.py!)")
    if not val.any() or not test.any():
        raise ValueError("Validation or test month has no rows. Check the month split.")

    art = load_artifacts()
    if art["ae_name"].startswith("TESTING-ONLY"):
        print("\n*** WARNING: testing-only stand-in autoencoder in use. "
              "Do NOT report these results as final. ***\n")

    # ---- score every row with the DNN and the autoencoder ----------------
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
    del Xdf

    # ---- calibrate on validation only ------------------------------------
    scorer = HybridScorer(
        dnn=art["dnn"], ae=art["ae"], target_fpr=TARGET_FPR,
        ae_input=art["ae_input"],
    )
    scorer.fit_components(p[val], e[val], y[val])
    scorer.save()
    h = scorer.score_components(p, e)
    print(f"[hybrid] AE weight={scorer.weight:.2f}  DNN weight={1 - scorer.weight:.2f}  "
          f"threshold={scorer.threshold_:.4f}  (chosen on validation month)")

    scorer.sweep_.to_csv(RESULTS_DIR / "hybrid_weight_sweep.csv", index=False)

    thresholds = {
        "DNN only": threshold_at_fpr(y[val], p[val], TARGET_FPR),
        "Autoencoder only": threshold_at_fpr(y[val], e[val], TARGET_FPR),
        "Hybrid": scorer.threshold_,
    }
    scores = {"DNN only": p, "Autoencoder only": e, "Hybrid": h}

    # ---- final unseen-month results --------------------------------------
    rows = []
    for name in scores:
        rows.append({"model": name, "threshold": thresholds[name],
                     **metrics_at(y[test], scores[name][test], thresholds[name])})
    final_df = pd.DataFrame(rows)
    final_df.to_csv(RESULTS_DIR / "final_unseen_month_results.csv", index=False)

    print(f"\n=== FINAL UNSEEN-MONTH RESULTS (thresholds set at "
          f"{TARGET_FPR:.0%} FPR on validation) ===")
    show(final_df, ["model", "pr_auc", "roc_auc", "precision", "recall", "f1",
                    "fpr", "recall_at_1pct_fpr", "recall_at_5pct_fpr"])

    # ---- per-month stability ---------------------------------------------
    monthly = []
    for mo in sorted(np.unique(months)):
        mk = months == mo
        for name in scores:
            monthly.append({"month": int(mo), "period": period_of(int(mo)),
                            "model": name,
                            **metrics_at(y[mk], scores[name][mk], thresholds[name])})
    monthly_df = pd.DataFrame(monthly)
    monthly_df.to_csv(RESULTS_DIR / "monthly_metrics.csv", index=False)

    print("\n=== PER-MONTH METRICS (months 0-5 were seen in training: optimistic) ===")
    show(monthly_df, ["month", "period", "model", "fraud_cases", "pr_auc",
                      "roc_auc", "recall", "fpr", "recall_at_1pct_fpr"])

    # ---- summary ----------------------------------------------------------
    summary = {
        "autoencoder": art["ae_name"],
        "ae_features": art["ae_features"],
        "validation_months": VAL_MONTHS,
        "test_months": TEST_MONTHS,
        "target_fpr": TARGET_FPR,
        "hybrid_ae_weight": scorer.weight,
        "hybrid_dnn_weight": 1.0 - scorer.weight,
        "thresholds": thresholds,
        "test_rows": int(test.sum()),
        "test_fraud_cases": int(y[test].sum()),
    }
    with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved: results/final_unseen_month_results.csv, monthly_metrics.csv, "
          "hybrid_weight_sweep.csv, evaluation_summary.json")
    print("Saved: models/hybrid_calibration.npz")
    print("EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()