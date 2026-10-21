"""
evaluate.py  (Member 4)
Run:  python3 -m src.evaluate            (from the repo root)
      python3 -m src.evaluate --val-months 6 --test-months 7

Produces
  results/test_metrics.csv       final unseen-month results (DNN vs AE vs Hybrid)
  results/per_month_metrics.csv  stability of the hybrid across every month
  results/timecv_folds.csv       walk-forward (time-based) CV of the hybrid layer
  results/weight_sweep.csv       blend-weight search on validation
  results/fairness_detail.csv / fairness_summary.csv
  results/results_tables.md      everything as markdown tables for the report/slides
  graphs/eval_*.png              PR curve, ROC curve, score distribution, weight sweep
  models/hybrid_calibration.npz  saved hybrid weight + threshold (used by explain.py and the app)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.hybrid import (GRAPHS_DIR, MONTH, RESULTS_DIR, TARGET_FPR, HybridScorer,
                        load_artifacts, load_processed, predict_components, threshold_at_fpr,
                        tpr_at_fpr)

# Protected attributes for the fairness check (Bank Account Fraud dataset columns).
# Missing columns are skipped automatically.
FAIRNESS_ATTRS = {
    "customer_age": lambda s: np.where(s >= 50, "age>=50", "age<50"),
    "income": lambda s: np.where(s >= 0.5, "income>=0.5", "income<0.5"),
    "employment_status": lambda s: s.astype(str).to_numpy(),
}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    has_both = len(np.unique(y)) == 2
    return {
        "n": len(y), "fraud_cases": int(y.sum()),
        "pr_auc": average_precision_score(y, s) if has_both else float("nan"),
        "roc_auc": roc_auc_score(y, s) if has_both else float("nan"),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "recall_at_5pct_fpr": tpr_at_fpr(y, s, 0.05),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Fairness
# ----------------------------------------------------------------------------
def add_employment_column(X: pd.DataFrame) -> pd.DataFrame:
    """Processed data has one-hot employment_status_* columns (first category dropped = 'CA')."""
    X = X.copy()
    cols = [c for c in X.columns if c.startswith("employment_status_")]
    if cols:
        arr = X[cols].to_numpy()
        names = np.array([c.replace("employment_status_", "") for c in cols])
        X["employment_status"] = np.where(arr.max(1) > 0, names[arr.argmax(1)], "CA (baseline)")
    return X


def fairness_report(df_raw: pd.DataFrame, y: np.ndarray, flag: np.ndarray, min_n: int = 50):
    rows = []
    for attr, fn in FAIRNESS_ATTRS.items():
        if attr not in df_raw.columns:
            continue
        g = np.asarray(fn(df_raw[attr]))
        for grp in np.unique(g):
            mk = g == grp
            n = int(mk.sum()); pos = int((y[mk] == 1).sum()); neg = n - pos
            tp = int((flag & mk & (y == 1)).sum()); fp = int((flag & mk & (y == 0)).sum())
            rows.append({"attribute": attr, "group": grp, "n": n,
                         "fraud_rate": pos / n, "flag_rate": float(flag[mk].mean()),
                         "tpr": tp / pos if pos else np.nan,
                         "fpr": fp / neg if neg else np.nan,
                         "precision": tp / (tp + fp) if tp + fp else np.nan})
    detail = pd.DataFrame(rows)
    summ = []
    for attr, d in detail.groupby("attribute"):
        d = d[d["n"] >= min_n]
        if len(d) < 2:
            continue
        di = d["flag_rate"].min() / d["flag_rate"].max() if d["flag_rate"].max() > 0 else np.nan
        fpr_ratio = d["fpr"].max() / d["fpr"].min() if d["fpr"].min() > 0 else np.inf
        tpr_gap = d["tpr"].max() - d["tpr"].min()
        summ.append({"attribute": attr,
                     "disparate_impact_ratio": di,       # >= 0.80 passes the "four-fifths" rule
                     "tpr_gap_equal_opportunity": tpr_gap,  # closer to 0 is fairer
                     "fpr_ratio_max_over_min": fpr_ratio,   # closer to 1 is fairer
                     "flag": "REVIEW" if (di < 0.8 or fpr_ratio > 1.25 or tpr_gap > 0.10) else "ok"})
    return detail, pd.DataFrame(summ)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def make_plots(y, scores: dict, hybrid_sweep: pd.DataFrame):
    GRAPHS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in scores.items():
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (unseen test month)")
    ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS_DIR / "eval_pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.axvline(TARGET_FPR, color="grey", ls=":", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (unseen test month)")
    ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS_DIR / "eval_roc_curve.png", dpi=150); plt.close(fig)

    h = scores["Hybrid"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(h[y == 0], bins=50, alpha=0.6, density=True, label="legit")
    ax.hist(h[y == 1], bins=50, alpha=0.6, density=True, label="fraud")
    ax.set(xlabel="Hybrid score", ylabel="Density", title="Hybrid score distribution (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS_DIR / "eval_score_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hybrid_sweep["w_dnn"], hybrid_sweep["tpr_at_fpr"], marker="o")
    ax.set(xlabel="DNN weight w  (0 = autoencoder only, 1 = DNN only)",
           ylabel=f"Recall @ {TARGET_FPR:.0%} FPR", title="Blend weight search (validation)")
    fig.tight_layout(); fig.savefig(GRAPHS_DIR / "eval_weight_sweep.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-months", type=int, nargs="+", default=None,
                    help="months used to calibrate the hybrid (default: second-to-last month)")
    ap.add_argument("--test-months", type=int, nargs="+", default=None,
                    help="final unseen month(s) (default: last month)")
    ap.add_argument("--cv-folds", type=int, default=4, help="walk-forward folds for time-based CV")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True); GRAPHS_DIR.mkdir(exist_ok=True)
    Xdf, ydf = load_processed()
    m = Xdf[MONTH].round().astype(int).to_numpy()
    y = ydf.to_numpy().astype(int)
    months = sorted(np.unique(m).tolist())
    test_months = args.test_months or [months[-1]]
    val_months = args.val_months or [months[-2]]
    print(f"[split] val months={val_months}  test months={test_months}  (make sure this matches preprocess.py!)")

    art = load_artifacts()
    print("[score] running DNN + autoencoder over the whole dataset ...")
    X, p, e = predict_components(art, Xdf)
    val, test = np.isin(m, val_months), np.isin(m, test_months)
    if y[val].sum() == 0 or y[test].sum() == 0:
        raise RuntimeError("Validation or test period has no fraud cases - check --val-months / --test-months")

    # ---- fit hybrid on validation, pick thresholds for all three systems on validation ----
    hyb = HybridScorer(art["dnn"], art["ae"]).fit_components(p[val], e[val], y[val])
    hyb.save()
    print(f"[hybrid] best DNN weight w={hyb.w:.2f}  threshold={hyb.threshold:.4f}")
    hyb.sweep_.to_csv(RESULTS_DIR / "weight_sweep.csv", index=False)

    systems = {
        "DNN only": (p, threshold_at_fpr(y[val], p[val], TARGET_FPR)),
        "Autoencoder only": (e, threshold_at_fpr(y[val], e[val], TARGET_FPR)),
        "Hybrid": (hyb.combine(p, e), hyb.threshold),
    }

    # ---- 1. final unseen-month results ----
    rows = []
    for name, (s_all, thr) in systems.items():
        rows.append({"model": name, "threshold": thr, **metrics_at(y[test], s_all[test], thr)})
    test_tbl = pd.DataFrame(rows)
    test_tbl.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print("\n=== FINAL UNSEEN-MONTH RESULTS ===")
    print(test_tbl[["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "fpr",
                    "recall_at_1pct_fpr", "recall_at_5pct_fpr"]].round(4).to_string(index=False))

    # ---- 2. per-month stability of the hybrid ----
    h_all, h_thr = systems["Hybrid"]
    rows = []
    for mo in months:
        mk = m == mo
        period = "test" if mo in test_months else "val" if mo in val_months else "train (models saw it)"
        rows.append({"month": mo, "period": period, **metrics_at(y[mk], h_all[mk], h_thr)})
    per_month = pd.DataFrame(rows)
    per_month.to_csv(RESULTS_DIR / "per_month_metrics.csv", index=False)

    # ---- 3. time-based (walk-forward) CV of the hybrid layer ----
    # Expanding window: calibrate w + threshold on months < k, evaluate on month k.
    # NOTE: DNN/AE weights are frozen (trained once by Members 2/3), so folds that fall in the
    # models' training period are optimistic - the 'models_saw_month' column makes that explicit.
    cv_months = [mo for mo in months if mo not in test_months][-(args.cv_folds + 1):]
    folds = []
    for i in range(1, len(cv_months)):
        cal, ev = np.isin(m, cv_months[:i]), m == cv_months[i]
        if y[cal].sum() == 0 or y[ev].sum() == 0:
            continue
        sc = HybridScorer().fit_components(p[cal], e[cal], y[cal])
        folds.append({"calibrate_on": f"{cv_months[0]}-{cv_months[i-1]}", "evaluate_on": cv_months[i],
                      "w_dnn": sc.w, "models_saw_month": bool(cv_months[i] not in val_months),
                      **{k: v for k, v in metrics_at(y[ev], sc.combine(p[ev], e[ev]), sc.threshold).items()
                         if k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "recall_at_5pct_fpr")}})
    cv_tbl = pd.DataFrame(folds)
    cv_tbl.to_csv(RESULTS_DIR / "timecv_folds.csv", index=False)
    if len(cv_tbl):
        print("\n=== TIME-BASED CV (walk-forward) ===")
        print(cv_tbl.round(4).to_string(index=False))
        print("mean:", cv_tbl[["pr_auc", "roc_auc", "recall", "fpr"]].mean().round(4).to_dict())
        print("std :", cv_tbl[["pr_auc", "roc_auc", "recall", "fpr"]].std().round(4).to_dict())

    # ---- 4. fairness on the test month ----
    df_test = add_employment_column(Xdf[test].reset_index(drop=True))
    flag = (h_all[test] >= h_thr)
    f_detail, f_summary = fairness_report(df_test, y[test], flag)
    f_detail.to_csv(RESULTS_DIR / "fairness_detail.csv", index=False)
    f_summary.to_csv(RESULTS_DIR / "fairness_summary.csv", index=False)
    print("\n=== FAIRNESS (hybrid @ test month) ===")
    print(f_summary.round(3).to_string(index=False) if len(f_summary) else "no protected attributes found")

    # ---- 5. plots + markdown tables for the report ----
    make_plots(y[test], {k: v[0][test] for k, v in systems.items()}, hyb.sweep_)
    with open(RESULTS_DIR / "results_tables.md", "w") as f:
        f.write(f"# Evaluation results\n\nAutoencoder file: {art['ae_name']}\n\nValidation months: {val_months} | Test (unseen) months: {test_months} "
                f"| Operating point: {TARGET_FPR:.0%} FPR set on validation | Hybrid DNN weight w={hyb.w:.2f}\n\n")
        f.write("## Final unseen-month results\n\n")
        f.write(md_table(test_tbl[["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "fpr",
                                   "recall_at_1pct_fpr", "recall_at_5pct_fpr", "tp", "fp", "fn", "tn"]]) + "\n\n")
        f.write("## Per-month stability (hybrid)\n\n")
        f.write(md_table(per_month[["month", "period", "pr_auc", "roc_auc", "precision", "recall", "fpr"]]) + "\n\n")
        if len(cv_tbl):
            f.write("## Time-based walk-forward CV (hybrid layer)\n\n" + md_table(cv_tbl) + "\n\n")
        f.write("## Fairness summary\n\n" + (md_table(f_summary) if len(f_summary) else "n/a") + "\n\n")
        f.write("## Fairness detail by group\n\n" + (md_table(f_detail) if len(f_detail) else "n/a") + "\n")
    print(f"\n[done] tables -> {RESULTS_DIR}/  plots -> {GRAPHS_DIR}/")


if __name__ == "__main__":
    main()