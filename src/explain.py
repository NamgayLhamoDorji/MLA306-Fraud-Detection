"""
explain.py  (Member 4)
Run AFTER evaluate.py:   python3 -m src.explain
      python3 -m src.explain --n-explain 200 --n-background 100

Produces
  graphs/shap_bar.png, graphs/shap_beeswarm.png        global importance of the HYBRID score
  graphs/shap_waterfall_case*.png                      why individual frauds were flagged
  graphs/ae_feature_error.png                          which features the autoencoder finds "unusual" in fraud
  results/shap_top_features.csv, results/ae_feature_error.csv
SHAP is computed on the full hybrid score (DNN + autoencoder), model-agnostic (Permutation explainer),
so it works with any Keras version.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.hybrid import (GRAPHS_DIR, MONTH, RESULTS_DIR, HybridScorer, load_artifacts,
                        load_processed, predict_components)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-months", type=int, nargs="+", default=None)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--n-explain", type=int, default=200, help="rows per class (fraud / legit) to explain")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True); GRAPHS_DIR.mkdir(exist_ok=True)

    Xdf, ydf = load_processed()
    m = Xdf[MONTH].round().astype(int).to_numpy()
    test_months = args.test_months or [int(m.max())]
    mask = np.isin(m, test_months)
    df_test = Xdf[mask].reset_index(drop=True)
    y = ydf[mask].to_numpy().astype(int)

    art = load_artifacts()
    hyb = HybridScorer.load(art["dnn"], art["ae"])
    X, p, e = predict_components(art, df_test)
    names = list(df_test.columns)

    # ------------------------------------------------------------------
    # Autoencoder: which features reconstruct worst for fraud vs legit? (cheap, no SHAP needed)
    # ------------------------------------------------------------------
    recon = np.asarray(art["ae"].predict(X, batch_size=8192, verbose=0))
    se = (X - recon) ** 2
    ae_tbl = pd.DataFrame({"feature": names,
                           "mse_legit": se[y == 0].mean(0),
                           "mse_fraud": se[y == 1].mean(0)})
    ae_tbl["fraud_to_legit_ratio"] = ae_tbl["mse_fraud"] / (ae_tbl["mse_legit"] + 1e-12)
    ae_tbl = ae_tbl.sort_values("fraud_to_legit_ratio", ascending=False)
    ae_tbl.to_csv(RESULTS_DIR / "ae_feature_error.csv", index=False)
    top = ae_tbl.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["fraud_to_legit_ratio"])
    ax.set(xlabel="Reconstruction error ratio (fraud / legit)", title="Autoencoder: features that look 'unusual' in fraud")
    fig.tight_layout(); fig.savefig(GRAPHS_DIR / "ae_feature_error.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------
    # SHAP on the hybrid score
    # ------------------------------------------------------------------
    fraud_idx, legit_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = np.concatenate([
        rng.choice(fraud_idx, min(args.n_explain, len(fraud_idx)), replace=False),
        rng.choice(legit_idx, min(args.n_explain, len(legit_idx)), replace=False)])
    X_explain, y_explain = X[pick], y[pick]
    X_bg = X[rng.choice(len(X), min(args.n_background, len(X)), replace=False)]

    f = lambda A: hyb.score(np.asarray(A, dtype=np.float32))
    explainer = shap.Explainer(f, shap.maskers.Independent(X_bg), algorithm="permutation")
    print(f"[shap] explaining {len(X_explain)} rows x {X.shape[1]} features - this can take several minutes ...")
    sv = explainer(X_explain, max_evals=2 * X.shape[1] + 1)
    sv = shap.Explanation(values=sv.values, base_values=sv.base_values, data=X_explain, feature_names=names)

    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv.values).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(RESULTS_DIR / "shap_top_features.csv", index=False)
    print(imp.head(15).to_string(index=False))

    plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(GRAPHS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(GRAPHS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    # Local explanations: the 3 highest-scoring true frauds
    scores = f(X_explain)
    fraud_rows = np.where(y_explain == 1)[0]
    for k, i in enumerate(fraud_rows[np.argsort(-scores[fraud_rows])][:3], 1):
        plt.figure(); shap.plots.waterfall(sv[i], max_display=10, show=False)
        plt.tight_layout(); plt.savefig(GRAPHS_DIR / f"shap_waterfall_case{k}.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"[done] SHAP plots -> {GRAPHS_DIR}/   tables -> {RESULTS_DIR}/")
    print("Note: the explained sample is balanced (fraud/legit) so importance reflects what separates "
          "fraud, not the natural ~1% base rate.")


if __name__ == "__main__":
    main()