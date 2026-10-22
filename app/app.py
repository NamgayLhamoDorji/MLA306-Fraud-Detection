"""Fraud detection demo app (MLA306 Member 5).

Run from the project root:
    streamlit run app/app.py

Uses the real trained models through src/hybrid.py:
    DNN (models/dnn_improved.keras) + autoencoder (models/autoencoder_best.keras),
    blended by the hybrid scorer calibrated in src.evaluate (models/hybrid_calibration.npz).

Two modes:
    - Single application: fill in the main fields; unlisted fields use training averages.
    - Batch: upload a CSV (processed format or raw Base.csv format) and download the scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.hybrid import (
    RESULTS_DIR,
    TARGET,
    HybridScorer,
    feature_names,
    load_artifacts,
    predict_components,
)

MAX_BATCH_ROWS = 50_000
ONE_HOT_PREFIXES = ("payment_type_", "employment_status_", "housing_status_",
                    "device_os_", "source_")

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------
# Loading (cached) with clear error states
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models ...")
def load_everything():
    art = load_artifacts()
    hyb = HybridScorer.load(art["dnn"], art["ae"], ae_input=art["ae_input"])
    return art, hyb


def load_or_stop():
    try:
        return load_everything()
    except FileNotFoundError as err:
        st.error(
            f"A required model file is missing:\n\n`{err}`\n\n"
            "Run the pipeline first: `python preprocess.py`, `python dnn_improved.py`, "
            "`python autoencoder.py`, then `python -m src.evaluate`."
        )
    except ValueError as err:
        st.error(f"The saved models do not match each other:\n\n`{err}`")
    except Exception as err:  # noqa: BLE001 - last-resort message for the demo
        st.error(f"Could not load the models: `{type(err).__name__}: {err}`")
    st.stop()


# ----------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------
def score_frame(art, hyb, frame):
    """Score rows with the DNN, the autoencoder and the hybrid blend."""
    _, p, e = predict_components(art, frame)
    ae_pct = hyb._normalize_ae(e)
    h = hyb.score_components(p, e)
    return pd.DataFrame({
        "dnn_score": p,
        "ae_error": e,
        "ae_percentile": ae_pct,
        "hybrid_score": h,
        "flagged": (h >= hyb.threshold_).astype(int),
    })


def default_row(art):
    """Training-average values for every model input (one-hot columns start at 0)."""
    scaler = art["scaler"]
    names = feature_names(scaler)
    row = dict(zip(names, (float(m) for m in scaler.mean_)))
    for name in names:
        if name.startswith(ONE_HOT_PREFIXES):
            row[name] = 0.0
    return row, names


def check_batch_columns(art, df):
    """Return (frame, missing_columns) or raise ValueError if the CSV does not fit."""
    expected = feature_names(art["scaler"])
    data = df.drop(columns=[TARGET, "Unnamed: 0"], errors="ignore")
    if not set(expected).issubset(data.columns):
        data = pd.get_dummies(data, drop_first=True, dtype=int)
    missing = [c for c in expected if c not in data.columns]
    if len(missing) > len(expected) // 2:
        raise ValueError(
            f"This file has only {len(expected) - len(missing)} of the {len(expected)} "
            "columns the model needs. Use a processed file (processed/X_*.csv) or the "
            "raw Base.csv format."
        )
    return data, missing


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.title("🛡️ Bank Account Fraud Detection")
st.caption("Hybrid model: deep neural network + autoencoder anomaly score. "
           "Course demo (MLA306), not a production system.")

art, hyb = load_or_stop()

if art["ae_name"].startswith("TESTING-ONLY"):
    st.warning("A testing-only stand-in autoencoder is loaded. Results are NOT final.")

with st.sidebar:
    st.header("Model info")
    st.write(f"**Autoencoder:** {art['ae_name']}")
    st.write(f"**Blend:** {1 - hyb.weight:.0%} DNN + {hyb.weight:.0%} autoencoder")
    st.write(f"**Alert threshold:** {hyb.threshold_:.4f}")
    st.caption(
        f"The threshold was set on the validation month so that about "
        f"{hyb.target_fpr:.0%} of legitimate applications get flagged."
    )
    st.write("**Autoencoder features:**")
    st.caption(", ".join(art["ae_features"]))

    results_path = RESULTS_DIR / "final_unseen_month_results.csv"
    if results_path.exists():
        st.subheader("Unseen-month results")
        res = pd.read_csv(results_path)
        cols = [c for c in ("model", "pr_auc", "roc_auc", "recall_at_1pct_fpr") if c in res]
        st.dataframe(res[cols].round(3), hide_index=True)
    st.caption("The DNN score is not a calibrated probability: the model was trained "
               "with class weights, so only the ranking and the threshold matter.")

tab_single, tab_batch = st.tabs(["Single application", "Batch (CSV)"])

# ----------------------------------------------------------------------
# Single application
# ----------------------------------------------------------------------
with tab_single:
    base, names = default_row(art)

    def clipped(name, lo, hi, fallback):
        return float(np.clip(base.get(name, fallback), lo, hi))

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.slider("Income (scaled 0-1)", 0.0, 1.0,
                               clipped("income", 0, 1, 0.5), 0.05)
            name_email = st.slider("Name / email similarity", 0.0, 1.0,
                                   clipped("name_email_similarity", 0, 1, 0.5), 0.05)
            age = st.selectbox("Customer age (decade)", list(range(10, 100, 10)), index=3)
            credit_risk = st.number_input(
                "Credit risk score", -200, 400,
                int(clipped("credit_risk_score", -200, 400, 100)))
            credit_limit = st.slider("Proposed credit limit", 200, 2000,
                                     int(round(clipped("proposed_credit_limit", 200, 2000, 500), -2)), 100)
        with c2:
            cur_addr = st.number_input(
                "Months at current address (-1 = unknown)", -1, 450,
                int(clipped("current_address_months_count", -1, 450, 50)))
            prev_addr = st.number_input(
                "Months at previous address (-1 = unknown)", -1, 450,
                int(clipped("prev_address_months_count", -1, 450, 50)))
            days_req = st.number_input("Days since request", 0.0, 80.0,
                                       clipped("days_since_request", 0, 80, 1.0), 0.5)
            device_os = st.selectbox("Device OS",
                                     ["linux", "macintosh", "other", "windows", "x11"], index=3)
            housing = st.selectbox("Housing status",
                                   ["BA", "BB", "BC", "BD", "BE", "BF", "BG"])
        with c3:
            keep_alive = st.checkbox("Keep-alive session", bool(base.get("keep_alive_session", 0) > 0.5))
            phone_home = st.checkbox("Home phone valid", bool(base.get("phone_home_valid", 0) > 0.5))
            email_free = st.checkbox("Free email provider", bool(base.get("email_is_free", 0) > 0.5))
            other_cards = st.checkbox("Has other cards", bool(base.get("has_other_cards", 0) > 0.5))
            foreign = st.checkbox("Foreign request", bool(base.get("foreign_request", 0) > 0.5))
        submitted = st.form_submit_button("Score application", type="primary")

    st.caption("Fields not shown here use training-set averages (one-hot categories start at 0).")

    if submitted:
        row = dict(base)
        row.update({
            "income": income, "name_email_similarity": name_email,
            "customer_age": age, "credit_risk_score": credit_risk,
            "proposed_credit_limit": credit_limit,
            "current_address_months_count": cur_addr,
            "prev_address_months_count": prev_addr,
            "days_since_request": days_req,
            "keep_alive_session": int(keep_alive), "phone_home_valid": int(phone_home),
            "email_is_free": int(email_free), "has_other_cards": int(other_cards),
            "foreign_request": int(foreign),
        })
        for col in (c for c in names if c.startswith("device_os_")):
            row[col] = 1.0 if col == f"device_os_{device_os}" else 0.0
        for col in (c for c in names if c.startswith("housing_status_")):
            row[col] = 1.0 if col == f"housing_status_{housing}" else 0.0

        try:
            frame = pd.DataFrame([{k: row[k] for k in names}])
            out = score_frame(art, hyb, frame).iloc[0]
        except Exception as err:  # noqa: BLE001
            st.error(f"Scoring failed: `{type(err).__name__}: {err}`")
        else:
            if out["flagged"]:
                st.error("🚩 Flag for manual review")
            else:
                st.success("✅ Looks normal, no alert")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("DNN score", f"{out['dnn_score']:.3f}")
            m2.metric("Autoencoder anomaly", f"{out['ae_percentile']:.0%}",
                      help="Share of validation applications with a lower reconstruction error.")
            m3.metric("Hybrid score", f"{out['hybrid_score']:.3f}")
            m4.metric("Alert threshold", f"{hyb.threshold_:.3f}")

# ----------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV in the processed format (like `processed/X_test.csv`) or the raw "
             "`Base.csv` format. If it has a `fraud_bool` column, quality metrics are shown.")
    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, nrows=MAX_BATCH_ROWS)
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not read that file as CSV: `{err}`")
            st.stop()

        if raw.empty:
            st.error("The file has no rows.")
            st.stop()
        if len(raw) == MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS:,} rows are scored in this demo.")

        try:
            data, missing = check_batch_columns(art, raw)
        except ValueError as err:
            st.error(str(err))
            st.stop()
        if missing:
            st.warning(f"{len(missing)} model columns were not in the file and were set to 0: "
                       + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))

        with st.spinner(f"Scoring {len(raw):,} rows ..."):
            try:
                scored = score_frame(art, hyb, data)
            except Exception as err:  # noqa: BLE001
                st.error(f"Scoring failed: `{type(err).__name__}: {err}`")
                st.stop()

        result = pd.concat([raw.reset_index(drop=True), scored], axis=1)
        n_flag = int(scored["flagged"].sum())
        a, b, c = st.columns(3)
        a.metric("Applications scored", f"{len(scored):,}")
        b.metric("Flagged for review", f"{n_flag:,}")
        c.metric("Flag rate", f"{n_flag / len(scored):.2%}")

        if TARGET in raw.columns:
            y = raw[TARGET].to_numpy().astype(int)
            if y.min() != y.max():
                tp = int(((scored["flagged"] == 1) & (y == 1)).sum())
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("PR-AUC", f"{average_precision_score(y, scored['hybrid_score']):.3f}")
                d2.metric("ROC-AUC", f"{roc_auc_score(y, scored['hybrid_score']):.3f}")
                d3.metric("Fraud caught", f"{tp / max(int(y.sum()), 1):.1%}")
                d4.metric("Flag precision", f"{tp / max(n_flag, 1):.1%}")
            else:
                st.info("The file has only one class in `fraud_bool`, so quality metrics are skipped.")

        st.dataframe(result.sort_values("hybrid_score", ascending=False).head(200),
                     use_container_width=True)
        st.caption("Showing the 200 highest-scoring rows. The download has every row.")
        st.download_button("Download all scores (CSV)",
                           result.to_csv(index=False).encode("utf-8"),
                           file_name="fraud_scores.csv", mime="text/csv")