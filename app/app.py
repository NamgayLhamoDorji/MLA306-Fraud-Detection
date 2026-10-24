"""Fraud detection demo app (MLA306 Member 5).

Run from the project root:
    streamlit run app/app.py

Uses the real trained models through src/hybrid.py:
    DNN (models/dnn_final.keras) + autoencoder (models/autoencoder_best.keras),
    blended by the hybrid scorer calibrated in src.evaluate (models/hybrid_calibration.npz).

Features
    - Single application: enter an application (or load a real one from the unseen test
      month) and get a score, a flag and a plain-language SHAP reason.
    - Batch: upload a CSV (processed format or raw Base.csv format), see quality metrics
      if it has fraud_bool, pick any flagged row to explain, download all scores.
    - Every single-application result and every flagged batch row is appended to
      logs/predictions.csv.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.hybrid import (
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET,
    HybridScorer,
    feature_names,
    load_artifacts,
    predict_components,
)

MAX_BATCH_ROWS = 50_000
N_EXAMPLE_ROWS = 20_000      # rows of the unseen test month searched for demo examples
N_BACKGROUND = 100           # reference rows for SHAP
LOG_PATH = ROOT / "logs" / "predictions.csv"
LOG_COLUMNS = ["timestamp", "mode", "source", "row", "dnn_score", "ae_percentile",
               "hybrid_score", "flagged", "top_reasons"]

ONE_HOT = {
    "payment_type_": "Payment type",
    "employment_status_": "Employment status",
    "housing_status_": "Housing status",
    "device_os_": "Device OS",
    "source_": "Application source",
}
FRIENDLY = {
    "income": "Applicant income (0-1)",
    "name_email_similarity": "Name / email similarity",
    "prev_address_months_count": "Months at previous address",
    "current_address_months_count": "Months at current address",
    "customer_age": "Customer age",
    "days_since_request": "Days since request",
    "intended_balcon_amount": "Intended balance transfer amount",
    "zip_count_4w": "Applications from the same ZIP (4 weeks)",
    "velocity_6h": "Application velocity (6 hours)",
    "velocity_24h": "Application velocity (24 hours)",
    "velocity_4w": "Application velocity (4 weeks)",
    "bank_branch_count_8w": "Applications at the same branch (8 weeks)",
    "date_of_birth_distinct_emails_4w": "Distinct emails for this date of birth (4 weeks)",
    "credit_risk_score": "Credit risk score",
    "email_is_free": "Free email provider",
    "phone_home_valid": "Home phone valid",
    "phone_mobile_valid": "Mobile phone valid",
    "bank_months_count": "Age of previous bank account (months)",
    "has_other_cards": "Has other cards",
    "proposed_credit_limit": "Proposed credit limit",
    "foreign_request": "Foreign request",
    "session_length_in_minutes": "Session length (minutes)",
    "keep_alive_session": "Keep-alive session",
    "device_distinct_emails_8w": "Distinct emails on this device (8 weeks)",
    "device_fraud_count": "Device fraud count",
}
BINARY = {"email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
          "foreign_request", "keep_alive_session"}
MISSING_FLAG = {"prev_address_months_count", "current_address_months_count",
                "bank_months_count", "session_length_in_minutes",
                "device_distinct_emails_8w"}
AGE_OPTIONS = list(range(10, 100, 10))
OS_LEVELS = ("macintosh", "other", "windows", "x11")      # "linux" is the baseline
HOUSING_LEVELS = ("BB", "BC", "BD", "BE", "BF", "BG")     # "BA" is the baseline


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
            "Run the pipeline first: `python preprocess.py`, `python tune_dnn_optuna.py`, "
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
    """Score rows with the DNN, the autoencoder and the hybrid blend.

    Returns (scores DataFrame, scaled feature matrix)."""
    X, p, e = predict_components(art, frame)
    ae_pct = hyb._normalize_ae(e)
    h = hyb.score_components(p, e)
    scored = pd.DataFrame({
        "dnn_score": p,
        "ae_error": e,
        "ae_percentile": ae_pct,
        "hybrid_score": h,
        "flagged": (h >= hyb.threshold_).astype(int),
    })
    return scored, X


def default_row(art):
    """Training-average values for every model input (one-hot columns start at 0)."""
    scaler = art["scaler"]
    names = feature_names(scaler)
    row = dict(zip(names, (float(m) for m in scaler.mean_)))
    for name in names:
        if name.startswith(tuple(ONE_HOT)):
            row[name] = 0.0
    return row, names


def check_batch_columns(art, df):
    """Return (frame, missing_columns) or raise ValueError if the CSV does not fit."""
    expected = feature_names(art["scaler"])
    data = df.drop(columns=[TARGET, "Unnamed: 0"], errors="ignore")
    if not set(expected).issubset(data.columns):
        data = pd.get_dummies(data, dtype=int)
    missing = [c for c in expected if c not in data.columns]
    if len(missing) > len(expected) // 2:
        raise ValueError(
            f"This file has only {len(expected) - len(missing)} of the {len(expected)} "
            "columns the model needs. Use a processed file (processed/X_*.csv) or the "
            "raw Base.csv format."
        )
    return data, missing


# ----------------------------------------------------------------------
# Explanations (SHAP) in plain language
# ----------------------------------------------------------------------
def describe(name, value):
    """Short plain-language description of one feature and its value."""
    for prefix, label in ONE_HOT.items():
        if name.startswith(prefix):
            level = name[len(prefix):]
            return f"{label} is {level}" if value >= 0.5 else f"{label} is not {level}"
    label = FRIENDLY.get(name, name.replace("_", " ").capitalize())
    if name in BINARY:
        return f"{label}: {'yes' if value >= 0.5 else 'no'}"
    if name in MISSING_FLAG and abs(value + 1) < 0.01:
        return f"{label}: not available"
    if abs(value - round(value)) < 1e-3 or abs(value) >= 100:
        return f"{label}: {value:,.0f}"
    return f"{label}: {value:.2f}"


def build_reasons(names, raw_values, shap_values):
    """Table of features sorted by how strongly they move the score."""
    df = pd.DataFrame({
        "feature": list(names),
        "description": [describe(n, float(v)) for n, v in zip(names, raw_values)],
        "shap": np.asarray(shap_values, dtype=float),
    })
    order = df["shap"].abs().sort_values(ascending=False).index
    return df.reindex(order).reset_index(drop=True)


def explain_application(art, hyb, background, x_scaled):
    """SHAP values of the hybrid score for one scaled application."""
    import shap  # imported here so the app still loads if shap is missing

    if background is None:
        raise RuntimeError("no reference data (processed/X_val.csv) found for SHAP")
    x = np.asarray(x_scaled, dtype=np.float32).reshape(1, -1)

    def f(z):
        return hyb.score(np.asarray(z, dtype=np.float32))

    explainer = shap.Explainer(f, shap.maskers.Independent(background),
                               algorithm="permutation")
    sv = explainer(x, max_evals=6 * x.shape[1])
    names = feature_names(art["scaler"])
    raw = art["scaler"].inverse_transform(x.astype(float))[0]
    return build_reasons(names, raw, np.asarray(sv.values)[0])


def top_reasons_text(reasons, k=3):
    up = reasons[reasons["shap"] > 0].head(k)
    return "; ".join(up["description"])


def show_reasons(reasons, flagged):
    st.subheader("Why this application was flagged" if flagged else "What drives this score")
    st.caption("Effect of each feature on the hybrid score compared with a typical application "
               "(SHAP values). Category codes in this dataset are anonymised, so they show "
               "which group matters, not what it means.")
    up = reasons[reasons["shap"] > 0].head(3)
    if len(up):
        st.markdown("**Main factors raising the risk score**")
        st.markdown("\n".join(f"- {r.description} (+{r.shap:.3f})" for r in up.itertuples()))
    else:
        st.info("No feature pushes this score above that of a typical application.")
    down = reasons[reasons["shap"] < 0].head(2)
    if len(down):
        st.markdown("**Main factors lowering it**")
        st.markdown("\n".join(f"- {r.description} ({r.shap:.3f})" for r in down.itertuples()))

    top = reasons.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(top["description"], top["shap"],
            color=["#d62728" if v > 0 else "#1f77b4" for v in top["shap"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect on hybrid score (red raises, blue lowers)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Prediction log
# ----------------------------------------------------------------------
def log_row(mode, source, row_id, scored_row, reasons_text=""):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "source": source,
        "row": row_id,
        "dnn_score": round(float(scored_row["dnn_score"]), 5),
        "ae_percentile": round(float(scored_row["ae_percentile"]), 5),
        "hybrid_score": round(float(scored_row["hybrid_score"]), 5),
        "flagged": int(scored_row["flagged"]),
        "top_reasons": reasons_text,
    }


def write_log(rows):
    """Append rows to logs/predictions.csv. Returns True on success."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(
            LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
        return True
    except OSError as err:
        st.warning(f"Could not write the prediction log ({err}). "
                   "If logs/predictions.csv is open in Excel, close it.")
        return False


def log_entry_count():
    if not LOG_PATH.exists():
        return 0
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


# ----------------------------------------------------------------------
# Demo examples: real applications from the unseen test month
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing explanations ...")
def get_background(_art):
    """Scaled reference rows for SHAP (validation month)."""
    for name in ("X_val.csv", "X_test.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            df = pd.read_csv(path, nrows=5000)
            df = df.sample(min(N_BACKGROUND, len(df)), random_state=42)
            X, _, _ = predict_components(_art, df)
            return X
    return None


@st.cache_resource(show_spinner="Preparing demo examples ...")
def load_examples(_art, _hyb):
    """Pick three real applications: a typical legitimate one, a fraud the model
    catches and a fraud it misses (from the first rows of the unseen test month)."""
    x_path, y_path = PROCESSED_DIR / "X_test.csv", PROCESSED_DIR / "y_test.csv"
    if not (x_path.exists() and y_path.exists()):
        return {}
    X = pd.read_csv(x_path, nrows=N_EXAMPLE_ROWS)
    y = pd.read_csv(y_path, nrows=N_EXAMPLE_ROWS).iloc[:, 0].to_numpy().astype(int)
    scored, _ = score_frame(_art, _hyb, X)
    h = scored["hybrid_score"].to_numpy()
    flag = scored["flagged"].to_numpy() == 1

    def pick(mask, quantile):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        order = idx[np.argsort(h[idx])]
        return int(order[min(int(quantile * len(order)), len(order) - 1)])

    picks = {
        "legit": ("Typical legitimate application", pick((y == 0) & ~flag, 0.5)),
        "caught": ("Fraud the model catches", pick((y == 1) & flag, 0.75)),
        "missed": ("Fraud the model misses", pick((y == 1) & ~flag, 0.5)),
    }
    out = {}
    for key, (label, i) in picks.items():
        if i is not None:
            out[key] = {
                "label": label,
                "row": {k: float(v) for k, v in X.iloc[i].items()},
                "is_fraud": bool(y[i]),
                "score": float(h[i]),
                "flagged": bool(flag[i]),
            }
    return out


def _clip_int(v, lo, hi):
    return int(np.clip(int(round(float(v))), lo, hi))


def _clip_float(v, lo, hi):
    return float(np.clip(float(v), lo, hi))


def set_widgets(row):
    """Copy the visible form fields from a full application row into session state."""
    ss = st.session_state
    ss["w_income"] = _clip_float(row.get("income", 0.5), 0.0, 1.0)
    ss["w_name_email"] = _clip_float(row.get("name_email_similarity", 0.5), 0.0, 1.0)
    ss["w_age"] = _clip_int(round(float(row.get("customer_age", 40)) / 10) * 10, 10, 90)
    ss["w_credit_risk"] = _clip_int(row.get("credit_risk_score", 100), -200, 500)
    ss["w_credit_limit"] = _clip_int(row.get("proposed_credit_limit", 500), 0, 5000)
    ss["w_cur_addr"] = _clip_int(row.get("current_address_months_count", 50), -1, 600)
    ss["w_prev_addr"] = _clip_int(row.get("prev_address_months_count", 50), -1, 600)
    ss["w_days_req"] = _clip_float(row.get("days_since_request", 1.0), 0.0, 100.0)
    ss["w_device_os"] = next(
        (lvl for lvl in OS_LEVELS if row.get(f"device_os_{lvl}", 0) >= 0.5), "linux")
    ss["w_housing"] = next(
        (lvl for lvl in HOUSING_LEVELS if row.get(f"housing_status_{lvl}", 0) >= 0.5), "BA")
    ss["w_keep_alive"] = bool(row.get("keep_alive_session", 0) >= 0.5)
    ss["w_phone_home"] = bool(row.get("phone_home_valid", 0) >= 0.5)
    ss["w_email_free"] = bool(row.get("email_is_free", 0) >= 0.5)
    ss["w_other_cards"] = bool(row.get("has_other_cards", 0) >= 0.5)
    ss["w_foreign"] = bool(row.get("foreign_request", 0) >= 0.5)


def apply_example(key, examples):
    ex = examples[key]
    st.session_state["base_row"] = dict(ex["row"])
    st.session_state["example_note"] = ex
    st.session_state.pop("single_result", None)
    set_widgets(ex["row"])


def reset_to_averages(art):
    row, _ = default_row(art)
    st.session_state["base_row"] = row
    st.session_state["example_note"] = None
    st.session_state.pop("single_result", None)
    set_widgets(row)


# ----------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------
def render_sidebar(art, hyb):
    with st.sidebar:
        st.header("Model info")
        st.write(f"**DNN:** {art['dnn_name']}")
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

        st.subheader("Prediction log")
        st.write(f"{log_entry_count()} entries in logs/predictions.csv")
        if LOG_PATH.exists():
            try:
                st.download_button("Download log", LOG_PATH.read_bytes(),
                                   file_name="predictions.csv", mime="text/csv")
            except OSError:
                pass


def render_single(art, hyb, examples, background):
    ss = st.session_state
    if "base_row" not in ss:
        ss["base_row"], _ = default_row(art)
        ss["example_note"] = None
        set_widgets(ss["base_row"])

    st.write("**Try a real application from the unseen test month**")
    cols = st.columns(len(examples) + 1)
    for col, key in zip(cols, examples):
        col.button(examples[key]["label"], key=f"btn_{key}",
                   on_click=apply_example, args=(key, examples))
    cols[-1].button("Reset to averages", key="btn_reset",
                    on_click=reset_to_averages, args=(art,))

    note = ss.get("example_note")
    if note:
        st.info(
            f"Loaded a real application from the unseen test month. In the dataset it was "
            f"**{'fraud' if note['is_fraud'] else 'legitimate'}**. All 46 model inputs come "
            "from that application; the form shows the main ones. Change them and score "
            "again to see how the result moves."
        )
    else:
        st.caption("Fields not shown in the form use training-set averages "
                   "(one-hot categories start at 0).")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Income (0-1)", min_value=0.0, max_value=1.0, step=0.1,
                            format="%.2f", key="w_income")
            st.number_input("Name / email similarity", min_value=0.0, max_value=1.0,
                            step=0.05, format="%.3f", key="w_name_email")
            st.selectbox("Customer age (decade)", AGE_OPTIONS, key="w_age")
            st.number_input("Credit risk score", min_value=-200, max_value=500, step=1,
                            key="w_credit_risk")
            st.number_input("Proposed credit limit", min_value=0, max_value=5000, step=10,
                            key="w_credit_limit")
        with c2:
            st.number_input("Months at current address (-1 = unknown)", min_value=-1,
                            max_value=600, step=1, key="w_cur_addr")
            st.number_input("Months at previous address (-1 = unknown)", min_value=-1,
                            max_value=600, step=1, key="w_prev_addr")
            st.number_input("Days since request", min_value=0.0, max_value=100.0,
                            step=0.5, format="%.3f", key="w_days_req")
            st.selectbox("Device OS", ["linux", *OS_LEVELS], key="w_device_os")
            st.selectbox("Housing status", ["BA", *HOUSING_LEVELS], key="w_housing")
        with c3:
            st.checkbox("Keep-alive session", key="w_keep_alive")
            st.checkbox("Home phone valid", key="w_phone_home")
            st.checkbox("Free email provider", key="w_email_free")
            st.checkbox("Has other cards", key="w_other_cards")
            st.checkbox("Foreign request", key="w_foreign")
        submitted = st.form_submit_button("Score application", type="primary")

    if submitted:
        row = dict(ss["base_row"])
        row.update({
            "income": ss["w_income"], "name_email_similarity": ss["w_name_email"],
            "customer_age": ss["w_age"], "credit_risk_score": ss["w_credit_risk"],
            "proposed_credit_limit": ss["w_credit_limit"],
            "current_address_months_count": ss["w_cur_addr"],
            "prev_address_months_count": ss["w_prev_addr"],
            "days_since_request": ss["w_days_req"],
            "keep_alive_session": int(ss["w_keep_alive"]),
            "phone_home_valid": int(ss["w_phone_home"]),
            "email_is_free": int(ss["w_email_free"]),
            "has_other_cards": int(ss["w_other_cards"]),
            "foreign_request": int(ss["w_foreign"]),
        })
        names = feature_names(art["scaler"])
        for col in (c for c in names if c.startswith("device_os_")):
            row[col] = 1.0 if col == f"device_os_{ss['w_device_os']}" else 0.0
        for col in (c for c in names if c.startswith("housing_status_")):
            row[col] = 1.0 if col == f"housing_status_{ss['w_housing']}" else 0.0

        try:
            frame = pd.DataFrame([{k: row[k] for k in names}])
            scored, Xs = score_frame(art, hyb, frame)
        except Exception as err:  # noqa: BLE001
            ss.pop("single_result", None)
            st.error(f"Scoring failed: `{type(err).__name__}: {err}`")
        else:
            out = scored.iloc[0]
            reasons, reason_error = None, None
            with st.spinner("Working out the reasons ..."):
                try:
                    reasons = explain_application(art, hyb, background, Xs[0])
                except Exception as err:  # noqa: BLE001
                    reason_error = f"{type(err).__name__}: {err}"
            ss["single_result"] = {"scored": out, "reasons": reasons,
                                   "reason_error": reason_error}
            write_log([log_row("single", "form", 0, out,
                               top_reasons_text(reasons) if reasons is not None else "")])

    result = ss.get("single_result")
    if result:
        out = result["scored"]
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
        if result["reasons"] is not None:
            show_reasons(result["reasons"], bool(out["flagged"]))
        elif result["reason_error"]:
            st.info(f"The reasons could not be computed ({result['reason_error']}). "
                    "The score above is still valid.")
        st.caption("Saved to the prediction log (logs/predictions.csv).")


def render_batch(art, hyb, background):
    ss = st.session_state
    st.write("Upload a CSV in the processed format (like `processed/X_test.csv`) or the raw "
             "`Base.csv` format. If it has a `fraud_bool` column, quality metrics are shown.")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is None:
        return

    key = (uploaded.name, uploaded.size)
    if ss.get("batch_key") != key:
        try:
            raw = pd.read_csv(uploaded, nrows=MAX_BATCH_ROWS)
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not read that file as CSV: `{err}`")
            return
        if raw.empty:
            st.error("The file has no rows.")
            return
        try:
            data, missing = check_batch_columns(art, raw)
        except ValueError as err:
            st.error(str(err))
            return
        with st.spinner(f"Scoring {len(raw):,} rows ..."):
            try:
                scored, Xs = score_frame(art, hyb, data)
            except Exception as err:  # noqa: BLE001
                st.error(f"Scoring failed: `{type(err).__name__}: {err}`")
                return
        ss["batch"] = {"raw": raw.reset_index(drop=True), "scored": scored, "X": Xs,
                       "missing": missing, "truncated": len(raw) == MAX_BATCH_ROWS}
        ss["batch_key"] = key
        ss.pop("batch_reasons", None)
        flagged_rows = [log_row("batch", uploaded.name, int(i), scored.iloc[i])
                        for i in np.where(scored["flagged"].to_numpy() == 1)[0]]
        if flagged_rows:
            write_log(flagged_rows)

    b = ss["batch"]
    raw, scored, Xs = b["raw"], b["scored"], b["X"]
    if b["truncated"]:
        st.warning(f"Only the first {MAX_BATCH_ROWS:,} rows are scored in this demo.")
    if b["missing"]:
        st.warning(f"{len(b['missing'])} model columns were not in the file and were set to 0: "
                   + ", ".join(b["missing"][:8]) + (" ..." if len(b["missing"]) > 8 else ""))

    n_flag = int(scored["flagged"].sum())
    a, c, d = st.columns(3)
    a.metric("Applications scored", f"{len(scored):,}")
    c.metric("Flagged for review", f"{n_flag:,}")
    d.metric("Flag rate", f"{n_flag / len(scored):.2%}")

    if TARGET in raw.columns:
        y = raw[TARGET].to_numpy().astype(int)
        if y.min() != y.max():
            tp = int(((scored["flagged"] == 1) & (y == 1)).sum())
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("PR-AUC", f"{average_precision_score(y, scored['hybrid_score']):.3f}")
            q2.metric("ROC-AUC", f"{roc_auc_score(y, scored['hybrid_score']):.3f}")
            q3.metric("Fraud caught", f"{tp / max(int(y.sum()), 1):.1%}")
            q4.metric("Flag precision", f"{tp / max(n_flag, 1):.1%}")
        else:
            st.info("The file has only one class in `fraud_bool`, so quality metrics are skipped.")

    result = pd.concat([raw, scored], axis=1)
    st.dataframe(result.sort_values("hybrid_score", ascending=False).head(200))
    st.caption("Showing the 200 highest-scoring rows. The left-most number is the row number "
               "in your file. The download has every row.")

    flag_idx = np.where(scored["flagged"].to_numpy() == 1)[0]
    if len(flag_idx):
        order = flag_idx[np.argsort(-scored["hybrid_score"].to_numpy()[flag_idx])][:50]
        st.subheader("Explain a flagged application")
        pick = st.selectbox(
            "Flagged row (top 50 by score)", [int(i) for i in order],
            format_func=lambda i: f"Row {i} - hybrid score {scored['hybrid_score'].iloc[i]:.3f}",
        )
        if st.button("Explain this application"):
            with st.spinner("Working out the reasons ..."):
                try:
                    reasons = explain_application(art, hyb, background, Xs[pick])
                    ss["batch_reasons"] = {"row": pick, "reasons": reasons, "error": None}
                except Exception as err:  # noqa: BLE001
                    ss["batch_reasons"] = {"row": pick, "reasons": None,
                                           "error": f"{type(err).__name__}: {err}"}
        shown = ss.get("batch_reasons")
        if shown and shown["row"] == pick:
            if shown["reasons"] is not None:
                show_reasons(shown["reasons"], True)
            else:
                st.info(f"The reasons could not be computed ({shown['error']}).")
    else:
        st.info("No application in this file was flagged.")

    st.download_button("Download all scores (CSV)", result.to_csv(index=False).encode("utf-8"),
                       file_name="fraud_scores.csv", mime="text/csv")
    st.caption("Every flagged row of an uploaded file is also saved to the prediction log.")


def main():
    st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")
    st.title("🛡️ Bank Account Fraud Detection")
    st.caption("Hybrid model: deep neural network + autoencoder anomaly score. "
               "Course demo (MLA306), not a production system. Flagged applications go to a "
               "human reviewer; nothing is approved or rejected automatically.")

    art, hyb = load_or_stop()
    if art["ae_name"].startswith("TESTING-ONLY"):
        st.warning("A testing-only stand-in autoencoder is loaded. Results are NOT final.")
    render_sidebar(art, hyb)

    try:
        background = get_background(art)
    except Exception as err:  # noqa: BLE001
        background = None
        st.caption(f"Explanations are unavailable: {type(err).__name__}: {err}")
    try:
        examples = load_examples(art, hyb)
    except Exception as err:  # noqa: BLE001
        examples = {}
        st.caption(f"Demo examples are unavailable: {type(err).__name__}: {err}")

    tab_single, tab_batch = st.tabs(["Single application", "Batch (CSV)"])
    with tab_single:
        render_single(art, hyb, examples, background)
    with tab_batch:
        render_batch(art, hyb, background)


if __name__ == "__main__":
    main()