# Financial Fraud Detection Using a Hybrid Deep Neural Network and Autoencoder

**Module:** MLA306 Improving Deep Neural Networks
**Assessment:** CA4 Deep Neural Network Optimisation Project, Part 2 (DL Model and Presentation)
**Institution:** Gedu College of Business Studies, Royal University of Bhutan
**Programme:** Bachelor of Business Intelligence, Year 3, Semester 1
**Module tutor:** Mr. Tashi Tobgay Sherpa

---

## 1. Project Summary

Banks and mobile payment services such as BOB Mobile, MyPay and ePay process millions of transactions, so fraud cannot be checked by hand, and fixed rules miss new fraud patterns. This project builds a deep learning system that screens **bank account applications** and returns a fraud-risk score, a flag and a short reason.

The system combines two models:

- **DNN classifier (supervised):** learns known fraud patterns from labelled applications and outputs a fraud score from 0 to 1.
- **Autoencoder (unsupervised):** trained only on normal applications; an unusual application is rebuilt poorly, and the reconstruction error is used as an anomaly score.
- **Hybrid scorer:** blends the DNN score with the autoencoder's anomaly percentile, with the blend weight and the alert threshold chosen on the validation month. The "flag if either model is suspicious" rule from the proposal is also implemented and compared with the blend (see section 9).

Flagged applications are explained with SHAP and passed to a human reviewer. **The system never approves or rejects an application on its own.**

## 2. Objectives

1. Prepare the BAF dataset for supervised and unsupervised fraud detection.
2. Build and train a DNN classifier that separates normal from fraudulent applications.
3. Build and train an autoencoder that detects anomalies by reconstruction error.
4. Combine both into a hybrid scoring system and test whether it beats either model alone.
5. Handle the class imbalance (about 1.1% fraud) so the models do not just predict "not fraud".
6. Evaluate with proper metrics, not accuracy alone.
7. Explain each flagged application instead of only returning a number.

## 3. Features of the System

| Feature | Description |
| --- | --- |
| Fraud risk scoring | A score from 0 (normal) to 1 (high risk) for each application |
| Anomaly detection | Reconstruction error from the autoencoder, needing no fraud labels |
| Hybrid decision logic | Blend of both scores (and an "either model" rule for comparison) |
| Threshold alerting | Flagged applications go to human review; nothing is auto-approved or rejected |
| Explainability | SHAP shows which features drove the flag, in plain language |
| Demo application | Streamlit app: single application and CSV batch modes, with a prediction log |

## 4. Dataset

**Bank Account Fraud (BAF) Dataset Suite, Base variant** (`Base.csv`), released at NeurIPS 2022 by Feedzai Research and hosted on Kaggle.

- 1,000,000 account-opening applications over 8 months (`month` column)
- 31 predictor columns plus the label `fraud_bool` (32 columns in total)
- About 1.1% fraud, so the data is heavily imbalanced
- Generated with a conditional GAN and differential privacy applied to a real, anonymised dataset, so it contains no real personal data
- Protected attributes used for fairness checks: age, employment status and income
- Only the Base variant is used. Variants I to V (different engineered bias patterns) are out of scope for this part.

**The dataset is NOT stored in this repository** because of its size. Download it separately and save it as:

```
data/Base.csv
```

See `scripts/download_data.md` for the link. Git ignores everything in `data/` except `.gitkeep`.

**Citation:** Jesus, S., Pombal, J., Alves, D., Cruz, A. F., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation. *Advances in Neural Information Processing Systems, 35.* https://arxiv.org/abs/2211.13358

**Licence:** CC BY-NC-ND 4.0, used here for academic, non-commercial purposes only.

## 5. System Workflow

```
Base.csv
   |
Split by month (train 0-5, validation 6, test 7), then encode categories
   |
Drop "month" (a time label, not a customer feature), scale on training data only
   |
   +--------------------+
   |                    |
 DNN classifier       Autoencoder (8 selected features,
 (fraud score)        trained on normal applications only)
   |                    |
   +----------+---------+
              |
      Hybrid scoring (blend, threshold set at 1% false-positive rate on validation)
              |
        Above threshold?
        /              \
      No               Yes
 Treat as normal    Flag for human review + SHAP reason
```

## 6. Model Design

**DNN classifier (multi-layer perceptron).** Input: 46 features after encoding (the 47 encoded columns minus `month`). Each hidden block is Dense, BatchNormalization, activation and Dropout, with a sigmoid output. Loss: binary cross-entropy with class weights computed from the training labels.

| Version | Script | Hidden layers |
| --- | --- | --- |
| Baseline | `dnn.py` | 128, 64 (ReLU, no regularisation) |
| Improved | `dnn_improved.py` | 128, 64 with BatchNorm and Dropout |
| Final | `tune_dnn_optuna.py` | Depth (1 to 3), widths, dropout, activation, optimiser, learning rate and batch size chosen by Optuna. The chosen values are in `results/optuna_best_params.json`. |

**Autoencoder** (`autoencoder.py`). Dense encoder, a bottleneck, and a mirrored decoder with a linear output. Trained on non-fraud rows only, with its own scaler fitted on normal training rows, clipped at +/-5. It uses the 8 features with the strongest individual relationship to fraud in the training data (the feature-set comparison is in `results/ae_feature_set_comparison.csv`). The bottleneck size and other settings come from an Optuna study (`results/ae_optuna_best_params.json`).

**Hybrid** (`src/hybrid.py`). Score = (1 - w) x DNN score + w x autoencoder percentile, where the autoencoder percentile is the share of validation applications with a smaller reconstruction error. The weight `w` and the alert threshold are chosen on validation month 6 (`results/hybrid_weight_sweep.csv`). The threshold is set so that about 1% of legitimate validation applications are flagged.

## 7. Data Preparation

1. **Data understanding:** size, types, fraud ratio and per-month fraud rates (`scripts/check_data.py`, `scripts/check_months.py`).
2. **Duplicates and missing values:** exact duplicate rows are removed; any missing values are filled with the training median. In this dataset, `-1` marks "not available" for some columns; it is kept as a numeric value and the models learn from it.
3. **Temporal split before any fitting:** months 0 to 5 train, month 6 validation, month 7 test. The test month is never used for training, tuning or threshold choice.
4. **Encoding:** categorical columns are one-hot encoded and aligned to the training columns.
5. **Scaling:** `StandardScaler` fitted on training rows only.
6. **`month` is not a model input.** It is only used to split the data and to report results per month.
7. **Class imbalance:** class weights computed on the training set.

## 8. Optimisation Experiments

| Area | Script | Output |
| --- | --- | --- |
| Activation functions: ReLU, LeakyReLU, GELU | `compare_activations.py` | `results/activation_comparison.csv` |
| Optimisers: Adam, AdamW | `compare_optimizers.py` | `results/optimizer_comparison.csv` |
| DNN hyperparameters, depth and width (architecture search) | `tune_dnn_optuna.py` | `results/optuna_trials.csv`, `optuna_best_params.json` |
| Autoencoder feature set, bottleneck, Optuna | `autoencoder.py` | `results/ae_*.csv`, `ae_*.json` |
| Regularisation | BatchNorm, Dropout, early stopping on validation PR-AUC | in each script |
| Alert threshold | validation analysis | `results/dnn_threshold_comparison.csv` |
| Tracking | TensorBoard (`logs/`) | `tensorboard --logdir logs` |

Common settings: seed 42, Adam or AdamW, early stopping on validation PR-AUC (patience 3 to 4), class weights.

## 9. Evaluation

Accuracy alone is misleading here: a model that says "not fraud" every time would score about 99%. `src/evaluate.py` reports, on the untouched test month (month 7):

- PR-AUC, ROC-AUC, precision, recall, F1, false-positive rate and confusion counts
- Recall at 1% and 5% false-positive rate
- 95% bootstrap intervals for PR-AUC and recall
- The same table for: DNN baseline, DNN improved, DNN final, autoencoder alone, hybrid blend and hybrid "either model" rule. Every alert threshold is chosen on validation month 6.
- Fairness by age, employment status and income band (`results/fairness_month7.csv`)
- Per-month metrics (months 0 to 5 were seen in training, so those numbers are optimistic and labelled as such)

`src/explain.py` produces SHAP plots for the hybrid score and a per-feature reconstruction-error table for the autoencoder.

**Deviations from the proposal**

| Proposal | What was done | Why |
| --- | --- | --- |
| Label `is_fraud`, 30 features | The real label is `fraud_bool`; 31 predictors | Matches the released dataset |
| 5-fold stratified cross-validation | Time-based train / validation / test split | Random folds would let the model train on later months than it is tested on |
| SMOTE, under-sampling or class weights | Class weights | Cheaper on 1M rows and keeps the real data distribution |
| Hybrid flags if either model flags | Weighted blend, with the "either" rule evaluated alongside | Compared on the same split and metrics; results in `final_unseen_month_results.csv` |
| Repository on GitLab | GitLab mirror of this repository (link in section 17) | Required by the brief |
| Bias tested across variants I to V | Base variant only | Scope limit |

## 10. Repository Structure

```
MLA306-Fraud-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── preprocess.py            Split by month, encode, impute, save processed/
├── dnn.py                   Baseline DNN
├── dnn_improved.py          DNN with BatchNorm and Dropout
├── compare_activations.py   ReLU vs LeakyReLU vs GELU
├── compare_optimizers.py    Adam vs AdamW
├── tune_dnn_optuna.py       Optuna search; saves the final DNN
├── autoencoder.py           Autoencoder, feature set, Optuna, threshold
├── src/
│   ├── hybrid.py            Loading, scoring and the hybrid scorer
│   ├── evaluate.py          Metrics, comparison, fairness, per-month results
│   └── explain.py           SHAP and autoencoder error analysis
├── app/app.py               Streamlit demo
├── scripts/                 Helper scripts and download_data.md
├── notebooks/               EDA
├── data/                    Base.csv goes here (git-ignored)
├── processed/               Created by preprocess.py (git-ignored)
├── models/                  Saved models, scalers, calibration (git-ignored)
├── results/                 Metrics tables and JSON files
├── graphs/                  Plots
└── logs/                    TensorBoard logs and the app's prediction log (git-ignored)
```

## 11. Setup

**Requirements:** Python 3.10 or 3.11, Git, and about 8 GB of RAM (16 GB or Google Colab is more comfortable for full training).

```bash
git clone <repository-url>
cd MLA306-Fraud-Detection

python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# .venv\Scripts\activate           # Command Prompt / PowerShell

pip install -r requirements.txt
```

Download `Base.csv` using `scripts/download_data.md` and save it as `data/Base.csv`.

## 12. How to Run

Run every command from the repository root, because all paths are relative. `models/` and `processed/` are not in the repository, so a fresh clone must run the pipeline in this order:

```bash
python preprocess.py              # processed/ train, validation and test files
python dnn.py                     # baseline DNN
python dnn_improved.py            # improved DNN
python compare_activations.py     # activation experiment
python compare_optimizers.py      # optimiser experiment
python tune_dnn_optuna.py         # Optuna search; saves models/dnn_final.keras
python autoencoder.py             # autoencoder, scaler and threshold
python -m src.evaluate            # hybrid calibration, final metrics, fairness
python -m src.explain --n-explain 100   # SHAP plots
streamlit run app/app.py          # demo application
```

Quick test on a laptop: `DNN_TRIALS=3 AE_TRIALS=3 AE_MAX_TRAIN_ROWS=100000 python ...` for the two tuning scripts.

After retraining any model, delete `results/score_cache.npz` before running `src.evaluate`.

TensorBoard: `tensorboard --logdir logs`

## 13. Team and Responsibilities

Team: Pema Lhamo, Rekhita Pradhan, Namgay Lhamo Dorji, Ugyel Tandin Wangmo, Pem Dechen.

| Member | Name | Role | Main files |
| --- | --- | --- | --- |
| Member 1 | <name> | Data engineering: EDA, cleaning, preprocessing | `preprocess.py`, `notebooks/` |
| Member 2 | <name> | DNN classifier, experiments and Optuna study | `dnn*.py`, `compare_*.py`, `tune_dnn_optuna.py` |
| Member 3 | <name> | Autoencoder, error threshold and Optuna study | `autoencoder.py` |
| Member 4 | <name> | Hybrid scorer, evaluation, SHAP, fairness | `src/` |
| Member 5 | <name> | Demo app, README, GitLab mirror, presentation | `app/`, `README.md` |

Team lead: <name>.

## 14. Git Workflow

- Work on a branch and open a pull request into `main` where practical; keep commit messages short and clear.
- Do not commit data, models, `processed/`, logs or anything over 50 MB.
- Random seed is 42 wherever randomness is used.
- The final submitted version is tagged `v1.0-part2`.

## 15. Ethics, Fairness and Privacy

- **Bias:** BAF deliberately contains bias patterns. We report alert rate, recall and false-positive rate per age group, employment status and income band, instead of one overall score (`results/fairness_month7.csv`).
- **Limitation:** a model that flags one group far more often without a matching rise in real fraud is reported as a limitation of this study.
- **Privacy:** only the anonymised, privacy-protected BAF data is used; no real customer data is collected or stored.
- **Human oversight:** every flagged application is reviewed by a person.
- **Compliance:** a real deployment would need to follow Bhutan's data protection rules and the policies of the financial institution.

## 16. Known Limitations

- BAF is synthetic data, so results may not match real Bhutanese payment data.
- Fraud patterns change over time; the test is a single month (month 7) with a limited number of fraud cases, so small differences between models are within noise (see the bootstrap intervals).
- Validation month 6 is used for early stopping, Optuna, autoencoder feature selection, the hybrid weight and the threshold, so validation scores are slightly optimistic. Only month 7 is untouched.
- Only the Base variant was used; fairness on the other variants was not tested.
- Fairness is measured on three attributes and is not proven.
- SHAP shows what the model uses, not what causes fraud. The explained sample is balanced between fraud and legitimate, and one-hot columns are treated as independent.
- The DNN score is not a calibrated probability because class weights were used; only the ranking and the threshold matter.
- The demo form shows the main inputs only; the rest use training averages unless a real example is loaded.

## 17. Submission

- **GitLab repository (required by the brief):** <GitLab link here>
- **Project report (PDF):** includes the GitLab link.
- **Presentation and demonstration:** at the time arranged with the supervisor.

## 18. References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623-2631.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321-357.
- Jesus, S., et al. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation. *NeurIPS 35*.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765-4774.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *NeurIPS 32*.