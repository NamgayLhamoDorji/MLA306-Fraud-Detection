# Financial Fraud Detection Using a Hybrid Deep Neural Network and Autoencoder

**Module:** MLA306 Improving Deep Neural Networks
**Assessment:** CA4 Deep Neural Network Optimisation Project, Part 2 (DL Model and Presentation)
**Institution:** Gedu College of Business Studies, Royal University of Bhutan
**Programme:** Bachelor of Business Intelligence, Year 3, Semester 1
**Module tutor:** Mr. Tashi Tobgye Sherpa

---

## 1. Project Summary

Banks and mobile payment services such as BOB Mobile, MyPay and ePay process millions of transactions, so fraud cannot be checked by hand, and fixed rules miss new fraud patterns. This project builds a deep learning system that screens **bank account applications** and returns a fraud-risk score, a flag and a short reason.

The system combines two models:

- **DNN classifier (supervised):** learns known fraud patterns from labelled applications and outputs a fraud probability from 0 to 1.
- **Autoencoder (unsupervised):** trained only on normal applications; an unusual application is rebuilt poorly, and the reconstruction error is used as an anomaly score.
- **Hybrid scorer:** flags an application if **either** model is suspicious, so new fraud types missed by the DNN can still be caught by the Autoencoder.

Flagged applications are explained with SHAP and passed to a human reviewer. **The system never approves or rejects an application on its own.**

## 2. Objectives

1. Prepare the BAF dataset for supervised and unsupervised fraud detection.
2. Build and train a DNN classifier that separates normal from fraudulent applications.
3. Build and train an Autoencoder that detects anomalies by reconstruction error.
4. Combine both into a hybrid scoring system and test whether it beats either model alone.
5. Handle the class imbalance (about 1.1% fraud) so the models do not just predict "not fraud".
6. Evaluate with proper metrics, not accuracy alone.
7. Explain each flagged application instead of only returning a number.

## 3. Features of the System

| Feature | Description |
| --- | --- |
| Fraud risk scoring | A probability from 0 (normal) to 1 (high risk) for each application |
| Anomaly detection | Reconstruction error from the Autoencoder, needing no fraud labels |
| Hybrid decision logic | An application is flagged if either model's score passes its threshold |
| Threshold alerting | Flagged applications go to human review; nothing is auto-approved or rejected |
| Basic explainability | SHAP shows which features drove the flag |
| Demo application | A reviewer enters an application and sees the score, the flag and the reason |

## 4. Dataset

**Bank Account Fraud (BAF) Dataset Suite, Base variant** (`Base.csv`), released at NeurIPS 2022 by Feedzai Research and hosted on Kaggle.

- 1,000,000 account-opening applications over 8 months (`month` column)
- 30 features plus the label `is_fraud`
- About 1.1% fraud, so the data is heavily imbalanced
- Generated with a conditional GAN and differential privacy applied to a real, anonymised dataset, so it contains no real personal data
- Three protected attributes, used for fairness checks: age group, employment status and income

**The dataset is NOT stored in this repository** because of its size. Every member downloads it separately and saves it as:

```
data/Base.csv
```

See `scripts/download_data.md` for the download link and steps. Git ignores everything in `data/` except `.gitkeep`, so the file cannot be committed by mistake.

**Citation:** Jesus, S., Pombal, J., Alves, D., Cruz, A. F., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation. *Advances in Neural Information Processing Systems, 35.* https://arxiv.org/abs/2211.13358

**Licence:** CC BY-NC-ND 4.0, used here for academic, non-commercial purposes only.

## 5. System Workflow

```
Application data
      |
Clean (flag -1 values as "not available")
      |
Scale numbers, encode categories
      |
      +--------------------+
      |                    |
 DNN classifier       Autoencoder
 (fraud probability)  (reconstruction error)
      |                    |
      +----------+---------+
                 |
          Hybrid scoring (flag if EITHER passes its threshold)
                 |
        Above threshold?
        /              \
      No               Yes
 Treat as normal    Flag for human review + SHAP reason
        \              /
      Show score, flag and reason
```

## 6. Model Design (Starting Architecture)

Layer sizes below are starting points. Optuna tunes them.

**DNN classifier (multi-layer perceptron)**

| Layer | Size | Notes |
| --- | --- | --- |
| Input | about 30 features | After encoding and scaling |
| Dense 1 | 128 | ReLU, BatchNorm, Dropout |
| Dense 2 | 64 | ReLU, Dropout |
| Dense 3 | 32 | ReLU |
| Output | 1 | Sigmoid, fraud probability |

Loss: binary cross-entropy with class weights.

**Autoencoder**

| Part | Layers |
| --- | --- |
| Encoder | Dense 64 (ReLU), Dense 32 (ReLU) |
| Bottleneck | 16 |
| Decoder | Dense 32 (ReLU), Dense 64 (ReLU), output = number of input features |

Loss: reconstruction error (mean squared error). Trained on **non-fraud applications only**. The alert threshold is a high percentile of the reconstruction error of normal validation applications.

## 7. Data Preparation

1. **Data understanding:** check size, types, fraud ratio and feature distributions.
2. **Missing values:** a value of `-1` means "not available" (for example `prev_address_months_count`, `bank_months_count`); flag it instead of treating it as a number.
3. **Scaling:** standardise numeric features such as income, credit risk score, velocity features and session length.
4. **Encoding:** convert categorical features (`payment_type`, `employment_status`, `housing_status`, `device_os`, `source`) to numbers.
5. **Temporal split:** use the `month` column so training data comes before test data. Proposed split: months 0 to 5 train, month 6 validation, month 7 test. The test month is never used for training or tuning.
6. **Class imbalance:** compare class weights, SMOTE and under-sampling on the training set only, never on validation or test data.
7. **Fairness groups:** keep age group, employment status and income band so results can be checked per group.

## 8. Optimisation Techniques

| Area | What we compare |
| --- | --- |
| Activation functions | ReLU, LeakyReLU, GELU |
| Optimisers | Adam, AdamW (and SGD with momentum as a baseline) |
| Hyperparameter search | Optuna: learning rate, batch size, depth, width, dropout |
| Neural architecture search | Optuna search over depth and width |
| Regularisation | Dropout, BatchNorm, early stopping |
| Imbalance handling | Class weights vs SMOTE vs under-sampling |
| Efficiency | Small networks that train on a laptop or free Colab |
| Tracking | TensorBoard logs for every trial |

## 9. Evaluation

Accuracy alone is misleading here: a model that says "not fraud" every time would score about 99%. We use:

- Precision, recall and F1-score
- ROC-AUC and PR-AUC
- Confusion matrix
- Reconstruction error (Autoencoder)
- 5-fold stratified cross-validation on the training data
- Final scores on the held-out test month (unseen data)
- Fairness checks by age group, employment status and income band
- A comparison table: DNN alone vs Autoencoder alone vs hybrid

## 10. Repository Structure

```
MLA306-Fraud-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/            Raw and processed data (git-ignored; only .gitkeep is tracked)
├── notebooks/       EDA and Optuna tuning notebooks
├── src/
│   ├── preprocess.py    Cleaning, scaling, encoding, splitting, saved pipeline
│   ├── dnn.py           DNN classifier
│   ├── autoencoder.py   Autoencoder
│   ├── hybrid.py        Hybrid scorer
│   ├── evaluate.py      Metrics, cross-validation, fairness
│   └── explain.py       SHAP explanations
├── app/             Demo application
├── models/          Small saved weights, thresholds, preprocessing pipeline
├── results/         Metrics tables and plots
└── scripts/         Helper scripts and download_data.md
```

## 11. Setup

**Requirements:** Python 3.10 or 3.11, Git, and about 8 GB of RAM (use a 100,000-row sample on a laptop and Google Colab for full training).

```bash
# 1. Clone the repository (pick a folder first, for example Documents)
git clone <repository-url>
cd MLA306-Fraud-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
# .venv\Scripts\activate           # Command Prompt / PowerShell

# 3. Install packages
pip install -r requirements.txt
```

**Get the data:** download `Base.csv` using `scripts/download_data.md` and save it as `data/Base.csv` inside your clone.

**Google Colab:** upload `Base.csv` to your Google Drive, mount Drive in the notebook and load the file from there. Use Colab for full-data training and Optuna runs.

## 12. How to Run

The commands below are the planned run order. Each owner updates this section when their module is merged.

```bash
python src/preprocess.py     # Member 1: cleaned, scaled, split data and saved pipeline
python src/dnn.py            # Member 2: train the DNN and save the best weights
python src/autoencoder.py    # Member 3: train the Autoencoder, save weights and threshold
python src/evaluate.py       # Member 4: hybrid scoring, metrics, SHAP, fairness
streamlit run app/demo.py    # Member 5: demo application
```

TensorBoard: `tensorboard --logdir results/tensorboard`

## 13. Team and Responsibilities

| Member | Name | Role | Main files |
| --- | --- | --- | --- |
| Member 1 | <name> | Data engineering: EDA, cleaning, preprocessing, saved pipeline | `preprocess.py`, `data/` files |
| Member 2 | <name> | DNN classifier and Optuna study | `dnn.py` |
| Member 3 | <name> | Autoencoder, error threshold and Optuna study | `autoencoder.py` |
| Member 4 | <name> | Hybrid scorer, metrics, cross-validation, SHAP, fairness | `hybrid.py`, `evaluate.py`, `explain.py` |
| Member 5 | <name> | Demo app, README, GitLab mirror, presentation | `app/`, `README.md` |

Team lead: <name>. The lead reviews and merges all pull requests and keeps the schedule.

## 14. Roadmap and Releases

| # | Owner | Dates | Work | Deliverable and tag |
| --- | --- | --- | --- | --- |
| 1 | Member 1 | 12-17 Oct | EDA, cleaning, -1 flags, scaling, encoding, month split, SMOTE vs class weights, saved preprocessing pipeline | `data/` files and `preprocess.py` (`v0.1-data`) |
| 2 | Member 2 | 18-23 Oct | DNN baseline, regularisation, activation and optimiser comparison, Optuna study (depth and width cover NAS) | `dnn.py` and best weights (`v0.2-dnn`) |
| 3 | Member 3 | 24-29 Oct | Autoencoder on non-fraud only, bottleneck size, error distribution, Optuna study, threshold choice | `autoencoder.py` and best weights (`v0.3-ae`) |
| 4 | Member 4 | 30 Oct-3 Nov | Hybrid scorer, metrics, time-based CV, final unseen-month results, SHAP, fairness checks | `hybrid.py`, `evaluate.py`, `explain.py`, results tables (`v0.4-eval`) |
| 5 | Member 5 | 4-8 Nov | Demo app with real models and error states, README, GitLab mirror, slide merge, rehearsals | Final release (`v1.0-part2`) |

Tutor check-ins are planned for 16 Oct, 23 Oct, 30 Oct and 6 Nov.

## 15. Git Workflow

- **Never push directly to `main`.** Work on a branch and open a pull request.
- **Branch names:** `feature/<role>-<task>`, for example `feature/dnn-baseline`, `feature/ae-threshold`, `fix/preprocess-missing-values`.
- **Start of each session:** `git checkout main`, `git pull origin main`, then `git checkout -b feature/<role>-<task>`.
- **Commit often** with clear messages, for example `Add class weights to DNN training`.
- **Before a pull request:** `git pull origin main` into your branch, run your code once, and make sure no data or large model files are included.
- **Pull request checklist:**
  - [ ] Code runs from a fresh clone with the documented steps
  - [ ] No dataset files and nothing over 50 MB committed
  - [ ] Random seed is 42 where randomness is used
  - [ ] Description says what changed and how to run it
- **Review:** the lead reviews and merges within 24 hours.
- **Releases:** the owner of each stage asks the lead to tag the merge (`v0.1-data` and so on).

## 16. Ethics, Fairness and Privacy

- **Bias:** BAF deliberately contains bias patterns across six variants. We report false-positive and false-negative rates per age group, employment status and income band, instead of one overall score.
- **Limitation:** a model that flags one group far more often without a matching rise in real fraud will be reported as a limitation of this study.
- **Privacy:** only the anonymised, privacy-protected BAF data is used; no real customer data is collected or stored.
- **Human oversight:** every flagged application is reviewed by a person.
- **Compliance:** a real deployment would need to follow Bhutan's data protection rules and the policies of the financial institution.

## 17. Known Limitations

- BAF is synthetic data, so results may not match real Bhutanese payment data.
- Fraud patterns change over time; the last month is a single test window.
- Fairness is measured on three protected attributes and is not proven.
- Training on the full 1,000,000 rows can be slow without Colab.

## 18. Submission

- **GitLab repository (required by the brief):** <GitLab link here>
- **Project report (PDF):** includes the GitLab link.
- **Presentation and demonstration:** at the time arranged with the supervisor.

## 19. References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623-2631.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321-357.
- Jesus, S., et al. (2022). Turning the tables: Biased, imbalanced, dynamic tabular datasets for ML evaluation. *NeurIPS 35*.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765-4774.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *NeurIPS 32*.