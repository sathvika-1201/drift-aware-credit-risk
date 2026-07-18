# Drift-Aware Credit Risk Monitoring & Automated Retraining System

A credit-risk modeling project built on Lending Club loan data. The system trains a leakage-safe LightGBM model, monitors drift across future loan cohorts, and triggers champion–challenger retraining when drift exceeds a defined threshold.

---

## Project Overview

Credit-risk models can lose reliability when borrower profiles, loan characteristics, credit quality, or lending conditions change over time. This project simulates that scenario by training on historical Lending Club loans and treating newer loan cohorts as production-like batches.

### Key Components

- Leakage-safe feature engineering
- Time-based validation
- LightGBM credit-risk classification
- Evidently-based feature drift detection
- Automated drift-triggered retraining
- Champion–challenger model comparison
- Metric-gated model promotion
- MLflow experiment and artifact tracking

---

## Business Problem

Financial institutions use credit-risk models to estimate the probability of borrower default or delinquency. A model trained on historical loans may become less reliable when newer applicants differ from the original training population.

This project builds a monitoring and retraining workflow to answer:

- Are newer loan applications drifting from historical training data?
- Which borrower or loan features are changing over time?
- Should the model be retrained?
- Is the retrained model better than the current champion?
- Should the challenger model be promoted or rejected?

---

## Tech Stack

| Area | Tools |
|------|------|
| Language | Python |
| Data Processing | Pandas |
| Modeling | LightGBM, scikit-learn |
| Drift Monitoring | Evidently |
| Experiment Tracking | MLflow |
| Pipeline Automation | Prefect |
| API Layer | FastAPI |
| Dashboard | Streamlit |
| Deployment | Docker |
| CI/CD | GitHub Actions |

---

## Dataset

**Dataset:** Lending Club Loan Data

Main file:

```text
accepted_2007_to_2018Q4.csv
```

Expected local path:

```text
data/raw/accepted_2007_to_2018Q4.csv
```

The raw dataset is excluded from version control because of its large file size.

---

## System Architecture

```text
Raw Lending Club Data
        │
        ▼
Chunk-Based Data Processing
        │
        ▼
Leakage Removal + Target Creation
        │
        ▼
Feature Selection + Preprocessing
        │
        ▼
Time-Based Train / Validation / Holdout Split
        │
        ▼
Champion LightGBM Model Training
        │
        ▼
Reference vs Current Drift Detection
        │
        ▼
Automatic Challenger Retraining
        │
        ▼
Champion vs Challenger Evaluation
        │
        ▼
Metric-Gated Promotion Decision
        │
        ▼
MLflow Tracking + Drift Reports
```

---

## Project Structure

```text
drift-aware-credit-risk/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── monthly_batches/
│   └── reports/
│
├── models/
│
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── clean_data.py
│   │   ├── load_data.py
│   │   └── split_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train_model.py
│   ├── monitoring/
│   │   └── drift_report.py
│   ├── pipelines/
│   │   └── retraining_pipeline.py
│   └── api/
│       └── main.py
│
├── dashboard/
│   └── app.py
│
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## Target Definition

The model predicts whether a loan is risky.

```text
0 = Fully Paid
1 = Charged Off / Default / Late
```

The binary target is derived from the `loan_status` column.

---

## Leakage-Safe Modeling

The Lending Club dataset contains several post-origination columns that reveal future repayment behavior. These features are removed before training to prevent target leakage.

### Removed Leakage Features

```text
loan_status
recoveries
collection_recovery_fee
last_pymnt_d
last_pymnt_amnt
total_pymnt
total_rec_prncp
total_rec_int
out_prncp
settlement_status
hardship_status
```

Only information available at or before loan origination is retained.

---

## Feature Engineering

The model uses borrower, loan, credit-history, and application-level information.

### Example Features

```text
loan_amnt
term
int_rate
installment
grade
sub_grade
emp_length
home_ownership
annual_inc
verification_status
purpose
addr_state
dti
fico_range_low
fico_range_high
open_acc
revol_bal
revol_util
application_type
mort_acc
pub_rec_bankruptcies
```

### Preprocessing

- Numerical imputation
- Categorical imputation
- One-hot encoding
- High-cardinality category handling
- Memory-safe selected feature loading

---

## Time-Based Validation

Chronological data splitting is used instead of random splitting.

```text
Training / Reference Data : Loans issued before 2016
Validation Data           : Loans issued in 2016
Holdout Data              : Loans issued in 2017
Current Data              : Loans issued from 2018 onward
```

This setup better reflects real-world deployment where models predict future loan applications.

---

## Champion Model

The first trained model becomes the **Champion Model**.

Saved model:

```text
models/champion_model.pkl
```

Tracked metrics:

```text
ROC-AUC
PR-AUC
F1 Score
Precision
Recall
```

PR-AUC and Recall are emphasized because credit-risk prediction is an imbalanced classification problem.

---

## Drift Detection

Feature drift is monitored by comparing historical reference data with newer production-like loan cohorts.

```text
Reference Data = Loans before 2016
Current Data   = Loans from 2018 onward
```

Evidently monitors distribution changes across important features.

### Monitored Features

```text
interest rate
loan amount
annual income
debt-to-income ratio
FICO score
loan grade
loan purpose
home ownership
state
revolving utilization
```

Drift score:

```text
drift_score = drifted_features / total_monitored_features
```

Retraining is triggered whenever the drift score exceeds the configured threshold.

---

## Automated Champion–Challenger Retraining

The retraining pipeline follows a Champion–Challenger strategy.

```text
1. Detect drift between reference and current cohorts
2. Trigger retraining when drift exceeds threshold
3. Train a Challenger model
4. Evaluate Champion and Challenger on the same holdout set
5. Promote Challenger only if it satisfies promotion criteria
6. Otherwise retain the Champion model
```

This prevents replacing a production model without proper validation.

---

## Promotion Rules

The Challenger model is promoted only if it satisfies predefined performance gates.

### Promotion Criteria

```text
PR-AUC must not decrease beyond tolerance
ROC-AUC must not decrease beyond tolerance
Recall must be maintained or improved
```

Decision file:

```text
data/reports/promotion_decision.json
```

Possible outcomes:

```text
promoted
rejected
skipped
```

---

## MLflow Tracking

MLflow records every stage of the workflow.

Tracked information:

```text
Model parameters
Validation metrics
Drift scores
Training artifacts
Drift reports
Champion metrics
Challenger metrics
Promotion gates
Promotion decision
```

Typical experiment runs:

```text
champion-training
drift-check
challenger-training-after-drift
champion-challenger-comparison
```

---

## Generated Artifacts

The workflow produces the following outputs:

```text
models/champion_model.pkl
models/challenger_model.pkl
data/reports/retraining_drift_report.html
data/reports/retraining_drift_summary.json
data/reports/promotion_decision.json
```

These artifacts document the complete lifecycle from training to monitoring, retraining, and promotion.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process the Raw Dataset

```bash
python -m src.data.load_data
```

### 3. Train the Champion Model

```bash
python -m src.models.train_model
```

### 4. Run the Automated Retraining Pipeline

```bash
python -m src.pipelines.retraining_pipeline
```

### 5. Launch MLflow

```bash
mlflow ui --workers 1 --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

---

## Future Improvements

- Incremental learning support
- Real-time streaming drift detection
- SHAP explainability dashboard
- Automated email or Slack alerts on drift detection
- Kubernetes deployment for scalable retraining
- Continuous monitoring using scheduled workflows
- Model registry integration for automated production deployment

---