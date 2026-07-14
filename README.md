# Drift-Aware Credit Risk Monitoring System

A credit-risk machine learning project that predicts risky loans using Lending Club data and monitors whether future loan batches drift from the original training data.

---

## Overview

Credit-risk models can become less reliable when borrower behavior, loan characteristics, or market conditions change. This project trains a LightGBM model on historical Lending Club loans and uses drift monitoring to compare older training data with newer production-like data.

The project includes:

* Data cleaning
* Leakage-safe feature selection
* Time-based train/validation split
* LightGBM model training
* MLflow experiment tracking
* Evidently drift report generation

---

## Dataset

Dataset: **Lending Club Loan Data**

https://www.kaggle.com/datasets/wordsforthewise/lending-club

Use the accepted loans file:

```text
accepted_2007_to_2018Q4.csv
```

Place it here:

```text
data/raw/accepted_2007_to_2018Q4.csv
```

The raw dataset is not included in this repository because of file size.

---

## Tech Stack

```text
Python
Pandas
LightGBM
scikit-learn
MLflow
Evidently
Prefect
FastAPI
Streamlit
Docker
GitHub Actions
```

---

## Project Structure

```text
drift-aware-credit-risk/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── models/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── pipelines/
│   └── api/
│
├── dashboard/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Target

The model predicts whether a loan is risky.

```text
0 = Fully Paid
1 = Charged Off / Default / Late
```

---

## Time-Based Split

Instead of a random split, the data is split by loan issue date:

```text
Train:       before 2016
Validation: 2016
Test:       2017
Current:    2018 onward
```

This better reflects how a model would behave on future data.

---

## How to Run

### 1. Create and activate virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Process the data

```bash
python -m src.data.load_data
```

### 4. Train the model

```bash
python -m src.models.train_model
```

The trained model is saved to:

```text
models/lightgbm_credit_risk_pipeline.pkl
```

### 5. Open MLflow

```bash
mlflow ui --workers 1 --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

### 6. Generate drift report

```bash
python -m src.monitoring.drift_report
```

The report is saved to:

```text
data/reports/data_drift_report.html
```

---

## Model Metrics

The model tracks:

```text
ROC-AUC
PR-AUC
F1 Score
Recall
```

For this problem, PR-AUC and recall are especially useful because risky loans are harder and more important to detect.

---

## Drift Monitoring

The drift report compares:

```text
Reference data = older training loans
Current data   = newer loan batches
```

It helps detect changes in features such as:

```text
loan amount
interest rate
annual income
debt-to-income ratio
FICO score
loan grade
home ownership
loan purpose
```

