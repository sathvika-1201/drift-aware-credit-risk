import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import MODEL_DIR
from src.features.build_features import prepare_features


app = FastAPI(title="Credit Risk Prediction API")

model_path = MODEL_DIR / "lightgbm_credit_risk_pipeline.pkl"
model = joblib.load(model_path)


class LoanApplication(BaseModel):
    loan_amnt: float
    term: float
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    addr_state: str
    dti: float
    delinq_2yrs: float
    earliest_cr_line: str
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    application_type: str
    mort_acc: float
    pub_rec_bankruptcies: float


@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running",
        "docs": "Go to /docs to test predictions",
    }


@app.post("/predict")
def predict(application: LoanApplication):
    input_df = pd.DataFrame([application.model_dump()])

    input_df = prepare_features(input_df)

    default_probability = model.predict_proba(input_df)[:, 1][0]

    if default_probability < 0.15:
        risk_bucket = "Low Risk"
    elif default_probability < 0.35:
        risk_bucket = "Medium Risk"
    else:
        risk_bucket = "High Risk"

    return {
        "default_probability": round(float(default_probability), 4),
        "risk_bucket": risk_bucket,
    }