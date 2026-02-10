"""
FastAPI Backend — Revenue Intelligence System
Endpoints:
  POST /predict          → Churn prediction for a customer
  POST /predict/batch    → Batch predictions
  GET  /model/info       → Model metadata
  GET  /shap/importance  → Top SHAP feature importances
  POST /retention        → Generate retention message for a customer
  GET  /financial        → Financial simulation results
  GET  /health           → Health check
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_retention import generate_retention_message_template, generate_retention_message_llm
from shap_explainability import explain_customer

app = FastAPI(
    title="Revenue Intelligence System API",
    description="Churn prediction, SHAP explainability, and AI-powered retention for Telco customers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load artifacts at startup ───
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

model = joblib.load(MODEL_DIR / "best_model.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())

try:
    shap_importances = json.loads((STATIC_DIR / "shap_importances.json").read_text())
except FileNotFoundError:
    shap_importances = []

try:
    financial_data = json.loads((STATIC_DIR / "financial_simulation.json").read_text())
except FileNotFoundError:
    financial_data = {}


# ─── Schemas ───
class CustomerInput(BaseModel):
    """Raw customer features for prediction."""
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 70.0


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    top_risk_factors: list


class RetentionRequest(BaseModel):
    customer: CustomerInput
    use_llm: bool = False
    api_key: Optional[str] = None


# ─── Feature Engineering (mirrors src/data_cleaning.py) ───
def engineer_single_customer(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as training pipeline to a single customer."""
    df = pd.DataFrame([data])

    # Engineered features
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["security_bundle_count"] = df[security_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["has_security_bundle"] = (df["security_bundle_count"] >= 2).astype(int)

    streaming_cols = ["StreamingTV", "StreamingMovies"]
    df["streaming_bundle_count"] = df[streaming_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["has_streaming_bundle"] = (df["streaming_bundle_count"] == 2).astype(int)

    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    df["contract_risk_score"] = df["Contract"].map(contract_risk)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No phone service", "No internet service"]),
        axis=1,
    ).clip(lower=1)
    df["monthly_charge_per_service"] = df["MonthlyCharges"] / df["num_services"]
    df["overpay_ratio"] = df["MonthlyCharges"] / 64.76  # Global mean from training

    # One-hot encode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Align with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


# ─── Endpoints ───
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": metadata.get("best_model"), "version": "1.0.0"}


@app.get("/model/info")
def model_info():
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    """Predict churn for a single customer."""
    try:
        data = customer.model_dump()
        X = engineer_single_customer(data)

        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= 0.5)
        risk = "Critical" if prob > 0.8 else "High" if prob > 0.6 else "Moderate" if prob > 0.4 else "Low"

        # SHAP explanation
        try:
            factors = explain_customer(X)
        except Exception:
            factors = []

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            risk_level=risk,
            top_risk_factors=factors[:5],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    """Batch predictions."""
    results = []
    for c in customers:
        try:
            result = predict_churn(c)
            results.append(result.model_dump())
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}


@app.get("/shap/importance")
def get_shap_importance():
    return {"features": shap_importances}


@app.post("/retention")
def generate_retention(req: RetentionRequest):
    """Generate personalized retention message."""
    try:
        data = req.customer.model_dump()
        X = engineer_single_customer(data)
        prob = float(model.predict_proba(X)[:, 1][0])

        profile = {
            "customer_id": "API-Customer",
            "tenure": data["tenure"],
            "contract": data["Contract"],
            "monthly_charges": data["MonthlyCharges"],
            "internet_service": data["InternetService"],
        }

        try:
            drivers = explain_customer(X)
        except Exception:
            drivers = [{"feature": "Contract_Month-to-month", "shap_impact": 0.3, "direction": "increases"}]

        if req.use_llm and req.api_key:
            message = generate_retention_message_llm(profile, prob, drivers, req.api_key)
        else:
            message = generate_retention_message_template(profile, prob, drivers)

        return {
            "churn_probability": round(prob, 4),
            "risk_level": "Critical" if prob > 0.8 else "High" if prob > 0.6 else "Moderate",
            "retention_message": message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/financial")
def get_financial_simulation():
    return financial_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
