"""
Data Cleaning & Feature Engineering Pipeline
Revenue Intelligence System — Telco Customer Churn
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load and return the raw dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco dataset:
    - Fix TotalCharges blank → NaN → impute with median
    - Encode Churn to binary
    - Drop customerID (non-predictive)
    """
    df = df.copy()

    # TotalCharges has 11 blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # SeniorCitizen is already 0/1
    # Drop customerID
    df.drop(columns=["customerID"], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-meaningful features:
    - tenure_group: bucketed tenure
    - avg_monthly_spend: TotalCharges / max(tenure, 1)
    - has_security_bundle: OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport
    - has_streaming_bundle: StreamingTV + StreamingMovies
    - is_new_customer: tenure <= 6
    - contract_risk_score: month-to-month=3, one year=2, two year=1
    - monthly_charge_per_service: MonthlyCharges / num_services
    - overpay_ratio: MonthlyCharges / avg for same contract type
    """
    df = df.copy()

    # Tenure groups
    bins = [0, 6, 12, 24, 48, 72, np.inf]
    labels = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

    # Avg monthly spend (prevents div-by-zero)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # Count yes-services for security bundle
    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["security_bundle_count"] = df[security_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["has_security_bundle"] = (df["security_bundle_count"] >= 2).astype(int)

    # Streaming bundle
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    df["streaming_bundle_count"] = df[streaming_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["has_streaming_bundle"] = (df["streaming_bundle_count"] == 2).astype(int)

    # New customer flag
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    # Contract risk score
    contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    df["contract_risk_score"] = df["Contract"].map(contract_risk)

    # Count total services
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

    # Overpay ratio vs. contract-type mean
    contract_avg = df.groupby("Contract")["MonthlyCharges"].transform("mean")
    df["overpay_ratio"] = df["MonthlyCharges"] / contract_avg

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining categorical columns."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def build_pipeline(raw_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Full pipeline: load → clean → engineer → encode → save."""
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    df_encoded = encode_features(df_feat)

    # Save processed
    out_path = Path("data/processed")
    out_path.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path / "cleaned.csv", index=False)
    df_feat.to_csv(out_path / "featured.csv", index=False)
    df_encoded.to_csv(out_path / "encoded.csv", index=False)

    print(f"✅ Pipeline complete — {df_encoded.shape[1]} features, {df_encoded.shape[0]} rows")
    return df_encoded


if __name__ == "__main__":
    build_pipeline()
