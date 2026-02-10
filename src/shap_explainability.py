"""
SHAP Explainability Module
Generates feature importance plots and per-customer explanations.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def generate_shap_analysis(model_path="models/best_model.pkl", test_path="models/X_test.csv"):
    """Generate SHAP summary and save artifacts."""
    print("🔍 Running SHAP explainability analysis...")

    model = joblib.load(model_path)
    X_test = pd.read_csv(test_path)

    # Use TreeExplainer for tree models, KernelExplainer as fallback
    model_type = type(model).__name__
    print(f"   Model type: {model_type}")

    if model_type in ["XGBClassifier", "LGBMClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # Neural network — use sample for speed
        sample = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, sample)
        shap_values = explainer.shap_values(sample)
        X_test = sample
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = churn

    out_dir = Path("static")
    out_dir.mkdir(exist_ok=True)

    # ─── Summary Bar Plot ───
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title("Top 20 Features — SHAP Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ shap_summary_bar.png saved")

    # ─── Beeswarm Plot ───
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title("SHAP Beeswarm — Feature Impact on Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ shap_beeswarm.png saved")

    # ─── Top feature importances as JSON (for dashboard) ───
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False)

    importance_dict = importance_df.head(20).to_dict(orient="records")
    with open(out_dir / "shap_importances.json", "w") as f:
        json.dump(importance_dict, f, indent=2, default=str)

    # Save shap values for per-customer explanations
    joblib.dump(shap_values, "models/shap_values.pkl")
    joblib.dump(explainer, "models/shap_explainer.pkl")

    print("✅ SHAP analysis complete")
    return importance_df


def explain_customer(customer_data: pd.DataFrame, model_path="models/best_model.pkl"):
    """Generate SHAP explanation for a single customer."""
    model = joblib.load(model_path)
    model_type = type(model).__name__

    if model_type in ["XGBClassifier", "LGBMClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(customer_data)
    else:
        explainer = joblib.load("models/shap_explainer.pkl")
        shap_vals = explainer.shap_values(customer_data)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    # Top contributing features
    sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
    feature_impacts = sorted(
        zip(customer_data.columns, sv, customer_data.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    explanation = []
    for feat, impact, val in feature_impacts[:5]:
        direction = "increases" if impact > 0 else "decreases"
        explanation.append({
            "feature": feat,
            "value": float(val),
            "shap_impact": round(float(impact), 4),
            "direction": direction,
        })

    return explanation


if __name__ == "__main__":
    generate_shap_analysis()
