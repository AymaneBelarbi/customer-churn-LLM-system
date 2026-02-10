"""
Streamlit Dashboard — Revenue Intelligence System
Full interactive dashboard with EDA, model comparison, SHAP,
financial simulation, and retention message generator.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_retention import generate_retention_message_template

# ─── Page Config ───
st.set_page_config(
    page_title="Revenue Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .risk-critical { color: #e74c3c; font-weight: bold; }
    .risk-high { color: #e67e22; font-weight: bold; }
    .risk-moderate { color: #f1c40f; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Models ───
@st.cache_data
def load_data():
    raw = pd.read_csv(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    featured = pd.read_csv(ROOT / "data" / "processed" / "featured.csv")
    return raw, featured


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "best_model.pkl")
    features = joblib.load(ROOT / "models" / "feature_names.pkl")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, features, meta


@st.cache_data
def load_model_comparison():
    return pd.read_csv(ROOT / "models" / "model_comparison.csv")


@st.cache_data
def load_shap_importances():
    try:
        return json.loads((ROOT / "static" / "shap_importances.json").read_text())
    except FileNotFoundError:
        return []


@st.cache_data
def load_financial():
    try:
        return json.loads((ROOT / "static" / "financial_simulation.json").read_text())
    except FileNotFoundError:
        return {}


raw_df, feat_df = load_data()
model, feature_names, meta = load_model()
comparison_df = load_model_comparison()
shap_data = load_shap_importances()
financial = load_financial()

# ─── Sidebar ───
st.sidebar.markdown("## 📊 Revenue Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🤖 Model Comparison", "🔍 SHAP Analysis",
     "💰 Financial Impact", "🎯 Predict & Retain", "📋 API Docs"],
)

# ═══════════════════════════════════════
# 🏠 OVERVIEW
# ═══════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">Revenue Intelligence System</p>', unsafe_allow_html=True)
    st.markdown("##### AI-Powered Churn Prediction & Customer Retention Platform")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = feat_df["Churn"].mean()
    with col1:
        st.metric("Total Customers", f"{len(feat_df):,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churn_rate:.1%} target")
    with col3:
        st.metric("Best Model", meta.get("best_model", "N/A"))
    with col4:
        st.metric("ROC-AUC", f"{meta.get('best_roc_auc', 0):.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Business KPIs")
        monthly_rev = feat_df["MonthlyCharges"].sum()
        churned_rev = feat_df[feat_df["Churn"] == 1]["MonthlyCharges"].sum()
        st.markdown(f"- **Monthly Revenue**: ${monthly_rev:,.0f}")
        st.markdown(f"- **Revenue at Risk (monthly)**: ${churned_rev:,.0f}")
        st.markdown(f"- **Revenue at Risk (annual)**: ${churned_rev * 12:,.0f}")
        st.markdown(f"- **Avg Revenue per Customer**: ${feat_df['MonthlyCharges'].mean():.2f}/mo")

    with col2:
        st.markdown("### 🎯 System Capabilities")
        st.markdown("""
        - ✅ **3 ML models** compared (XGBoost, LightGBM, Neural Net)
        - ✅ **SHAP explainability** for every prediction
        - ✅ **AI retention messages** personalized per customer
        - ✅ **Financial simulation** with ROI projections
        - ✅ **REST API** for production integration
        - ✅ **Docker-ready** deployment
        """)

# ═══════════════════════════════════════
# 📊 EDA
# ═══════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlations", "Churn Drivers"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(feat_df, x="Churn", color="Churn",
                             title="Churn Distribution",
                             color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
                             labels={"Churn": "Churned"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(feat_df, x="tenure", color="Churn", nbins=30,
                             title="Tenure Distribution by Churn",
                             color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
                             barmode="overlay", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(feat_df, x="Churn", y="MonthlyCharges", color="Churn",
                        title="Monthly Charges by Churn Status",
                        color_discrete_map={0: "#27ae60", 1: "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(feat_df, x="MonthlyCharges", color="Churn", nbins=30,
                             title="Monthly Charges Distribution",
                             color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
                             barmode="overlay", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
        corr = feat_df[numeric_cols].corr()
        fig = px.imshow(corr, title="Feature Correlation Heatmap",
                       color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Churn rate by contract
        contract_churn = feat_df.groupby("Contract")["Churn"].mean().reset_index()
        fig = px.bar(contract_churn, x="Contract", y="Churn",
                    title="Churn Rate by Contract Type",
                    color="Churn", color_continuous_scale="Reds")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            internet_churn = feat_df.groupby("InternetService")["Churn"].mean().reset_index()
            fig = px.bar(internet_churn, x="InternetService", y="Churn",
                        title="Churn Rate by Internet Service",
                        color="Churn", color_continuous_scale="Reds")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            payment_churn = feat_df.groupby("PaymentMethod")["Churn"].mean().reset_index()
            fig = px.bar(payment_churn, x="PaymentMethod", y="Churn",
                        title="Churn Rate by Payment Method",
                        color="Churn", color_continuous_scale="Reds")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════
# 🤖 MODEL COMPARISON
# ═══════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("## 🤖 Model Comparison")
    st.markdown("---")

    # Metrics table
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["accuracy", "precision", "recall", "f1", "roc_auc"],
            color="#d4edda"
        ),
        use_container_width=True,
    )

    # Visual comparison
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()
    colors = ["#667eea", "#764ba2", "#f093fb"]
    for i, _, in enumerate(comparison_df.itertuples()):
        fig.add_trace(go.Bar(
            name=_.model,
            x=metrics_to_plot,
            y=[getattr(_, m) for m in metrics_to_plot],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group", yaxis_range=[0, 1],
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best model callout
    best = comparison_df.loc[comparison_df["roc_auc"].idxmax()]
    st.success(f"🏆 **Best Model: {best['model']}** — ROC-AUC: {best['roc_auc']:.4f}, F1: {best['f1']:.4f}")

    # SHAP plots
    col1, col2 = st.columns(2)
    shap_bar = ROOT / "static" / "shap_summary_bar.png"
    shap_bee = ROOT / "static" / "shap_beeswarm.png"
    if shap_bar.exists():
        with col1:
            st.image(str(shap_bar), caption="SHAP Feature Importance")
    if shap_bee.exists():
        with col2:
            st.image(str(shap_bee), caption="SHAP Beeswarm Plot")

# ═══════════════════════════════════════
# 🔍 SHAP ANALYSIS
# ═══════════════════════════════════════
elif page == "🔍 SHAP Analysis":
    st.markdown("## 🔍 SHAP Explainability")
    st.markdown("---")

    if shap_data:
        shap_df = pd.DataFrame(shap_data)
        fig = px.bar(
            shap_df.head(15), x="importance", y="feature",
            orientation="h", title="Top 15 Features by SHAP Importance",
            color="importance", color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📖 Feature Interpretation")
        for i, row in shap_df.head(5).iterrows():
            st.markdown(f"**{i+1}. {row['feature']}** — Avg. SHAP impact: `{row['importance']:.4f}`")
    else:
        st.warning("SHAP analysis not yet generated. Run the training pipeline first.")

    # Show SHAP images
    for img_name, title in [("shap_summary_bar.png", "SHAP Bar Plot"), ("shap_beeswarm.png", "SHAP Beeswarm")]:
        img_path = ROOT / "static" / img_name
        if img_path.exists():
            st.image(str(img_path), caption=title, use_container_width=True)

# ═══════════════════════════════════════
# 💰 FINANCIAL IMPACT
# ═══════════════════════════════════════
elif page == "💰 Financial Impact":
    st.markdown("## 💰 Financial Impact Simulation")
    st.markdown("---")

    if financial:
        current = financial.get("current_metrics", {})
        intervention = financial.get("intervention", {})
        impact = financial.get("financial_impact", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Churn Rate", f"{current.get('churn_rate', 0)}%")
        with col2:
            st.metric("Annual Revenue at Risk", f"${current.get('annual_revenue_loss_from_churn', 0):,.0f}")
        with col3:
            st.metric("Customers Saved", f"{intervention.get('customers_saved', 0)}")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Revenue Recovered", f"${impact.get('annual_revenue_recovered', 0):,.0f}")
        with col2:
            st.metric("Total Retention Cost", f"${impact.get('total_retention_cost', 0):,.0f}")
        with col3:
            net = impact.get("net_annual_impact", 0)
            st.metric("Net Annual Impact", f"${net:,.0f}", delta=f"{impact.get('roi_pct', 0)}% ROI")

        # Sensitivity chart
        sensitivity = financial.get("sensitivity_analysis", [])
        if sensitivity:
            sens_df = pd.DataFrame(sensitivity)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sens_df["reduction_pct"], y=sens_df["annual_revenue_recovered"],
                name="Revenue Recovered", marker_color="#27ae60",
            ))
            fig.add_trace(go.Bar(
                x=sens_df["reduction_pct"], y=sens_df["total_cost"],
                name="Retention Cost", marker_color="#e74c3c",
            ))
            fig.add_trace(go.Scatter(
                x=sens_df["reduction_pct"], y=sens_df["net_impact"],
                name="Net Impact", mode="lines+markers",
                line=dict(color="#667eea", width=3),
            ))
            fig.update_layout(
                title="Sensitivity Analysis: Revenue vs. Cost by Churn Reduction %",
                xaxis_title="Churn Reduction (%)",
                yaxis_title="Amount ($)",
                barmode="group", template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(sens_df, use_container_width=True)
    else:
        st.warning("Financial simulation not yet generated.")

# ═══════════════════════════════════════
# 🎯 PREDICT & RETAIN
# ═══════════════════════════════════════
elif page == "🎯 Predict & Retain":
    st.markdown("## 🎯 Customer Churn Predictor & Retention Engine")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col3:
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col3:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    col1, col2 = st.columns(2)
    with col1:
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    with col2:
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, monthly * max(tenure, 1))

    if st.button("🔮 Predict Churn & Generate Retention Strategy", type="primary", use_container_width=True):
        # Build input
        customer_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        # Engineer features (same as API)
        df = pd.DataFrame([customer_data])
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
        sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        df["security_bundle_count"] = df[sec_cols].apply(lambda r: sum(1 for v in r if v == "Yes"), axis=1)
        df["has_security_bundle"] = (df["security_bundle_count"] >= 2).astype(int)
        str_cols = ["StreamingTV", "StreamingMovies"]
        df["streaming_bundle_count"] = df[str_cols].apply(lambda r: sum(1 for v in r if v == "Yes"), axis=1)
        df["has_streaming_bundle"] = (df["streaming_bundle_count"] == 2).astype(int)
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        cr = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        df["contract_risk_score"] = df["Contract"].map(cr)
        svc_cols = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df["num_services"] = df[svc_cols].apply(
            lambda r: sum(1 for v in r if v not in ["No", "No phone service", "No internet service"]), axis=1
        ).clip(lower=1)
        df["monthly_charge_per_service"] = df["MonthlyCharges"] / df["num_services"]
        df["overpay_ratio"] = df["MonthlyCharges"] / 64.76

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= 0.5)
        risk = "Critical" if prob > 0.8 else "High" if prob > 0.6 else "Moderate" if prob > 0.4 else "Low"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col2:
            risk_color = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}
            st.metric("Risk Level", f"{risk_color.get(risk, '')} {risk}")
        with col3:
            st.metric("Prediction", "Will Churn ⚠️" if pred else "Will Stay ✅")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 40], "color": "#d4edda"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#ffeeba"},
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Retention message
        if prob > 0.4:
            st.markdown("### 📧 AI-Generated Retention Message")
            profile = {
                "customer_id": "Dashboard-User",
                "tenure": tenure,
                "contract": contract,
                "monthly_charges": monthly,
                "internet_service": internet,
            }
            drivers = [
                {"feature": "Contract_Month-to-month", "shap_impact": 0.3, "direction": "increases"},
                {"feature": "tenure", "shap_impact": -0.2, "direction": "increases"},
                {"feature": "MonthlyCharges", "shap_impact": 0.15, "direction": "increases"},
            ]
            message = generate_retention_message_template(profile, prob, drivers)
            st.text_area("Retention Email", message, height=400)

# ═══════════════════════════════════════
# 📋 API DOCS
# ═══════════════════════════════════════
elif page == "📋 API Docs":
    st.markdown("## 📋 API Documentation")
    st.markdown("---")
    st.markdown("""
    ### Base URL: `http://localhost:8000`

    | Endpoint | Method | Description |
    |----------|--------|-------------|
    | `/health` | GET | Health check |
    | `/model/info` | GET | Model metadata and performance |
    | `/predict` | POST | Single customer churn prediction |
    | `/predict/batch` | POST | Batch predictions |
    | `/shap/importance` | GET | SHAP feature importances |
    | `/retention` | POST | Generate retention message |
    | `/financial` | GET | Financial simulation results |

    ### Example: Predict Churn
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 268.50
      }'
    ```
    """)

# ─── Footer ───
st.sidebar.markdown("---")
st.sidebar.markdown("**Revenue Intelligence System** v1.0")
st.sidebar.markdown("Built with ❤️ by Aymane Ait Belarbi")
st.sidebar.markdown(f"Best Model: `{meta.get('best_model', 'N/A')}`")
st.sidebar.markdown(f"ROC-AUC: `{meta.get('best_roc_auc', 0):.4f}`")
