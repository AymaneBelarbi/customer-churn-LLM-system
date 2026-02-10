# 📊 Revenue Intelligence System

**An end-to-end ML platform that predicts customer churn, explains why customers leave, and auto-generates personalized retention campaigns — recovering $73K+ in annual revenue.**

Built as a production-grade system, not a notebook exercise.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-0.846_AUC-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

---

## 🎯 Business Problem

A telecom company with 7,043 customers loses **$1.67M annually** from a 26.5% churn rate. The executive team needs to know:

1. **Who** is about to leave?
2. **Why** are they leaving?
3. **What** can we do to keep them?
4. **How much** will intervention cost vs. save?

This system answers all four questions.

---

## 💰 Results

| Metric | Value |
|--------|-------|
| Best Model ROC-AUC | **0.846** (XGBoost) |
| Customers Saved (5% churn reduction) | 93 |
| Annual Revenue Recovered | **$80,163** |
| Retention Campaign Cost | $6,654 |
| **Net Annual Impact** | **$73,509** |
| **ROI** | **1,105%** |

---

## 🏗️ System Architecture

```
Raw Data → Data Cleaning → Feature Engineering (46 features)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                 XGBoost        LightGBM      Neural Network
                 (0.846)        (0.846)         (0.842)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                          Best Model Selection
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              SHAP Analysis   FastAPI Backend   Financial Sim
                    ↓               ↓               ↓
              Explainability   REST Endpoints   ROI Projections
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Streamlit UI    LLM Retention    Docker Deploy
                              Message Engine
```

---

## 🔬 Technical Highlights

### Feature Engineering
Engineered 46 features from 21 raw columns, including:
- **contract_risk_score** — Month-to-month (3) vs One year (2) vs Two year (1)
- **overpay_ratio** — Customer's charges vs contract-type average
- **security_bundle_count** — Number of protection services active
- **monthly_charge_per_service** — Cost efficiency per service
- **is_new_customer** — Tenure ≤ 6 months flag

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGBoost** | 0.799 | 0.655 | 0.513 | 0.576 | **0.846** |
| LightGBM | 0.737 | 0.503 | **0.821** | **0.623** | 0.846 |
| Neural Net | 0.798 | 0.665 | 0.484 | 0.560 | 0.842 |

**Key insight:** LightGBM has the highest recall (82%) — better when the priority is catching every potential churner. XGBoost wins on precision-AUC balance.

### SHAP Explainability
Every prediction comes with feature-level explanations. Top churn drivers:
1. Contract type (month-to-month = highest risk)
2. Tenure (new customers churn 3x more)
3. Monthly charges (higher spend without bundled value)
4. Fiber optic internet (quality/price perception gap)
5. Lack of tech support and online security

### LLM-Powered Retention
Generates personalized retention emails using customer profile + SHAP drivers. Supports Claude API with template fallback.

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/revenue-intelligence-system.git
cd revenue-intelligence-system

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline
python -c "from src.data_cleaning import build_pipeline; build_pipeline()"
python -c "from src.model_training import run_training_pipeline; run_training_pipeline(tune=True)"
python -c "from src.shap_explainability import generate_shap_analysis; generate_shap_analysis()"
python -c "from src.financial_simulation import simulate_financial_impact; simulate_financial_impact(data_path='data/processed/featured.csv', churn_reduction_pct=5.0)"

# Launch
uvicorn api.main:app --port 8000          # Terminal 1
streamlit run dashboard/app.py             # Terminal 2
```

Dashboard: http://localhost:8501 | API Docs: http://localhost:8000/docs

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Churn prediction + SHAP explanation |
| `/predict/batch` | POST | Batch predictions |
| `/retention` | POST | AI-generated retention message |
| `/shap/importance` | GET | Global feature importances |
| `/financial` | GET | Financial simulation + sensitivity |
| `/model/info` | GET | Model metadata |
| `/health` | GET | Health check |

---

## 📁 Project Structure

```
├── api/main.py                    # FastAPI backend (7 endpoints)
├── dashboard/app.py               # Streamlit dashboard (7 pages)
├── src/
│   ├── data_cleaning.py           # Pipeline: clean → engineer → encode
│   ├── model_training.py          # XGBoost, LightGBM, Neural Net + tuning
│   ├── shap_explainability.py     # Global + per-customer SHAP
│   ├── financial_simulation.py    # ROI with sensitivity analysis
│   └── llm_retention.py           # Claude API + template retention
├── models/                        # Saved models + artifacts
├── static/                        # SHAP plots + JSON data
├── docker-compose.yml
├── Dockerfile.api / Dockerfile.dashboard
└── requirements.txt
```

---

## 🛠️ Tech Stack

**ML:** XGBoost · LightGBM · scikit-learn · SHAP  
**Backend:** FastAPI · Pydantic · Uvicorn  
**Frontend:** Streamlit · Plotly  
**LLM:** Anthropic Claude API  
**Infra:** Docker · docker-compose  

---

## 📊 Financial Sensitivity Analysis

| Churn Reduction | Customers Saved | Revenue Recovered | Net Impact | ROI |
|----------------|-----------------|-------------------|------------|-----|
| 1% | 18 | $15,876 | $14,560 | 1,106% |
| 3% | 56 | $49,345 | $45,253 | 1,106% |
| **5%** | **93** | **$80,163** | **$73,509** | **1,105%** |
| 10% | 186 | $163,820 | $150,240 | 1,106% |
| 15% | 280 | $246,762 | $226,308 | 1,107% |

---

## Author

Built by **Kira** — ML Engineer & Full-Stack Developer

---

## License

MIT
