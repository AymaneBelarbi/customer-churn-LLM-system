"""
Model Training & Comparison
Compares XGBoost, LightGBM, and a Neural Network (MLPClassifier)
with hyperparameter tuning via Optuna-style grid/random search.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


def load_data(path: str = "data/processed/encoded.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def train_xgboost(X_train, y_train, tune=True):
    """Train XGBoost with optional hyperparameter tuning."""
    if tune:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "scale_pos_weight": [1, 2, 3],
        }
        base = XGBClassifier(
            eval_metric="logloss", use_label_encoder=False, random_state=42, verbosity=0
        )
        search = RandomizedSearchCV(
            base, param_dist, n_iter=40, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"  XGBoost best params: {search.best_params_}")
        return search.best_estimator_
    else:
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False, random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        return model


def train_lightgbm(X_train, y_train, tune=True):
    """Train LightGBM with optional tuning."""
    if tune:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20],
            "scale_pos_weight": [1, 2, 3],
        }
        base = LGBMClassifier(random_state=42, verbose=-1)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=40, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"  LightGBM best params: {search.best_params_}")
        return search.best_estimator_
    else:
        model = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model


def train_neural_network(X_train, y_train, tune=True):
    """Train MLPClassifier with scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if tune:
        param_dist = {
            "hidden_layer_sizes": [(128, 64), (256, 128, 64), (128, 64, 32), (64, 32)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.005, 0.01],
            "batch_size": [32, 64, 128],
        }
        base = MLPClassifier(max_iter=500, early_stopping=True, random_state=42, verbose=False)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=20, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_scaled, y_train)
        print(f"  Neural Net best params: {search.best_params_}")
        return search.best_estimator_, scaler
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, random_state=42, verbose=False,
        )
        model.fit(X_scaled, y_train)
        return model, scaler


def run_training_pipeline(tune=True):
    """Full training pipeline with comparison."""
    print("📊 Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Churn rate: {y.mean():.2%}\n")

    results = []
    models = {}

    # ─── XGBoost ───
    print("🚀 Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, tune=tune)
    metrics_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(metrics_xgb)
    models["xgboost"] = xgb_model
    print(f"   ROC-AUC: {metrics_xgb['roc_auc']}, F1: {metrics_xgb['f1']}\n")

    # ─── LightGBM ───
    print("🚀 Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, tune=tune)
    metrics_lgbm = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
    results.append(metrics_lgbm)
    models["lightgbm"] = lgbm_model
    print(f"   ROC-AUC: {metrics_lgbm['roc_auc']}, F1: {metrics_lgbm['f1']}\n")

    # ─── Neural Network ───
    print("🚀 Training Neural Network...")
    nn_model, scaler = train_neural_network(X_train, y_train, tune=tune)
    # Scale test data for NN
    X_test_scaled = scaler.transform(X_test)

    y_pred_nn = nn_model.predict(X_test_scaled)
    y_prob_nn = nn_model.predict_proba(X_test_scaled)[:, 1]
    metrics_nn = {
        "model": "NeuralNetwork",
        "accuracy": round(accuracy_score(y_test, y_pred_nn), 4),
        "precision": round(precision_score(y_test, y_pred_nn), 4),
        "recall": round(recall_score(y_test, y_pred_nn), 4),
        "f1": round(f1_score(y_test, y_pred_nn), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob_nn), 4),
    }
    results.append(metrics_nn)
    models["neural_network"] = nn_model
    models["nn_scaler"] = scaler
    print(f"   ROC-AUC: {metrics_nn['roc_auc']}, F1: {metrics_nn['f1']}\n")

    # ─── Compare & Save Best ───
    results_df = pd.DataFrame(results)
    print("=" * 60)
    print("📋 MODEL COMPARISON:")
    print(results_df.to_string(index=False))
    print("=" * 60)

    best_idx = results_df["roc_auc"].idxmax()
    best_model_name = results_df.loc[best_idx, "model"]
    print(f"\n🏆 Best model: {best_model_name} (ROC-AUC: {results_df.loc[best_idx, 'roc_auc']})")

    # Save models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(models["xgboost"], model_dir / "xgboost_model.pkl")
    joblib.dump(models["lightgbm"], model_dir / "lightgbm_model.pkl")
    joblib.dump(models["neural_network"], model_dir / "neural_network_model.pkl")
    joblib.dump(models["nn_scaler"], model_dir / "nn_scaler.pkl")

    # Save best model reference
    best_key = best_model_name.lower().replace(" ", "_").replace("neuralnetwork", "neural_network")
    if best_key == "neuralnetwork":
        best_key = "neural_network"
    joblib.dump(models.get(best_key, models["xgboost"]), model_dir / "best_model.pkl")

    # Save feature names and results
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, model_dir / "feature_names.pkl")
    results_df.to_csv(model_dir / "model_comparison.csv", index=False)

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "best_roc_auc": float(results_df.loc[best_idx, "roc_auc"]),
        "best_f1": float(results_df.loc[best_idx, "f1"]),
        "n_features": len(feature_names),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "churn_rate": float(y.mean()),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test data for SHAP
    X_test.to_csv(model_dir / "X_test.csv", index=False)
    y_test.to_csv(model_dir / "y_test.csv", index=False)

    print("✅ All models saved to /models/")
    return results_df, models


if __name__ == "__main__":
    run_training_pipeline(tune=True)
