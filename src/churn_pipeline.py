"""
Customer Churn Prediction - End to End Pipeline
=================================================
Dataset: Telco Customer Churn (Kaggle)
https://www.kaggle.com/blastchar/telco-customer-churn

Download the CSV and place it at: data/telco_churn.csv

Run:
    python src/churn_pipeline.py

This script walks through:
1. Loading & cleaning data
2. EDA (saves plots to outputs/)
3. Feature engineering
4. Handling class imbalance
5. Model training (Logistic Regression, Random Forest, XGBoost)
6. Evaluation (precision/recall/F1/ROC-AUC, confusion matrix)
7. SHAP explainability
8. Saving the best model for the Streamlit app
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # TotalCharges is often read as object due to blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customer ID, it carries no signal
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target column standardization
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def run_eda(df):
    sns.set_theme(style="whitegrid")

    # Churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution (0 = Stayed, 1 = Churned)")
    plt.savefig(os.path.join(OUTPUT_DIR, "churn_distribution.png"), bbox_inches="tight")
    plt.close()

    # Churn by contract type
    if "Contract" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Contract", hue="Churn")
        plt.title("Churn by Contract Type")
        plt.savefig(os.path.join(OUTPUT_DIR, "churn_by_contract.png"), bbox_inches="tight")
        plt.close()

    # Tenure vs churn
    if "tenure" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="Churn", y="tenure")
        plt.title("Tenure vs Churn")
        plt.savefig(os.path.join(OUTPUT_DIR, "tenure_vs_churn.png"), bbox_inches="tight")
        plt.close()

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    print(f"EDA plots saved to {OUTPUT_DIR}/")


def feature_engineer(df):
    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
    )

    # Average monthly spend proxy
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].replace(0, 1))

    return df


def encode_features(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train_models(X_train, y_train):
    models = {}

    # Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE
    )
    xgb_clf.fit(X_train, y_train)
    models["XGBoost"] = xgb_clf

    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
        print(f"ROC-AUC: {auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"), bbox_inches="tight")
        plt.close()

        results[name] = {"model": model, "auc": auc, "f1": f1}

    return results


def explain_best_model(model, X_train, X_test, feature_names):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {OUTPUT_DIR}/shap_summary.png")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        print("Download it from https://www.kaggle.com/blastchar/telco-customer-churn")
        print("and place the CSV at data/telco_churn.csv")
        return

    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)

    print("Running EDA...")
    run_eda(df)

    print("Engineering features...")
    df = feature_engineer(df)

    print("Encoding categorical features...")
    df, encoders = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numeric features (helps logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE (train set only)
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    print("Training models...")
    models = train_models(X_train_bal, y_train_bal)

    print("Evaluating models...")
    results = evaluate_models(models, X_test_scaled, y_test)

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['auc']:.4f})")

    print("Generating SHAP explainability plot...")
    explain_best_model(best_model, X_train_bal, X_test_scaled, X.columns.tolist())

    # Save artifacts for the Streamlit app
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"\nModel artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
