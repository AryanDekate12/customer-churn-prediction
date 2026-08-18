"""
Streamlit App - Customer Churn Predictor
==========================================
Run with: streamlit run app.py

Requires model artifacts in models/ (produced by src/churn_pipeline.py):
    - best_model.pkl
    - scaler.pkl
    - encoders.pkl
    - feature_names.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "models"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, encoders, feature_names


def main():
    st.title("📉 Customer Churn Predictor")
    st.write(
        "Enter a customer's details to estimate their probability of churning, "
        "and see which factors are driving the prediction."
    )

    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        st.warning(
            "No trained model found. Run `python src/churn_pipeline.py` first "
            "to train the model and generate the required artifacts."
        )
        return

    model, scaler, encoders, feature_names = load_artifacts()

    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict Churn Risk", type="primary"):
        # Build a raw input row matching training schema.
        # NOTE: fill remaining categorical fields with sensible defaults —
        # adapt this section to match your exact training feature set.
        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet_service,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure_group": "0-1yr",
            "avg_monthly_spend": total_charges / max(tenure, 1),
        }

        input_df = pd.DataFrame([raw_input])

        # Apply saved label encoders to categorical columns
        for col, encoder in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    # Unseen category fallback
                    input_df[col] = 0

        # Align column order with training features
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        st.subheader("Result")
        risk_pct = prob * 100

        if pred == 1:
            st.error(f"⚠️ High churn risk: {risk_pct:.1f}% probability")
        else:
            st.success(f"✅ Low churn risk: {risk_pct:.1f}% probability")

        st.progress(min(int(risk_pct), 100))

        st.caption(
            "This is a demo model trained on the Telco Customer Churn dataset. "
            "Predictions are for illustration, not real business decisions."
        )


if __name__ == "__main__":
    main()
