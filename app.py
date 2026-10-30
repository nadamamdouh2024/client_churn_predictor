
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Churn Predictor", page_icon="📞")

st.title("📞 Customer Churn Predictor")
st.caption("Predict whether a customer is likely to leave the company.")

if not (MODELS_DIR / "churn_model.joblib").exists():
    st.error("No trained model found. Please run: python train.py")
    st.stop()

model = joblib.load(MODELS_DIR / "churn_model.joblib")
best_name = (MODELS_DIR / "best_model_name.txt").read_text()

tab1, tab2 = st.tabs(["Customer Prediction", "Models Comparison"])

# ---------------- Tab 1: Prediction Form ----------------
with tab1:
    st.write(f"**Model in use:** {best_name}")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col2:
        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        )

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(monthly * tenure)
    )

    if st.button("Predict Churn", use_container_width=True):

        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prob = model.predict_proba(row)[0][1]

        st.metric("Churn Probability", f"{prob:.1%}")
        st.progress(float(prob))

        if prob >= 0.5:
            st.error(
                "⚠️ High Churn Risk! "
                "This customer is likely to leave. "
                "Consider offering a retention incentive."
            )
        else:
            st.success(
                "✅ Low Churn Risk! "
                "This customer is likely to stay."
            )


# ---------------- Tab 2: Models Comparison ----------------
with tab2:

    results = pd.read_csv(MODELS_DIR / "results.csv")

    st.subheader("Model Performance Comparison")

    st.dataframe(
        results.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Score vs ROC-AUC")

    st.bar_chart(
        results.set_index("Model")[["Accuracy", "ROC_AUC"]]
    )


