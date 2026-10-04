from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# Works no matter where Streamlit is launched from
_HERE = Path(__file__).resolve().parent
MODEL_PATH = next(
    (p for p in [
        _HERE.parent / "models" / "customer_churn_xgboost.pkl",
        _HERE.parent / "customer_churn_xgboost.pkl",
        _HERE / "customer_churn_xgboost.pkl",
    ] if p.exists()),
    _HERE.parent / "notebooks" / "customer_churn_xgboost.pkl",
)

# Column order/names exactly as used when training the pipeline (X in the notebook)
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges", "TenureGroup",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
    )
    return df


def risk_label(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.7):
        return "🔴 High"
    if p >= threshold:
        return "🟠 Medium"
    return "🟢 Low"


model = load_model()

st.title("📉 Customer Churn Predictor")
st.caption("XGBoost pipeline trained on the Telco Customer Churn dataset")

threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.95, 0.32, 0.01,
    help="Customers with churn probability >= threshold are flagged as churners. "
         "Lower it to catch more churners (higher recall, lower precision). "
         "Default 0.32 is the threshold tuned on the validation set.",
)

tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Batch (CSV)", "ℹ️ About project"])

# ---------------------------------------------------------------- single
with tab_single:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_net = internet == "No"
        opts = ["No internet service"] if no_net else ["No", "Yes"]
        security = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
        protection = st.selectbox("Device protection", opts)
        support = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)

    with c3:
        st.subheader("Charges")
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, 0.5)
        total = st.number_input("Total charges", 0.0, 10000.0, float(round(monthly * tenure, 2)), 1.0)

    st.divider()
    predict = st.button("🔮 Predict churn", type="primary", use_container_width=True)

    if predict:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        row = add_tenure_group(row)[FEATURES]
        prob = float(model.predict_proba(row)[0, 1])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Prediction", "Will churn" if prob >= threshold else "Will stay")
        m3.metric("Risk level", risk_label(prob, threshold))
        st.progress(min(prob, 1.0))

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset (`customerID` optional, `Churn` ignored).")
    file = st.file_uploader("CSV file", type="csv")

    if file:
        data = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in data.columns and c != "TenureGroup"]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
            data = data[data["tenure"] > 0].dropna(subset=["TotalCharges"]).reset_index(drop=True)
            X = add_tenure_group(data)[FEATURES]

            data["ChurnProbability"] = model.predict_proba(X)[:, 1]
            data["Prediction"] = (data["ChurnProbability"] >= threshold).map({True: "Churn", False: "Stay"})
            data["Risk"] = data["ChurnProbability"].apply(lambda p: risk_label(p, threshold))

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", len(data))
            k2.metric("Predicted churners", int((data["Prediction"] == "Churn").sum()))
            k3.metric("Avg churn probability", f"{data['ChurnProbability'].mean():.1%}")

            st.dataframe(
                data.sort_values("ChurnProbability", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download predictions",
                data.to_csv(index=False).encode(),
                "churn_predictions.csv",
                "text/csv",
            )

# ---------------------------------------------------------------- about
with tab_about:
    st.header("About this project")
    st.write(
        "Customer churn is when a subscriber leaves the company. Losing a customer costs more than "
        "keeping one, so this project predicts which telecom customers are likely to churn, "
        "letting the business act early with offers or support."
    )

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Dataset")
        st.markdown(
            "- **Source:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)\n"
            "- **Target:** `Churn` (Yes / No)\n"
            "- **Features:** demographics, services subscribed, contract, billing and charges\n"
            "- **Cleaning:** 11 rows with blank `TotalCharges` (tenure = 0, new customers) were dropped; "
            "`TotalCharges` converted to numeric\n"
            "- **Feature engineering:** `TenureGroup` (0-12, 13-24, 25-48, 49-72 months)"
        )
        st.subheader("Preprocessing")
        st.markdown(
            "- One-hot encoding for categorical features\n"
            "- Stratified 80/20 train-test split\n"
            "- Class imbalance handled with class weights and SMOTENC experiments"
        )
    with a2:
        st.subheader("Models compared")
        st.markdown(
            "- Logistic Regression (with and without SMOTENC)\n"
            "- Decision Tree (GridSearchCV)\n"
            "- Random Forest (GridSearchCV)\n"
            "- **XGBoost (RandomizedSearchCV)** ← selected, ROC-AUC ≈ 0.84"
        )
        st.subheader("Threshold tuning")
        st.write(
            "Missing a churner is costlier than a false alarm, so the decision threshold was tuned on a "
            "validation set using the precision-recall curve to improve recall. Use the sidebar slider "
            "to explore this trade-off."
        )

    st.subheader("Key findings from EDA")
    st.markdown(
        "- **Month-to-month contracts** churn far more than one- or two-year contracts\n"
        "- **Low tenure** customers are most likely to leave\n"
        "- **Higher monthly charges** are associated with churn\n"
        "- Lack of **Tech Support** / **Online Security** and **Fiber optic** internet relate to higher churn\n"
        "- Gender and phone service have little effect"
    )

    st.subheader("Tech stack")
    st.write("Python, pandas, scikit-learn, imbalanced-learn, XGBoost, matplotlib, Streamlit")