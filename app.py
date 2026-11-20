import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave the service "
    "using a Logistic Regression model."
)

# --------------------------------------------------
# Expected dataset columns
# --------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]


# --------------------------------------------------
# Data preparation and model training
# --------------------------------------------------
@st.cache_resource
def train_model(file_bytes):
    from io import BytesIO

    data = pd.read_csv(BytesIO(file_bytes))

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()

    # Convert numeric columns correctly.
    # Invalid/blank TotalCharges values become NaN and are imputed below.
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Normalize target labels
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET].isin(["Yes", "No"])].copy()
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})

    if data.empty or data[TARGET].nunique() < 2:
        raise ValueError("The dataset must contain both Yes and No churn records.")

    X = data[FEATURES]
    y = data[TARGET]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    return model, metrics, len(data)


# --------------------------------------------------
# Dataset upload
# --------------------------------------------------
st.sidebar.header("⚙️ Model Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Telco Customer Churn CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload the Telco Customer Churn CSV from the sidebar to train the model. "
        "For deployment, keep the CSV in the project folder or upload it here."
    )
    st.stop()

try:
    model, metrics, row_count = train_model(uploaded_file.getvalue())
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

st.sidebar.success(f"Model trained on {row_count:,} records")

# --------------------------------------------------
# Model performance
# --------------------------------------------------
st.subheader("📈 Model Performance")

metric_columns = st.columns(5)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name, f"{value:.2%}")

st.caption(
    "Metrics are calculated on a held-out 20% test set. "
    "They may differ from the original notebook because TotalCharges "
    "is converted to numeric and missing values are imputed."
)

st.divider()

# --------------------------------------------------
# Customer details form
# --------------------------------------------------
st.subheader("🧑‍💼 Enter Customer Information")
st.write("Fill in the customer's details and click Predict Churn.")

with st.form("customer_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12, step=1
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=1000.0,
            value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=100000.0,
            value=float(tenure * monthly_charges), step=10.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction result
# --------------------------------------------------
if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])[FEATURES]

    prediction = int(model.predict(customer)[0])
    churn_probability = float(model.predict_proba(customer)[0][1])

    st.divider()
    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ High Risk: This customer is predicted to churn.")
        else:
            st.success("✅ Low Risk: This customer is predicted not to churn.")

    with result_col2:
        st.metric("Churn Probability", f"{churn_probability:.2%}")

    st.progress(churn_probability)
    st.caption(
        "This is a model-based estimate, not a guarantee of customer behavior. "
        "Use it as decision support rather than as the sole basis for action."
    )

st.divider()
st.caption("Customer Churn Prediction | Built with Python, Pandas, Scikit-learn & Streamlit")
