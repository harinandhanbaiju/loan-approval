
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LoanGuard AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_DIR / "loan_model.pkl")
    education_encoder = joblib.load(
        MODEL_DIR / "education_encoder.pkl"
    )
    employment_encoder = joblib.load(
        MODEL_DIR / "employment_encoder.pkl"
    )
    target_encoder = joblib.load(
        MODEL_DIR / "target_encoder.pkl"
    )
    return model, education_encoder, employment_encoder, target_encoder

try:
    model, education_encoder, employment_encoder, target_encoder = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Check that all four .pkl files exist inside the model folder.")
    st.stop()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
    color: #17243b;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.topbar {
    background: #101f38;
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.brand {
    font-size: 24px;
    font-weight: 800;
}
.muted {
    color: #8ea2bf;
    font-size: 12px;
}
.panel {
    background: white;
    padding: 22px;
    border: 1px solid #dfe7f0;
    border-radius: 14px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 12px;
    color: #142641;
}
.result {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 10px 0 18px;
}
.good {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857;
}
.bad {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #be123c;
}
.result-title {
    font-size: 25px;
    font-weight: 800;
}
div.stButton > button[kind="primary"] {
    background: #059669;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 750;
    min-height: 48px;
}
div.stButton > button[kind="primary"]:hover {
    background: #047857;
}
div[data-testid="stMetric"] {
    background: #f8fafc;
    padding: 13px;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="topbar">
    <div class="brand">✓ LoanGuard AI</div>
    <div class="muted">
        INSTITUTIONAL RISK • LOAN ASSESSMENT & UNDERWRITING
    </div>
</div>
""", unsafe_allow_html=True)

header1, header2, header3 = st.columns([2, 2, 2])
with header1:
    st.caption("APPLICATION")
    st.write("**LN-2026-001**")
with header2:
    st.caption("INFERENCE STATUS")
    st.success("Model loaded")
with header3:
    st.caption("MODEL")
    st.write("**Random Forest**")

st.title("Loan Application Assessment & Underwriting")
st.caption(
    "Enter applicant details and run the trained model to obtain a prediction."
)

# ---------------- FORM ----------------
with st.form("loan_application"):
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">01 · Applicant Profile</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            dependents = st.number_input(
                "Number of Dependents", 0, 20, 2
            )
            education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"]
            )
        with c2:
            self_employed = st.selectbox(
                "Self Employed", ["Yes", "No"]
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">02 · Loan & Credit Details</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Annual Income (₹)", min_value=0,
                value=5000000, step=100000
            )
            loan_amount = st.number_input(
                "Loan Amount (₹)", min_value=0,
                value=1000000, step=100000
            )
        with c2:
            loan_term = st.number_input(
                "Loan Term (Years)", 1, 50, 10
            )
            cibil = st.number_input(
                "CIBIL Score", 300, 900, 700
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">03 · Asset Valuation</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            residential = st.number_input(
                "Residential Assets (₹)", min_value=0,
                value=1000000, step=100000
            )
            luxury = st.number_input(
                "Luxury Assets (₹)", min_value=0,
                value=500000, step=100000
            )
        with c2:
            commercial = st.number_input(
                "Commercial Assets (₹)", min_value=0,
                value=500000, step=100000
            )
            bank_assets = st.number_input(
                "Bank Assets (₹)", min_value=0,
                value=500000, step=100000
            )

        total_assets = (
            residential + commercial + luxury + bank_assets
        )
        st.metric("Total Declared Assets", f"₹{total_assets:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">NEURAL RISK ENGINE</div>',
            unsafe_allow_html=True
        )
        st.subheader("Prediction Result")
        st.write(
            "Submit the application to see the model's prediction "
            "and its confidence."
        )

        st.metric("CIBIL Score", f"{cibil}/900")
        ratio = loan_amount / income if income > 0 else None
        st.metric(
            "Loan-to-Income Ratio",
            f"{ratio:.2f}x" if ratio is not None else "N/A"
        )

        coverage = (
            total_assets / loan_amount if loan_amount > 0 else None
        )
        st.metric(
            "Asset-to-Loan Ratio",
            f"{coverage:.2f}x" if coverage is not None else "N/A"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "✨  PREDICT LOAN STATUS",
            type="primary",
            use_container_width=True
        )

# ---------------- PREDICTION ----------------
if submitted:
    try:
        education_value = education_encoder.transform([education])[0]
        employment_value = employment_encoder.transform([self_employed])[0]

        # Column names and order must match train_model.py
        input_data = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": education_value,
            "self_employed": employment_value,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": residential,
            "commercial_assets_value": commercial,
            "luxury_assets_value": luxury,
            "bank_asset_value": bank_assets
        }])

        # Reorder features to match the trained model
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[list(model.feature_names_in_)]

        prediction = model.predict(input_data)[0]
        result = str(
            target_encoder.inverse_transform([prediction])[0]
        ).strip()

        st.markdown("---")
        st.subheader("Assessment Outcome")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            class_names = [
                str(x).strip() for x in target_encoder.classes_
            ]
            predicted_class = result
            predicted_index = class_names.index(predicted_class)
            confidence = float(probabilities[predicted_index]) * 100
        else:
            confidence = None

        if result.lower() == "approved":
            st.markdown(
                f'<div class="result good">'
                f'<div class="result-title">✓ LOAN APPROVED</div>'
                f'<div>{confidence:.1f}% model confidence</div>'
                f'</div>' if confidence is not None else
                '<div class="result good">'
                '<div class="result-title">✓ LOAN APPROVED</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result bad">'
                f'<div class="result-title">✕ LOAN REJECTED</div>'
                f'<div>{confidence:.1f}% model confidence</div>'
                f'</div>' if confidence is not None else
                '<div class="result bad">'
                '<div class="result-title">✕ LOAN REJECTED</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.caption(
            "This is a machine-learning prediction, not a guarantee "
            "of an actual lending decision. Model confidence is not "
            "necessarily the real-world probability of repayment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check the saved encoders, feature names, and training "
            "column order in train_model.py."
        )

st.divider()
st.caption(
    "LoanGuard AI · Machine Learning Decision Support System"
)
