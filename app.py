from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root{--blue:#168cff;--cyan:#12c8ff;--text:#fff;--muted:#bfd6f3}

/* Background pure page ka */
.stApp {
    background: radial-gradient(circle at 85% 8%, rgba(23,142,255,.22), transparent 28%),
                radial-gradient(circle at 8% 75%, rgba(0,197,255,.14), transparent 30%),
                linear-gradient(135deg, #04132f 0%, #06204a 48%, #0a3470 100%) !important;
    color: #fff !important;
}

.block-container {
    max-width: 1080px;
    padding: 2.2rem 2rem 3rem;
}
#MainMenu, footer {visibility: hidden;} 
header {background: transparent !important;}

/* =========================================================
   CARDS STYLING (Hero Card, Car Info Card, Technical Details Card)
   ========================================================= */
.custom-card {
    position: relative !important;
    overflow: hidden !important;
    background: radial-gradient(circle at 88% 15%, rgba(42, 164, 255, 0.38), transparent 28%),
                linear-gradient(120deg, #071a3d 0%, #0b3675 52%, #126de0 100%) !important;
    color: #fff !important;
    border-radius: 20px !important;
    padding: 2rem 2.3rem 1.8rem !important;
    margin-bottom: 24px !important;
    border: 1.5px solid rgba(46, 157, 255, 0.75) !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28), 0 0 28px rgba(18, 140, 255, 0.15) !important;
}

.custom-card::before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -75px;
    top: -115px;
    border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: 0 0 0 22px rgba(255, 255, 255, 0.025), 0 0 0 45px rgba(255, 255, 255, 0.014);
}

.custom-card h1 {
    position: relative;
    z-index: 1;
    margin: 0;
    font-size: clamp(2.15rem, 4.2vw, 3rem);
    line-height: 1.08;
    letter-spacing: -0.035em;
    font-weight: 850;
    color: #fff !important;
}

.custom-card p {
    position: relative;
    z-index: 1;
    margin: 0.75rem 0 0;
    max-width: 800px;
    color: #d3e7ff !important;
    font-size: 1rem;
    line-height: 1.55;
}

/* Titles inside section cards */
.section-title {
    position: relative;
    z-index: 1;
    display: flex !important;
    align-items: center !important;
    gap: 0.65rem !important;
    color: #ffffff !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.35rem 0 !important;
}

.section-title::before {
    content: "" !important;
    width: 5px !important;
    height: 24px !important;
    border-radius: 8px !important;
    background: linear-gradient(180deg, #16a7ff, #00d2ff) !important;
    box-shadow: 0 0 12px rgba(22, 167, 255, 0.45) !important;
}

.section-subtitle {
    position: relative;
    z-index: 1;
    color: #bcd5f3 !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
    margin: 0 0 1.25rem 0 !important;
}

/* Input & Select fields */
label {
    font-weight: 650 !important;
    color: #e5f1ff !important;
}

div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div, 
input {
    min-height: 46px !important;
    border-radius: 11px !important;
    border: 1px solid rgba(105, 157, 216, 0.45) !important;
    background: #092652 !important;
    color: #fff !important;
}

div[data-baseweb="select"] > div:hover, input:hover {
    border-color: #27a5ff !important;
}

div[data-baseweb="select"] * {
    color: #fff !important;
}

.field-caption {
    color: #8fb0d5 !important;
    font-size: 0.75rem !important;
    margin-top: 0.25rem;
}

/* Form remove border */
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* Button */
button[kind="primaryFormSubmit"], div[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    min-height: 3.35rem !important;
    border-radius: 13px !important;
    border: 1px solid rgba(77, 202, 255, 0.65) !important;
    background: linear-gradient(100deg, #087cff 0%, #119bea 52%, #08c5ef 100%) !important;
    color: #fff !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 25px rgba(0, 132, 255, 0.27) !important;
    cursor: pointer !important;
}

button[kind="primaryFormSubmit"]:hover, div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    filter: brightness(1.08);
    box-shadow: 0 14px 30px rgba(0, 157, 255, 0.36) !important;
}

/* Result Box */
.result {
    position: relative;
    overflow: hidden;
    background: radial-gradient(circle at 92% 8%, rgba(45, 164, 255, 0.22), transparent 28%),
                linear-gradient(135deg, #092653 0%, #0d3974 100%);
    border: 1.5px solid rgba(55, 175, 255, 0.65);
    border-radius: 18px;
    padding: 1.45rem 1.7rem;
    margin-top: 1.15rem;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.20);
}
.result-label {
    color: #b8d7f7;
    font-size: .77rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .10em;
}
.result-price {
    color: #fff;
    font-size: clamp(2.1rem, 4vw, 3rem);
    line-height: 1.04;
    font-weight: 880;
    margin: .4rem 0 .25rem;
    letter-spacing: -.04em;
}
.result-secondary {
    color: #b9d2ed;
    font-size: .9rem;
}
.footer {
    text-align: center;
    color: #8ca9ca;
    font-size: 0.78rem;
    margin-top: 1.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = [
        "car_price_model.pkl",
        "encoders.pkl",
        "cat_cols.pkl",
        "feature_columns.pkl",
        "numeric_ranges.pkl",
    ]
    missing = [name for name in required if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    return (
        joblib.load(BASE_DIR / "car_price_model.pkl"),
        joblib.load(BASE_DIR / "encoders.pkl"),
        joblib.load(BASE_DIR / "cat_cols.pkl"),
        joblib.load(BASE_DIR / "feature_columns.pkl"),
        joblib.load(BASE_DIR / "numeric_ranges.pkl"),
    )


try:
    model, encoders, cat_cols, feature_columns, numeric_ranges = load_artifacts()
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.exception(exc)
    st.stop()

# ================== CARD 1: TOP HERO CARD ==================
st.markdown(
    """
<div class="custom-card">
  <h1>🚗 Car Price Predictor</h1>
  <p>Enter the car details below and get an estimated selling price in Indian Rupees.</p>
</div>
""",
    unsafe_allow_html=True,
)

user_input = {}
errors = []
WIDGET_VERSION = "v5"

with st.form("car_form", clear_on_submit=False):

    # ================== CARD 2: CAR INFORMATION CARD ==================
    card2 = st.empty()
    with card2.container():
        st.markdown(
            """
            <div class="custom-card">
              <div class="section-title">Car Information</div>
              <div class="section-subtitle">Choose the vehicle details. The model field is searchable.</div>
            """,
            unsafe_allow_html=True,
        )

        categorical = [c for c in feature_columns if c in cat_cols]
        categorical_order = [
            c for c in ["Brand", "Body", "Engine Type", "Registration", "Model"]
            if c in categorical
        ] + [c for c in categorical if c not in {"Brand", "Body", "Engine Type", "Registration", "Model"}]

        ccols = st.columns(2, gap="large")
        for i, col in enumerate(categorical_order):
            with ccols[i % 2]:
                classes = [str(x) for x in encoders[col].classes_]
                placeholder = {
                    "Brand": "Select a brand",
                    "Body": "Select body type",
                    "Engine Type": "Select engine type",
                    "Registration": "Select registration status",
                    "Model": "Search or select car model",
                }.get(col, f"Select {col.lower()}")

                empty_label = f"— {placeholder} —"
                options = [empty_label] + classes
                choice = st.selectbox(
                    col,
                    options=options,
                    index=0,
                    key=f"{WIDGET_VERSION}_select_{col}",
                    help=("Type to search models." if col == "Model" else None),
                )

                if choice == empty_label:
                    user_input[col] = None
                else:
                    user_input[col] = encoders[col].transform([choice])[0]

        if len(categorical_order) % 2 == 1:
            st.caption(" ")

        st.markdown("</div>", unsafe_allow_html=True)

    # ================== CARD 3: TECHNICAL DETAILS CARD ==================
    card3 = st.empty()
    with card3.container():
        st.markdown(
            """
            <div class="custom-card">
              <div class="section-title">Technical Details</div>
              <div class="section-subtitle">Enter values using the examples below. Each value is checked against the model training range.</div>
            """,
            unsafe_allow_html=True,
        )

        numeric = [c for c in feature_columns if c not in cat_cols]
        ncols = st.columns(3, gap="large")
        numeric_labels = {
            "Year": ("Manufacturing Year", "e.g. 2014"),
            "Mileage": ("Mileage (thousand km)", "e.g. 120"),
            "EngineV": ("Engine Volume (L)", "e.g. 2.0"),
        }

        for i, col in enumerate(numeric):
            with ncols[i % 3]:
                lo, hi = numeric_ranges.get(col, (0.0, 100.0))
                label, placeholder = numeric_labels.get(col, (col, "Enter a value"))
                raw = st.text_input(
                    label,
                    placeholder=placeholder,
                    value="",
                    key=f"{WIDGET_VERSION}_input_{col}",
                )

                if raw.strip() == "":
                    user_input[col] = None
                else:
                    try:
                        value = float(raw.strip())
                        if col == "Year" and value.is_integer():
                            value = int(value)
                        if value < lo or value > hi:
                            errors.append(f"{label} must be between {lo:g} and {hi:g}.")
                        user_input[col] = value
                    except ValueError:
                        errors.append(f"{label} must be a valid number.")
                        user_input[col] = None

                st.markdown(
                    f'<div class="field-caption">Allowed range: {lo:g} – {hi:g}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("  Predict Car Price", type="primary", use_container_width=True)

if submitted:
    missing_fields = [c for c, v in user_input.items() if v is None]
    if missing_fields:
        errors.append("Please complete all fields before predicting.")

    if errors:
        for msg in dict.fromkeys(errors):
            st.warning(msg)
    else:
        try:
            input_df = pd.DataFrame([user_input], columns=feature_columns)
            log_price_pred = float(model.predict(input_df)[0])
            price_pred = float(np.exp(log_price_pred))

            USD_TO_INR = 90.0
            price_pred_inr = price_pred * USD_TO_INR
            lakh = price_pred_inr / 100000

            st.markdown(
                f"""
<div class="result">
  <div class="result-label">Estimated Car Price</div>
  <div class="result-price">₹{price_pred_inr:,.0f}</div>
  <div class="result-secondary">Approximately ₹{lakh:,.2f} lakh • ${price_pred:,.0f} USD</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.caption(
                "ML estimate only. Actual market value can vary with condition, location, ownership history and other factors."
            )
        except Exception as exc:
            st.error("Prediction failed. Please check your inputs and try again.")
            st.exception(exc)

st.markdown(
    '<div class="footer">Car Price Predictor • Machine Learning Regression Project</div>',
    unsafe_allow_html=True,
)