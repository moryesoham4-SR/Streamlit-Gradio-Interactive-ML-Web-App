import os
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Canonical feature list expected by the trained pipeline
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang"
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# ---------------------------------------------------------
# Cached Model Loader (Rule 3: Load once with @st.cache_resource)
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """
    Loads and caches the model from disk.
    Cached so it does not reload on every user interaction.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found.")
    
    loaded = joblib.load(MODEL_PATH)
    # Support both dictionary bundle and direct estimator formats
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], loaded.get("features", FEATURES)
    return loaded, FEATURES

# ---------------------------------------------------------
# Header & Context
# ---------------------------------------------------------
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown(
    """
    An interactive machine learning application that predicts the estimated risk of 
    cardiovascular disease based on clinical biomarkers and patient vital signs.
    """
)

# ---------------------------------------------------------
# Safe Model Loading with Friendly Error Handling (Rule 7)
# ---------------------------------------------------------
try:
    model, model_features = load_trained_model()
except FileNotFoundError:
    st.error(
        "⚠️ **Model file is missing.**\n\n"
        "Please ensure `model.joblib` exists in the application root directory. "
        "If you haven't trained the model yet, run `python train_model.py` in your terminal first."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ **Failed to load model:** {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Presets and Quick Test Samples
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 Quick Presets")
    st.caption("Load pre-configured sample profiles for quick testing:")

    preset = st.radio(
        "Select Profile:",
        ["Custom Inputs", "Healthy Young Adult", "Middle-Aged Active", "High-Risk Patient"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 About the Model")
    st.info(
        "• **Algorithm:** Random Forest Classifier with Feature Scaling\n"
        "• **Framework:** Scikit-Learn Pipeline & Joblib\n"
        "• **Inference:** Cached via `@st.cache_resource`\n"
        "• **Output:** Real-time probability estimation"
    )

# Assign default values based on chosen preset
if preset == "Healthy Young Adult":
    default_age, default_sex, default_cp = 28, "Female", "Typical Angina (0)"
    default_trestbps, default_chol, default_fbs = 112, 175, "No (<= 120 mg/dl)"
    default_restecg, default_thalach, default_exang = "Normal (0)", 175, "No"
elif preset == "Middle-Aged Active":
    default_age, default_sex, default_cp = 48, "Male", "Atypical Angina (1)"
    default_trestbps, default_chol, default_fbs = 128, 210, "No (<= 120 mg/dl)"
    default_restecg, default_thalach, default_exang = "Normal (0)", 150, "No"
elif preset == "High-Risk Patient":
    default_age, default_sex, default_cp = 65, "Male", "Asymptomatic (3)"
    default_trestbps, default_chol, default_fbs = 165, 310, "Yes (> 120 mg/dl)"
    default_restecg, default_thalach, default_exang = "ST-T Wave Abnormality (1)", 110, "Yes"
else: # Custom Inputs
    default_age, default_sex, default_cp = 52, "Male", "Non-anginal Pain (2)"
    default_trestbps, default_chol, default_fbs = 130, 240, "No (<= 120 mg/dl)"
    default_restecg, default_thalach, default_exang = "Normal (0)", 148, "No"

# ---------------------------------------------------------
# Input Section with Sensible Limits & Clear Layout (Rule 4)
# ---------------------------------------------------------
st.subheader("Patient Clinical Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "Age (years)",
        min_value=18,
        max_value=90,
        value=default_age,
        step=1,
        help="Patient age in years."
    )
    sex_label = st.radio(
        "Biological Sex",
        options=["Male", "Female"],
        index=0 if default_sex == "Male" else 1,
        horizontal=True,
        help="Sex at birth."
    )
    cp_label = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina (0)",
            "Atypical Angina (1)",
            "Non-anginal Pain (2)",
            "Asymptomatic (3)"
        ],
        index=[
            "Typical Angina (0)",
            "Atypical Angina (1)",
            "Non-anginal Pain (2)",
            "Asymptomatic (3)"
        ].index(default_cp),
        help="Type of chest pain reported by the patient."
    )

with col2:
    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        min_value=90,
        max_value=210,
        value=default_trestbps,
        step=1,
        help="Resting blood pressure measured in mm Hg upon hospital admission."
    )
    chol = st.slider(
        "Serum Cholesterol (mg/dl)",
        min_value=120,
        max_value=500,
        value=default_chol,
        step=5,
        help="Serum cholesterol level in mg/dl."
    )
    fbs_label = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
        index=0 if "No" in default_fbs else 1,
        horizontal=True,
        help="Indicates whether fasting blood sugar exceeds 120 mg/dl."
    )

with col3:
    restecg_label = st.selectbox(
        "Resting ECG Results",
        options=[
            "Normal (0)",
            "ST-T Wave Abnormality (1)",
            "Left Ventricular Hypertrophy (2)"
        ],
        index=[
            "Normal (0)",
            "ST-T Wave Abnormality (1)",
            "Left Ventricular Hypertrophy (2)"
        ].index(default_restecg),
        help="Electrocardiographic diagnostic readout."
    )
    thalach = st.slider(
        "Max Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=default_thalach,
        step=1,
        help="Maximum heart rate recorded during exercise stress test."
    )
    exang_label = st.radio(
        "Exercise-Induced Angina",
        options=["No", "Yes"],
        index=0 if default_exang == "No" else 1,
        horizontal=True,
        help="Did the patient experience angina during exercise?"
    )

# ---------------------------------------------------------
# Feature Transformation
# ---------------------------------------------------------
# Map user-friendly labels to numerical values
sex_val = 1 if sex_label == "Male" else 0
cp_val = int(cp_label.split("(")[-1].replace(")", ""))
fbs_val = 1 if "Yes" in fbs_label else 0
restecg_val = int(restecg_label.split("(")[-1].replace(")", ""))
exang_val = 1 if exang_label == "Yes" else 0

input_data = {
    "age": age,
    "sex": sex_val,
    "cp": cp_val,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_val,
    "restecg": restecg_val,
    "thalach": thalach,
    "exang": exang_val
}

# ---------------------------------------------------------
# Prediction Button & Results Display (Rule 5 & 6)
# ---------------------------------------------------------
st.markdown("---")
predict_btn = st.button("🔮 Calculate Risk Prediction", type="primary", use_container_width=True)

if predict_btn:
    try:
        # Build one-row DataFrame in the exact feature column order
        row = pd.DataFrame([input_data])[model_features]

        # Predict probability
        probabilities = model.predict_proba(row)[0]
        # Class 1 is Heart Disease Risk
        disease_probability = probabilities[1]
        predicted_class = model.predict(row)[0]

        st.subheader("Prediction Results")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])

        with res_col1:
            st.metric(
                label="Estimated Risk of Heart Disease",
                value=f"{disease_probability:.1%}",
                delta=f"Risk: {'Elevated' if disease_probability >= 0.5 else 'Low'}",
                delta_color="inverse" if disease_probability >= 0.5 else "normal"
            )

        with res_col2:
            verdict = "High Risk" if predicted_class == 1 else "Low Risk"
            st.metric(
                label="Model Classification",
                value=verdict,
                delta="Positive Detection" if predicted_class == 1 else "Negative Detection",
                delta_color="inverse" if predicted_class == 1 else "normal"
            )

        with res_col3:
            st.write("**Risk Gauge:**")
            st.progress(float(disease_probability))
            if disease_probability >= 0.65:
                st.error("⚠️ The model estimates high probability of cardiovascular risk. Clinical follow-up recommended.")
            elif disease_probability >= 0.40:
                st.warning("⚡ The model estimates moderate risk factors present. Lifestyle and preventative checks advised.")
            else:
                st.success("✅ The model estimates low cardiovascular disease risk based on the input biomarkers.")

        with st.expander("🔍 View Processed Feature Vector"):
            st.dataframe(row, use_container_width=True)

    except Exception as err:
        st.error(f"An unexpected error occurred during prediction: {err}")

# ---------------------------------------------------------
# Educational Caption & Disclaimer (Rule 6)
# ---------------------------------------------------------
st.caption(
    "ℹ️ **Educational Disclaimer:** This web application is a machine learning learning exercise developed for "
    "the InternCircle Data Science & AI Program. It uses a Random Forest classifier trained on clinical cardiovascular "
    "biomarkers. This tool is for educational and portfolio demonstration purposes only and does NOT constitute medical advice."
)
