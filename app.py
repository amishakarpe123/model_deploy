import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Diagnostic AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR ATTRACTIVE UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        transform: scale(1.01);
        border: none;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #2E7D32;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load Assets
lottie_medical = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v83.json")
model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    if lottie_medical:
        st_lottie(lottie_medical, height=150, key="sidebar_anim")
    else:
        st.title("🏥 Health AI")
    
    st.info("This system uses a K-Nearest Neighbors (KNN) model to analyze patient health metrics.")
    st.divider()
    st.markdown("### How to use:")
    st.write("1. Enter patient vitals in the form.\n2. Click 'Run Diagnostic'.\n3. Review the AI prediction result.")

# --- MAIN INTERFACE ---
st.title("Medical Diagnostic Prediction Engine")
st.markdown("##### Fill in the biological parameters below for a real-time analysis.")

if model is None:
    st.error("⚠️ Error: 'model.pkl' not found. Please upload the file to the app directory.")
    st.stop()

# --- INPUT FORM ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🩸 Blood Metrics")
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        blood_p = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

    with col2:
        st.markdown("#### 👤 Physical Metrics")
        age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
        bmi = st.number_input("BMI (Weight in kg/(m)^2)", min_value=0.0, max_value=70.0, value=25.0)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("✨ RUN DIAGNOSTIC ANALYSIS")

# --- PREDICTION LOGIC ---
if submit:
    # Prepare the data in the exact order the model was trained on
    # Order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    features = np.array([[pregnancies, glucose, blood_p, skin_thickness, insulin, bmi, dpf, age]])
    
    with st.spinner("Analyzing metrics against training data..."):
        prediction = model.predict(features)
        
    st.divider()
    
    # Attractive Result Display
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction[0] == 1:
            st.error("### Result: POSITIVE")
        else:
            st.success("### Result: NEGATIVE")
    
    with res_col2:
        if prediction[0] == 1:
            st.warning("The model predicts a high correlation with diabetic indicators. Further clinical testing is recommended.")
        else:
            st.info("The patient metrics provided are consistent with low-risk health profiles in the current model.")

    st.balloons()
