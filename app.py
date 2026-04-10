import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="Health Prediction Engine", page_icon="🏥", layout="wide")

# Function to load animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a medical-themed animation
lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v83.json")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar and Header
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🏥 Diagnostics Prediction App")
        st.markdown("### Intelligent Analysis Powered by Machine Learning")
        st.write("Complete the patient profile below to generate a diagnostic prediction.")
    with col2:
        st_lottie(lottie_health, height=200, key="health_anim")

st.divider()

# Input Form
with st.form("prediction_form"):
    st.subheader("📋 Patient Biological Metrics")
    
    # Feature inputs based on your model's required fields
    c1, c2, c3 = st.columns(3)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    
    with c2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
        
    with c3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    submit = st.form_submit_button("🚀 Generate Prediction")

# Prediction Logic
if submit:
    try:
        # Load the user-provided model.pkl
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Format inputs for the model
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, dpf, age]])
        
        with st.spinner('Analyzing patient data...'):
            prediction = model.predict(input_data)
            
        # UI Feedback
        st.balloons()
        if prediction[0] == 1:
            st.error(f"### Result: Positive Indication (Class {prediction[0]})")
            st.info("The model suggests a high probability of diabetes based on these metrics.")
        else:
            st.success(f"### Result: Negative Indication (Class {prediction[0]})")
            st.info("The metrics provided fall within the low-risk range according to the model.")
            
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the file is in the same directory as this script.")
