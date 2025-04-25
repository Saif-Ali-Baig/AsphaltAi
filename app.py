import streamlit as st
import numpy as np
from load_model import load_model
from utils import preprocess_input

# Load models
gb_model = load_model('gradient_boosting_model.pkl', model_type='sklearn')
xgb_model = load_model('xgboost_model.pkl', model_type='xgboost')

st.title("🛣️ Asphalt AI - Road Repair Prediction")

st.header("Enter Road & Environment Details")
last_laid_year = st.number_input("Last Laid Year", min_value=1950, max_value=2025, value=2005)
last_repair_year = st.number_input("Last Repair Year", min_value=1950, max_value=2025, value=2015)

material = st.selectbox("Material Type", ['asphalt', 'concrete', 'gravel'])
weather = st.selectbox("Weather Condition", ['hot', 'humid', 'rainy'])
usage_type = st.selectbox("Usage Type", ['urban', 'rural', 'highway'])
traffic_level = st.selectbox("Traffic Level", ['low', 'medium', 'high'])

accidents_reported = st.slider("Accidents Reported (last year)", 0, 50, 5)

if st.button("Predict Repair Need"):
    features = preprocess_input(
        last_laid_year, last_repair_year, material,
        weather, usage_type, traffic_level, accidents_reported
    )

    gb_pred = gb_model.predict([features])[0]
    xgb_pred = xgb_model.predict([features])[0]

    repair_labels = {0: "Good Condition", 1: "Needs Repair"}
    st.subheader("🔧 Prediction Results")
    st.write(f"**Gradient Boosting:** {repair_labels[gb_pred]}")
    st.write(f"**XGBoost:** {repair_labels[int(xgb_pred)]}")
