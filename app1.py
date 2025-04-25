!pip install streamlit scikit-learn pandas numpy xgboostimport streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
# Replace with actual file paths or load from your training code
gb_model = joblib.load('gradient_boosting_model.pkl') 
xgb_model = joblib.load('xgboost_model.pkl')
st.title("Road Repair Prediction")

# Input fields
road_age = st.number_input("Road Age (years)", min_value=0)
years_since_repair = st.number_input("Years Since Last Repair", min_value=0)
material = st.selectbox("Material Type", ["Asphalt", "Concrete", "Gravel"])
weather = st.selectbox("Weather Condition", ["Hot", "Humid", "Rainy", "Dry", "Cold"])
usage_type = st.selectbox("Usage Type", ["Residential", "Commercial", "Highway"])
traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
accidents_reported = st.number_input("Number of Accidents Reported", min_value=0)

# Preprocessing (if needed)
# ... (One-hot encoding, scaling, etc.) ...
# Replace with your preprocessing logic

material_mapping = {'Asphalt': 0, 'Concrete': 1, 'Gravel': 2}
weather_mapping = {'Hot': 0, 'Humid': 1, 'Rainy': 2, 'Dry': 3, 'Cold': 4}
usage_type_mapping = {'Residential': 0, 'Commercial': 1, 'Highway': 2}
traffic_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

material_encoded = material_mapping[material]
weather_encoded = weather_mapping[weather]
usage_type_encoded = usage_type_mapping[usage_type]
# ... (other code) ...
traffic_level_encoded = traffic_level_mapping[traffic_level]  # Changed usage_type to traffic_level
# ... (rest of the code) ...

# Model selection
model_choice = st.selectbox("Choose Model", ["Gradient Boosting", "XGBoost"])

# Make prediction
input_data = np.array([[road_age, years_since_repair, material_encoded, weather_encoded,
                        usage_type_encoded, traffic_level_encoded, accidents_reported]])

if model_choice == "Gradient Boosting":
    prediction = gb_model.predict(input_data)[0]
else:
    prediction = xgb_model.predict(input_data)[0]

# Display prediction
if prediction == 1:
    st.write("**This road is likely in need of repair.**")
else:
    st.write("**This road is likely not in need of repair.**")
