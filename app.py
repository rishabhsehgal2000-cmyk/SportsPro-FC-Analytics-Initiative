import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SportsProFC Analytics", page_icon="⚽")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_objects():
    # These must match the filenames you saved in your notebook
    model = joblib.load('SportsProFC_Model.pkl')
    scaler = joblib.load('SportsProFC_Scaler.pkl')
    return model, scaler

try:
    best_gb_model, scaler = load_model_objects()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🏆 SportsProFC Performance Predictor")
st.markdown("Enter athlete data to predict successful performance output.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=50, value=25)
        agility = st.slider("Agility Score", 0, 100, 70)
        assists = st.number_input("Recent Assists", value=5)
        training_hours = st.number_input("Weekly Training Hours", value=20)

    with col2:
        contract_years = st.number_input("Contract Years Left", value=2)
        experience_level = st.slider("Experience Level (1-10)", 1, 10, 5)
        recovery_rate = st.slider("Recovery Rate (%)", 0, 100, 80)
        intensity = st.selectbox("Intensity", [1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x-1])

    submit_button = st.form_submit_button("Predict Result")

if submit_button:
    # 1. Map inputs to EXACT lowercase names used in training
    input_dict = {
        'age': age,
        'agility': agility,
        'assists': assists,
        'contract_years': contract_years,
        'experience_level': experience_level,
        'training_hours': training_hours,
        'recovery_rate': recovery_rate,
        'intensity': intensity
    }
    
    # 2. Create DataFrame
    input_df = pd.DataFrame([input_dict])

    try:
        # 3. Scale the data (Scaler expects same feature names as fit time)
        input_scaled = scaler.transform(input_df)

        # 4. Make Prediction
        prediction = best_gb_model.predict(input_scaled)
        prob = best_gb_model.predict_proba(input_scaled)

        st.divider()
        if prediction[0] == 1:
            st.success(f"### Result: SUCCESS ✅ ({np.max(prob)*100:.1f}% confidence)")
        else:
            st.error(f"### Result: TARGET NOT MET ❌ ({np.max(prob)*100:.1f}% confidence)")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
