import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SportsProFC Analytics", page_icon="⚽", layout="centered")

# --- LOAD ASSETS ---
# Using cache_resource so the model doesn't reload on every click
@st.cache_resource
def load_model_objects():
    # Ensure these files are in your GitHub root folder
    model = joblib.load('SportsProFC_Model.pkl')
    scaler = joblib.load('SportsProFC_Scaler.pkl')
    return model, scaler

try:
    best_gb_model, scaler = load_model_objects()
except Exception as e:
    st.error(f"Error loading model files. Make sure .pkl files are in the repo. Error: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🏆 SportsProFC Performance Predictor")
st.markdown("""
This app predicts athlete success based on training and performance metrics 
using the **Gradient Boosting** model trained in our initiative.
""")

with st.form("input_form"):
    st.subheader("Athlete Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=50, value=25)
        training_hours = st.number_input("Weekly Training Hours", min_value=1, max_value=100, value=20)
        performance_score = st.slider("Current Performance Score", 0, 100, 70)

    with col2:
        years_exp = st.number_input("Years of Experience", min_value=0, max_value=30, value=5)
        recovery_rate = st.slider("Recovery Rate (%)", 0, 100, 85)
        intensity = st.selectbox("Training Intensity Level", options=[1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x-1])

    submit_button = st.form_submit_button("Analyze Performance")

# --- PREDICTION LOGIC ---
if submit_button:
    # IMPORTANT: The dictionary keys MUST match the column names from your notebook exactly
    input_dict = {
        'Age': age,
        'Training_Hours': training_hours,
        'Performance_Score': performance_score,
        'Years_Experience': years_exp,
        'Recovery_Rate': recovery_rate,
        'Intensity': intensity
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 1. Scale the data using the loaded scaler
    try:
        input_scaled = scaler.transform(input_df)

        # 2. Make Prediction
        prediction = best_gb_model.predict(input_scaled)
        probability = best_gb_model.predict_proba(input_scaled)

        # 3. Display Results
        st.divider()
        confidence = np.max(probability) * 100

        if prediction[0] == 1:
            st.balloons()
            st.success(f"### Prediction: **SUCCESSFUL PERFORMANCE** ✅")
            st.write(f"The model is **{confidence:.2f}%** confident in this athlete's success.")
        else:
            st.error(f"### Prediction: **PERFORMANCE BELOW TARGET** ❌")
            st.write(f"The model is **{confidence:.2f}%** confident this athlete may need more training.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- FOOTER ---
st.sidebar.info("Developed for the SportsProFC Analytics Initiative.")
