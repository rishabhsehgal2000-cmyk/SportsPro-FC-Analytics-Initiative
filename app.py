import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="SportsProFC Predictor", layout="centered")

# 1. Load the saved model and scaler
# On GitHub/Streamlit Cloud, they must be in the same root folder
@st.cache_resource
def load_artifacts():
    model = joblib.load('SportsProFC_Model.pkl')
    scaler = joblib.load('SportsProFC_Scaler.pkl')
    return model, scaler

try:
    best_gb_model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. App UI
st.title("🏆 SportsProFC Performance Predictor")
st.markdown("Enter athlete statistics below to predict performance success.")

# 3. Input Fields (Based on your notebook features)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=5, max_value=60, value=25)
        training_hours = st.number_input("Training Hours", min_value=0, max_value=100, value=20)
        performance_score = st.slider("Previous Performance Score", 0, 100, 50)
        
    with col2:
        # Assuming these are other numerical columns from your X_train
        years_exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
        recovery_rate = st.slider("Recovery Rate (%)", 0, 100, 80)
        intensity = st.selectbox("Training Intensity", [1, 2, 3], help="1=Low, 3=High")

    submit = st.form_submit_button("Predict Result")

# 4. Prediction Logic
if submit:
    # Create the input array in the EXACT same order as your X_train columns
    # Adjust these names to match your notebook's X.columns exactly
    input_data = pd.DataFrame([[
        age, training_hours, performance_score, years_exp, recovery_rate, intensity
    ]], columns=['Age', 'Training_Hours', 'Performance_Score', 'Years_Experience', 'Recovery_Rate', 'Intensity'])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = best_gb_model.predict(input_scaled)
    probability = best_gb_model.predict_proba(input_scaled)

    # Display Result
    st.divider()
    if prediction[0] == 1:
        st.success(f"### Result: SUCCESS ✅")
    else:
        st.error(f"### Result: FAILURE ❌")
        
    st.write(f"**Confidence Level:** {np.max(probability) * 100:.2f}%")
