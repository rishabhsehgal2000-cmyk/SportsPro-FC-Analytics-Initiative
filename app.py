import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SportsProFC Analytics", page_icon="⚽", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_objects():
    # These must match the filenames you saved in your folder
    model = joblib.load('SportsProFC_Model.pkl')
    scaler = joblib.load('SportsProFC_Scaler.pkl')
    return model, scaler

try:
    best_gb_model, scaler = load_model_objects()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🏆 SportsProFC Star Performance Predictor")
st.markdown("Enter the key athlete performance metrics to predict 'Star Player' status.")

with st.form("prediction_form"):
    st.subheader("Top Contributing Features")
    col1, col2 = st.columns(2)
    
    with col1:
        minutes_played = st.number_input("Minutes Played", min_value=0, value=1500)
        goals_scored = st.number_input("Goals Scored", min_value=0, value=10)
        stamina = st.slider("Stamina Score", 0, 100, 75)
        assists = st.number_input("Assists", min_value=0, value=5)
        pass_accuracy = st.slider("Pass Accuracy (%)", 0, 100, 80)

    with col2:
        market_value = st.number_input("Market Value (Millions)", min_value=0.0, value=25.0)
        strength = st.slider("Strength Score", 0, 100, 70)
        sprint_speed = st.slider("Sprint Speed Score", 0, 100, 75)
        agility = st.slider("Agility Score", 0, 100, 70)
        matches_played = st.number_input("Matches Played", min_value=0, value=20)

    submit = st.form_submit_button("Predict Star Status")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Create a dictionary with EXACT lowercase names from training
    # Even if you use fewer features in the UI, the dataframe must match the scaler's 'fit' structure.
    # We map the inputs to the feature names identified in the notebook.
    input_data = pd.DataFrame([{
        'minutes_played': minutes_played,
        'goals_scored': goals_scored,
        'stamina': stamina,
        'assists': assists,
        'pass_accuracy': pass_accuracy,
        'market_value_million': market_value,
        'strength': strength,
        'sprint_speed': sprint_speed,
        'agility': agility,
        'matches_played': matches_played
    }])

    try:
        # 2. Scale the input data
        input_scaled = scaler.transform(input_data)

        # 3. Predict using the Gradient Boosting model
        prediction = best_gb_model.predict(input_scaled)
        probability = best_gb_model.predict_proba(input_scaled)

        # 4. Display Results
        st.divider()
        confidence = np.max(probability) * 100
        
        if prediction[0] == 1:
            st.success(f"### Result: STAR PLAYER ✅")
            st.write(f"Confidence Level: **{confidence:.2f}%**")
            st.balloons()
        else:
            st.error(f"### Result: REGULAR PLAYER ❌")
            st.write(f"Confidence Level: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your Scaler was trained with more features than these 10. If so, you must include all training columns in the dataframe.")
