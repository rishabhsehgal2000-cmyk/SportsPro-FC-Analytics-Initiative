import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="SportsProFC Analytics", page_icon="⚽", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_objects():
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
st.markdown("Enter the athlete's performance metrics below to predict **Star Player** status.")

with st.form("prediction_form"):
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
    # 1. Create the dictionary excluding 'star_player', 'star_probability', and 'predicted_star_label'
    # These are filled with neutral defaults so the Scaler sees the correct number of features
    full_features = {
        'age': 25, 'height_cm': 180, 'weight_kg': 75, 'nationality': 0, 'position': 0,
        'sprint_speed': sprint_speed, 
        'stamina': stamina, 
        'strength': strength, 
        'agility': agility, 
        'jump_height_cm': 50,
        'injury_prone': 0, 
        'matches_played': matches_played, 
        'goals_scored': goals_scored, 
        'assists': assists,
        'yellow_cards': 0, 'red_cards': 0, 
        'minutes_played': minutes_played, 
        'pass_accuracy': pass_accuracy,
        'tackles': 0, 'saves': 0, 'team': 0, 'contract_years': 2,
        'market_value_million': market_value, 
        'experience_level': 5
    }

    # 2. Define the CORRECT ORDER (the 24 features the model was fit on)
    # Note: We have removed the 3 columns causing the 'unseen' error
    correct_order = [
        'age', 'height_cm', 'weight_kg', 'nationality', 'position',
        'sprint_speed', 'stamina', 'strength', 'agility', 'jump_height_cm',
        'injury_prone', 'matches_played', 'goals_scored', 'assists',
        'yellow_cards', 'red_cards', 'minutes_played', 'pass_accuracy',
        'tackles', 'saves', 'team', 'contract_years',
        'market_value_million', 'experience_level'
    ]

    # Convert to DataFrame and reorder columns
    input_df = pd.DataFrame([full_features])
    input_df = input_df[correct_order]

    try:
        # 3. Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = best_gb_model.predict(input_scaled)
        probability = best_gb_model.predict_proba(input_scaled)

        # 4. Display Results
        st.divider()
        confidence = np.max(probability) * 100
        
        if prediction[0] == 1 or str(prediction[0]) == 'True':
            st.balloons()
            st.success(f"### Result: STAR PLAYER ✅")
            st.write(f"The model is **{confidence:.2f}%** confident.")
        else:
            st.error(f"### Result: REGULAR PLAYER ❌")
            st.write(f"The model is **{confidence:.2f}%** confident.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Technical Detail: The model expected specific features. Ensure your .pkl files match the current feature list.")
