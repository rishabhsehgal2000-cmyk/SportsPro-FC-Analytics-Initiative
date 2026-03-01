import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG (Must be the first Streamlit command)
# ---------------------------------------------------
st.set_page_config(page_title="SportsProFC Analytics Platform", page_icon="⚽", layout="wide")

# Light Grey Background Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA & ML ASSETS
# ---------------------------------------------------
@st.cache_data
def load_data():
    players = pd.read_csv("players_new.csv")
    performance = pd.read_csv("performance_new.csv")
    contracts = pd.read_csv("contracts_new.csv")
    
    df = performance.merge(players, on="player_id").merge(contracts, on="player_id")
    df = df[df["is_valid"] == "Valid"]
    
    # Create Contract Risk
    df["contract_risk"] = pd.cut(
        df["contract_years"],
        bins=[-1, 1, 3, 10],
        labels=["High Risk", "Medium Risk", "Low Risk"]
    )
    return df

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('SportsProFC_Model.pkl')
        scaler = joblib.load('SportsProFC_Scaler.pkl')
        return model, scaler
    except:
        return None, None

df_raw = load_data()

# ---------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS
# ---------------------------------------------------
st.sidebar.title("Navigation & Filters")

dashboard = st.sidebar.radio(
    "Select Dashboard",
    ["Squad Investment Overview",
     "Performance Fitness Overview",
     "Talent Scouting Insights",
     "Star Player Predictor (ML)"]
)

# Only show data filters for the first three dashboards
if dashboard != "Star Player Predictor (ML)":
    team_filter = st.sidebar.multiselect("Team", df_raw["team"].unique(), default=df_raw["team"].unique())
    position_filter = st.sidebar.multiselect("Position", df_raw["position"].unique(), default=df_raw["position"].unique())
    age_filter = st.sidebar.slider("Age Range", int(df_raw["age"].min()), int(df_raw["age"].max()), (18, 35))

    df = df_raw[
        (df_raw["team"].isin(team_filter)) &
        (df_raw["position"].isin(position_filter)) &
        (df_raw["age"].between(age_filter[0], age_filter[1]))
    ]

# ============================================================
# DASHBOARD 1: SQUAD INVESTMENT
# ============================================================
if dashboard == "Squad Investment Overview":
    st.title("🏢 Squad Investment Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Market Value (M)", f"{df['market_value_million'].sum():,.1f}")
    col2.metric("Star Players %", f"{df['star_player'].mean()*100:.1f}%")
    col3.metric("Avg Contract Years", f"{df['contract_years'].mean():.1f}")
    col4.metric("Average Age", f"{df['age'].mean():.1f}")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    exp_val = df.groupby("experience_level")["market_value_million"].mean().reset_index()
    fig1 = px.bar(exp_val, x="experience_level", y="market_value_million", title="Avg Market Value by Experience")
    c1.plotly_chart(fig1, use_container_width=True)
    
    star_dist = df["star_player"].value_counts().reset_index()
    fig2 = px.pie(star_dist, names="index", values="star_player", title="Star Distribution")
    c2.plotly_chart(fig2, use_container_width=True)

# ============================================================
# DASHBOARD 2: PERFORMANCE FITNESS
# ============================================================
elif dashboard == "Performance Fitness Overview":
    st.title("🏃 Performance Fitness Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Goals per Match", f"{(df['goals_scored'].sum()/df['matches_played'].sum()):.2f}")
    col2.metric("Avg Pass Accuracy", f"{df['pass_accuracy'].mean():.1f}%")
    col3.metric("Injury Rate", f"{df['injury_prone'].mean()*100:.1f}%")
    col4.metric("Fitness Score", f"{(df['stamina'].mean()+df['agility'].mean()):.1f}")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    goals_pos = df.groupby("position")["goals_scored"].mean().reset_index()
    c1.plotly_chart(px.bar(goals_pos, x="position", y="goals_scored", title="Avg Goals by Position"), use_container_width=True)
    
    injury = df["injury_prone"].value_counts().reset_index()
    c2.plotly_chart(px.pie(injury, names="index", values="injury_prone", title="Availability Status"), use_container_width=True)

# ============================================================
# DASHBOARD 3: SCOUTING
# ============================================================
elif dashboard == "Talent Scouting Insights":
    st.title("🔎 Talent Scouting Insights")
    # ... (Keep your existing scouting logic here)
    st.info("Visualizing high-potential scouting data based on nationality and age yield.")
    # (Existing Scouting code from your first block goes here)

# ============================================================
# DASHBOARD 4: STAR PLAYER PREDICTOR (ML)
# ============================================================
elif dashboard == "Star Player Predictor (ML)":
    st.title("🏆 Star Player Status Predictor")
    st.markdown("Enter the athlete's metrics to run the **Gradient Boosting** prediction model.")

    best_gb_model, scaler = load_ml_model()

    if best_gb_model is None:
        st.error("Model files (.pkl) not found. Please ensure they are in the root directory.")
    else:
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
            
            submit = st.form_submit_button("Run ML Prediction")

        if submit:
            full_features = {
                'age': 25, 'height_cm': 180, 'weight_kg': 75, 'nationality': 0, 'position': 0,
                'sprint_speed': sprint_speed, 'stamina': stamina, 'strength': strength, 
                'agility': agility, 'jump_height_cm': 50, 'injury_prone': 0, 
                'matches_played': matches_played, 'goals_scored': goals_scored, 'assists': assists,
                'yellow_cards': 0, 'red_cards': 0, 'minutes_played': minutes_played, 
                'pass_accuracy': pass_accuracy, 'tackles': 0, 'saves': 0, 'team': 0, 
                'contract_years': 2, 'market_value_million': market_value, 'experience_level': 5
            }

            correct_order = [
                'age', 'height_cm', 'weight_kg', 'nationality', 'position',
                'sprint_speed', 'stamina', 'strength', 'agility', 'jump_height_cm',
                'injury_prone', 'matches_played', 'goals_scored', 'assists',
                'yellow_cards', 'red_cards', 'minutes_played', 'pass_accuracy',
                'tackles', 'saves', 'team', 'contract_years',
                'market_value_million', 'experience_level'
            ]

            input_df = pd.DataFrame([full_features])[correct_order]

            try:
                input_scaled = scaler.transform(input_df)
                prediction = best_gb_model.predict(input_scaled)
                probability = best_gb_model.predict_proba(input_scaled)
                confidence = np.max(probability) * 100

                st.divider()
                if prediction[0] == 1 or str(prediction[0]) == 'True':
                    st.balloons()
                    st.success(f"### Result: STAR PLAYER ✅ (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"### Result: REGULAR PLAYER ❌ (Confidence: {confidence:.2f}%)")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Sports Analytics Platform v2.0 | Integrated ML Module | Built with Streamlit")
