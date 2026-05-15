import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ---------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h1 style='text-align: center;'>🏆 SportsProFC Login</h1>", unsafe_allow_html=True)
    with st.container():
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == "coach" and pw == "coach@123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------------------------------------------
# PAGE CONFIG & PROFESSIONAL THEME (CSS)
# ---------------------------------------------------
st.set_page_config(page_title="SportsProFC | Executive Analytics", page_icon="⚽", layout="wide")

# Custom CSS for Professional Branding
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Sidebar Styling - White Background with Black Text */
    [data-testid="stSidebar"] { 
        background-color: #ffffff !important; 
        border-right: 1px solid #e9ecef;
    }
    
    /* Force all text in sidebar to Black */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Card-like styling for metrics */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1f3b4d; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
    
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }

    /* Professional Font */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f3b4d;
        color: white;
        border-radius: 5px;
        width: 100%;
        border: none;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Define Corporate Colors
COLORS = ["#1f3b4d", "#ffc107", "#007bff", "#28a745", "#dc3545"]

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
    df["star_label"] = df["star_player"].map({True: "Star Player", False: "Regular Player", 1: "Star Player", 0: "Regular Player"})
    df["injury_label"] = df["injury_prone"].map({True: "Injury Prone", False: "Available", 1: "Injury Prone", 0: "Available"})
    df["contract_risk"] = pd.cut(df["contract_years"], bins=[-1, 1, 3, 10], labels=["High Risk", "Medium Risk", "Low Risk"])
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
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5329/5329949.png", width=80) # Placeholder Logo
    st.title("SPORTSPRO FC")
    st.markdown("---")
    dashboard = st.radio("DASHBOARD SELECTOR", 
        ["Squad Investment", "Performance Fitness", "Scouting Insights", "Star Predictor (ML)"])

if dashboard != "Star Predictor (ML)":
    st.sidebar.markdown("### DATA FILTERS")
    team_f = st.sidebar.multiselect("Team", df_raw["team"].unique(), default=df_raw["team"].unique())
    pos_f = st.sidebar.multiselect("Position", df_raw["position"].unique(), default=df_raw["position"].unique())
    age_f = st.sidebar.slider("Age Range", 18, 40, (18, 35))
    df = df_raw[(df_raw["team"].isin(team_f)) & (df_raw["position"].isin(pos_f)) & (df_raw["age"].between(age_f[0], age_f[1]))]

# ============================================================
# DASHBOARD 1: SQUAD INVESTMENT
# ============================================================
if dashboard == "Squad Investment":
    st.header("🏢 Squad Investment Strategy")
    
    # Custom Styled Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_market_value_billion = df["market_value_million"].sum() / 1000
    with m1: st.metric("Market Value", f"${total_market_value_billion:,.1f}B")
    with m2: st.metric("Star Ratio", f"{df['star_player'].mean()*100:.1f}%")
    with m3: st.metric("Avg Contract", f"{df['contract_years'].mean():.1f} Yrs")
    with m4: st.metric("Avg Age", f"{df['age'].mean():.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    exp_val = df.groupby("experience_level")["market_value_million"].mean().reset_index()
    fig1 = px.bar(exp_val, x="experience_level", y="market_value_million", text_auto=".2f", 
                  title="Valuation by Experience", color_discrete_sequence=[COLORS[0]])
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    c1.plotly_chart(fig1, use_container_width=True)

    star_dist = df["star_label"].value_counts().reset_index(); star_dist.columns = ["Status", "Count"]
    fig2 = px.pie(star_dist, names="Status", values="Count", title="Elite Status Distribution",
                  color_discrete_sequence=[COLORS[1], COLORS[0]], hole=0.5)
    c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    age_dist = (df["age"].value_counts(normalize=True).sort_index() * 100).reset_index(); age_dist.columns = ["Age", "Dist"]
    c3.plotly_chart(px.bar(age_dist, x="Age", y="Dist", text_auto=".1f", title="Age Demographics (%)", color_discrete_sequence=[COLORS[2]]), use_container_width=True)
    
    risk_dist = (df["contract_risk"].value_counts(normalize=True) * 100).reset_index(); risk_dist.columns = ["Risk", "Dist"]
    c4.plotly_chart(px.bar(risk_dist, x="Risk", y="Dist", text_auto=".1f", title="Contractual Risk (%)", color_discrete_sequence=[COLORS[4]]), use_container_width=True)

# ============================================================
# DASHBOARD 2: PERFORMANCE FITNESS
# ============================================================
elif dashboard == "Performance Fitness":
    st.header("🏃 Athletic Performance & Medical")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Goals/Match", f"{(df['goals_scored'].sum()/df['matches_played'].sum()):.2f}")
    m2.metric("Pass Precision", f"{df['pass_accuracy'].mean():.1f}%")
    m3.metric("Injury Propensity", f"{df['injury_prone'].mean()*100:.1f}%")
    m4.metric("Squad Fitness", f"{(df['stamina'].mean()+df['agility'].mean()):.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    goals_pos = df.groupby("position")["goals_scored"].mean().reset_index()
    c1.plotly_chart(px.bar(goals_pos, x="position", y="goals_scored", text_auto=".2f", title="Scoring Prowess by Position", color_discrete_sequence=[COLORS[0]]), use_container_width=True)
    
    df["acc_band"] = pd.cut(df["pass_accuracy"], bins=[50, 70, 85, 100], labels=["Low", "Med", "Elite"])
    acc_data = df.groupby("acc_band")["assists"].mean().reset_index()
    c2.plotly_chart(px.bar(acc_data, x="acc_band", y="assists", text_auto=".2f", title="Playmaking vs Accuracy", color_discrete_sequence=[COLORS[1]]), use_container_width=True)

    c3, c4 = st.columns(2)
    inj_dist = df["injury_label"].value_counts().reset_index(); inj_dist.columns = ["Status", "Count"]
    c3.plotly_chart(px.pie(inj_dist, names="Status", values="Count", title="Squad Availability", hole=0.5, color_discrete_sequence=[COLORS[3], COLORS[4]]), use_container_width=True)
    
    match_dist = df["matches_played"].value_counts().sort_index().reset_index(); match_dist.columns = ["Matches", "Count"]
    c4.plotly_chart(px.line(match_dist, x="Matches", y="Count", markers=True, title="Game Time Exposure", color_discrete_sequence=[COLORS[0]]), use_container_width=True)

# ============================================================
# DASHBOARD 3: SCOUTING INSIGHTS
# ============================================================
elif dashboard == "Scouting Insights":
    st.header("🔎 Global Scouting & Talent Acquisition")
    m1, m2, m3, m4 = st.columns(4)
    young = df[df["age"] <= 26]
    m1.metric("U26 Star Yield", f"{young['star_player'].mean()*100:.1f}%")
    m2.metric("Avg Sprint", f"{df['sprint_speed'].mean():.1f}km/hr")
    m3.metric("Avg Vertical Jump", f"{df['jump_height_cm'].mean():.1f}cm")
    star_avg_minutes = df[df["star_player"] == 1]["minutes_played"].median()
    star_avg_matches = df[df["star_player"] == 1]["matches_played"].median()

    breakout = df[
    (df["star_player"] == 0) &
    (df["minutes_played"] >= star_avg_minutes) &
    (df["matches_played"] >= star_avg_matches)
    ]

    breakout_index = (len(breakout) / len(df) * 100)
    m4.metric("Breakout Index", f"{breakout_index:.1f}%", delta=f"{len(breakout)} hidden gems")

    
    st.markdown("---")
    c1, c2 = st.columns(2)
    nat = (df[df["star_player"]==1]["nationality"].value_counts(normalize=True)*100).reset_index(); nat.columns = ["Nation", "Share"]
    c1.plotly_chart(px.bar(nat, x="Nation", y="Share", text_auto=".1f", title="Elite Yield by Region (%)", color_discrete_sequence=[COLORS[0]]), use_container_width=True)
    
    exp_s = (df.groupby("experience_level")["star_player"].mean()*100).reset_index(); exp_s.columns = ["Exp", "Prob"]
    c2.plotly_chart(px.bar(exp_s, x="Exp", y="Prob", text_auto=".1f", title="Exp Level vs Peak Performance", color_discrete_sequence=[COLORS[1]]), use_container_width=True)

    c3, c4 = st.columns(2)
    age_g = df.groupby("age")["goals_scored"].mean().reset_index()
    c3.plotly_chart(px.line(age_g, x="age", y="goals_scored", markers=True, title="Goal Maturity Curve", color_discrete_sequence=[COLORS[2]]), use_container_width=True)
    
    df["v_band"] = pd.cut(df["market_value_million"], bins=[0,10,30,60,200], labels=["Core", "Key", "Star", "Elite"])
    v_dist = (df["v_band"].value_counts(normalize=True)*100).reset_index(); v_dist.columns = ["Band", "Dist"]
    c4.plotly_chart(px.bar(v_dist, x="Band", y="Dist", text_auto=".1f", title="Asset Value Tiers (%)", color_discrete_sequence=[COLORS[0]]), use_container_width=True)

# ============================================================
# DASHBOARD 4: ML PREDICTOR
# ============================================================
else:
    st.header("🏆 Star Talent Predictive Engine")
    st.write("Deep neural prediction based on Gradient Boosting Classifier.")
    
    model, scaler = load_ml_model()
    if model:
        with st.form("ml_professional"):
            col1, col2 = st.columns(2)
            with col1:
                min_p = st.number_input("Minutes Played", value=1500); goals = st.number_input("Goals", value=10)
                stam = st.slider("Stamina", 0, 100, 75); ass = st.number_input("Assists", value=5); pass_a = st.slider("Pass Accuracy %", 0, 100, 80)
            with col2:
                mval = st.number_input("Market Value (M)", value=25.0); stre = st.slider("Strength", 0, 100, 70)
                spr = st.slider("Sprint Speed", 0, 100, 75); agi = st.slider("Agility", 0, 100, 70); mat = st.number_input("Matches Played", value=20)
            
            predict_btn = st.form_submit_button("RUN CLASSIFICATION ENGINE")
            
            if predict_btn:
                row = [25, 180, 75, 0, 0, spr, stam, stre, agi, 50, 0, mat, goals, ass, 0, 0, min_p, pass_a, 0, 0, 0, 2, mval, 5]
                scaled = scaler.transform(pd.DataFrame([row], columns=['age', 'height_cm', 'weight_kg', 'nationality', 'position', 'sprint_speed', 'stamina', 'strength', 'agility', 'jump_height_cm', 'injury_prone', 'matches_played', 'goals_scored', 'assists', 'yellow_cards', 'red_cards', 'minutes_played', 'pass_accuracy', 'tackles', 'saves', 'team', 'contract_years', 'market_value_million', 'experience_level']))
                res = model.predict(scaled)[0]
                
                st.markdown("### Prediction Result")
                if res == 1: 
                    st.success("✅ CLASSIFIED AS: **STAR PLAYER**")
                    st.balloons()
                else: 
                    st.error("❌ CLASSIFIED AS: **REGULAR SQUAD PLAYER**")




