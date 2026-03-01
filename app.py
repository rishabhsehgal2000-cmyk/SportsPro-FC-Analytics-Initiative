import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG & PROFESSIONAL THEME
# ---------------------------------------------------
st.set_page_config(page_title="SportsProFC | Executive Platform", page_icon="⚽", layout="wide")

# Custom CSS for Professional Branding
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Sidebar Styling - Dark Theme with White Text */
    [data-testid="stSidebar"] { 
        background-color: #0e1117; 
    }
    [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Login Box Styling */
    .login-box {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: auto;
    }

    /* Professional Font */
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* Chart Tooltip Styling */
    .js-plotly-plot .plotly .cursor-crosshair { cursor: default; }
    </style>
""", unsafe_allow_html=True)

COLORS = ["#1f3b4d", "#ffc107", "#007bff", "#28a745", "#dc3545"]

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
# LOAD DATA & ML ASSETS (Only if Logged In)
# ---------------------------------------------------
@st.cache_data
def load_data():
    players = pd.read_csv("players_new.csv")
    performance = pd.read_csv("performance_new.csv")
    contracts = pd.read_csv("contracts_new.csv")
    df = performance.merge(players, on="player_id").merge(contracts, on="player_id")
    df = df[df["is_valid"] == "Valid"]
    df["star_label"] = df["star_player"].map({True: "Star", False: "Regular", 1: "Star", 0: "Regular"})
    df["injury_label"] = df["injury_prone"].map({True: "Injured", False: "Available", 1: "Injured", 0: "Available"})
    df["contract_risk"] = pd.cut(df["contract_years"], bins=[-1, 1, 3, 10], labels=["High Risk", "Medium Risk", "Low Risk"])
    return df

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('SportsProFC_Model.pkl')
        scaler = joblib.load('SportsProFC_Scaler.pkl')
        return model, scaler
    except: return None, None

df_raw = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.title("⚽ SPORTSPRO FC")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown("---")
    dashboard = st.radio("DASHBOARD", ["Investment", "Performance", "Scouting", "Predictor (ML)"])

if dashboard != "Predictor (ML)":
    st.sidebar.markdown("### FILTERS")
    team_f = st.sidebar.multiselect("Team", df_raw["team"].unique(), default=df_raw["team"].unique())
    age_f = st.sidebar.slider("Age", 18, 40, (18, 35))
    df = df_raw[(df_raw["team"].isin(team_f)) & (df_raw["age"].between(age_f[0], age_f[1]))]

# ---------------------------------------------------
# REFINED CHART HELPER
# ---------------------------------------------------
def apply_clean_theme(fig, x_title, y_title):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
        title_font=dict(size=18, color='#1f3b4d')
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor='#6c757d')
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor='#e9ecef', linecolor='#6c757d')
    return fig

# ============================================================
# DASHBOARD 1: INVESTMENT
# ============================================================
if dashboard == "Investment":
    st.header("🏢 Squad Investment Strategy")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Value", f"${df['market_value_million'].sum():,.1f}M")
    m2.metric("Star Ratio", f"{df['star_player'].mean()*100:.1f}%")
    m3.metric("Avg Contract", f"{df['contract_years'].mean():.1f}Y")
    m4.metric("Avg Age", f"{df['age'].mean():.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    exp_val = df.groupby("experience_level")["market_value_million"].mean().reset_index()
    fig1 = px.bar(exp_val, x="experience_level", y="market_value_million", text_auto=".2f", color_discrete_sequence=[COLORS[0]])
    c1.plotly_chart(apply_clean_theme(fig1, "Experience Level", "Avg Value ($M)"), use_container_width=True)

    star_dist = df["star_label"].value_counts().reset_index(); star_dist.columns = ["Status", "Count"]
    fig2 = px.pie(star_dist, names="Status", values="Count", color_discrete_sequence=[COLORS[1], COLORS[0]], hole=0.5)
    c2.plotly_chart(fig2, use_container_width=True)

# ============================================================
# DASHBOARD 2: PERFORMANCE
# ============================================================
elif dashboard == "Performance":
    st.header("🏃 Athletic Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Goals/Match", f"{(df['goals_scored'].sum()/df['matches_played'].sum()):.2f}")
    m2.metric("Pass Precision", f"{df['pass_accuracy'].mean():.1f}%")
    m3.metric("Injury Rate", f"{df['injury_prone'].mean()*100:.1f}%")
    m4.metric("Fitness", f"{(df['stamina'].mean()+df['agility'].mean()):.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    goals_pos = df.groupby("position")["goals_scored"].mean().reset_index()
    fig_goals = px.bar(goals_pos, x="position", y="goals_scored", text_auto=".2f", color_discrete_sequence=[COLORS[0]])
    c1.plotly_chart(apply_clean_theme(fig_goals, "Field Position", "Average Goals"), use_container_width=True)
    
    match_dist = df["matches_played"].value_counts().sort_index().reset_index(); match_dist.columns = ["Matches", "Count"]
    fig_matches = px.line(match_dist, x="Matches", y="Count", markers=True, color_discrete_sequence=[COLORS[0]])
    c2.plotly_chart(apply_clean_theme(fig_matches, "Total Matches Played", "Number of Athletes"), use_container_width=True)

# ============================================================
# DASHBOARD 3: SCOUTING
# ============================================================
elif dashboard == "Scouting":
    st.header("🔎 Global Scouting Insights")
    c1, c2 = st.columns(2)
    nat = (df[df["star_player"]==1]["nationality"].value_counts(normalize=True)*100).reset_index(); nat.columns = ["Nation", "Share"]
    fig_nat = px.bar(nat, x="Nation", y="Share", text_auto=".1f", color_discrete_sequence=[COLORS[0]])
    c1.plotly_chart(apply_clean_theme(fig_nat, "Nationality", "Star Yield %"), use_container_width=True)
    
    age_g = df.groupby("age")["goals_scored"].mean().reset_index()
    fig_age = px.line(age_g, x="age", y="goals_scored", markers=True, color_discrete_sequence=[COLORS[2]])
    c2.plotly_chart(apply_clean_theme(fig_age, "Athlete Age", "Avg Scored"), use_container_width=True)

# ============================================================
# DASHBOARD 4: ML PREDICTOR
# ============================================================
else:
    st.header("🏆 Star Talent Predictive Engine")
    model, scaler = load_ml_model()
    if model:
        with st.form("ml_pro"):
            col1, col2 = st.columns(2)
            with col1:
                min_p = st.number_input("Minutes", value=1500); goals = st.number_input("Goals", value=10)
                stam = st.slider("Stamina", 0, 100, 75); ass = st.number_input("Assists", value=5)
            with col2:
                mval = st.number_input("Value ($M)", value=25.0); pass_a = st.slider("Pass %", 0, 100, 80)
                spr = st.slider("Sprint", 0, 100, 75); agi = st.slider("Agility", 0, 100, 70); mat = st.number_input("Matches", value=20)
            
            if st.form_submit_button("RUN ANALYSIS"):
                row = [25, 180, 75, 0, 0, spr, stam, 70, agi, 50, 0, mat, goals, ass, 0, 0, min_p, pass_a, 0, 0, 0, 2, mval, 5]
                scaled = scaler.transform(pd.DataFrame([row], columns=['age', 'height_cm', 'weight_kg', 'nationality', 'position', 'sprint_speed', 'stamina', 'strength', 'agility', 'jump_height_cm', 'injury_prone', 'matches_played', 'goals_scored', 'assists', 'yellow_cards', 'red_cards', 'minutes_played', 'pass_accuracy', 'tackles', 'saves', 'team', 'contract_years', 'market_value_million', 'experience_level']))
                res = model.predict(scaled)[0]
                if res == 1: st.success("✅ PREDICTED STATUS: **STAR PLAYER**"); st.balloons()
                else: st.error("❌ PREDICTED STATUS: **REGULAR PLAYER**")

st.sidebar.markdown("---")
st.sidebar.caption("Secured Coach Access Only")
