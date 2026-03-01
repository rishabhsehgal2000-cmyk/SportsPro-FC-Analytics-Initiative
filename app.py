import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="SportsProFC Analytics Platform", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
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
    
    # Pre-process labels
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
st.sidebar.title("Navigation")
dashboard = st.sidebar.radio("Select Dashboard", 
    ["Squad Investment Overview", "Performance Fitness Overview", "Talent Scouting Insights", "Star Player Predictor (ML)"])

if dashboard != "Star Player Predictor (ML)":
    team_f = st.sidebar.multiselect("Team", df_raw["team"].unique(), default=df_raw["team"].unique())
    pos_f = st.sidebar.multiselect("Position", df_raw["position"].unique(), default=df_raw["position"].unique())
    age_f = st.sidebar.slider("Age Range", int(df_raw["age"].min()), int(df_raw["age"].max()), (18, 40))
    df = df_raw[(df_raw["team"].isin(team_f)) & (df_raw["position"].isin(pos_f)) & (df_raw["age"].between(age_f[0], age_f[1]))]

# ============================================================
# DASHBOARD 1: SQUAD INVESTMENT (4 Charts Restored)
# ============================================================
if dashboard == "Squad Investment Overview":
    st.title("🏢 Squad Investment Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Market Value (M)", f"{df['market_value_million'].sum():,.1f}")
    m2.metric("Star Players %", f"{df['star_player'].mean()*100:.1f}%")
    m3.metric("Avg Contract Years", f"{df['contract_years'].mean():.1f}")
    m4.metric("Average Age", f"{df['age'].mean():.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    # Chart 1: Market Value by Experience
    exp_val = df.groupby("experience_level")["market_value_million"].mean().reset_index()
    c1.plotly_chart(px.bar(exp_val, x="experience_level", y="market_value_million", text_auto=".2f", title="Avg Market Value by Experience"), use_container_width=True)
    # Chart 2: Star Distribution
    star_dist = df["star_label"].value_counts().reset_index(); star_dist.columns = ["Status", "Count"]
    c2.plotly_chart(px.pie(star_dist, names="Status", values="Count", title="Star vs Non-Star Distribution"), use_container_width=True)

    c3, c4 = st.columns(2)
    # Chart 3: Age Distribution
    age_dist = (df["age"].value_counts(normalize=True).sort_index() * 100).reset_index(); age_dist.columns = ["Age", "Dist"]
    c3.plotly_chart(px.bar(age_dist, x="Age", y="Dist", text_auto=".1f", title="Age Distribution (%)"), use_container_width=True)
    # Chart 4: Contract Risk
    risk_dist = (df["contract_risk"].value_counts(normalize=True) * 100).reset_index(); risk_dist.columns = ["Risk", "Dist"]
    c4.plotly_chart(px.bar(risk_dist, x="Risk", y="Dist", text_auto=".1f", title="Contract Risk Distribution (%)"), use_container_width=True)

# ============================================================
# DASHBOARD 2: PERFORMANCE FITNESS (4 Charts Restored)
# ============================================================
elif dashboard == "Performance Fitness Overview":
    st.title("🏃 Performance Fitness Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Goals per Match", f"{(df['goals_scored'].sum()/df['matches_played'].sum()):.2f}")
    m2.metric("Avg Pass Accuracy", f"{df['pass_accuracy'].mean():.1f}%")
    m3.metric("Injury Rate", f"{df['injury_prone'].mean()*100:.1f}%")
    m4.metric("Fitness Score", f"{(df['stamina'].mean()+df['agility'].mean()):.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    # Chart 1: Goals by Position
    goals_pos = df.groupby("position")["goals_scored"].mean().reset_index()
    c1.plotly_chart(px.bar(goals_pos, x="position", y="goals_scored", text_auto=".2f", title="Avg Goals by Position"), use_container_width=True)
    # Chart 2: Assists by Pass Accuracy
    df["acc_band"] = pd.cut(df["pass_accuracy"], bins=[50, 70, 85, 100], labels=["Low", "Medium", "High"])
    acc_data = df.groupby("acc_band")["assists"].mean().reset_index()
    c2.plotly_chart(px.bar(acc_data, x="acc_band", y="assists", text_auto=".2f", title="Avg Assists by Accuracy Band"), use_container_width=True)

    c3, c4 = st.columns(2)
    # Chart 3: Availability
    inj_dist = df["injury_label"].value_counts().reset_index(); inj_dist.columns = ["Status", "Count"]
    c3.plotly_chart(px.pie(inj_dist, names="Status", values="Count", title="Player Availability"), use_container_width=True)
    # Chart 4: Matches Played Distribution
    match_dist = df["matches_played"].value_counts().sort_index().reset_index(); match_dist.columns = ["Matches", "Count"]
    c4.plotly_chart(px.line(match_dist, x="Matches", y="Count", markers=True, title="Matches Played Distribution"), use_container_width=True)

# ============================================================
# DASHBOARD 3: SCOUTING (4 Charts Restored)
# ============================================================
elif dashboard == "Talent Scouting Insights":
    st.title("🔎 Talent Scouting Insights")
    m1, m2, m3, m4 = st.columns(4)
    young = df[df["age"] <= 26]
    m1.metric("Star Potential %", f"{young['star_player'].mean()*100:.1f}%")
    m2.metric("Avg Sprint Speed", f"{df['sprint_speed'].mean():.1f}")
    m3.metric("Avg Jump Height", f"{df['jump_height_cm'].mean():.1f}")
    m4.metric("Breakout Index", f"{(len(df[df['star_player']==0])/len(df)*100):.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    # Chart 1: Nationality Yield
    nat = (df[df["star_player"]==1]["nationality"].value_counts(normalize=True)*100).reset_index(); nat.columns = ["Nation", "Share"]
    c1.plotly_chart(px.bar(nat, x="Nation", y="Share", text_auto=".1f", title="Star Yield by Nationality (%)"), use_container_width=True)
    # Chart 2: Experience vs Star Prob
    exp_s = (df.groupby("experience_level")["star_player"].mean()*100).reset_index(); exp_s.columns = ["Exp", "Prob"]
    c2.plotly_chart(px.bar(exp_s, x="Exp", y="Prob", text_auto=".1f", title="Exp Level vs Star Prob (%)"), use_container_width=True)

    c3, c4 = st.columns(2)
    # Chart 3: Age vs Goals
    age_g = df.groupby("age")["goals_scored"].mean().reset_index()
    c3.plotly_chart(px.line(age_g, x="age", y="goals_scored", markers=True, title="Age vs Average Goals"), use_container_width=True)
    # Chart 4: Market Value Distribution
    df["v_band"] = pd.cut(df["market_value_million"], bins=[0,10,30,60,200], labels=["Low", "Medium", "High", "Elite"])
    v_dist = (df["v_band"].value_counts(normalize=True)*100).reset_index(); v_dist.columns = ["Band", "Dist"]
    c4.plotly_chart(px.bar(v_dist, x="Band", y="Dist", text_auto=".1f", title="Market Value Distribution (%)"), use_container_width=True)

# ============================================================
# DASHBOARD 4: ML PREDICTOR
# ============================================================
else:
    st.title("🏆 Star Player Predictor")
    model, scaler = load_ml_model()
    if model:
        with st.form("ml"):
            col1, col2 = st.columns(2)
            with col1:
                min_p = st.number_input("Minutes Played", value=1500); goals = st.number_input("Goals", value=10)
                stam = st.slider("Stamina", 0, 100, 75); ass = st.number_input("Assists", value=5); pass_a = st.slider("Pass %", 0, 100, 80)
            with col2:
                mval = st.number_input("Value (M)", value=25.0); stre = st.slider("Strength", 0, 100, 70)
                spr = st.slider("Sprint", 0, 100, 75); agi = st.slider("Agility", 0, 100, 70); mat = st.number_input("Matches", value=20)
            if st.form_submit_button("Predict"):
                # All 24 columns for model
                row = [25, 180, 75, 0, 0, spr, stam, stre, agi, 50, 0, mat, goals, ass, 0, 0, min_p, pass_a, 0, 0, 0, 2, mval, 5]
                scaled = scaler.transform(pd.DataFrame([row], columns=['age', 'height_cm', 'weight_kg', 'nationality', 'position', 'sprint_speed', 'stamina', 'strength', 'agility', 'jump_height_cm', 'injury_prone', 'matches_played', 'goals_scored', 'assists', 'yellow_cards', 'red_cards', 'minutes_played', 'pass_accuracy', 'tackles', 'saves', 'team', 'contract_years', 'market_value_million', 'experience_level']))
                res = model.predict(scaled)[0]
                st.divider()
                if res == 1: st.success("Result: STAR PLAYER ✅"); st.balloons()
                else: st.error("Result: REGULAR PLAYER ❌")
