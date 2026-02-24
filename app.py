import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Sports Analytics Platform", layout="wide")

# Light Grey Background
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
players = pd.read_csv("players_new.csv")
performance = pd.read_csv("performance_new.csv")
contracts = pd.read_csv("contracts_new.csv")

df = performance.merge(players, on="player_id") \
                .merge(contracts, on="player_id")

# Only VALID records
df = df[df["is_valid"] == "Valid"]

# Create Contract Risk
df["contract_risk"] = pd.cut(
    df["contract_years"],
    bins=[-1, 1, 3, 10],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.title("Filters")

dashboard = st.sidebar.radio(
    "Select Dashboard",
    ["Executive Dashboard",
     "Coach Dashboard",
     "Scouting Dashboard"]
)

team_filter = st.sidebar.multiselect(
    "Team",
    df["team"].unique(),
    default=df["team"].unique()
)

position_filter = st.sidebar.multiselect(
    "Position",
    df["position"].unique(),
    default=df["position"].unique()
)

age_filter = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (18, 35)
)

df = df[
    (df["team"].isin(team_filter)) &
    (df["position"].isin(position_filter)) &
    (df["age"].between(age_filter[0], age_filter[1]))
]

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================
if dashboard == "Executive Dashboard":

    st.title("🏢 Club Management Dashboard")

    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Market Value (M)", f"{df['market_value_million'].sum():,.1f}")
    col2.metric("Star Players %", f"{df['star_player'].mean()*100:.1f}%")
    col3.metric("Avg Contract Years", f"{df['contract_years'].mean():.1f}")
    col4.metric("Average Age", f"{df['age'].mean():.1f}")

    st.markdown("---")

    col5, col6 = st.columns(2)

    # Market Value by Experience
    exp_val = df.groupby("experience_level")["market_value_million"].mean().reset_index()
    fig1 = px.bar(exp_val,
                  x="experience_level",
                  y="market_value_million",
                  text_auto=".2f",
                  labels={"market_value_million":"Avg Market Value (M)"},
                  title="Market Value by Experience Level")
    col5.plotly_chart(fig1, use_container_width=True)

    # Star Distribution Pie (Safe Mapping)
    star_dist = df["star_player"].replace(
        {True:"Star Player", False:"Non-Star Player",
         1:"Star Player", 0:"Non-Star Player"}
    ).value_counts().reset_index()
    star_dist.columns = ["Star Status", "Count"]

    fig2 = px.pie(star_dist,
                  names="Star Status",
                  values="Count",
                  title="Star vs Non-Star Distribution")
    col6.plotly_chart(fig2, use_container_width=True)

    col7, col8 = st.columns(2)

    # Age Distribution
    age_dist = df["age"].value_counts().sort_index().reset_index()
    age_dist.columns = ["Age", "Number of Players"]
    fig3 = px.bar(age_dist,
                  x="Age",
                  y="Number of Players",
                  text_auto=True,
                  title="Age Distribution")
    col7.plotly_chart(fig3, use_container_width=True)

    # Contract Risk
    risk_dist = df["contract_risk"].value_counts().reset_index()
    risk_dist.columns = ["Risk Level", "Number of Players"]
    fig4 = px.bar(risk_dist,
                  x="Risk Level",
                  y="Number of Players",
                  text_auto=True,
                  title="Contract Risk Distribution")
    col8.plotly_chart(fig4, use_container_width=True)

# ============================================================
# COACH DASHBOARD
# ============================================================
elif dashboard == "Coach Dashboard":

    st.title("🏃 Performance & Coaching Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Goals per Match",
                f"{(df['goals_scored'].sum()/df['matches_played'].sum()):.2f}")

    col2.metric("Avg Pass Accuracy",
                f"{df['pass_accuracy'].mean():.1f}%")

    col3.metric("Injury Rate",
                f"{df['injury_prone'].mean()*100:.1f}%")

    col4.metric("Fitness Score",
                f"{(df['stamina'].mean()+df['agility'].mean()):.1f}")

    st.markdown("---")

    col5, col6 = st.columns(2)

    # Goals by Position
    goals_pos = df.groupby("position")["goals_scored"].mean().reset_index()
    fig5 = px.bar(goals_pos,
                  x="position",
                  y="goals_scored",
                  text_auto=".2f",
                  labels={"goals_scored":"Average Goals"},
                  title="Average Goals by Position")
    col5.plotly_chart(fig5, use_container_width=True)

    # Assists by Accuracy Band
    df["accuracy_band"] = pd.cut(
        df["pass_accuracy"],
        bins=[50, 60, 70, 80, 90, 100],
        labels=["50-60","60-70","70-80","80-90","90-100"]
    )
    assist_band = df.groupby("accuracy_band")["assists"].mean().reset_index()
    fig6 = px.bar(assist_band,
                  x="accuracy_band",
                  y="assists",
                  text_auto=".2f",
                  labels={"assists":"Average Assists"},
                  title="Average Assists by Pass Accuracy Band")
    col6.plotly_chart(fig6, use_container_width=True)

    col7, col8 = st.columns(2)

    # Availability Pie
    injury = df["injury_prone"].replace(
        {True:"Injury Prone", False:"Available",
         1:"Injury Prone", 0:"Available"}
    ).value_counts().reset_index()
    injury.columns = ["Status", "Count"]

    fig7 = px.pie(injury,
                  names="Status",
                  values="Count",
                  title="Player Availability")
    col7.plotly_chart(fig7, use_container_width=True)

    # Matches Played Distribution
    matches = df["matches_played"].value_counts().sort_index().reset_index()
    matches.columns = ["Matches Played", "Number of Players"]
    fig8 = px.line(matches,
                   x="Matches Played",
                   y="Number of Players",
                   markers=True,
                   title="Matches Played Distribution")
    col8.plotly_chart(fig8, use_container_width=True)
# ============================================================
# SCOUTING DASHBOARD
# ============================================================
else:

    st.title("🔎 Scouting & Talent Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    young = df[df["age"] <= 26]

    col1.metric("Star Potential %",
                f"{young['star_player'].mean()*100:.1f}%")

    col2.metric("Avg Sprint Speed",
                f"{df['sprint_speed'].mean():.1f}")

    col3.metric("Avg Jump Height",
                f"{df['jump_height_cm'].mean():.1f}")

    breakout = df[(df["star_player"] == 0) & (df["age"] <= 26)]
    bpi = (len(breakout)/len(df))*100 if len(df)>0 else 0
    col4.metric("Breakout Potential Index",
                f"{bpi:.1f}%")

    st.markdown("---")

    # ---------------- Row 1 ----------------
    col5, col6 = st.columns(2)

    # 1️⃣ Star Yield by Nationality (%)
    nat = df.groupby("nationality")["star_player"].mean().reset_index()
    nat["star_player"] = nat["star_player"] * 100

    fig9 = px.bar(
        nat,
        x="nationality",
        y="star_player",
        text_auto=".1f",
        labels={"star_player":"Star Percentage (%)"},
        title="Star Yield by Nationality (%)"
    )
    col5.plotly_chart(fig9, use_container_width=True)

    # 2️⃣ Experience vs Star Probability (%)
    exp_star = df.groupby("experience_level")["star_player"].mean().reset_index()
    exp_star["star_player"] = exp_star["star_player"] * 100

    fig10 = px.bar(
        exp_star,
        x="experience_level",
        y="star_player",
        text_auto=".1f",
        labels={"star_player":"Star Probability (%)"},
        title="Experience Level vs Star Probability (%)"
    )
    col6.plotly_chart(fig10, use_container_width=True)

    # ---------------- Row 2 ----------------
    col7, col8 = st.columns(2)

    # 3️⃣ Age vs Average Goals
    age_goal = df.groupby("age")["goals_scored"].mean().reset_index()

    fig11 = px.line(
        age_goal,
        x="age",
        y="goals_scored",
        markers=True,
        labels={"goals_scored":"Average Goals"},
        title="Age vs Average Goals"
    )
    col7.plotly_chart(fig11, use_container_width=True)

    # 4️⃣ Market Value Distribution (%)
    df["value_band"] = pd.cut(
        df["market_value_million"],
        bins=[0,5,20,50,100],
        labels=["Low","Medium","High","Elite"]
    )

    val_band = df["value_band"].value_counts(normalize=True) * 100
    val_band = val_band.reset_index()
    val_band.columns = ["Value Band","Percentage"]

    fig12 = px.bar(
        val_band,
        x="Value Band",
        y="Percentage",
        text_auto=".1f",
        labels={"Percentage":"Player Distribution (%)"},
        title="Market Value Distribution (%)"
    )
    col8.plotly_chart(fig12, use_container_width=True)
    
st.markdown("---")
st.caption("Sports Analytics Platform | Built with Streamlit")