# ⚽ SportsPro FC — Predictive Analytics & Player Recruitment Intelligence

## 📋 Project Overview

Professional football clubs have historically relied on subjective judgement and scouting intuition when making recruitment and contract decisions. **SportsPro FC** faced a critical operational gap — coaches had no systematic, data-driven framework to distinguish genuine *Star Players* from squad fillers, and the finance team had no quantitative method to assess whether high-value contracts were delivering measurable ROI on the pitch.

This project transitions that paradigm from **gut-feel recruitment to evidence-based squad intelligence** by building an end-to-end data science solution — from a 9-point SQL data audit through to a live XGBoost ML classifier deployable by non-technical coaching staff in real time.

---

## 🌐 Interactive Dashboards & Applications

Explore the live deployments of this project to see the predictive model and data insights in action:

- **🖥️ Interactive Web Application:** [Streamlit Live App](https://sportspro-fc-analytics-initiative-gpr2cyojbqqd6d6pczgjbx.streamlit.app/) — Input player attributes across physical, performance, and contract dimensions to generate a real-time Star vs. Regular classification using the trained XGBoost model.
  > 🔐 **Login credentials:** Username: `coach` · Password: `coach@123`

- **📊 Executive Business Intelligence Dashboard:** [Power BI Service Dashboard](https://app.powerbi.com/view?r=eyJrIjoiYWUwNzFiZWUtODRkMi00OWUyLTkzNGQtZTZlZDgyZDgyNDZkIiwidCI6Ijg1NTg5NWNlLTdjMzEtNGQ5My1iNjFjLTNlODM2NTEyNTk5MyJ9&pageName=32fc9b1351237b0c7c0b) — Squad investment overview, market value segmentation by team and experience level, contract risk heat maps, star player distribution, and injury exposure visuals — tailored for coaching staff and club finance managers.

---

## 🎯 Business & Operational Objectives

- **Star Player Identification:** Classify players as Star or Regular using performance, physical, and contract attributes — giving coaches a validated, objective shortlisting tool before recruitment decisions.
- **ROI-Driven Squad Management:** Quantify each player's performance output relative to their market value, enabling finance teams to flag overpaid underperformers and optimise contract renewals.
- **Contract Risk Monitoring:** Track players with ≤1 year remaining on contracts and cross-reference with star classification to prioritise urgent retention planning.
- **Non-Technical Deployment:** Deliver ML predictions through a Streamlit web app operable by coaches with zero data science background — input 10 attributes, get an instant prediction.

---

## 📊 Dataset & Feature Architecture

The analytical pipeline merges three relational tables tracking player attributes across demographics, performance, and financial domains:

| Table | Records | Key Features |
|---|---|---|
| `players.csv` | 15,000 raw | age, nationality, position, height_cm, sprint_speed, stamina, agility, injury_prone |
| `performance.csv` | 15,000 raw | matches_played, goals_scored, assists, minutes_played, pass_accuracy, yellow/red_cards, **star_player ← TARGET** |
| `contracts.csv` | 15,000 raw | team, market_value_million, contract_years, experience_level |

**After merge & null removal:** 9,937 clean records · **Train / Test split:** 7,452 / 2,485 (80/20) · **Class imbalance:** ~14% Star Players

**Target Variable:** `star_player` — Binary classification: `0 = Regular Player`, `1 = Star Player`

---

## ⚙️ Analytical & Modelling Pipeline

### Phase 1 — SQL Data Audit & Cleaning

A rigorous **9-point data audit** was executed before any analysis:

- Row count reconciliation across all 3 tables
- Column-level NULL checks and duplicate primary key detection
- Referential integrity validation across join keys
- Domain & range validation (e.g. red cards > matches played flagged as logically impossible)
- Cross-table logic checks: goals + assists exceeding 3× matches played
- Outlier detection for sprint speed, stamina, and market value

Post-audit, cleaned `*_new` tables were created with an `is_valid` flag column tagging each invalid row with a named quality rule (e.g. `ZERO_MATCH_NONZERO_MINUTES`, `RED_CARDS_GT_MATCHES`).

### Phase 2 — KPI Analysis (SQL)

Eight business KPIs were engineered directly in SQL:

| KPI | Formula | Business Purpose |
|---|---|---|
| Player ROI | AVG((goals + assists + minutes/100) / market_value) | Flags overpaid underperformers |
| Player Utilisation % | AVG(minutes / (matches × 90)) × 100 | Identifies under-rotated squad members |
| Star Dependency Ratio | SUM(star=1) / COUNT(*) × 100 | Signals single-point-of-failure risk |
| Contract Risk % | SUM(contract_years ≤ 1) / COUNT(*) × 100 | Drives urgent retention planning |
| Injury Risk % | SUM(injury_prone=1) / COUNT(*) × 100 | Flags high-value players needing insurance |

### Phase 3 — Exploratory Data Analysis (Python)

Jupyter Notebook EDA covering:
- Distribution plots and correlation heatmaps across all features
- Position-wise goal and assist breakdowns
- Age curve analysis vs. star classification
- Stamina and sprint speed scatter plots overlaid with the star label
- Market value segmentation by team and experience level

### Phase 4 — Predictive Modelling (XGBoost)

**Algorithm:** XGBoost Binary Classifier — chosen for its robustness to tabular data with mixed feature types, native handling of class imbalance, and interpretability via feature importance scores.

**Preprocessing pipeline:** LabelEncoder on 4 categorical columns (nationality, position, team, experience_level) → StandardScaler → XGBClassifier (`eval_metric='logloss'`, `random_state=42`)

**Model performance validated on 2,485 held-out test records.**

---

## 📈 Key Business Findings

| Metric | Star Players | Regular Players |
|---|---|---|
| Avg Goals Scored | **14.1** | 3.0 |
| Avg Assists | **9.4** | 2.8 |
| Avg Minutes Played | **2,288** | 364 |
| Avg Stamina | **80.0** | 74.6 |
| Avg Market Value | $51M | $48M |

> Star players averaged **14.1 goals vs 3.0** for Regular — giving the recruitment team a validated quantitative shortlisting benchmark. The finance team used the Player ROI metric to identify underperformers and re-prioritise contract investment.

---

## 🔐 Model Explainability & Trade-offs

XGBoost was selected over Logistic Regression for higher predictive accuracy on this tabular dataset. The acknowledged trade-off: a coach asking *"why is this player a Star?"* receives feature importance rankings rather than a simple equation. A future iteration would integrate **SHAP values** to deliver per-prediction natural language explanations directly in the Streamlit app.

---

## 📂 Repository Structure

```
├── data/                        # Raw and cleaned CSVs (players, performance, contracts)
├── sql/
│   ├── data_audit.sql           # 9-point data quality audit
│   ├── data_cleaning.sql        # Cleaned table creation + is_valid flagging
│   └── kpi_analysis.sql         # 8 business KPI queries
├── notebooks/
│   └── SportsPro_EDA.ipynb      # Full exploratory data analysis
├── models/
│   ├── xgboost_model.pkl        # Serialised trained XGBoost classifier
│   └── scaler.pkl               # Serialised StandardScaler
├── streamlit_app/
│   └── app.py                   # 4-page Streamlit app with login + ML predictor
├── powerbi/                     # Power BI .pbix report file
├── outputs/
│   ├── risk_report.csv
│   ├── star_classification_output.csv
│   ├── contract_impact.csv
│   └── summary_metrics.csv
└── README.md
```
