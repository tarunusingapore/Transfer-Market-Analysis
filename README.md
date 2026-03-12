# ⚽ FootballIQ — Football Analytics Startup

> **Analytics-driven transfer intelligence for football clubs**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Goal

Football clubs overspend on high-profile players while missing hidden talent in smaller leagues.
**FootballIQ** uses data analytics to identify undervalued players — those with high performance
metrics but relatively low market values — helping clubs make smarter, data-driven transfer decisions.

---

## 📊 Dataset Description

A synthetic dataset of **1,000 professional football players** is generated with realistic
position-specific distributions across 25 columns:

| Column | Description |
|--------|-------------|
| `Player_ID` | Unique identifier |
| `Position` | GK / Defender / Midfielder / Forward |
| `Club_Size` | Small / Medium / Big |
| `Goals`, `Assists` | Direct contribution metrics |
| `xG`, `xA` | Expected goals and assists |
| `Market_Value_Million_Euros` | Transfer market value |
| `Injury_Days_Last_Season` | Fitness indicator |
| `Performance_Index` | Composite weighted score |
| `Transfer_Score` | AI recommendation score (0–100) |

---

## 🔬 Analytics Process

### 1. Data Generation (`generate_data.py`)
Generates 1,000 players with position-realistic distributions using NumPy and Faker.

### 2. Data Cleaning (`data_cleaning.py`)
- Removes duplicates
- Imputes missing values using position-group medians
- Creates `Age_Group` categories

### 3. Feature Engineering (`data_cleaning.py`)
- **Performance_Index** — position-weighted composite score
- **Value_for_Money** — performance per € of market value
- **Availability_Index** — minutes played vs injury exposure
- **Potential_Index** — age-adjusted future ceiling

### 4. Transfer Model (`transfer_model.py`)
Min-Max scaled weighted combination → **Transfer_Score (0–100)**

| Score | Tier |
|-------|------|
| 80–100 | 🏆 Elite Transfer Target |
| 60–79  | ✅ Strong Signing Opportunity |
| 40–59  | ⚠️ Moderate Value |
| < 40   | ❌ Low Priority |

### 5. Visualisations (`eda_analysis.py`, `radar_charts.py`)
- Correlation heatmap
- Value vs Performance scatter
- Quadrant analysis
- Player radar charts

---

## 💡 Key Business Insights

1. **Small-club players are systematically underpriced** — strong performance at a fraction of big-club valuations.
2. **Young players aged 18–23 offer the best long-term value** — high potential, lower current price.
3. **Midfielders provide the best value-for-money ratio** — dual contributions in attack and defence.
4. **Availability matters more than peak performance** — an injured elite player is worth less than a fit good player.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Generate Dataset
```bash
python generate_data.py
```

### Step 2 — Launch Dashboard
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
football-analytics/
├── app.py               # Streamlit dashboard
├── generate_data.py     # Synthetic dataset generation
├── data_cleaning.py     # Cleaning + feature engineering
├── eda_analysis.py      # EDA visualisation functions
├── radar_charts.py      # Radar chart functions
├── transfer_model.py    # AI Transfer Score model
├── players_dataset.csv  # Generated dataset (auto-created)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — MinMax scaling
- **Plotly** — interactive visualisations
- **Streamlit** — web dashboard
- **Faker** — realistic name generation

---

## 📜 License

MIT © FootballIQ Analytics
