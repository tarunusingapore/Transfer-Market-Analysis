"""
Football Analytics Startup — Streamlit Dashboard
Run:  streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from generate_data   import generate_football_dataset
from data_cleaning   import clean_data, engineer_features
from transfer_model  import (compute_transfer_score, quadrant_analysis,
                              top_recommendations, best_young_talents,
                              best_value_signings)
from eda_analysis    import (plot_correlation_heatmap, plot_value_vs_performance,
                              plot_goals_vs_assists, plot_age_performance_curve,
                              plot_position_performance, plot_club_size_value,
                              plot_value_for_money_dist, plot_transfer_score_dist,
                              plot_quadrant_chart, plot_age_group_pie)
from radar_charts    import build_radar, build_position_comparison

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='⚽ Football Analytics Startup',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a5f, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-card .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 0.2rem; }
    .insight-box {
        background: #f0f8ff;
        border-left: 4px solid #45B7D1;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.92rem;
        color: #1e3a5f;
    }
    section[data-testid="stSidebar"] { background-color: #0d1b2a; }
    section[data-testid="stSidebar"] .css-1d391kg { color: white; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading dataset…')
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'players_dataset.csv')
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
    else:
        df_raw = generate_football_dataset(1000)

    df = clean_data(df_raw)
    df = engineer_features(df)
    df = compute_transfer_score(df)
    df = quadrant_analysis(df)
    return df_raw, df


df_raw, df = load_data()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FootballIQ")
    st.markdown("*Analytics-Driven Transfer Intelligence*")
    st.divider()
    page = st.radio(
        'Navigate',
        [
            '🏠 Startup Overview',
            '📊 Dataset Explorer',
            '🧹 Data Cleaning',
            '📈 EDA Visualisations',
            '🔍 Player Scouting Tool',
            '🎯 Player Radar Chart',
            '🤖 AI Transfer Recommendations',
        ],
    )
    st.divider()
    st.caption(f"Dataset: **{len(df):,}** players across **{df['Nationality'].nunique()}** nationalities")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STARTUP OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == '🏠 Startup Overview':
    st.markdown('<p class="main-header">⚽ FootballIQ Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Helping clubs identify undervalued players using data-driven scouting</p>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (f"{len(df):,}", "Players Analysed"),
        (f"{df['Nationality'].nunique()}", "Nationalities"),
        (f"€{df['Market_Value_Million_Euros'].mean():.1f}M", "Avg Market Value"),
        (f"{(df['Transfer_Score'] >= 80).sum()}", "Elite Targets Identified"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], kpis):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("🎯 The Business Problem")
        st.write("""
Football clubs spend hundreds of millions of euros in transfer windows, yet many overpay for
**high-profile players** while overlooking hidden talent in smaller leagues.  
This creates a structural market inefficiency:

- **Big clubs** overspend on brand names, inflating the market.
- **Small clubs** lose their best players for below-market fees due to poor data.
- **Scouts** rely on subjective observation rather than objective metrics.

**FootballIQ** solves this by aggregating 25+ player metrics and applying quantitative scoring
models to surface the best **value-for-money transfers** on the market.
        """)

        st.subheader("💡 How It Works")
        st.markdown("""
| Step | Description |
|------|-------------|
| **1. Data Collection** | 25 performance & market metrics per player |
| **2. Feature Engineering** | Composite scores: Performance, Availability, Potential |
| **3. AI Scoring** | Min-Max weighted Transfer Score (0–100) |
| **4. Scouting Dashboard** | Interactive filters, radar charts, ranked lists |
        """)

    with c2:
        st.subheader("📊 Dataset Snapshot")
        pos_dist = df['Position'].value_counts().reset_index()
        pos_dist.columns = ['Position', 'Count']
        fig_pos = px.pie(pos_dist, names='Position', values='Count',
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         hole=0.45)
        fig_pos.update_layout(height=320, margin=dict(t=20, b=10))
        st.plotly_chart(fig_pos, use_container_width=True)

        tier_counts = df['Transfer_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_tier = px.bar(tier_counts, x='Tier', y='Count',
                          color='Tier', title='Players by Transfer Tier',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_tier.update_layout(height=280, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_tier, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Dataset Explorer':
    st.header('📊 Dataset Explorer')

    tab1, tab2, tab3 = st.tabs(['Raw Data', 'Engineered Features', 'Summary Statistics'])

    with tab1:
        st.subheader('Raw Dataset')
        pos_f  = st.multiselect('Filter by Position', df_raw['Position'].unique(),
                                 default=list(df_raw['Position'].unique()))
        size_f = st.multiselect('Filter by Club Size', df_raw['Club_Size'].unique(),
                                 default=list(df_raw['Club_Size'].unique()))
        raw_view = df_raw[(df_raw['Position'].isin(pos_f)) & (df_raw['Club_Size'].isin(size_f))]
        st.dataframe(raw_view, use_container_width=True, height=450)
        st.caption(f"Showing {len(raw_view):,} of {len(df_raw):,} rows")

    with tab2:
        st.subheader('Engineered Features')
        feature_cols = ['Player_Name', 'Age', 'Position', 'Club_Size',
                         'Performance_Index', 'Value_for_Money', 'Availability_Index',
                         'Potential_Index', 'Scouting_Score', 'Transfer_Score', 'Transfer_Tier']
        st.dataframe(df[feature_cols], use_container_width=True, height=450)

    with tab3:
        st.subheader('Summary Statistics')
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True, height=520)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🧹 Data Cleaning':
    st.header('🧹 Data Cleaning & Transformation')

    st.markdown("""
### Why Data Cleaning?
Raw data from diverse sources often contains **missing values**, **duplicates**, and
**inconsistent formats**. Without cleaning, analytical results are unreliable.
    """)

    steps = [
        ('1. Remove Duplicates', 
         'Duplicate Player_IDs skew aggregations. We drop rows with repeated IDs.',
         'pandas `drop_duplicates()`'),
        ('2. Handle Missing Values',
         'Pass_Accuracy, Key_Passes_per_Game, and Fan_Popularity_Index have ~3% missing values. '
         'We impute with **position-group medians** to preserve distributional integrity.',
         'pandas `groupby().transform("median")`'),
        ('3. Standardise Column Names',
         'Whitespace trimmed; snake_case enforced for consistent programmatic access.',
         'pandas `str.strip()`'),
        ('4. Age Group Categorisation',
         'Continuous age is binned into strategic groups: Young Talent (≤23), '
         'Prime Player (24-29), Veteran (≥30). These align with typical club strategies.',
         'pandas `pd.cut()`'),
    ]

    for title, explanation, method in steps:
        with st.expander(title, expanded=True):
            st.write(explanation)
            st.code(method, language='python')

    st.divider()
    st.subheader('Missing Values Before Cleaning')
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ['Column', 'Missing Count']
    missing['Missing %'] = (missing['Missing Count'] / len(df_raw) * 100).round(2)
    st.dataframe(missing, use_container_width=True)

    st.subheader('Age Group Distribution After Cleaning')
    st.plotly_chart(plot_age_group_pie(df), use_container_width=True)

    st.subheader('Feature Engineering Formulae')
    st.latex(r"\text{Performance Index} = \sum_i w_i \cdot \text{metric}_i \quad (\text{position-specific weights})")
    st.latex(r"\text{Value for Money} = \frac{\text{Performance Index}}{\text{Market Value (€M)}}")
    st.latex(r"\text{Availability Index} = \frac{\text{Minutes Played}}{\text{Minutes Played} + \text{Injury Days} \times 90}")
    st.latex(r"\text{Potential Index} = \begin{cases} 1.0 & \text{age} \le 22 \\ 1 - 0.04(\text{age}-22) & 23 \le \text{age} \le 29 \\ 0.72 - 0.06(\text{age}-29) & \text{age} > 29 \end{cases}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📈 EDA Visualisations':
    st.header('📈 Exploratory Data Analysis')

    vis_choice = st.selectbox(
        'Select Visualisation',
        [
            'Correlation Heatmap',
            'Market Value vs Performance',
            'Goals vs Assists',
            'Age Performance Curve',
            'Position vs Performance',
            'Club Size vs Market Value',
            'Value-for-Money Distribution',
            'Transfer Score Distribution',
            'Quadrant Analysis',
            'Position Radar Comparison',
        ],
    )

    insights = {
        'Correlation Heatmap':
            "Goals, xG, and Performance_Index show strong positive correlations. "
            "Market value is moderately correlated with performance but also depends on age and club size.",
        'Market Value vs Performance':
            "Players in small clubs frequently exhibit strong performance at a fraction of the market value "
            "of equivalent big-club players — a clear scouting opportunity.",
        'Goals vs Assists':
            "Forwards cluster in the high-goals/low-assists quadrant; midfielders exhibit a balanced spread. "
            "Top midfielders are especially valuable for dual contributions.",
        'Age Performance Curve':
            "Performance peaks between ages 24–27. Young players (18–23) show high variance — "
            "some breakout performers offer exceptional value before their market price rises.",
        'Position vs Performance':
            "Forwards and Midfielders score highest on the Performance Index due to direct goal contributions. "
            "Defenders and GKs require alternative metrics (tackles, clean sheets) to assess accurately.",
        'Club Size vs Market Value':
            "Big-club players command significantly higher market values regardless of performance, "
            "indicating a brand/exposure premium. Small-club players are structurally underpriced.",
        'Value-for-Money Distribution':
            "Defenders and Midfielders in smaller clubs often provide the highest value-for-money ratios, "
            "suggesting clubs should prioritise these positions in budget transfer windows.",
        'Transfer Score Distribution':
            "Only ~15% of players reach the Elite tier (score ≥ 80). The majority sit in the Moderate band, "
            "offering room for clubs to uncover bargains in the 60-80 range.",
        'Quadrant Analysis':
            "Undervalued Gems (high performance, low market value) represent the highest-priority targets. "
            "These players deliver top-tier output at below-market cost.",
        'Position Radar Comparison':
            "Positional radars confirm distinct skill profiles. Midfielders are the most versatile. "
            "Forwards dominate attacking metrics while defenders lead in defensive duels.",
    }

    fig_map = {
        'Correlation Heatmap':          plot_correlation_heatmap(df),
        'Market Value vs Performance':  plot_value_vs_performance(df),
        'Goals vs Assists':             plot_goals_vs_assists(df),
        'Age Performance Curve':        plot_age_performance_curve(df),
        'Position vs Performance':      plot_position_performance(df),
        'Club Size vs Market Value':    plot_club_size_value(df),
        'Value-for-Money Distribution': plot_value_for_money_dist(df),
        'Transfer Score Distribution':  plot_transfer_score_dist(df),
        'Quadrant Analysis':            plot_quadrant_chart(df),
        'Position Radar Comparison':    build_position_comparison(df),
    }

    st.plotly_chart(fig_map[vis_choice], use_container_width=True)
    st.markdown(
        f'<div class="insight-box">💡 <strong>Business Insight:</strong> {insights[vis_choice]}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PLAYER SCOUTING TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🔍 Player Scouting Tool':
    st.header('🔍 Player Scouting Tool')
    st.write('Filter players and surface the best undervalued targets.')

    with st.form('scouting_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_sel   = st.selectbox('Position', ['All'] + sorted(df['Position'].unique().tolist()))
            size_sel  = st.multiselect('Club Size', ['Small', 'Medium', 'Big'],
                                        default=['Small', 'Medium', 'Big'])
        with col2:
            age_range = st.slider('Age Range', 17, 38, (18, 30))
            max_val   = st.slider('Max Market Value (€M)', 1.0, 120.0, 30.0, step=0.5)
        with col3:
            min_perf  = st.slider('Min Performance Index', 0.0,
                                   float(df['Performance_Index'].max()), 0.0, step=1.0)
            top_n     = st.slider('Number of Results', 5, 50, 15)

        submitted = st.form_submit_button('🔍 Find Players', use_container_width=True)

    if submitted:
        mask = (
            (df['Age'] >= age_range[0]) &
            (df['Age'] <= age_range[1]) &
            (df['Market_Value_Million_Euros'] <= max_val) &
            (df['Performance_Index'] >= min_perf) &
            (df['Club_Size'].isin(size_sel))
        )
        if pos_sel != 'All':
            mask &= df['Position'] == pos_sel

        results = df[mask].sort_values('Transfer_Score', ascending=False).head(top_n)

        st.success(f"Found **{len(results)}** players matching your criteria")

        display_cols = ['Player_Name', 'Age', 'Position', 'Nationality', 'Club_Size',
                         'Market_Value_Million_Euros', 'Performance_Index',
                         'Value_for_Money', 'Transfer_Score', 'Transfer_Tier']

        def colour_tier(val):
            if 'Elite' in str(val):      return 'background-color: #d4edda; color: #155724'
            elif 'Strong' in str(val):   return 'background-color: #cce5ff; color: #004085'
            elif 'Moderate' in str(val): return 'background-color: #fff3cd; color: #856404'
            else:                         return 'background-color: #f8d7da; color: #721c24'

        styled = (
            results[display_cols]
            .reset_index(drop=True)
            .style
            .map(colour_tier, subset=['Transfer_Tier'])
            .format({'Market_Value_Million_Euros': '€{:.1f}M',
                     'Performance_Index': '{:.1f}',
                     'Value_for_Money': '{:.3f}',
                     'Transfer_Score': '{:.1f}'})
        )
        st.dataframe(styled, use_container_width=True, height=450)

        # Quick bar chart
        fig = px.bar(
            results.head(10),
            x='Player_Name', y='Transfer_Score',
            color='Transfer_Tier',
            title='Top 10 Scouting Results by Transfer Score',
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(xaxis_tickangle=-30, height=400)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PLAYER RADAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🎯 Player Radar Chart':
    st.header('🎯 Player Radar Chart')

    col1, col2 = st.columns([1, 2])
    with col1:
        player_names = sorted(df['Player_Name'].tolist())
        selected_player = st.selectbox('Select a Player', player_names)
        compare_avg = st.checkbox('Compare vs Position Average', value=True)

        if selected_player:
            player_row = df[df['Player_Name'] == selected_player].iloc[0]
            st.markdown(f"""
**Position:** {player_row['Position']}  
**Age:** {player_row['Age']}  
**Nationality:** {player_row['Nationality']}  
**Club Size:** {player_row['Club_Size']}  
**Market Value:** €{player_row['Market_Value_Million_Euros']}M  
**Transfer Score:** {player_row['Transfer_Score']} — {player_row['Transfer_Tier']}  
**Performance Index:** {player_row['Performance_Index']}  
**Scouting Score:** {player_row['Scouting_Score']}
            """)

    with col2:
        if selected_player:
            try:
                fig_radar = build_radar(df, selected_player, compare_avg)
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.subheader('Position Performance Comparison')
    st.plotly_chart(build_position_comparison(df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI TRANSFER RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🤖 AI Transfer Recommendations':
    st.header('🤖 AI Transfer Recommendation Panel')
    st.markdown(
        "Players are ranked using a **weighted Transfer Score** combining Performance, "
        "Value-for-Money, Availability, and Potential."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ['🏆 Top 10 Overall', '🌟 Best Young Talents', '💰 Best Value Signings', '📋 Full Rankings']
    )

    with tab1:
        st.subheader('🏆 Top 10 Transfer Targets')
        top10 = top_recommendations(df, n=10)
        st.dataframe(
            top10.reset_index(drop=True)
            .style.format({'Market_Value_Million_Euros': '€{:.1f}M',
                           'Performance_Index': '{:.1f}',
                           'Value_for_Money': '{:.3f}',
                           'Transfer_Score': '{:.1f}'}),
            use_container_width=True,
        )
        fig = px.bar(
            top10, x='Player_Name', y='Transfer_Score',
            color='Position',
            color_discrete_sequence=px.colors.qualitative.Bold,
            title='Top 10 Transfer Targets — Transfer Score',
        )
        fig.update_layout(xaxis_tickangle=-30, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader('🌟 Best Young Talents (≤ 23)')
        young = best_young_talents(df, n=10)
        st.dataframe(
            young.reset_index(drop=True)
            .style.format({'Market_Value_Million_Euros': '€{:.1f}M',
                           'Transfer_Score': '{:.1f}',
                           'Potential_Index': '{:.3f}'}),
            use_container_width=True,
        )
        fig2 = px.scatter(
            young, x='Market_Value_Million_Euros', y='Transfer_Score',
            size='Potential_Index', color='Position',
            hover_data=['Player_Name', 'Age'],
            title='Young Talents: Market Value vs Transfer Score',
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader('💰 Best Value Signings')
        value = best_value_signings(df, n=10)
        st.dataframe(
            value.reset_index(drop=True)
            .style.format({'Market_Value_Million_Euros': '€{:.1f}M',
                           'Performance_Index': '{:.1f}',
                           'Value_for_Money': '{:.3f}',
                           'Transfer_Score': '{:.1f}'}),
            use_container_width=True,
        )
        fig3 = px.bar(
            value, x='Player_Name', y='Value_for_Money',
            color='Position',
            title='Best Value Signings — Value-for-Money Index',
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig3.update_layout(xaxis_tickangle=-30, height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader('📋 Full Transfer Rankings')
        pos_f  = st.selectbox('Filter Position', ['All'] + sorted(df['Position'].unique().tolist()),
                               key='full_pos')
        size_f = st.multiselect('Filter Club Size', ['Small', 'Medium', 'Big'],
                                 default=['Small', 'Medium', 'Big'], key='full_size')
        max_v  = st.slider('Max Market Value (€M)', 1.0, 200.0, 200.0, key='full_val')

        mask = (
            df['Club_Size'].isin(size_f) &
            (df['Market_Value_Million_Euros'] <= max_v)
        )
        if pos_f != 'All':
            mask &= df['Position'] == pos_f

        full_ranked = df[mask].sort_values('Transfer_Score', ascending=False)[
            ['Player_Name', 'Age', 'Position', 'Nationality', 'Club_Size',
             'Market_Value_Million_Euros', 'Performance_Index',
             'Transfer_Score', 'Transfer_Tier']
        ].reset_index(drop=True)

        st.dataframe(
            full_ranked.style.format(
                {'Market_Value_Million_Euros': '€{:.1f}M',
                 'Performance_Index': '{:.1f}',
                 'Transfer_Score': '{:.1f}'}
            ),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(full_ranked):,} players")
