"""FootballIQ — Streamlit Dashboard  |  streamlit run app.py"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from generate_data  import generate_football_dataset
from data_cleaning  import clean_data, engineer_features
from transfer_model import (compute_transfer_score, quadrant_analysis,
                             top_recommendations, best_young_talents,
                             best_value_signings)
from eda_analysis   import (plot_correlation_heatmap, plot_value_vs_performance,
                             plot_goals_vs_assists, plot_age_performance_curve,
                             plot_position_performance, plot_club_size_value,
                             plot_value_for_money_dist, plot_transfer_score_dist,
                             plot_quadrant_chart, plot_age_group_pie,
                             plot_league_distribution)
from radar_charts   import build_two_player_radar, build_position_comparison

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='⚽ FootballIQ Analytics',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global dark base ── */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0a0e1a !important;
      color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
      border-right: 1px solid #1e293b;
  }
  [data-testid="stHeader"] { background: #0a0e1a !important; }

  /* ── KPI cards ── */
  .kpi-card {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,212,255,0.08);
  }
  .kpi-card .kpi-val  { font-size: 2rem; font-weight: 800; color: #00d4ff; }
  .kpi-card .kpi-lab  { font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; }

  /* ── Section heading ── */
  .page-title {
      font-size: 2.2rem; font-weight: 800;
      background: linear-gradient(90deg, #00d4ff, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 0.1rem;
  }
  .page-sub { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }

  /* ── Insight box ── */
  .insight {
      background: #0f172a;
      border-left: 3px solid #00d4ff;
      padding: 0.75rem 1rem;
      border-radius: 6px;
      font-size: 0.91rem;
      color: #94a3b8;
      margin: 0.4rem 0 1rem 0;
  }
  .insight strong { color: #00d4ff; }

  /* ── Tier badges ── */
  .tier-elite    { color: #4ade80; font-weight: 700; }
  .tier-strong   { color: #00d4ff; font-weight: 700; }
  .tier-moderate { color: #fbbf24; font-weight: 700; }
  .tier-low      { color: #f87171; font-weight: 700; }

  /* ── Tables ── */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Plotly dark defaults ───────────────────────────────────────────────────────
import plotly.io as pio
pio.templates.default = "plotly_dark"


# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading dataset…')
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'players_dataset.csv')
    df_raw = pd.read_csv(csv_path) if os.path.exists(csv_path) \
             else generate_football_dataset(150)
    df = clean_data(df_raw)
    df = engineer_features(df)
    df = compute_transfer_score(df)
    df = quadrant_analysis(df)
    return df_raw, df

df_raw, df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FootballIQ")
    st.markdown("<span style='color:#64748b;font-size:0.85rem'>Analytics-Driven Transfer Intelligence</span>",
                unsafe_allow_html=True)
    st.divider()
    page = st.radio('Navigate', [
        '🏠  Overview',
        '📊  Dataset Explorer',
        '🧹  Data Cleaning',
        '📈  EDA Visualisations',
        '🔍  Player Scouting',
        '🎯  Player Comparison',
        '🤖  Transfer Rankings',
    ])
    st.divider()
    st.caption(f"**{len(df)}** players · **{df['Nationality'].nunique()}** nationalities · **{df['League'].nunique()}** leagues")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == '🏠  Overview':
    st.markdown('<p class="page-title">⚽ FootballIQ Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Helping clubs find undervalued players through data science</p>',
                unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (f"{len(df)}", "Players in Database"),
        (f"{df['League'].nunique()}", "Leagues Covered"),
        (f"€{df['Market_Value_Million_Euros'].median():.0f}M", "Median Market Value"),
        (f"{(df['Transfer_Score'] >= 60).sum()}", "Recommended Targets"),
    ]
    for col,(val,lab) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-lab">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("🎯 The Problem")
        st.markdown("""
Clubs routinely overpay for famous names while elite performers in smaller leagues go unnoticed.
**FootballIQ** quantifies 27 metrics per player and produces a single **Transfer Score (0–100)**
so scouts can compare across positions, leagues, and club contexts objectively.

| Step | What we do |
|------|------------|
| **Data** | 150 real players, 27 metrics, 2023/24 season |
| **Engineering** | Performance, Value-for-Money, Availability, Potential indexes |
| **Scoring** | Weighted min-max Transfer Score |
| **Dashboard** | Interactive scouting, radar comparisons, ranked shortlists |
        """)

    with right:
        fig_pos = px.pie(df['Position'].value_counts().reset_index(),
                         names='Position', values='count', hole=0.5,
                         color='Position',
                         color_discrete_map={'GK':'#f97316','Defender':'#22d3ee',
                                             'Midfielder':'#a78bfa','Forward':'#4ade80'})
        fig_pos.update_layout(height=300, margin=dict(t=20,b=10),
                              paper_bgcolor='#111827', font_color='#e2e8f0',
                              legend=dict(bgcolor='#0f172a'))
        st.plotly_chart(fig_pos, use_container_width=True)

        tier_df = df['Transfer_Tier'].value_counts().reset_index()
        tier_df.columns = ['Tier','Count']
        fig_tier = px.bar(tier_df, x='Count', y='Tier', orientation='h',
                          color='Count', color_continuous_scale='Blues',
                          title='Players by Transfer Tier')
        fig_tier.update_layout(height=260, paper_bgcolor='#111827',
                               plot_bgcolor='#0f172a', font_color='#e2e8f0',
                               yaxis={'categoryorder':'total ascending'},
                               showlegend=False, coloraxis_showscale=False,
                               margin=dict(t=40))
        st.plotly_chart(fig_tier, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📊  Dataset Explorer':
    st.header('📊 Dataset Explorer')
    tab_raw, tab_feat, tab_stats = st.tabs(['Raw Data','Engineered Features','Summary Stats'])

    with tab_raw:
        c1,c2,c3 = st.columns(3)
        pos_f  = c1.multiselect('Position', df['Position'].unique(), default=list(df['Position'].unique()))
        size_f = c2.multiselect('Club Size', ['Small','Medium','Big'], default=['Small','Medium','Big'])
        league_f = c3.multiselect('League', sorted(df['League'].unique()), default=list(df['League'].unique()))
        view = df[(df['Position'].isin(pos_f)) & (df['Club_Size'].isin(size_f)) & (df['League'].isin(league_f))]
        st.dataframe(view.reset_index(drop=True), use_container_width=True, height=480)
        st.caption(f"Showing **{len(view)}** of **{len(df)}** players")

    with tab_feat:
        feat_cols = ['Player_Name','Age','Position','Club','League',
                     'Performance_Index','Value_for_Money','Availability_Index',
                     'Potential_Index','Scouting_Score','Transfer_Score','Transfer_Tier']
        st.dataframe(df[feat_cols].reset_index(drop=True), use_container_width=True, height=480)

    with tab_stats:
        num_cols = df.select_dtypes('number').columns.tolist()
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True, height=520)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🧹  Data Cleaning':
    st.header('🧹 Data Cleaning & Feature Engineering')

    for title, body, code in [
        ('1 · Duplicate Removal',
         'Player_ID is the unique key. Duplicates inflate aggregates.',
         "df = df.drop_duplicates(subset=['Player_ID']).reset_index(drop=True)"),
        ('2 · Missing Value Imputation',
         'Pass_Accuracy, Key_Passes_per_Game, Fan_Popularity_Index have ~3% missing values.\n'
         'We fill with **position-group medians** to preserve distributional integrity.',
         "df[col] = df[col].fillna(df.groupby('Position')[col].transform('median'))"),
        ('3 · Age Group Categorisation',
         'Continuous age → three strategic buckets aligned with recruitment philosophy.',
         "df['Age_Group'] = pd.cut(df['Age'], bins=[0,23,29,99],\n"
         "    labels=['Young Talent','Prime Player','Veteran']).astype(str)"),
        ('4 · Feature Engineering',
         'Four composite indexes derived from raw metrics — see formulae below.',
         "df['Performance_Index'] = Σ (position_weight × metric)"),
    ]:
        with st.expander(title, expanded=True):
            st.write(body)
            st.code(code, language='python')

    st.subheader('Formulae')
    st.latex(r"\text{Value for Money} = \frac{\text{Performance Index}}{\text{Market Value}}")
    st.latex(r"\text{Availability} = \frac{\text{Minutes Played}}{\text{Minutes}+\text{Injury Days}\times 90}")
    st.latex(r"\text{Transfer Score} = 0.35\,P + 0.30\,V + 0.20\,A + 0.15\,\text{Pot}")

    st.divider()
    st.subheader('Age Group Distribution')
    st.plotly_chart(plot_age_group_pie(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📈  EDA Visualisations':
    st.header('📈 Exploratory Data Analysis')

    VIS = {
        'Correlation Heatmap':           (plot_correlation_heatmap,
            "Goals, xG and Performance Index are tightly correlated. "
            "Market Value correlates only moderately (~0.5) — confirming the brand premium."),
        'Market Value vs Performance':   (plot_value_vs_performance,
            "Small-club players (smaller bubbles) with high Performance Index sit far below "
            "the market value of equivalent big-club players — clear scouting opportunities."),
        'Goals vs Assists':              (plot_goals_vs_assists,
            "Forwards cluster top-left (goals-heavy). Elite midfielders balance both axes. "
            "Players above both medians are dual-threat match-winners."),
        'Age Performance Curve':         (plot_age_performance_curve,
            "Peak performance lands between 24–27. Players 18–22 show high variance — "
            "some break out early before the market re-prices them."),
        'Position vs Performance':       (plot_position_performance,
            "Forwards lead on Performance Index due to goal weighting. "
            "Defenders with high scores are rare and extremely valuable."),
        'Club Size vs Market Value':     (plot_club_size_value,
            "Big-club players command 3–5× the market value of statistically similar "
            "small/medium-club players — the core inefficiency FootballIQ targets."),
        'Value-for-Money Distribution':  (plot_value_for_money_dist,
            "Medium-club midfielders and forwards dominate the high-VfM tail — "
            "these are the profiles most frequently overlooked by top-club scouts."),
        'Transfer Score Distribution':   (plot_transfer_score_dist,
            "The Elite tier (≥80) is intentionally rare — a strict quality filter. "
            "The 60–80 band offers the richest pool of actionable, affordable targets."),
        'Quadrant Analysis':             (plot_quadrant_chart,
            "Undervalued Gems (green) — high performance, low value — are the primary "
            "transfer targets. Declining Veterans (amber) should be avoided."),
        'League Distribution':           (plot_league_distribution,
            "Premier League dominates the dataset; Bundesliga and La Liga follow. "
            "Underrepresented leagues hide the best value-for-money players."),
    }

    choice = st.selectbox('Select chart', list(VIS.keys()))
    fn, insight = VIS[choice]
    st.plotly_chart(fn(df), use_container_width=True)
    st.markdown(f'<div class="insight"><strong>💡 Insight:</strong> {insight}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — SCOUTING TOOL
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🔍  Player Scouting':
    st.header('🔍 Player Scouting Tool')

    with st.form('scout'):
        c1,c2,c3 = st.columns(3)
        pos_s   = c1.selectbox('Position', ['All']+sorted(df['Position'].unique().tolist()))
        size_s  = c1.multiselect('Club Size', ['Small','Medium','Big'], default=['Small','Medium','Big'])
        age_r   = c2.slider('Age Range', 16, 40, (18, 30))
        max_v   = c2.slider('Max Market Value (€M)', 0.5, 200.0, 50.0, step=0.5)
        min_p   = c3.slider('Min Performance Index', 0.0, float(df['Performance_Index'].max()), 0.0, step=1.0)
        top_n   = c3.slider('Results to show', 5, 50, 15)
        go_btn  = st.form_submit_button('🔍 Find Players', use_container_width=True)

    if go_btn:
        mask = ((df['Age'] >= age_r[0]) & (df['Age'] <= age_r[1]) &
                (df['Market_Value_Million_Euros'] <= max_v) &
                (df['Performance_Index'] >= min_p) &
                (df['Club_Size'].isin(size_s)))
        if pos_s != 'All':
            mask &= df['Position'] == pos_s
        res = df[mask].sort_values('Transfer_Score', ascending=False).head(top_n)

        st.success(f"**{len(res)}** players match your criteria")

        cols = ['Player_Name','Age','Position','Club','League','Nationality',
                'Market_Value_Million_Euros','Performance_Index','Transfer_Score','Transfer_Tier']

        display = res[cols].copy().reset_index(drop=True)
        display.index = display.index + 1   # start at 1

        st.dataframe(
            display.style
            .format({'Market_Value_Million_Euros':'€{:.1f}M',
                     'Performance_Index':'{:.1f}',
                     'Transfer_Score':'{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=460
        )

        fig = px.bar(res.head(10), x='Player_Name', y='Transfer_Score',
                     color='Position',
                     color_discrete_map={'GK':'#f97316','Defender':'#22d3ee',
                                         'Midfielder':'#a78bfa','Forward':'#4ade80'},
                     title='Top 10 Scouting Results')
        fig.update_layout(xaxis_tickangle=-30, height=420,
                          paper_bgcolor='#111827', plot_bgcolor='#0f172a',
                          font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — PLAYER COMPARISON RADAR
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🎯  Player Comparison':
    st.header('🎯 Player Comparison — Radar Chart')
    st.markdown("Select **two players** to compare their normalised stats head-to-head.")

    all_names = sorted(df['Player_Name'].tolist())
    c1, c2 = st.columns(2)

    with c1:
        p1 = st.selectbox('Player 1', all_names, index=0)
        if p1:
            r1 = df[df['Player_Name'] == p1].iloc[0]
            st.markdown(f"""
<div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:1rem;">
<p style="color:#00d4ff;font-weight:700;font-size:1.1rem;">{r1['Player_Name']}</p>
<p style="color:#94a3b8;margin:0">🏟 {r1['Club']} · {r1['League']}</p>
<p style="color:#94a3b8;margin:0">📍 {r1['Position']} · Age {r1['Age']} · {r1['Nationality']}</p>
<p style="color:#94a3b8;margin:0">💶 €{r1['Market_Value_Million_Euros']}M</p>
<p style="margin:0"><span style="color:#00d4ff;font-weight:700">Transfer Score: {r1['Transfer_Score']}</span>
 — <span style="color:#94a3b8">{r1['Transfer_Tier']}</span></p>
</div>""", unsafe_allow_html=True)

    with c2:
        default_p2 = all_names[1] if len(all_names) > 1 else all_names[0]
        p2 = st.selectbox('Player 2', all_names, index=1)
        if p2:
            r2 = df[df['Player_Name'] == p2].iloc[0]
            st.markdown(f"""
<div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:1rem;">
<p style="color:#4ade80;font-weight:700;font-size:1.1rem;">{r2['Player_Name']}</p>
<p style="color:#94a3b8;margin:0">🏟 {r2['Club']} · {r2['League']}</p>
<p style="color:#94a3b8;margin:0">📍 {r2['Position']} · Age {r2['Age']} · {r2['Nationality']}</p>
<p style="color:#94a3b8;margin:0">💶 €{r2['Market_Value_Million_Euros']}M</p>
<p style="margin:0"><span style="color:#4ade80;font-weight:700">Transfer Score: {r2['Transfer_Score']}</span>
 — <span style="color:#94a3b8">{r2['Transfer_Tier']}</span></p>
</div>""", unsafe_allow_html=True)

    if p1 and p2:
        try:
            st.plotly_chart(build_two_player_radar(df, p1, p2), use_container_width=True)
        except Exception as e:
            st.error(f"Radar error: {e}")

    st.divider()
    st.subheader('Position Averages Comparison')
    st.plotly_chart(build_position_comparison(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — TRANSFER RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🤖  Transfer Rankings':
    st.header('🤖 AI Transfer Recommendation Panel')
    st.markdown("Ranked by **weighted Transfer Score**: Performance 35% · Value-for-Money 30% · Availability 20% · Potential 15%")

    t1,t2,t3,t4 = st.tabs(['🏆 Top 10 Overall','🌟 Young Talents','💰 Best Value','📋 Full Rankings'])

    FMT = {'Market_Value_Million_Euros':'€{:.1f}M',
           'Performance_Index':'{:.1f}',
           'Value_for_Money':'{:.3f}',
           'Transfer_Score':'{:.1f}',
           'Potential_Index':'{:.3f}'}

    with t1:
        top10 = top_recommendations(df, n=10).reset_index(drop=True)
        top10.index = top10.index + 1
        st.dataframe(top10.style.format(FMT)
                     .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
                     use_container_width=True)
        fig = px.bar(top10.reset_index().rename(columns={'index':'Rank'}),
                     x='Player_Name', y='Transfer_Score', color='Position',
                     color_discrete_map={'GK':'#f97316','Defender':'#22d3ee',
                                         'Midfielder':'#a78bfa','Forward':'#4ade80'},
                     title='Top 10 Transfer Targets')
        fig.update_layout(xaxis_tickangle=-30, height=420,
                          paper_bgcolor='#111827', plot_bgcolor='#0f172a', font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        young = best_young_talents(df, n=10).reset_index(drop=True)
        young.index = young.index + 1
        st.dataframe(young.style.format(FMT)
                     .background_gradient(subset=['Transfer_Score'], cmap='Purples'),
                     use_container_width=True)
        fig2 = px.scatter(young.reset_index(), x='Market_Value_Million_Euros',
                          y='Transfer_Score', size='Potential_Index', color='Position',
                          hover_data=['Player_Name','Age'],
                          color_discrete_map={'GK':'#f97316','Defender':'#22d3ee',
                                              'Midfielder':'#a78bfa','Forward':'#4ade80'},
                          title='Young Talents: Value vs Score')
        fig2.update_layout(paper_bgcolor='#111827', plot_bgcolor='#0f172a', font_color='#e2e8f0')
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        val = best_value_signings(df, n=10).reset_index(drop=True)
        val.index = val.index + 1
        st.dataframe(val.style.format(FMT)
                     .background_gradient(subset=['Value_for_Money'], cmap='Greens'),
                     use_container_width=True)
        fig3 = px.bar(val.reset_index(), x='Player_Name', y='Value_for_Money',
                      color='Position',
                      color_discrete_map={'GK':'#f97316','Defender':'#22d3ee',
                                          'Midfielder':'#a78bfa','Forward':'#4ade80'},
                      title='Best Value Signings')
        fig3.update_layout(xaxis_tickangle=-30, height=420,
                           paper_bgcolor='#111827', plot_bgcolor='#0f172a', font_color='#e2e8f0')
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        fc1,fc2 = st.columns(2)
        pos_f  = fc1.selectbox('Position', ['All']+sorted(df['Position'].unique().tolist()), key='rk_pos')
        size_f = fc2.multiselect('Club Size', ['Small','Medium','Big'],
                                  default=['Small','Medium','Big'], key='rk_size')
        max_v  = st.slider('Max Market Value (€M)', 0.5, 200.0, 200.0, key='rk_val')

        mask = df['Club_Size'].isin(size_f) & (df['Market_Value_Million_Euros'] <= max_v)
        if pos_f != 'All':
            mask &= df['Position'] == pos_f

        ranked = (df[mask]
                  .sort_values('Transfer_Score', ascending=False)
                  [['Player_Name','Age','Position','Club','League','Nationality',
                    'Market_Value_Million_Euros','Performance_Index',
                    'Transfer_Score','Transfer_Tier']]
                  .reset_index(drop=True))
        ranked.index = ranked.index + 1   # start at 1

        st.dataframe(
            ranked.style.format({'Market_Value_Million_Euros':'€{:.1f}M',
                                  'Performance_Index':'{:.1f}',
                                  'Transfer_Score':'{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=520)
        st.caption(f"**{len(ranked)}** players shown")
