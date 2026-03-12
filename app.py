"""FootballIQ — Streamlit Dashboard  |  streamlit run app.py"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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

# ── Light theme — use plotly's clean white template ───────────────────────────
pio.templates.default = "plotly_white"

# ── Light CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── KPI cards ── */
  .kpi-card {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      border: 1px solid #bfdbfe;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      text-align: center;
      box-shadow: 0 2px 8px rgba(29,78,216,0.08);
  }
  .kpi-card .kpi-val  { font-size: 2rem; font-weight: 800; color: #1d4ed8; }
  .kpi-card .kpi-lab  { font-size: 0.82rem; color: #475569; margin-top: 0.3rem; }

  /* ── Page title ── */
  .page-title {
      font-size: 2.2rem; font-weight: 800; color: #1e3a5f;
      margin-bottom: 0.1rem;
  }
  .page-sub { color: #475569; font-size: 1rem; margin-bottom: 1.5rem; }

  /* ── Insight box ── */
  .insight {
      background: #eff6ff;
      border-left: 3px solid #1d4ed8;
      padding: 0.75rem 1rem;
      border-radius: 6px;
      font-size: 0.91rem;
      color: #1e3a5f;
      margin: 0.4rem 0 1rem 0;
  }
  .insight strong { color: #1d4ed8; }

  /* ── Player comparison cards ── */
  .player-card-blue {
      background: #eff6ff;
      border: 2px solid #1d4ed8;
      border-radius: 10px;
      padding: 1rem 1.2rem;
  }
  .player-card-green {
      background: #f0fdf4;
      border: 2px solid #16a34a;
      border-radius: 10px;
      padding: 1rem 1.2rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Position / tier colour maps ───────────────────────────────────────────────
POS_COLOURS = {
    'GK':         '#f97316',
    'Defender':   '#3b82f6',
    'Midfielder': '#8b5cf6',
    'Forward':    '#16a34a',
}

# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading dataset…')
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'players_dataset.csv')
    df_raw = pd.read_csv(csv_path) if os.path.exists(csv_path) \
             else generate_football_dataset(310)
    df = clean_data(df_raw)
    df = engineer_features(df)
    df = compute_transfer_score(df)
    df = quadrant_analysis(df)
    return df_raw, df

df_raw, df = load_data()
N = len(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FootballIQ")
    st.markdown("<span style='color:#475569;font-size:0.85rem'>Analytics-Driven Transfer Intelligence</span>",
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
    st.caption(f"**{N}** players · **{df['Nationality'].nunique()}** nationalities · **{df['League'].nunique()}** leagues")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == '🏠  Overview':
    st.markdown('<p class="page-title">⚽ FootballIQ Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Helping clubs find undervalued players through data science</p>',
                unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{N}",                                    "Players in Database"),
        (f"{df['League'].nunique()}",               "Leagues Covered"),
        (f"€{df['Market_Value_Million_Euros'].median():.0f}M", "Median Market Value"),
        (f"{(df['Transfer_Score'] >= 60).sum()}",   "Recommended Targets"),
    ]
    for col, (val, lab) in zip([c1, c2, c3, c4], kpis):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lab">{lab}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("🎯 The Problem")
        st.markdown(f"""
Clubs routinely overpay for famous names while elite performers in smaller leagues go unnoticed.
**FootballIQ** quantifies 27 metrics per player and produces a single **Transfer Score (0–100)**
so scouts can compare across positions, leagues, and club contexts objectively.

| Step | What we do |
|------|------------|
| **Data** | {N} real players, 27 metrics, 2023/24 season |
| **Engineering** | Performance, Value-for-Money, Availability, Potential indexes |
| **Scoring** | Weighted min-max Transfer Score |
| **Dashboard** | Interactive scouting, radar comparisons, ranked shortlists |
        """)

    with right:
        fig_pos = px.pie(
            df['Position'].value_counts().reset_index(),
            names='Position', values='count', hole=0.5,
            color='Position', color_discrete_map=POS_COLOURS,
        )
        fig_pos.update_layout(height=300, margin=dict(t=20, b=10),
                              paper_bgcolor='white', font_color='#0f172a',
                              legend=dict(bgcolor='#f8fafc'))
        st.plotly_chart(fig_pos, use_container_width=True)

        tier_df = df['Transfer_Tier'].value_counts().reset_index()
        tier_df.columns = ['Tier', 'Count']
        fig_tier = px.bar(tier_df, x='Count', y='Tier', orientation='h',
                          color='Count', color_continuous_scale='Blues',
                          title='Players by Transfer Tier')
        fig_tier.update_layout(
            height=260, paper_bgcolor='white', plot_bgcolor='#f8fafc',
            font_color='#0f172a', yaxis={'categoryorder': 'total ascending'},
            showlegend=False, coloraxis_showscale=False, margin=dict(t=40),
        )
        st.plotly_chart(fig_tier, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📊  Dataset Explorer':
    st.header('📊 Dataset Explorer')
    tab_raw, tab_feat, tab_stats = st.tabs(['Raw Data', 'Engineered Features', 'Summary Stats'])

    with tab_raw:
        c1, c2, c3 = st.columns(3)
        pos_f    = c1.multiselect('Position',  df['Position'].unique(),         default=list(df['Position'].unique()))
        size_f   = c2.multiselect('Club Size', ['Small', 'Medium', 'Big'],       default=['Small', 'Medium', 'Big'])
        league_f = c3.multiselect('League',    sorted(df['League'].unique()),    default=list(df['League'].unique()))
        view = df[
            df['Position'].isin(pos_f) &
            df['Club_Size'].isin(size_f) &
            df['League'].isin(league_f)
        ]
        st.dataframe(view.reset_index(drop=True), use_container_width=True, height=480)
        st.caption(f"Showing **{len(view)}** of **{N}** players")

    with tab_feat:
        feat_cols = ['Player_Name', 'Age', 'Position', 'Club', 'League',
                     'Performance_Index', 'Value_for_Money', 'Availability_Index',
                     'Potential_Index', 'Scouting_Score', 'Transfer_Score', 'Transfer_Tier']
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
        'Market Value vs Performance':   (plot_value_vs_performance,
            'Higher-performing players command higher valuations — but significant '
            'outliers exist, especially in smaller leagues, representing potential targets.'),
        'Goals vs Assists':              (plot_goals_vs_assists,
            'Forwards cluster top-right; midfielders show wider assist spread. '
            'Players above the trend line offer dual attacking threat.'),
        'Age vs Performance Curve':      (plot_age_performance_curve,
            'Performance peaks 24–28. The ±1 std band widens at extremes, '
            'indicating higher variance among young talents and veterans.'),
        'Performance by Position':       (plot_position_performance,
            'Midfielders score highest on the composite index due to multi-metric contributions. '
            'GK scores reflect availability and passing rather than attacking output.'),
        'Market Value by Club Size':     (plot_club_size_value,
            'Big-club players command a 3× median premium. '
            'This premium creates the value gap FootballIQ exploits.'),
        'Value-for-Money Distribution':  (plot_value_for_money_dist,
            'Right-skewed distribution highlights rare undervalued gems. '
            'Targets in the top 5% of this metric offer exceptional ROI.'),
        'Transfer Score Distribution':   (plot_transfer_score_dist,
            'Most players score 40–60 (Moderate). '
            'The Elite tier (≥80) is deliberately scarce to maintain exclusivity.'),
        'Quadrant Analysis':             (plot_quadrant_chart,
            'Undervalued Gems (high performance, low cost) are the primary recruitment targets. '
            'Overpriced Stars may be liabilities despite their reputation.'),
        'Age Group Pie':                 (plot_age_group_pie,
            'The dataset skews toward Prime Players (24–29), reflecting top-flight squad compositions. '
            'Young Talents represent future upside.'),
        'Players by League':             (plot_league_distribution,
            'Premier League and La Liga dominate the database. '
            'Smaller leagues offer the highest concentration of undervalued players.'),
        'Correlation Heatmap':           (plot_correlation_heatmap,
            'Goals and xG are tightly correlated (r≈0.91). '
            'Market Value correlates strongly with Performance Index (r≈0.76).'),
    }

    choice = st.selectbox('Select Visualisation', list(VIS.keys()))
    fn, insight = VIS[choice]
    st.plotly_chart(fn(df), use_container_width=True)
    st.markdown(f'<div class="insight">💡 <strong>Insight:</strong> {insight}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — PLAYER SCOUTING
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🔍  Player Scouting':
    st.header('🔍 Player Scouting Tool')
    st.markdown("Filter by criteria and rank players by **Transfer Score**.")

    with st.form('scout_form'):
        c1, c2, c3 = st.columns(3)
        pos_s  = c1.selectbox('Position', ['All'] + sorted(df['Position'].unique().tolist()))
        age_r  = c2.slider('Age Range', 15, 40, (18, 30))
        max_v  = c3.slider('Max Market Value (€M)', 1.0, 200.0, 80.0)
        c4, c5, c6 = st.columns(3)
        min_p  = c4.slider('Min Performance Index', 0.0, float(df['Performance_Index'].max()), 0.0)
        size_s = c5.multiselect('Club Size', ['Small', 'Medium', 'Big'], default=['Small', 'Medium', 'Big'])
        top_n  = c6.slider('Results to show', 5, 50, 15)
        go_btn = st.form_submit_button('🔍 Find Players', use_container_width=True)

    if go_btn:
        mask = (
            (df['Age'] >= age_r[0]) & (df['Age'] <= age_r[1]) &
            (df['Market_Value_Million_Euros'] <= max_v) &
            (df['Performance_Index'] >= min_p) &
            (df['Club_Size'].isin(size_s))
        )
        if pos_s != 'All':
            mask &= df['Position'] == pos_s
        res = df[mask].sort_values('Transfer_Score', ascending=False).head(top_n)

        st.success(f"**{len(res)}** players match your criteria")

        cols = ['Player_Name', 'Age', 'Position', 'Club', 'League', 'Nationality',
                'Market_Value_Million_Euros', 'Performance_Index', 'Transfer_Score', 'Transfer_Tier']
        display = res[cols].copy().reset_index(drop=True)
        display.index = display.index + 1

        st.dataframe(
            display.style
            .format({'Market_Value_Million_Euros': '€{:.1f}M',
                     'Performance_Index': '{:.1f}',
                     'Transfer_Score': '{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=460,
        )

        fig = px.bar(res.head(10), x='Player_Name', y='Transfer_Score',
                     color='Position', color_discrete_map=POS_COLOURS,
                     title='Top 10 Scouting Results',
                     labels={'Transfer_Score': 'Transfer Score', 'Player_Name': ''})
        fig.update_layout(xaxis_tickangle=-30, height=420,
                          paper_bgcolor='white', plot_bgcolor='#f8fafc',
                          font_color='#0f172a')
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — PLAYER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🎯  Player Comparison':
    st.header('🎯 Player Comparison — Radar Chart')
    st.markdown("Select **two players** to compare their normalised attributes head-to-head.")

    all_names = sorted(df['Player_Name'].unique().tolist())
    c1, c2 = st.columns(2)

    with c1:
        p1 = st.selectbox('🔵 Player 1', all_names, index=0)
    with c2:
        p2 = st.selectbox('🟢 Player 2', all_names, index=min(1, len(all_names) - 1))

    # Show player cards
    col1, col2 = st.columns(2)
    if p1:
        r1 = df[df['Player_Name'] == p1].iloc[0]
        with col1:
            st.markdown(f"""
<div class="player-card-blue">
  <p style="color:#1d4ed8;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0">🔵 {r1['Player_Name']}</p>
  <p style="color:#1e3a5f;margin:0">🏟 <b>{r1['Club']}</b> · {r1['League']}</p>
  <p style="color:#1e3a5f;margin:0">📍 {r1['Position']} · Age {r1['Age']} · {r1['Nationality']}</p>
  <p style="color:#1e3a5f;margin:0">💶 €{r1['Market_Value_Million_Euros']}M</p>
  <p style="color:#1d4ed8;font-weight:700;margin:0.4rem 0 0 0">Transfer Score: {r1['Transfer_Score']:.1f} — {r1['Transfer_Tier']}</p>
</div>""", unsafe_allow_html=True)

    if p2:
        r2 = df[df['Player_Name'] == p2].iloc[0]
        with col2:
            st.markdown(f"""
<div class="player-card-green">
  <p style="color:#16a34a;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0">🟢 {r2['Player_Name']}</p>
  <p style="color:#14532d;margin:0">🏟 <b>{r2['Club']}</b> · {r2['League']}</p>
  <p style="color:#14532d;margin:0">📍 {r2['Position']} · Age {r2['Age']} · {r2['Nationality']}</p>
  <p style="color:#14532d;margin:0">💶 €{r2['Market_Value_Million_Euros']}M</p>
  <p style="color:#16a34a;font-weight:700;margin:0.4rem 0 0 0">Transfer Score: {r2['Transfer_Score']:.1f} — {r2['Transfer_Tier']}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("####")

    # Radar chart — always show both players
    if p1 and p2:
        try:
            fig_radar = build_two_player_radar(df, p1, p2)
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.error(f"Radar error: {e}")

    st.divider()
    st.subheader('📊 Position Averages Comparison')
    st.plotly_chart(build_position_comparison(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — TRANSFER RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == '🤖  Transfer Rankings':
    st.header('🤖 AI Transfer Recommendation Panel')
    st.markdown("Ranked by **weighted Transfer Score**: Performance 35% · Value-for-Money 30% · Availability 20% · Potential 15%")

    t1, t2, t3, t4 = st.tabs(['🏆 Top 10 Overall', '🌟 Young Talents', '💰 Best Value', '📋 Full Rankings'])

    FMT = {
        'Market_Value_Million_Euros': '€{:.1f}M',
        'Performance_Index': '{:.1f}',
        'Value_for_Money': '{:.3f}',
        'Transfer_Score': '{:.1f}',
        'Potential_Index': '{:.3f}',
    }

    def light_bar(df_in, x, y, title):
        fig = px.bar(df_in, x=x, y=y, color='Position',
                     color_discrete_map=POS_COLOURS, title=title)
        fig.update_layout(xaxis_tickangle=-30, height=420,
                          paper_bgcolor='white', plot_bgcolor='#f8fafc',
                          font_color='#0f172a')
        return fig

    with t1:
        top10 = top_recommendations(df, n=10).reset_index(drop=True)
        top10.index = top10.index + 1
        st.dataframe(top10.style.format(FMT)
                     .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
                     use_container_width=True)
        st.plotly_chart(
            light_bar(top10.reset_index(), 'Player_Name', 'Transfer_Score', 'Top 10 Transfer Targets'),
            use_container_width=True,
        )

    with t2:
        young = best_young_talents(df, n=10).reset_index(drop=True)
        young.index = young.index + 1
        st.dataframe(young.style.format(FMT)
                     .background_gradient(subset=['Transfer_Score'], cmap='Purples'),
                     use_container_width=True)
        fig2 = px.scatter(young.reset_index(), x='Market_Value_Million_Euros',
                          y='Transfer_Score', size='Potential_Index', color='Position',
                          hover_data=['Player_Name', 'Age'],
                          color_discrete_map=POS_COLOURS,
                          title='Young Talents: Value vs Score')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='#f8fafc', font_color='#0f172a')
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        val = best_value_signings(df, n=10).reset_index(drop=True)
        val.index = val.index + 1
        st.dataframe(val.style.format(FMT)
                     .background_gradient(subset=['Value_for_Money'], cmap='Greens'),
                     use_container_width=True)
        st.plotly_chart(
            light_bar(val.reset_index(), 'Player_Name', 'Value_for_Money', 'Best Value Signings'),
            use_container_width=True,
        )

    with t4:
        fc1, fc2 = st.columns(2)
        pos_f  = fc1.selectbox('Position', ['All'] + sorted(df['Position'].unique().tolist()), key='rk_pos')
        size_f = fc2.multiselect('Club Size', ['Small', 'Medium', 'Big'],
                                 default=['Small', 'Medium', 'Big'], key='rk_size')
        max_v  = st.slider('Max Market Value (€M)', 0.5, 200.0, 200.0, key='rk_val')

        mask = df['Club_Size'].isin(size_f) & (df['Market_Value_Million_Euros'] <= max_v)
        if pos_f != 'All':
            mask &= df['Position'] == pos_f

        ranked = (
            df[mask]
            .sort_values('Transfer_Score', ascending=False)
            [['Player_Name', 'Age', 'Position', 'Club', 'League', 'Nationality',
              'Market_Value_Million_Euros', 'Performance_Index',
              'Transfer_Score', 'Transfer_Tier']]
            .reset_index(drop=True)
        )
        ranked.index = ranked.index + 1

        st.dataframe(
            ranked.style
            .format({'Market_Value_Million_Euros': '€{:.1f}M',
                     'Performance_Index': '{:.1f}',
                     'Transfer_Score': '{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=520,
        )
        st.caption(f"**{len(ranked)}** players shown")
