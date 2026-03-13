"""FootballIQ — Streamlit Dashboard  |  streamlit run app.py"""

import os, warnings
warnings.filterwarnings('ignore')

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
from ml_models      import (
    # K-Means
    run_kmeans, plot_kmeans_pca, plot_archetype_radar,
    plot_cluster_composition, plot_elbow,
    # Classification
    train_classifiers, plot_clf_comparison, plot_confusion_matrix,
    plot_clf_feature_importance, CLF_FEATURES, STRONG_SIGNING_THRESHOLD,
    # Regression
    train_regressors, plot_regression_comparison, plot_reg_actual_vs_predicted,
    plot_residuals, plot_reg_feature_importance, predict_player_value,
    MV_FEATURES, REGRESSORS,
    # Club Fit
    compute_club_fit, plot_club_fit,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='⚽ FootballIQ Analytics',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded',
)
pio.templates.default = 'plotly_white'

# ── Light CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* KPI cards */
  .kpi-card {
      background: linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
      border:1px solid #bfdbfe; border-radius:12px;
      padding:1.2rem 1.4rem; text-align:center;
      box-shadow:0 2px 8px rgba(29,78,216,.08);
  }
  .kpi-card .kpi-val { font-size:2rem; font-weight:800; color:#1d4ed8; }
  .kpi-card .kpi-lab { font-size:.82rem; color:#475569; margin-top:.3rem; }

  /* Section title */
  .page-title { font-size:2.2rem; font-weight:800; color:#1e3a5f; margin-bottom:.1rem; }
  .page-sub   { color:#475569; font-size:1rem; margin-bottom:1.5rem; }

  /* Insight box */
  .insight {
      background:#eff6ff; border-left:3px solid #1d4ed8;
      padding:.75rem 1rem; border-radius:6px;
      font-size:.91rem; color:#1e3a5f; margin:.4rem 0 1rem 0;
  }
  .insight strong { color:#1d4ed8; }

  /* Data-profiling stat box */
  .stat-box {
      background:#f8fafc; border:1px solid #e2e8f0;
      border-radius:8px; padding:.8rem 1rem; text-align:center;
  }
  .stat-box .sval { font-size:1.4rem; font-weight:700; color:#1d4ed8; }
  .stat-box .slab { font-size:.78rem; color:#64748b; }

  /* Player comparison cards */
  .card-blue  { background:#eff6ff; border:2px solid #1d4ed8; border-radius:10px; padding:1rem 1.2rem; }
  .card-green { background:#f0fdf4; border:2px solid #16a34a; border-radius:10px; padding:1rem 1.2rem; }

  /* Metric badge */
  .badge {
      display:inline-block; padding:.25rem .6rem;
      border-radius:99px; font-size:.78rem; font-weight:600;
  }
  .badge-blue  { background:#dbeafe; color:#1d4ed8; }
  .badge-green { background:#dcfce7; color:#16a34a; }
  .badge-amber { background:#fef3c7; color:#92400e; }
  .badge-red   { background:#fee2e2; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# ── Colour maps ───────────────────────────────────────────────────────────────
POS_COLOURS = {'GK':'#f97316','Defender':'#3b82f6','Midfielder':'#8b5cf6','Forward':'#16a34a'}

def light_bar(df_in, x, y, title, colour_col='Position'):
    fig = px.bar(df_in, x=x, y=y, color=colour_col,
                 color_discrete_map=POS_COLOURS, title=title)
    fig.update_layout(xaxis_tickangle=-30, height=420,
                      paper_bgcolor='white', plot_bgcolor='#f8fafc', font_color='#0f172a')
    return fig

# ── Data pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading dataset…')
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'players_dataset.csv')
    df_raw = pd.read_csv(csv_path) if os.path.exists(csv_path) \
             else generate_football_dataset(310)
    # Deduplicate on Player_Name (keep first occurrence)
    df_raw = df_raw.drop_duplicates(subset='Player_Name').reset_index(drop=True)
    df = clean_data(df_raw)
    df = engineer_features(df)
    df = compute_transfer_score(df)
    df = quadrant_analysis(df)
    return df_raw, df

@st.cache_resource(show_spinner='Training ML models…')
def load_ml(df):
    # K-Means archetypes
    df_cl, km, scaler_km, arch_map, centers = run_kmeans(df, n_clusters=6)
    # Classification — Strong Signing prediction
    clf_results, clf_fitted, clf_scaler, pos_frac, X_tr, X_te, y_tr, y_te = train_classifiers(df)
    # Regression — Market Value prediction
    reg_results, reg_fitted, reg_scaler, X_te_reg, y_te_reg = train_regressors(df)
    return (df_cl, km, scaler_km, arch_map,
            clf_results, clf_fitted, clf_scaler, pos_frac,
            reg_results, reg_fitted, reg_scaler)

df_raw, df = load_data()
N = len(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ⚽ FootballIQ')
    st.markdown("<span style='color:#475569;font-size:.85rem'>Analytics-Driven Transfer Intelligence</span>",
                unsafe_allow_html=True)
    st.divider()
    page = st.radio('Navigate', [
        '🏠  Overview',
        '📋  Data Profile',
        '🧹  Data Cleaning',
        '📈  EDA Visualisations',
        '🔍  Player Scouting',
        '🎯  Player Comparison',
        '🤖  Transfer Rankings',
        '🧠  Machine Learning',
    ])
    st.divider()
    st.caption(f"**{N}** players · **{df['Nationality'].nunique()}** nationalities · **{df['League'].nunique()}** leagues")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == '🏠  Overview':
    st.markdown('<p class="page-title">⚽ FootballIQ Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Helping clubs find undervalued players through data science</p>',
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (f"{N}",                                          'Players in Database'),
        (f"{df['League'].nunique()}",                     'Leagues Covered'),
        (f"€{df['Market_Value_Million_Euros'].median():.0f}M", 'Median Market Value'),
        (f"{(df['Transfer_Score'] >= 60).sum()}",         'Recommended Targets'),
    ]
    for col,(val,lab) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-lab">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('---')
    left, right = st.columns([1.3,1])
    with left:
        st.subheader('🎯 The Problem')
        st.markdown(f"""
Clubs routinely overpay for famous names while elite performers in smaller leagues go unnoticed.
**FootballIQ** quantifies 27 metrics per player and produces a single **Transfer Score (0–100)**
so scouts can compare across positions, leagues, and club contexts objectively.

| Step | What we do |
|------|------------|
| **Data** | {N} real players, 27 metrics, 2023/24 season |
| **Engineering** | Performance, Value-for-Money, Availability, Potential indexes |
| **ML** | K-Means archetypes, RF market value prediction, Club Fit scoring |
| **Dashboard** | Interactive scouting, radar comparisons, ranked shortlists |
        """)
    with right:
        fig_pos = px.pie(df['Position'].value_counts().reset_index(),
                         names='Position', values='count', hole=0.5,
                         color='Position', color_discrete_map=POS_COLOURS)
        fig_pos.update_layout(height=300, margin=dict(t=20,b=10),
                              paper_bgcolor='white', font_color='#0f172a')
        st.plotly_chart(fig_pos, use_container_width=True)

        tier_df = df['Transfer_Tier'].value_counts().reset_index()
        tier_df.columns = ['Tier','Count']
        fig_tier = px.bar(tier_df, x='Count', y='Tier', orientation='h',
                          color='Count', color_continuous_scale='Blues',
                          title='Players by Transfer Tier')
        fig_tier.update_layout(height=260, paper_bgcolor='white', plot_bgcolor='#f8fafc',
                               font_color='#0f172a', yaxis={'categoryorder':'total ascending'},
                               showlegend=False, coloraxis_showscale=False, margin=dict(t=40))
        st.plotly_chart(fig_tier, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA PROFILE (replaces Dataset Explorer)
# ═════════════════════════════════════════════════════════════════════════════
elif page == '📋  Data Profile':
    st.header('📋 Data Profile & Overview')
    st.markdown("A structured overview of the raw dataset — shape, types, missingness, and distributions.")

    tab1, tab2, tab3, tab4 = st.tabs(['📐 Shape & Types', '🔍 Sample Data', '❓ Missing Values', '📊 Distributions'])

    with tab1:
        st.subheader('Dataset Dimensions')
        s1,s2,s3,s4 = st.columns(4)
        num_numeric = df_raw.select_dtypes('number').shape[1]
        num_cat     = df_raw.select_dtypes('object').shape[1]
        s1.markdown(f'<div class="stat-box"><div class="sval">{df_raw.shape[0]}</div><div class="slab">Rows</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-box"><div class="sval">{df_raw.shape[1]}</div><div class="slab">Columns</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-box"><div class="sval">{num_numeric}</div><div class="slab">Numeric Features</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="stat-box"><div class="sval">{num_cat}</div><div class="slab">Categorical Features</div></div>', unsafe_allow_html=True)

        st.markdown('---')
        st.subheader('Column Types & Non-Null Counts')
        dtype_df = pd.DataFrame({
            'Column':    df_raw.columns,
            'Data Type': df_raw.dtypes.astype(str).values,
            'Non-Null':  df_raw.count().values,
            'Null':      df_raw.isnull().sum().values,
            'Null %':    (df_raw.isnull().mean() * 100).round(2).values,
            'Unique':    df_raw.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=500)

    with tab2:
        st.subheader('Data Preview')
        n_rows = st.slider('Rows to display', 5, 50, 20)
        order  = st.radio('Show', ['First N rows','Last N rows','Random sample'], horizontal=True)
        if order == 'First N rows':
            sample = df_raw.head(n_rows)
        elif order == 'Last N rows':
            sample = df_raw.tail(n_rows)
        else:
            sample = df_raw.sample(n_rows, random_state=42)
        st.dataframe(sample.reset_index(drop=True), use_container_width=True)

        st.markdown('---')
        st.subheader('Descriptive Statistics')
        st.dataframe(df_raw.describe().T.round(3), use_container_width=True, height=480)

    with tab3:
        st.subheader('Missing Value Analysis')
        miss = pd.DataFrame({
            'Column':  df_raw.columns,
            'Missing': df_raw.isnull().sum().values,
            'Pct (%)': (df_raw.isnull().mean() * 100).round(2).values,
        }).query('Missing > 0').sort_values('Missing', ascending=False)

        if miss.empty:
            st.success('✅ No missing values detected across all columns!')
        else:
            st.dataframe(miss.reset_index(drop=True), use_container_width=True)
            fig_miss = px.bar(miss, x='Pct (%)', y='Column', orientation='h',
                              color='Pct (%)', color_continuous_scale='Reds',
                              title='Missing Value % by Column')
            fig_miss.update_layout(paper_bgcolor='white', plot_bgcolor='#f8fafc',
                                   font_color='#0f172a', coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)

        st.markdown('---')
        st.subheader('Duplicate Analysis')
        n_dupes = df_raw.duplicated().sum()
        n_name_dupes = df_raw.duplicated(subset='Player_Name').sum()
        c1,c2 = st.columns(2)
        c1.metric('Full-row duplicates', n_dupes)
        c2.metric('Duplicate player names', n_name_dupes)
        if n_name_dupes > 0:
            st.dataframe(df_raw[df_raw.duplicated(subset='Player_Name', keep=False)]
                         .sort_values('Player_Name'), use_container_width=True)
        else:
            st.success('✅ No duplicate player names detected.')

    with tab4:
        st.subheader('Feature Distributions')
        num_cols = df_raw.select_dtypes('number').columns.tolist()
        chosen = st.selectbox('Select numeric column', num_cols)
        col_a, col_b = st.columns(2)
        with col_a:
            fig_h = px.histogram(df_raw, x=chosen, color='Position' if 'Position' in df_raw.columns else None,
                                 color_discrete_map=POS_COLOURS, nbins=35, title=f'{chosen} — Distribution')
            fig_h.update_layout(paper_bgcolor='white', plot_bgcolor='#f8fafc', font_color='#0f172a')
            st.plotly_chart(fig_h, use_container_width=True)
        with col_b:
            fig_b = px.box(df_raw, y=chosen, color='Position' if 'Position' in df_raw.columns else None,
                           color_discrete_map=POS_COLOURS, title=f'{chosen} — Box Plot')
            fig_b.update_layout(paper_bgcolor='white', plot_bgcolor='#f8fafc', font_color='#0f172a')
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown('---')
        st.subheader('Categorical Breakdowns')
        cat_cols = df_raw.select_dtypes('object').columns.drop(['Player_ID','Player_Name'], errors='ignore').tolist()
        chosen_cat = st.selectbox('Select categorical column', cat_cols)
        vc = df_raw[chosen_cat].value_counts().reset_index()
        vc.columns = [chosen_cat,'Count']
        fig_cat = px.bar(vc.head(30), x=chosen_cat, y='Count',
                         title=f'{chosen_cat} — Value Counts (top 30)',
                         color='Count', color_continuous_scale='Blues')
        fig_cat.update_layout(xaxis_tickangle=-30, paper_bgcolor='white',
                               plot_bgcolor='#f8fafc', font_color='#0f172a',
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA CLEANING (richer, with before/after)
# ═════════════════════════════════════════════════════════════════════════════
elif page == '🧹  Data Cleaning':
    st.header('🧹 Data Cleaning & Feature Engineering')
    st.markdown("Step-by-step pipeline showing **before** and **after** for each transformation.")

    tab_steps, tab_formulas, tab_result = st.tabs(['🔧 Cleaning Steps','📐 Formulae','✅ Final Dataset'])

    with tab_steps:
        # Step 1 — Duplicates
        with st.expander('**Step 1 — Duplicate Removal**', expanded=True):
            col1,col2 = st.columns(2)
            with col1:
                st.markdown('**Before** — raw row count')
                st.metric('Rows', df_raw.shape[0])
                st.code("df.duplicated(subset='Player_Name').sum()", language='python')
            with col2:
                st.markdown('**After** — deduplicated')
                cleaned = df_raw.drop_duplicates(subset='Player_Name')
                st.metric('Rows', cleaned.shape[0], delta=f"{-(df_raw.shape[0]-cleaned.shape[0])}" if df_raw.shape[0]!=cleaned.shape[0] else "0 removed")
            st.markdown('<div class="insight">💡 <strong>Why:</strong> Duplicate players inflate performance aggregates and bias model training.</div>', unsafe_allow_html=True)

        # Step 2 — Missing values
        with st.expander('**Step 2 — Missing Value Imputation**', expanded=True):
            miss_before = df_raw.isnull().sum().sum()
            col1,col2 = st.columns(2)
            with col1:
                st.markdown('**Before**')
                st.metric('Total missing cells', miss_before)
                top_miss = df_raw.isnull().sum().sort_values(ascending=False).head(5)
                st.dataframe(top_miss.rename('Missing'), use_container_width=True)
            with col2:
                st.markdown('**After** — position-group median fill')
                miss_after = df.isnull().sum().sum()
                st.metric('Total missing cells', miss_after, delta=f"{-(miss_before-miss_after)}")
                st.code("df[col] = df[col].fillna(\n    df.groupby('Position')[col].transform('median')\n)", language='python')
            st.markdown('<div class="insight">💡 <strong>Why:</strong> Position-group medians preserve distributional integrity better than global means — a GK\'s pass accuracy differs greatly from a Forward\'s.</div>', unsafe_allow_html=True)

        # Step 3 — Outlier clipping
        with st.expander('**Step 3 — Outlier Clipping**', expanded=False):
            col1,col2 = st.columns(2)
            with col1:
                st.markdown('**Before** — raw Market Value range')
                st.metric('Max Market Value', f"€{df_raw['Market_Value_Million_Euros'].max():.0f}M")
                st.metric('Std Dev', f"€{df_raw['Market_Value_Million_Euros'].std():.1f}M")
            with col2:
                st.markdown('**After** — IQR clipping applied in Value_for_Money')
                st.metric('Availability Index range', f"0 – 1 (clipped)")
                st.code("df['Availability_Index'] = df['Availability_Index'].clip(0, 1)", language='python')
            st.markdown('<div class="insight">💡 <strong>Why:</strong> Extreme values distort min-max scaling and skew the Transfer Score calculation.</div>', unsafe_allow_html=True)

        # Step 4 — Age grouping
        with st.expander('**Step 4 — Age Group Categorisation**', expanded=False):
            grp = df['Age_Group'].value_counts().reset_index()
            grp.columns = ['Age Group','Count']
            st.dataframe(grp, use_container_width=True)
            st.code("df['Age_Group'] = pd.cut(df['Age'], bins=[0,23,29,99],\n"
                    "    labels=['Young Talent','Prime Player','Veteran'])", language='python')
            st.markdown('<div class="insight">💡 <strong>Why:</strong> Age buckets enable strategic recruitment filters — clubs targeting resale value need Young Talents; title chasers need Prime Players.</div>', unsafe_allow_html=True)

        # Step 5 — Feature engineering
        with st.expander('**Step 5 — Feature Engineering**', expanded=False):
            eng_features = ['Performance_Index','Value_for_Money','Availability_Index',
                            'Potential_Index','Scouting_Score']
            before_cols = list(df_raw.columns)
            after_cols  = list(df.columns)
            new_cols    = [c for c in after_cols if c not in before_cols]
            col1,col2 = st.columns(2)
            with col1:
                st.markdown('**Before** — original columns')
                st.metric('Columns', len(before_cols))
            with col2:
                st.markdown(f'**After** — {len(new_cols)} new features added')
                st.metric('Columns', len(after_cols), delta=f"+{len(new_cols)}")
                st.write(', '.join(new_cols))
            st.markdown('<div class="insight">💡 <strong>Why:</strong> Raw stats are not comparable across positions. Composite indexes normalise performance into position-agnostic scores.</div>', unsafe_allow_html=True)

    with tab_formulas:
        st.subheader('Engineered Feature Formulae')
        st.markdown("All indexes are min-max scaled to [0, 1] or [0, 100] unless stated.")

        feats = [
            ('Performance Index',
             r"\text{PI} = \sum_{i} w_i \cdot \text{metric}_i \quad \text{(position-specific weights)}",
             'Weighted sum of position-relevant stats. Forwards weight Goals/xG; Defenders weight Tackles/Interceptions.'),
            ('Value for Money',
             r"\text{VfM} = \frac{\text{Performance Index}}{\text{Market Value (€M)}}",
             'Higher = more output per euro spent. The primary signal for undervalued player detection.'),
            ('Availability Index',
             r"\text{Avail} = \frac{\text{Minutes Played}}{\text{Minutes Played} + \text{Injury Days} \times 90}",
             'Clips to [0,1]. Penalises injury-prone players regardless of peak performance.'),
            ('Potential Index',
             r"\text{Pot} = \text{Scouting Score} \times e^{-0.05 \cdot \max(0,\, \text{Age} - 23)}",
             'Exponential age decay applied after 23. Young high-scorers retain full potential.'),
            ('Transfer Score',
             r"\text{TS} = 0.35 \cdot P + 0.30 \cdot V + 0.20 \cdot A + 0.15 \cdot \text{Pot}",
             'Final composite score [0–100]. Weights reflect scouting priority: performance first, value second.'),
        ]
        for name, formula, explanation in feats:
            st.markdown(f'**{name}**')
            st.latex(formula)
            st.markdown(f'_{explanation}_')
            st.markdown('---')

    with tab_result:
        st.subheader('Cleaned & Engineered Dataset')
        eng_cols = ['Player_Name','Age','Age_Group','Position','Club','League',
                    'Performance_Index','Value_for_Money','Availability_Index',
                    'Potential_Index','Scouting_Score','Transfer_Score','Transfer_Tier']
        c1,c2,c3 = st.columns(3)
        pos_filter  = c1.multiselect('Position',  df['Position'].unique(), default=list(df['Position'].unique()))
        tier_filter = c2.multiselect('Tier', df['Transfer_Tier'].unique(), default=list(df['Transfer_Tier'].unique()))
        sort_col    = c3.selectbox('Sort by', ['Transfer_Score','Performance_Index','Value_for_Money','Market_Value_Million_Euros'])

        view = df[df['Position'].isin(pos_filter) & df['Transfer_Tier'].isin(tier_filter)]
        view = view.sort_values(sort_col, ascending=False)[eng_cols].reset_index(drop=True)
        view.index = view.index + 1
        st.dataframe(
            view.style.format({'Performance_Index':'{:.2f}','Value_for_Money':'{:.3f}',
                               'Availability_Index':'{:.2f}','Potential_Index':'{:.3f}',
                               'Scouting_Score':'{:.1f}','Transfer_Score':'{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=520,
        )
        st.caption(f"Showing **{len(view)}** of **{N}** players")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == '📈  EDA Visualisations':
    st.header('📈 Exploratory Data Analysis')

    VIS = {
        'Market Value vs Performance':   (plot_value_vs_performance,
            'Higher-performing players command higher valuations — but significant outliers '
            'exist in smaller leagues, representing prime scouting targets.'),
        'Goals vs Assists':              (plot_goals_vs_assists,
            'Forwards cluster top-right; midfielders show wider assist spread. '
            'Players above the trend line offer dual attacking threat.'),
        'Age vs Performance Curve':      (plot_age_performance_curve,
            'Performance peaks 24–28. The ±1 std band widens at extremes, '
            'indicating higher variance among young talents and veterans.'),
        'Performance by Position':       (plot_position_performance,
            'Midfielders score highest on the composite index due to multi-metric contributions.'),
        'Market Value by Club Size':     (plot_club_size_value,
            'Big-club players command a 3× median premium — creating the value gap FootballIQ exploits.'),
        'Value-for-Money Distribution':  (plot_value_for_money_dist,
            'Right-skewed distribution highlights rare undervalued gems. '
            'Targets in the top 5% of this metric offer exceptional ROI.'),
        'Transfer Score Distribution':   (plot_transfer_score_dist,
            'Most players score 40–60 (Moderate). The Strong tier (≥60) is reserved for elite targets.'),
        'Quadrant Analysis':             (plot_quadrant_chart,
            'Undervalued Gems (high performance, low cost) are the primary recruitment targets.'),
        'Age Group Breakdown':           (plot_age_group_pie,
            'The dataset skews toward Prime Players (24–29), reflecting top-flight squad compositions.'),
        'Players by League':             (plot_league_distribution,
            'Premier League and La Liga dominate. Smaller leagues offer higher concentrations of undervalued players.'),
        'Correlation Heatmap':           (plot_correlation_heatmap,
            'Goals and xG tightly correlated (r≈0.91). Market Value correlates with Performance Index (r≈0.76).'),
    }

    choice = st.selectbox('Select Visualisation', list(VIS.keys()))
    fn, insight = VIS[choice]
    st.plotly_chart(fn(df), use_container_width=True)
    st.markdown(f'<div class="insight">💡 <strong>Insight:</strong> {insight}</div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PLAYER SCOUTING
# ═════════════════════════════════════════════════════════════════════════════
elif page == '🔍  Player Scouting':
    st.header('🔍 Player Scouting Tool')
    st.markdown("Filter by criteria and rank players by **Transfer Score**.")

    with st.form('scout_form'):
        c1,c2,c3 = st.columns(3)
        pos_s  = c1.selectbox('Position', ['All'] + sorted(df['Position'].unique().tolist()))
        age_r  = c2.slider('Age Range', 15, 40, (18, 30))
        max_v  = c3.slider('Max Market Value (€M)', 1.0, 200.0, 80.0)
        c4,c5,c6 = st.columns(3)
        min_p  = c4.slider('Min Performance Index', 0.0, float(df['Performance_Index'].max()), 0.0)
        size_s = c5.multiselect('Club Size', ['Small','Medium','Big'], default=['Small','Medium','Big'])
        top_n  = c6.slider('Results to show', 5, 50, 15)
        go_btn = st.form_submit_button('🔍 Find Players', use_container_width=True)

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
        display.index = display.index + 1
        st.dataframe(
            display.style
            .format({'Market_Value_Million_Euros':'€{:.1f}M','Performance_Index':'{:.1f}','Transfer_Score':'{:.1f}'})
            .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
            use_container_width=True, height=460,
        )
        st.plotly_chart(light_bar(res.head(10), 'Player_Name', 'Transfer_Score', 'Top 10 Scouting Results'),
                        use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PLAYER COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == '🎯  Player Comparison':
    st.header('🎯 Player Comparison — Radar Chart')
    st.markdown("Select **two players** to compare their normalised attributes head-to-head.")

    all_names = sorted(df['Player_Name'].unique().tolist())

    # ── Independent columns so changing one does NOT reset the other ──
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox('🔵 Player 1', all_names, index=0, key='cmp_p1')
    with c2:
        # Default to index 1 (different player) but fully independent
        p2_default = 1 if len(all_names) > 1 else 0
        p2 = st.selectbox('🟢 Player 2', all_names, index=p2_default, key='cmp_p2')

    # ── Player cards — rendered independently ──
    card1, card2 = st.columns(2)
    r1 = df[df['Player_Name'] == p1].iloc[0]
    r2 = df[df['Player_Name'] == p2].iloc[0]

    with card1:
        st.markdown(f"""
<div class="card-blue">
  <p style="color:#1d4ed8;font-weight:700;font-size:1.1rem;margin:0 0 .5rem 0">🔵 {r1['Player_Name']}</p>
  <p style="color:#1e3a5f;margin:0">🏟 <b>{r1['Club']}</b> · {r1['League']}</p>
  <p style="color:#1e3a5f;margin:0">📍 {r1['Position']} · Age {r1['Age']} · {r1['Nationality']}</p>
  <p style="color:#1e3a5f;margin:0">💶 €{r1['Market_Value_Million_Euros']:.0f}M</p>
  <p style="color:#1d4ed8;font-weight:700;margin:.4rem 0 0 0">
     Transfer Score: {r1['Transfer_Score']:.1f} — {r1['Transfer_Tier']}</p>
</div>""", unsafe_allow_html=True)

    with card2:
        st.markdown(f"""
<div class="card-green">
  <p style="color:#16a34a;font-weight:700;font-size:1.1rem;margin:0 0 .5rem 0">🟢 {r2['Player_Name']}</p>
  <p style="color:#14532d;margin:0">🏟 <b>{r2['Club']}</b> · {r2['League']}</p>
  <p style="color:#14532d;margin:0">📍 {r2['Position']} · Age {r2['Age']} · {r2['Nationality']}</p>
  <p style="color:#14532d;margin:0">💶 €{r2['Market_Value_Million_Euros']:.0f}M</p>
  <p style="color:#16a34a;font-weight:700;margin:.4rem 0 0 0">
     Transfer Score: {r2['Transfer_Score']:.1f} — {r2['Transfer_Tier']}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('####')

    # ── Radar — always shows BOTH players independently ──
    try:
        fig_radar = build_two_player_radar(df, p1, p2)
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.error(f"Radar error: {e}")

    # ── Head-to-head stat table ──
    st.divider()
    st.subheader('📊 Head-to-Head Stats')
    stat_cols = ['Goals','Assists','Shots_per_Game','Key_Passes_per_Game','Pass_Accuracy',
                 'Dribbles_per_Game','Tackles_per_Game','Interceptions_per_Game',
                 'Performance_Index','Transfer_Score']
    h2h = pd.DataFrame({
        'Metric': stat_cols,
        p1:       [r1[c] for c in stat_cols],
        p2:       [r2[c] for c in stat_cols],
    })
    h2h['Winner'] = h2h.apply(lambda row: p1 if row[p1] > row[p2]
                               else (p2 if row[p2] > row[p1] else 'Draw'), axis=1)
    st.dataframe(h2h.style.format({p1:'{:.2f}', p2:'{:.2f}'}),
                 use_container_width=True, height=380)

    st.divider()
    st.subheader('📊 Position Averages Comparison')
    st.plotly_chart(build_position_comparison(df), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — TRANSFER RANKINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == '🤖  Transfer Rankings':
    st.header('🤖 AI Transfer Recommendation Panel')
    st.markdown("Ranked by **weighted Transfer Score**: Performance 35% · Value-for-Money 30% · Availability 20% · Potential 15%")

    t1,t2,t3,t4 = st.tabs(['🏆 Top 10 Overall','🌟 Young Talents','💰 Best Value','📋 Full Rankings'])
    FMT = {'Market_Value_Million_Euros':'€{:.1f}M','Performance_Index':'{:.1f}',
           'Value_for_Money':'{:.3f}','Transfer_Score':'{:.1f}','Potential_Index':'{:.3f}'}

    with t1:
        top10 = top_recommendations(df, n=10).reset_index(drop=True)
        top10.index = top10.index + 1
        st.dataframe(top10.style.format(FMT).background_gradient(subset=['Transfer_Score'], cmap='Blues'),
                     use_container_width=True)
        st.plotly_chart(light_bar(top10.reset_index(), 'Player_Name', 'Transfer_Score', 'Top 10 Transfer Targets'),
                        use_container_width=True)

    with t2:
        young = best_young_talents(df, n=10).reset_index(drop=True)
        young.index = young.index + 1
        st.dataframe(young.style.format(FMT).background_gradient(subset=['Transfer_Score'], cmap='Purples'),
                     use_container_width=True)
        fig2 = px.scatter(young.reset_index(), x='Market_Value_Million_Euros',
                          y='Transfer_Score', size='Potential_Index', color='Position',
                          hover_data=['Player_Name','Age'], color_discrete_map=POS_COLOURS,
                          title='Young Talents: Value vs Score')
        fig2.update_layout(paper_bgcolor='white', plot_bgcolor='#f8fafc', font_color='#0f172a')
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        val = best_value_signings(df, n=10).reset_index(drop=True)
        val.index = val.index + 1
        st.dataframe(val.style.format(FMT).background_gradient(subset=['Value_for_Money'], cmap='Greens'),
                     use_container_width=True)
        st.plotly_chart(light_bar(val.reset_index(), 'Player_Name', 'Value_for_Money', 'Best Value Signings'),
                        use_container_width=True)

    with t4:
        fc1,fc2 = st.columns(2)
        pos_f  = fc1.selectbox('Position', ['All']+sorted(df['Position'].unique().tolist()), key='rk_pos')
        size_f = fc2.multiselect('Club Size', ['Small','Medium','Big'], default=['Small','Medium','Big'], key='rk_size')
        max_v  = st.slider('Max Market Value (€M)', 0.5, 200.0, 200.0, key='rk_val')
        mask   = df['Club_Size'].isin(size_f) & (df['Market_Value_Million_Euros'] <= max_v)
        if pos_f != 'All':
            mask &= df['Position'] == pos_f
        ranked = (df[mask].sort_values('Transfer_Score', ascending=False)
                  [['Player_Name','Age','Position','Club','League','Nationality',
                    'Market_Value_Million_Euros','Performance_Index','Transfer_Score','Transfer_Tier']]
                  .reset_index(drop=True))
        ranked.index = ranked.index + 1
        st.dataframe(ranked.style
                     .format({'Market_Value_Million_Euros':'€{:.1f}M','Performance_Index':'{:.1f}','Transfer_Score':'{:.1f}'})
                     .background_gradient(subset=['Transfer_Score'], cmap='Blues'),
                     use_container_width=True, height=520)
        st.caption(f"**{len(ranked)}** players shown")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8 — MACHINE LEARNING
# ═════════════════════════════════════════════════════════════════════════════
elif page == '🧠  Machine Learning':
    st.header('🧠 Machine Learning Models')
    st.markdown("""
Four ML modules covering **unsupervised learning**, **classification**, **regression**, and **similarity scoring** —
each with multi-algorithm comparison, evaluation metrics, and visualisations.
    """)

    with st.spinner('Training all models on dataset…'):
        (df_cl, km, scaler_km, arch_map,
         clf_results, clf_fitted, clf_scaler, pos_frac,
         reg_results, reg_fitted, reg_scaler) = load_ml(df)

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        '🎨 Player Archetypes (K-Means)',
        '🏆 Signing Prediction (Classification)',
        '💶 Market Value (Regression)',
        '🏟 Club Fit Score',
    ])

    # ── TAB 1: K-MEANS ───────────────────────────────────────────────────────
    with ml_tab1:
        st.subheader('🎨 Player Archetype Profiling — K-Means Clustering')
        st.markdown("""
**Algorithm:** K-Means unsupervised clustering groups players by 9 on-pitch features
(goals, assists, shots, key passes, pass accuracy, dribbles, tackles, interceptions, minutes).
Each cluster is automatically labelled by its dominant statistical trait.
        """)

        col_info, col_k = st.columns([2,1])
        with col_k:
            st.metric('Optimal k (Elbow)', '6')
            if st.button('↺ Re-run Clustering', use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        with col_info:
            archetypes_found = df_cl['Archetype'].value_counts()
            st.markdown('**Archetypes discovered:**')
            arch_cols = st.columns(min(len(archetypes_found), 4))
            for i,(arch,cnt) in enumerate(archetypes_found.items()):
                arch_cols[i % 4].metric(arch, f'{cnt} players')

        st.divider()
        st.subheader('PCA Scatter — Player Archetypes')
        st.caption('Each point is a player. Axes = first two principal components of the 9 clustering features.')
        st.plotly_chart(plot_kmeans_pca(df_cl), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader('Elbow Method')
            st.caption('Inertia drops sharply until k=6, then plateaus — justifying our choice.')
            st.plotly_chart(plot_elbow(df), use_container_width=True)
        with col_b:
            st.subheader('Position Composition per Archetype')
            st.plotly_chart(plot_cluster_composition(df_cl), use_container_width=True)

        st.divider()
        st.subheader('Archetype Deep Dive')
        arch_list   = sorted(df_cl['Archetype'].unique().tolist())
        chosen_arch = st.selectbox('Select Archetype to inspect', arch_list, key='km_arch')
        col_r, col_t = st.columns([1,1])
        with col_r:
            st.plotly_chart(plot_archetype_radar(df_cl, chosen_arch), use_container_width=True)
        with col_t:
            arch_players = (df_cl[df_cl['Archetype']==chosen_arch]
                            [['Player_Name','Age','Position','Club','League','Transfer_Score']]
                            .sort_values('Transfer_Score', ascending=False).reset_index(drop=True))
            arch_players.index = arch_players.index + 1
            st.dataframe(arch_players, use_container_width=True, height=380)

    # ── TAB 2: CLASSIFICATION ────────────────────────────────────────────────
    with ml_tab2:
        st.subheader('🏆 Strong Signing Prediction — Classification')
        st.markdown(f"""
**Problem:** Predict whether a player is a **Strong Signing** (binary classification).

**Target variable:** `Strong_Signing = 1` if `Transfer_Score ≥ {STRONG_SIGNING_THRESHOLD}`, else `0`

**Class balance:** {pos_frac*100:.1f}% positive (strong signings) · {(1-pos_frac)*100:.1f}% negative

**Features used ({len(CLF_FEATURES)}):** Age, Goals, Assists, xG, xA, Key Passes, Dribbles, Tackles, Interceptions, Pass Accuracy, Minutes, Market Value, Injury Days, League Quality Index

**Evaluation:** Stratified 80/20 train-test split
        """)
        st.markdown('<div class="insight">💡 <strong>Why compare models?</strong> Logistic Regression is the linear baseline; Decision Tree is interpretable; Random Forest and Gradient Boosting capture non-linear patterns via ensembles. Comparing all four identifies the most robust approach for this dataset.</div>',
                    unsafe_allow_html=True)

        # ── Model comparison table ──────────────────────────────────────────
        st.divider()
        st.subheader('Model Comparison Table')
        clf_table = pd.DataFrame({
            'Model':     list(clf_results.keys()),
            'Accuracy':  [clf_results[m]['Accuracy']  for m in clf_results],
            'Precision': [clf_results[m]['Precision'] for m in clf_results],
            'Recall':    [clf_results[m]['Recall']    for m in clf_results],
            'F1 Score':  [clf_results[m]['F1 Score']  for m in clf_results],
        }).sort_values('F1 Score', ascending=False).reset_index(drop=True)
        clf_table.index = clf_table.index + 1

        st.dataframe(
            clf_table.style
            .format({'Accuracy':'{:.3f}','Precision':'{:.3f}','Recall':'{:.3f}','F1 Score':'{:.3f}'})
            .background_gradient(subset=['Accuracy','Precision','Recall','F1 Score'], cmap='Blues')
            .highlight_max(subset=['Accuracy','Precision','Recall','F1 Score'],
                           props='font-weight:bold;color:#16a34a'),
            use_container_width=True,
        )

        best_model = clf_table.iloc[0]['Model']
        best_f1    = clf_table.iloc[0]['F1 Score']
        st.success(f"**Best model:** {best_model} — F1 Score: **{best_f1:.3f}**")

        # ── Bar chart ──────────────────────────────────────────────────────
        st.subheader('Visual Comparison')
        st.plotly_chart(plot_clf_comparison(clf_results), use_container_width=True)

        # ── Confusion matrix + feature importance ─────────────────────────
        st.divider()
        st.subheader('Confusion Matrix & Feature Importance')
        cm_col, fi_col = st.columns(2)
        with cm_col:
            cm_model = st.selectbox('Select model', list(clf_results.keys()), key='cm_model')
            st.plotly_chart(plot_confusion_matrix(clf_results, cm_model), use_container_width=True)
            r = clf_results[cm_model]
            tn,fp,fn,tp = confusion_matrix(r['y_te'], r['y_pred']).ravel()
            st.caption(f"TP={tp}  FP={fp}  FN={fn}  TN={tn} — "
                       f"**True Positives** = correctly identified strong signings")
        with fi_col:
            fi_model = st.selectbox('Feature importance for', list(clf_results.keys()), key='fi_clf')
            st.plotly_chart(plot_clf_feature_importance(clf_fitted, fi_model), use_container_width=True)

    # ── TAB 3: REGRESSION ────────────────────────────────────────────────────
    with ml_tab3:
        st.subheader('💶 Market Value Prediction — Regression')
        st.markdown(f"""
**Problem:** Predict a player's **Market Value (€M)** from their statistics.

**Target variable:** `Market_Value_Million_Euros`

**Features used ({len(MV_FEATURES)}):** Age, Goals, Assists, Shots, Key Passes, Pass Accuracy, Dribbles, Tackles, Interceptions, xG, xA, Matches Played, Minutes, League Quality, Fan Popularity

**Models compared:** Linear, Ridge, Lasso (linear), Random Forest, Gradient Boosting (ensemble)

**Evaluation:** 80/20 train-test split · Metrics: R², MAE (€M), RMSE (€M)
        """)
        st.markdown('<div class="insight">💡 <strong>Why compare models?</strong> Linear models assume a straight-line relationship between features and value. Ridge and Lasso add regularisation to reduce overfitting. Ensemble methods capture complex non-linear patterns — useful because market value depends on many interacting factors.</div>',
                    unsafe_allow_html=True)

        # ── Model comparison table ──────────────────────────────────────────
        st.divider()
        st.subheader('Model Comparison Table')
        reg_table = pd.DataFrame({
            'Model': list(reg_results.keys()),
            'R²':    [reg_results[m]['R²']   for m in reg_results],
            'MAE':   [reg_results[m]['MAE']  for m in reg_results],
            'RMSE':  [reg_results[m]['RMSE'] for m in reg_results],
        }).sort_values('R²', ascending=False).reset_index(drop=True)
        reg_table.index = reg_table.index + 1

        st.dataframe(
            reg_table.style
            .format({'R²':'{:.3f}','MAE':'€{:.2f}M','RMSE':'€{:.2f}M'})
            .background_gradient(subset=['R²'], cmap='Blues')
            .background_gradient(subset=['MAE','RMSE'], cmap='Reds_r')
            .highlight_max(subset=['R²'], props='font-weight:bold;color:#16a34a')
            .highlight_min(subset=['MAE','RMSE'], props='font-weight:bold;color:#16a34a'),
            use_container_width=True,
        )

        best_reg_name = reg_table.iloc[0]['Model']
        best_r2       = reg_table.iloc[0]['R²']
        best_mae      = reg_table.iloc[0]['MAE']
        st.success(f"**Best model:** {best_reg_name} — R²: **{best_r2:.3f}** · MAE: **€{best_mae:.1f}M**")

        # ── Visual comparison ──────────────────────────────────────────────
        st.subheader('Metric Comparison Charts')
        st.plotly_chart(plot_regression_comparison(reg_results), use_container_width=True)

        # ── Per-model deep dive ─────────────────────────────────────────────
        st.divider()
        st.subheader('Model Deep Dive')
        reg_model_choice = st.selectbox('Select model to inspect', list(reg_results.keys()), key='reg_model')

        dv1, dv2 = st.columns(2)
        with dv1:
            st.plotly_chart(plot_reg_actual_vs_predicted(reg_results, reg_model_choice),
                            use_container_width=True)
        with dv2:
            st.plotly_chart(plot_residuals(reg_results, reg_model_choice), use_container_width=True)
            st.caption('Ideal residuals: symmetric, centred at zero, no skew.')

        st.plotly_chart(plot_reg_feature_importance(reg_fitted, reg_model_choice),
                        use_container_width=True)

        # ── Per-player predictions ─────────────────────────────────────────
        st.divider()
        st.subheader('🔮 Predict a Player\'s Market Value — All Models')
        all_names_mv = sorted(df['Player_Name'].unique().tolist())
        chosen_player = st.selectbox('Select player', all_names_mv, key='reg_player')

        preds_all  = predict_player_value(reg_fitted, reg_scaler, df, chosen_player)
        actual_val = float(df[df['Player_Name']==chosen_player].iloc[0]['Market_Value_Million_Euros'])

        pred_df = pd.DataFrame({
            'Model':    list(preds_all.keys()),
            'Predicted (€M)': list(preds_all.values()),
        })
        pred_df['Delta vs Actual'] = (pred_df['Predicted (€M)'] - actual_val).round(1)
        pred_df['Verdict'] = pred_df['Delta vs Actual'].apply(
            lambda d: '🟢 Undervalued' if d > 5 else ('🔴 Overvalued' if d < -5 else '⚪ Fair Value')
        )

        st.metric('Actual Market Value', f'€{actual_val:.1f}M')
        st.dataframe(pred_df.style
                     .format({'Predicted (€M)':'€{:.1f}M','Delta vs Actual':'{:+.1f}M'})
                     .background_gradient(subset=['Predicted (€M)'], cmap='Blues'),
                     use_container_width=True, hide_index=True)
        st.markdown('<div class="insight">💡 A positive delta means the model thinks this player is <strong>undervalued</strong> relative to their current market price.</div>',
                    unsafe_allow_html=True)

    # ── TAB 4: CLUB FIT ──────────────────────────────────────────────────────
    with ml_tab4:
        st.subheader('🏟 Club Fit Score — Cosine Similarity')
        st.markdown("""
**Algorithm:** Cosine similarity computes the angle between a player's statistical vector
and the mean vector of each club's existing squad.
A score of 100% means the player's profile is identical to the club's average output.

**Features used:** Goals, Assists, Key Passes, Pass Accuracy, Dribbles, Tackles, Interceptions, Shots, xG
        """)

        all_names_fit = sorted(df['Player_Name'].unique().tolist())
        fit_player = st.selectbox('Select player to find best club fits', all_names_fit, key='fit_player')
        top_n_fit  = st.slider('Top N clubs to show', 5, 20, 10, key='fit_topn')

        fit_df = compute_club_fit(df, fit_player, top_n=top_n_fit)
        if not fit_df.empty:
            current = df[df['Player_Name']==fit_player].iloc[0]
            st.info(f"**Current club:** {current['Club']} ({current['League']}) — excluded from results")
            col_chart, col_table = st.columns([1.4,1])
            with col_chart:
                st.plotly_chart(plot_club_fit(fit_df, fit_player), use_container_width=True)
            with col_table:
                fit_df.index = fit_df.index + 1
                st.dataframe(fit_df.style.format({'Fit_Score':'{:.1f}%'})
                             .background_gradient(subset=['Fit_Score'], cmap='Blues'),
                             use_container_width=True, height=400)
            st.markdown('<div class="insight">💡 <strong>Interpretation:</strong> A Fit Score ≥ 80% indicates strong stylistic compatibility. This is a data-driven indicator only and does not account for tactical system, squad needs, or transfer budget.</div>',
                        unsafe_allow_html=True)
        else:
            st.warning('Player not found in dataset.')
