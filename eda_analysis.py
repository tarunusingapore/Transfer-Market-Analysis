import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
POS_COLOUR = {
    'GK':         '#FF6B6B',
    'Defender':   '#4ECDC4',
    'Midfielder': '#45B7D1',
    'Forward':    '#96CEB4',
}

# ── 1. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include='number').drop(
        columns=[c for c in df.columns if c.endswith('_z')], errors='ignore'
    )
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Correlation Heatmap of Player Metrics',
    )
    fig.update_layout(height=700)
    return fig


# ── 2. Market value vs performance ───────────────────────────────────────────
def plot_value_vs_performance(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x='Performance_Index',
        y='Market_Value_Million_Euros',
        color='Position',
        color_discrete_map=POS_COLOUR,
        hover_data=['Player_Name', 'Age', 'Club_Size'],
        size='Matches_Played',
        title='Market Value vs Performance Index',
        labels={
            'Performance_Index': 'Performance Index',
            'Market_Value_Million_Euros': 'Market Value (€M)',
        },
    )
    fig.update_layout(height=520)
    return fig


# ── 3. Goals vs assists ───────────────────────────────────────────────────────
def plot_goals_vs_assists(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x='Goals', y='Assists',
        color='Position',
        color_discrete_map=POS_COLOUR,
        hover_data=['Player_Name', 'Market_Value_Million_Euros'],
        title='Goals vs Assists by Position',
        trendline='ols',
    )
    fig.update_layout(height=480)
    return fig


# ── 4. Age performance curve ──────────────────────────────────────────────────
def plot_age_performance_curve(df: pd.DataFrame) -> go.Figure:
    agg = (
        df.groupby('Age')['Performance_Index']
        .agg(['mean', 'std'])
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg['Age'], y=agg['mean'],
        mode='lines+markers',
        name='Avg Performance',
        line=dict(color='#45B7D1', width=3),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([agg['Age'], agg['Age'][::-1]]),
        y=pd.concat([agg['mean'] + agg['std'], (agg['mean'] - agg['std'])[::-1]]),
        fill='toself', fillcolor='rgba(69,183,209,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
    ))
    fig.update_layout(
        title='Age vs Performance Index Curve',
        xaxis_title='Age', yaxis_title='Average Performance Index',
        height=480,
    )
    return fig


# ── 5. Position vs performance bar chart ─────────────────────────────────────
def plot_position_performance(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby('Position')['Performance_Index'].mean().reset_index()
    fig = px.bar(
        agg, x='Position', y='Performance_Index',
        color='Position', color_discrete_map=POS_COLOUR,
        title='Average Performance Index by Position',
        labels={'Performance_Index': 'Avg Performance Index'},
    )
    fig.update_layout(height=420, showlegend=False)
    return fig


# ── 6. Club size vs market value ──────────────────────────────────────────────
def plot_club_size_value(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x='Club_Size', y='Market_Value_Million_Euros',
        color='Club_Size',
        category_orders={'Club_Size': ['Small', 'Medium', 'Big']},
        title='Market Value Distribution by Club Size',
        labels={'Market_Value_Million_Euros': 'Market Value (€M)'},
    )
    fig.update_layout(height=460, showlegend=False)
    return fig


# ── 7. Value-for-money distribution ─────────────────────────────────────────
def plot_value_for_money_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x='Value_for_Money', color='Position',
        color_discrete_map=POS_COLOUR,
        nbins=50, barmode='overlay', opacity=0.7,
        title='Value-for-Money Distribution by Position',
        labels={'Value_for_Money': 'Value for Money Index'},
    )
    fig.update_layout(height=460)
    return fig


# ── 8. Transfer score distribution ───────────────────────────────────────────
def plot_transfer_score_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x='Transfer_Score', color='Position',
        color_discrete_map=POS_COLOUR,
        nbins=40, barmode='overlay', opacity=0.75,
        title='Transfer Score Distribution',
    )
    for tier, colour in [(80, 'green'), (60, 'orange'), (40, 'red')]:
        fig.add_vline(x=tier, line_dash='dash', line_color=colour,
                      annotation_text=f'{tier}', annotation_position='top right')
    fig.update_layout(height=460)
    return fig


# ── 9. Quadrant chart ─────────────────────────────────────────────────────────
def plot_quadrant_chart(df: pd.DataFrame) -> go.Figure:
    quad_colours = {
        'Undervalued Gems': '#2ecc71',
        'Overpriced Stars': '#e74c3c',
        'Average Players':  '#95a5a6',
        'Declining Veterans': '#f39c12',
    }
    fig = px.scatter(
        df,
        x='Performance_Index',
        y='Market_Value_Million_Euros',
        color='Quadrant',
        color_discrete_map=quad_colours,
        hover_data=['Player_Name', 'Age', 'Position'],
        title='Player Quadrant Analysis',
        labels={
            'Performance_Index': 'Performance Index',
            'Market_Value_Million_Euros': 'Market Value (€M)',
        },
    )
    med_perf  = df['Performance_Index'].median()
    med_value = df['Market_Value_Million_Euros'].median()
    fig.add_hline(y=med_value, line_dash='dash', line_color='grey')
    fig.add_vline(x=med_perf,  line_dash='dash', line_color='grey')
    fig.update_layout(height=560)
    return fig


# ── 10. Age group pie ─────────────────────────────────────────────────────────
def plot_age_group_pie(df: pd.DataFrame) -> go.Figure:
    counts = df['Age_Group'].value_counts().reset_index()
    counts.columns = ['Age_Group', 'Count']
    fig = px.pie(counts, names='Age_Group', values='Count',
                 title='Age Group Distribution',
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=420)
    return fig
