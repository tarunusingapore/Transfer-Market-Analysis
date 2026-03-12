"""eda_analysis.py — All EDA charts, fully light-themed."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

POS_COLOUR = {
    'GK':         '#f97316',
    'Defender':   '#3b82f6',
    'Midfielder': '#8b5cf6',
    'Forward':    '#16a34a',
}
QUAD_COLOUR = {
    'Undervalued Gems':   '#16a34a',
    'Overpriced Stars':   '#ef4444',
    'Average Players':    '#64748b',
    'Declining Veterans': '#f59e0b',
}

def _L(fig, h=500):
    """Apply consistent light layout to any Plotly figure."""
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(color='#0f172a', family='sans-serif'),
        height=h,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    # Force ALL axes that might have been set dark
    fig.update_xaxes(color='#0f172a', gridcolor='#e2e8f0', linecolor='#cbd5e1')
    fig.update_yaxes(color='#0f172a', gridcolor='#e2e8f0', linecolor='#cbd5e1')
    return fig


def plot_correlation_heatmap(df):
    drop = [c for c in df.columns if c.endswith('_z')]
    skip = ['Player_ID','Player_Name','Nationality','Position','Club','Club_Size',
            'League','Age_Group','Transfer_Tier','Quadrant','Cluster','Cluster_Label',
            'PCA_1','PCA_2','Profile']
    num = df.drop(columns=drop+skip, errors='ignore').select_dtypes('number')
    corr = num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    aspect='auto', title='Correlation Heatmap — Numeric Features')
    return _L(fig, 720)

def plot_value_vs_performance(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Position', color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Age','Club'],
                     size='Matches_Played',
                     title='Market Value vs Performance Index',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    return _L(fig, 540)

def plot_goals_vs_assists(df):
    fig = px.scatter(df, x='Goals', y='Assists', color='Position',
                     color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Club','Market_Value_Million_Euros'],
                     title='Goals vs Assists by Position')
    return _L(fig, 500)

def plot_age_performance_curve(df):
    agg = df.groupby('Age')['Performance_Index'].agg(['mean','std']).reset_index()
    agg['std'] = agg['std'].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([agg['Age'], agg['Age'][::-1]]),
        y=pd.concat([agg['mean']+agg['std'], (agg['mean']-agg['std'])[::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='±1 Std Dev',
    ))
    fig.add_trace(go.Scatter(
        x=agg['Age'], y=agg['mean'], mode='lines+markers',
        name='Avg Performance', line=dict(color='#1d4ed8', width=3),
        marker=dict(color='#1d4ed8', size=6),
    ))
    fig.update_layout(title='Age vs Performance Curve',
                      xaxis_title='Age', yaxis_title='Avg Performance Index')
    return _L(fig, 500)

def plot_position_performance(df):
    agg = df.groupby('Position')['Performance_Index'].mean().reset_index()
    fig = px.bar(agg, x='Position', y='Performance_Index',
                 color='Position', color_discrete_map=POS_COLOUR,
                 title='Avg Performance Index by Position',
                 labels={'Performance_Index':'Avg Performance Index'})
    fig.update_layout(showlegend=False)
    return _L(fig, 440)

def plot_club_size_value(df):
    fig = px.box(df, x='Club_Size', y='Market_Value_Million_Euros',
                 color='Club_Size',
                 category_orders={'Club_Size':['Small','Medium','Big']},
                 title='Market Value Distribution by Club Size',
                 labels={'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.update_layout(showlegend=False)
    return _L(fig, 480)

def plot_value_for_money_dist(df):
    fig = px.histogram(df, x='Value_for_Money', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=40, barmode='overlay', opacity=0.75,
                       title='Value-for-Money Distribution by Position',
                       labels={'Value_for_Money':'Value for Money Index'})
    return _L(fig, 480)

def plot_transfer_score_dist(df):
    fig = px.histogram(df, x='Transfer_Score', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=30, barmode='overlay', opacity=0.75,
                       title='Transfer Score Distribution by Position')
    for tier, col, lbl in [(80,'#16a34a','Elite'),(60,'#f59e0b','Strong'),(40,'#ef4444','Moderate')]:
        fig.add_vline(x=tier, line_dash='dash', line_color=col,
                      annotation_text=lbl, annotation_font_color=col,
                      annotation_position='top right')
    return _L(fig, 480)

def plot_quadrant_chart(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Quadrant', color_discrete_map=QUAD_COLOUR,
                     hover_data=['Player_Name','Age','Position','Club'],
                     title='Player Quadrant Analysis',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.add_hline(y=df['Market_Value_Million_Euros'].median(),
                  line_dash='dash', line_color='#94a3b8',
                  annotation_text='Median Value', annotation_font_color='#64748b')
    fig.add_vline(x=df['Performance_Index'].median(),
                  line_dash='dash', line_color='#94a3b8',
                  annotation_text='Median Perf', annotation_font_color='#64748b')
    return _L(fig, 580)

def plot_age_group_pie(df):
    counts = df['Age_Group'].value_counts().reset_index()
    counts.columns = ['Age_Group','Count']
    fig = px.pie(counts, names='Age_Group', values='Count',
                 title='Squad Age Group Distribution',
                 color_discrete_sequence=['#1d4ed8','#8b5cf6','#16a34a'])
    fig.update_traces(textfont_color='#0f172a')
    return _L(fig, 440)

def plot_league_distribution(df):
    cnt = df['League'].value_counts().reset_index()
    cnt.columns = ['League','Count']
    fig = px.bar(cnt, x='Count', y='League', orientation='h',
                 title='Player Count by League',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder':'total ascending'},
                      coloraxis_showscale=False)
    return _L(fig, 520)

def plot_missing_values(df_raw):
    """Show missing value counts per column — for Dataset Overview tab."""
    miss = df_raw.isnull().sum().reset_index()
    miss.columns = ['Column','Missing']
    miss = miss[miss['Missing'] > 0].sort_values('Missing', ascending=False)
    if miss.empty:
        fig = go.Figure()
        fig.add_annotation(text='✓ No missing values in this dataset',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           font=dict(size=18, color='#16a34a'), showarrow=False)
        return _L(fig, 300)
    miss['Pct'] = (miss['Missing'] / len(df_raw) * 100).round(1)
    fig = px.bar(miss, x='Column', y='Pct',
                 title='Missing Values (% of rows)',
                 labels={'Pct':'Missing (%)'},
                 color='Pct', color_continuous_scale='Reds')
    fig.update_layout(coloraxis_showscale=False)
    return _L(fig, 400)

def plot_numeric_distributions(df, selected_cols: list):
    """Small-multiples histogram grid."""
    n = len(selected_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=cols_per_row,
                        subplot_titles=selected_cols)
    for idx, col in enumerate(selected_cols):
        r, c = divmod(idx, cols_per_row)
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col,
                         marker_color='#3b82f6', opacity=0.8,
                         showlegend=False),
            row=r+1, col=c+1,
        )
    fig.update_layout(template='plotly_white', paper_bgcolor='white',
                      plot_bgcolor='#f8fafc', font_color='#0f172a',
                      height=max(300, rows * 250), showlegend=False)
    fig.update_xaxes(color='#0f172a', gridcolor='#e2e8f0')
    fig.update_yaxes(color='#0f172a', gridcolor='#e2e8f0')
    return fig
