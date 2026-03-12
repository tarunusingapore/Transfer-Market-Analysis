import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DARK = "plotly_dark"

POS_COLOUR = {
    'GK':         '#f97316',
    'Defender':   '#22d3ee',
    'Midfielder': '#a78bfa',
    'Forward':    '#4ade80',
}
QUAD_COLOUR = {
    'Undervalued Gems':   '#4ade80',
    'Overpriced Stars':   '#f87171',
    'Average Players':    '#94a3b8',
    'Declining Veterans': '#fbbf24',
}

def _dark(fig, h=None):
    upd = dict(template=DARK, paper_bgcolor='#111827', plot_bgcolor='#0f172a',
               font_color='#e2e8f0')
    if h:
        upd['height'] = h
    fig.update_layout(**upd)
    return fig

def plot_correlation_heatmap(df):
    drop = [c for c in df.columns if c.endswith('_z')]
    skip = ['Player_ID','Player_Name','Nationality','Position','Club','Club_Size',
            'League','Age_Group','Transfer_Tier','Quadrant']
    num = df.drop(columns=drop+skip, errors='ignore').select_dtypes('number')
    corr = num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    aspect='auto', title='Correlation Heatmap')
    return _dark(fig, 720)

def plot_value_vs_performance(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Position', color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Age','Club'],
                     size='Matches_Played',
                     title='Market Value vs Performance Index',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    return _dark(fig, 540)

def plot_goals_vs_assists(df):
    # No trendline to avoid statsmodels dependency
    fig = px.scatter(df, x='Goals', y='Assists', color='Position',
                     color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Market_Value_Million_Euros'],
                     title='Goals vs Assists by Position')
    return _dark(fig, 500)

def plot_age_performance_curve(df):
    agg = df.groupby('Age')['Performance_Index'].agg(['mean','std']).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg['Age'], y=agg['mean'], mode='lines+markers',
                             name='Avg Performance',
                             line=dict(color='#00d4ff', width=3)))
    fig.add_trace(go.Scatter(
        x=pd.concat([agg['Age'], agg['Age'][::-1]]),
        y=pd.concat([agg['mean']+agg['std'], (agg['mean']-agg['std'])[::-1]]),
        fill='toself', fillcolor='rgba(0,212,255,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='±1 Std Dev'))
    fig.update_layout(title='Age vs Performance Curve',
                      xaxis_title='Age', yaxis_title='Avg Performance Index')
    return _dark(fig, 500)

def plot_position_performance(df):
    agg = df.groupby('Position')['Performance_Index'].mean().reset_index()
    fig = px.bar(agg, x='Position', y='Performance_Index',
                 color='Position', color_discrete_map=POS_COLOUR,
                 title='Avg Performance Index by Position',
                 labels={'Performance_Index':'Avg Performance Index'})
    fig.update_layout(showlegend=False)
    return _dark(fig, 440)

def plot_club_size_value(df):
    fig = px.box(df, x='Club_Size', y='Market_Value_Million_Euros',
                 color='Club_Size',
                 category_orders={'Club_Size':['Small','Medium','Big']},
                 title='Market Value by Club Size',
                 labels={'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.update_layout(showlegend=False)
    return _dark(fig, 480)

def plot_value_for_money_dist(df):
    fig = px.histogram(df, x='Value_for_Money', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=40, barmode='overlay', opacity=0.75,
                       title='Value-for-Money Distribution',
                       labels={'Value_for_Money':'Value for Money Index'})
    return _dark(fig, 480)

def plot_transfer_score_dist(df):
    fig = px.histogram(df, x='Transfer_Score', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=30, barmode='overlay', opacity=0.75,
                       title='Transfer Score Distribution')
    for tier, col in [(80,'#4ade80'),(60,'#fbbf24'),(40,'#f87171')]:
        fig.add_vline(x=tier, line_dash='dash', line_color=col,
                      annotation_text=str(tier), annotation_font_color=col)
    return _dark(fig, 480)

def plot_quadrant_chart(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Quadrant', color_discrete_map=QUAD_COLOUR,
                     hover_data=['Player_Name','Age','Position'],
                     title='Player Quadrant Analysis',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.add_hline(y=df['Market_Value_Million_Euros'].median(),
                  line_dash='dash', line_color='#475569')
    fig.add_vline(x=df['Performance_Index'].median(),
                  line_dash='dash', line_color='#475569')
    return _dark(fig, 580)

def plot_age_group_pie(df):
    counts = df['Age_Group'].value_counts().reset_index()
    counts.columns = ['Age_Group','Count']
    fig = px.pie(counts, names='Age_Group', values='Count',
                 title='Age Group Distribution',
                 color_discrete_sequence=['#00d4ff','#a78bfa','#4ade80'])
    return _dark(fig, 440)

def plot_league_distribution(df):
    cnt = df['League'].value_counts().reset_index()
    cnt.columns = ['League','Count']
    fig = px.bar(cnt, x='Count', y='League', orientation='h',
                 title='Players by League',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return _dark(fig, 480)
