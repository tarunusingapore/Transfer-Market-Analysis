import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

LIGHT = "plotly_white"

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

def _light(fig, h=None):
    upd = dict(template=LIGHT, paper_bgcolor='white', plot_bgcolor='#f8fafc',
               font_color='#0f172a')
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
    return _light(fig, 720)

def plot_value_vs_performance(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Position', color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Age','Club'],
                     size='Matches_Played',
                     title='Market Value vs Performance Index',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    return _light(fig, 540)

def plot_goals_vs_assists(df):
    fig = px.scatter(df, x='Goals', y='Assists', color='Position',
                     color_discrete_map=POS_COLOUR,
                     hover_data=['Player_Name','Market_Value_Million_Euros'],
                     title='Goals vs Assists by Position')
    return _light(fig, 500)

def plot_age_performance_curve(df):
    agg = df.groupby('Age')['Performance_Index'].agg(['mean','std']).reset_index()
    agg['std'] = agg['std'].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([agg['Age'], agg['Age'][::-1]]),
        y=pd.concat([agg['mean']+agg['std'], (agg['mean']-agg['std'])[::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='rgba(0,0,0,0)'), name='±1 Std Dev',
    ))
    fig.add_trace(go.Scatter(
        x=agg['Age'], y=agg['mean'], mode='lines+markers',
        name='Avg Performance', line=dict(color='#1d4ed8', width=3),
    ))
    fig.update_layout(title='Age vs Performance Curve',
                      xaxis_title='Age', yaxis_title='Avg Performance Index')
    return _light(fig, 500)

def plot_position_performance(df):
    agg = df.groupby('Position')['Performance_Index'].mean().reset_index()
    fig = px.bar(agg, x='Position', y='Performance_Index',
                 color='Position', color_discrete_map=POS_COLOUR,
                 title='Avg Performance Index by Position',
                 labels={'Performance_Index':'Avg Performance Index'})
    fig.update_layout(showlegend=False)
    return _light(fig, 440)

def plot_club_size_value(df):
    fig = px.box(df, x='Club_Size', y='Market_Value_Million_Euros',
                 color='Club_Size',
                 category_orders={'Club_Size':['Small','Medium','Big']},
                 title='Market Value by Club Size',
                 labels={'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.update_layout(showlegend=False)
    return _light(fig, 480)

def plot_value_for_money_dist(df):
    fig = px.histogram(df, x='Value_for_Money', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=40, barmode='overlay', opacity=0.75,
                       title='Value-for-Money Distribution',
                       labels={'Value_for_Money':'Value for Money Index'})
    return _light(fig, 480)

def plot_transfer_score_dist(df):
    fig = px.histogram(df, x='Transfer_Score', color='Position',
                       color_discrete_map=POS_COLOUR,
                       nbins=30, barmode='overlay', opacity=0.75,
                       title='Transfer Score Distribution')
    for tier, col in [(80,'#16a34a'),(60,'#f59e0b'),(40,'#ef4444')]:
        fig.add_vline(x=tier, line_dash='dash', line_color=col,
                      annotation_text=str(tier), annotation_font_color=col)
    return _light(fig, 480)

def plot_quadrant_chart(df):
    fig = px.scatter(df, x='Performance_Index', y='Market_Value_Million_Euros',
                     color='Quadrant', color_discrete_map=QUAD_COLOUR,
                     hover_data=['Player_Name','Age','Position'],
                     title='Player Quadrant Analysis',
                     labels={'Performance_Index':'Performance Index',
                             'Market_Value_Million_Euros':'Market Value (€M)'})
    fig.add_hline(y=df['Market_Value_Million_Euros'].median(),
                  line_dash='dash', line_color='#94a3b8')
    fig.add_vline(x=df['Performance_Index'].median(),
                  line_dash='dash', line_color='#94a3b8')
    return _light(fig, 580)

def plot_age_group_pie(df):
    counts = df['Age_Group'].value_counts().reset_index()
    counts.columns = ['Age_Group','Count']
    fig = px.pie(counts, names='Age_Group', values='Count',
                 title='Age Group Distribution',
                 color_discrete_sequence=['#1d4ed8','#8b5cf6','#16a34a'])
    return _light(fig, 440)

def plot_league_distribution(df):
    cnt = df['League'].value_counts().reset_index()
    cnt.columns = ['League','Count']
    fig = px.bar(cnt, x='Count', y='League', orientation='h',
                 title='Players by League',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return _light(fig, 500)
