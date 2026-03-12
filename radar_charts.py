import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Light theme colours
P1_COLOR  = '#1d4ed8'   # blue  — Player 1
P2_COLOR  = '#16a34a'   # green — Player 2
P1_FILL   = 'rgba(29,78,216,0.18)'
P2_FILL   = 'rgba(22,163,74,0.18)'
POS_COLORS = ['#f97316','#3b82f6','#8b5cf6','#16a34a']

RADAR_METRICS = {
    'GK':         ['Pass_Accuracy', 'Availability_Index', 'Tackles_per_Game',
                   'Interceptions_per_Game', 'Minutes_Played'],
    'Defender':   ['Tackles_per_Game', 'Interceptions_per_Game', 'Pass_Accuracy',
                   'Dribbles_per_Game', 'Goals', 'Assists'],
    'Midfielder': ['Key_Passes_per_Game', 'Assists', 'Goals',
                   'Dribbles_per_Game', 'Pass_Accuracy', 'Tackles_per_Game'],
    'Forward':    ['Goals', 'Assists', 'Shots_per_Game',
                   'Expected_Goals_xG', 'Dribbles_per_Game', 'Key_Passes_per_Game'],
}

def _scale(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    return pd.DataFrame(
        scaler.fit_transform(df[metrics].fillna(0)),
        columns=metrics, index=df.index,
    )

def _light_polar(fig, title=''):
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor='#e2e8f0', tickfont=dict(color='#475569', size=10),
                linecolor='#cbd5e1',
            ),
            angularaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1',
                             tickfont=dict(color='#0f172a', size=11)),
            bgcolor='#f8fafc',
        ),
        title=dict(text=title, font=dict(color='#0f172a', size=16)),
        paper_bgcolor='white',
        font_color='#0f172a',
        legend=dict(bgcolor='#f1f5f9', bordercolor='#cbd5e1', borderwidth=1),
        height=520,
    )
    return fig


def build_two_player_radar(df: pd.DataFrame, player1: str, player2: str) -> go.Figure:
    """Overlay radar chart comparing two players. Always shows both traces."""
    r1 = df[df['Player_Name'] == player1]
    r2 = df[df['Player_Name'] == player2]
    if r1.empty or r2.empty:
        raise ValueError("One or both players not found.")

    r1 = r1.iloc[0]
    r2 = r2.iloc[0]

    # Union of both positions' metrics, deduplicated, preserving order
    pos1, pos2 = r1['Position'], r2['Position']
    m1 = RADAR_METRICS.get(pos1, RADAR_METRICS['Forward'])
    m2 = RADAR_METRICS.get(pos2, RADAR_METRICS['Forward'])
    metrics = list(dict.fromkeys(m1 + [x for x in m2 if x not in m1]))

    # Scale against the full dataset so values are comparable
    scaled = _scale(df, metrics)
    v1 = scaled.loc[r1.name].tolist()
    v2 = scaled.loc[r2.name].tolist()

    # Close the polygon
    theta = metrics + [metrics[0]]
    v1_loop = v1 + [v1[0]]
    v2_loop = v2 + [v2[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=v1_loop, theta=theta,
        fill='toself', name=player1,
        line=dict(color=P1_COLOR, width=2.5),
        fillcolor=P1_FILL,
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2_loop, theta=theta,
        fill='toself', name=player2,
        line=dict(color=P2_COLOR, width=2.5),
        fillcolor=P2_FILL,
    ))

    return _light_polar(fig, f'{player1}  vs  {player2}')


def build_position_comparison(df: pd.DataFrame) -> go.Figure:
    metrics = ['Goals', 'Assists', 'Key_Passes_per_Game',
               'Tackles_per_Game', 'Dribbles_per_Game', 'Pass_Accuracy']
    scaled = _scale(df, metrics)
    scaled['Position'] = df['Position'].values

    positions = ['GK', 'Defender', 'Midfielder', 'Forward']
    theta = metrics + [metrics[0]]

    fig = go.Figure()
    for pos, colour in zip(positions, POS_COLORS):
        vals = scaled[scaled['Position'] == pos][metrics].mean().tolist()
        loop = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=loop, theta=theta, fill='toself', name=pos,
            line=dict(color=colour, width=2),
        ))

    return _light_polar(fig, 'Position Comparison (Normalised Averages)')
