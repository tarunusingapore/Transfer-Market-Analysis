import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

DARK_BG   = '#111827'
PLOT_BG   = '#0f172a'
FONT_COL  = '#e2e8f0'

RADAR_METRICS = {
    'GK':         ['Pass_Accuracy','Availability_Index','Tackles_per_Game',
                   'Interceptions_per_Game','Minutes_Played'],
    'Defender':   ['Tackles_per_Game','Interceptions_per_Game','Pass_Accuracy',
                   'Dribbles_per_Game','Goals','Assists'],
    'Midfielder': ['Key_Passes_per_Game','Assists','Goals',
                   'Dribbles_per_Game','Pass_Accuracy','Tackles_per_Game'],
    'Forward':    ['Goals','Assists','Shots_per_Game',
                   'Expected_Goals_xG','Dribbles_per_Game','Key_Passes_per_Game'],
}

COLOURS = ['#00d4ff','#4ade80','#f97316','#a78bfa','#f87171','#fbbf24']


def _scale(df, metrics):
    scaler = MinMaxScaler()
    return pd.DataFrame(
        scaler.fit_transform(df[metrics].fillna(0)),
        columns=metrics, index=df.index
    )


def build_two_player_radar(df: pd.DataFrame,
                            player1: str,
                            player2: str) -> go.Figure:
    """Radar chart comparing two selected players."""
    r1 = df[df['Player_Name'] == player1]
    r2 = df[df['Player_Name'] == player2]
    if r1.empty or r2.empty:
        raise ValueError("One or both players not found.")

    r1 = r1.iloc[0]
    r2 = r2.iloc[0]

    # Use the union of both positions' metrics, deduplicated
    pos1, pos2 = r1['Position'], r2['Position']
    m1 = RADAR_METRICS.get(pos1, RADAR_METRICS['Forward'])
    m2 = RADAR_METRICS.get(pos2, RADAR_METRICS['Forward'])
    metrics = list(dict.fromkeys(m1 + [x for x in m2 if x not in m1]))

    scaled = _scale(df, metrics)
    v1 = scaled.loc[r1.name].tolist()
    v2 = scaled.loc[r2.name].tolist()
    labels = metrics + [metrics[0]]

    fig = go.Figure()
    for vals, name, colour in [(v1, player1, COLOURS[0]), (v2, player2, COLOURS[1])]:
        loop = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=loop, theta=labels, fill='toself', name=name,
            line=dict(color=colour, width=2.5),
            fillcolor=colour.replace('ff', '33') if '#' in colour else colour,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#1e293b', tickfont_color='#94a3b8'),
            angularaxis=dict(gridcolor='#1e293b'),
            bgcolor=PLOT_BG,
        ),
        title=dict(text=f'{player1}  vs  {player2}', font_color=FONT_COL),
        paper_bgcolor=DARK_BG,
        font_color=FONT_COL,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        height=520,
    )
    return fig


def build_position_comparison(df: pd.DataFrame) -> go.Figure:
    metrics = ['Goals','Assists','Key_Passes_per_Game',
               'Tackles_per_Game','Dribbles_per_Game','Pass_Accuracy']
    scaled = _scale(df, metrics)
    scaled['Position'] = df['Position'].values

    positions = ['GK','Defender','Midfielder','Forward']
    colours   = [COLOURS[2], COLOURS[0], COLOURS[3], COLOURS[1]]
    labels    = metrics + [metrics[0]]

    fig = go.Figure()
    for pos, colour in zip(positions, colours):
        vals = scaled[scaled['Position'] == pos][metrics].mean().tolist()
        loop = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=loop, theta=labels, fill='toself', name=pos,
            line=dict(color=colour, width=2),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#1e293b', tickfont_color='#94a3b8'),
            angularaxis=dict(gridcolor='#1e293b'),
            bgcolor=PLOT_BG,
        ),
        title=dict(text='Position Comparison (Normalised Averages)', font_color=FONT_COL),
        paper_bgcolor=DARK_BG,
        font_color=FONT_COL,
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        height=520,
    )
    return fig
