import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


RADAR_METRICS = {
    'GK':         ['Pass_Accuracy', 'Availability_Index', 'Performance_Index',
                   'Tackles_per_Game', 'Interceptions_per_Game'],
    'Defender':   ['Tackles_per_Game', 'Interceptions_per_Game', 'Pass_Accuracy',
                   'Dribbles_per_Game', 'Performance_Index'],
    'Midfielder': ['Key_Passes_per_Game', 'Assists', 'Goals',
                   'Dribbles_per_Game', 'Pass_Accuracy', 'Performance_Index'],
    'Forward':    ['Goals', 'Assists', 'Shots_per_Game',
                   'Expected_Goals_xG', 'Dribbles_per_Game', 'Performance_Index'],
}

DEFAULT_METRICS = ['Goals', 'Assists', 'Key_Passes_per_Game',
                   'Tackles_per_Game', 'Dribbles_per_Game', 'Performance_Index']


def build_radar(df: pd.DataFrame,
                player_name: str,
                compare_avg: bool = True) -> go.Figure:
    """
    Build a radar chart for a given player.
    Optionally overlay the position-average values.
    """
    row = df[df['Player_Name'] == player_name]
    if row.empty:
        raise ValueError(f"Player '{player_name}' not found.")

    row = row.iloc[0]
    pos = row['Position']
    metrics = RADAR_METRICS.get(pos, DEFAULT_METRICS)

    # Normalise metrics 0-1 using the full dataset
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df[metrics].fillna(0)),
        columns=metrics, index=df.index
    )

    player_vals = scaled_df.loc[row.name].tolist()
    labels = metrics + [metrics[0]]  # close the loop
    player_vals_loop = player_vals + [player_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_vals_loop,
        theta=labels,
        fill='toself',
        name=player_name,
        line=dict(color='#45B7D1', width=2),
        fillcolor='rgba(69,183,209,0.25)',
    ))

    if compare_avg:
        pos_avg = scaled_df.loc[df['Position'] == pos].mean().tolist()
        pos_avg_loop = pos_avg + [pos_avg[0]]
        fig.add_trace(go.Scatterpolar(
            r=pos_avg_loop,
            theta=labels,
            fill='toself',
            name=f'{pos} Average',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            fillcolor='rgba(255,107,107,0.10)',
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f'Player Radar — {player_name} vs {pos} Average',
        showlegend=True,
        height=500,
    )
    return fig


def build_position_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped radar comparing positional averages across shared metrics."""
    metrics = ['Goals', 'Assists', 'Key_Passes_per_Game',
               'Tackles_per_Game', 'Dribbles_per_Game', 'Pass_Accuracy']

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[metrics].fillna(0)),
        columns=metrics
    )
    scaled['Position'] = df['Position'].values

    positions = ['GK', 'Defender', 'Midfielder', 'Forward']
    colours   = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    fig = go.Figure()
    labels = metrics + [metrics[0]]
    for pos, colour in zip(positions, colours):
        vals = scaled[scaled['Position'] == pos][metrics].mean().tolist()
        vals_loop = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_loop, theta=labels,
            fill='toself', name=pos,
            line=dict(color=colour, width=2),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Position Performance Comparison (Normalised Averages)',
        height=520,
    )
    return fig
