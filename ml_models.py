"""
ml_models.py — Machine Learning module for FootballIQ
Three models:
  1. K-Means Player Profiling — cluster players into tactical archetypes
  2. Market Value Predictor   — Random Forest regression
  3. Club Fit Score           — cosine similarity of player vs club avg profile
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ── Colour maps ───────────────────────────────────────────────────────────────
CLUSTER_COLOURS = [
    '#1d4ed8','#16a34a','#dc2626','#f59e0b',
    '#8b5cf6','#06b6d4','#ec4899','#84cc16',
]
POS_COLOUR = {'GK':'#f97316','Defender':'#3b82f6','Midfielder':'#8b5cf6','Forward':'#16a34a'}

def _light(fig, h=None):
    upd = dict(template='plotly_white', paper_bgcolor='white',
               plot_bgcolor='#f8fafc', font_color='#0f172a')
    if h: upd['height'] = h
    fig.update_layout(**upd)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS PLAYER PROFILING
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_FEATURES = [
    'Goals','Assists','Shots_per_Game','Key_Passes_per_Game',
    'Pass_Accuracy','Dribbles_per_Game','Tackles_per_Game',
    'Interceptions_per_Game','Minutes_Played',
]

ARCHETYPE_NAMES = {
    0: 'Goal Machine',
    1: 'Creative Playmaker',
    2: 'Box-to-Box Engine',
    3: 'Defensive Anchor',
    4: 'Wide Attacker',
    5: 'Deep-Lying Playmaker',
    6: 'Utility Player',
    7: 'Shot-Stopper',
}

def run_kmeans(df: pd.DataFrame, n_clusters: int = 6):
    """Fit K-Means, return df with Cluster + Archetype columns."""
    X = df[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)

    # Label each cluster by its dominant stat
    cluster_df = pd.DataFrame(Xs, columns=CLUSTER_FEATURES)
    cluster_df['Cluster'] = labels
    centers = cluster_df.groupby('Cluster')[CLUSTER_FEATURES].mean()

    # Assign human-readable archetype based on top stat per cluster
    stat_map = {
        'Goals':                  'Goal Machine',
        'Key_Passes_per_Game':    'Creative Playmaker',
        'Tackles_per_Game':       'Defensive Anchor',
        'Interceptions_per_Game': 'Ball Winner',
        'Dribbles_per_Game':      'Dribbling Wizard',
        'Pass_Accuracy':          'Deep-Lying Playmaker',
        'Assists':                'Clinical Provider',
        'Shots_per_Game':         'Penalty Box Hunter',
        'Minutes_Played':         'Workhorse',
    }
    archetype_map = {}
    for c in range(n_clusters):
        top_stat = centers.loc[c].idxmax()
        archetype_map[c] = stat_map.get(top_stat, f'Profile {c+1}')

    result = df.copy()
    result['Cluster'] = labels
    result['Archetype'] = result['Cluster'].map(archetype_map)
    return result, km, scaler, archetype_map, centers

def plot_kmeans_pca(df_cl: pd.DataFrame):
    """2-D PCA scatter of clusters."""
    X = df_cl[CLUSTER_FEATURES].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)

    plot_df = pd.DataFrame({
        'PC1': coords[:, 0], 'PC2': coords[:, 1],
        'Archetype': df_cl['Archetype'],
        'Player': df_cl['Player_Name'],
        'Position': df_cl['Position'],
        'Club': df_cl['Club'],
    })
    ev = pca.explained_variance_ratio_
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Archetype',
        hover_data=['Player', 'Position', 'Club'],
        title=f'Player Archetypes — PCA (explains {ev[0]*100:.1f}% + {ev[1]*100:.1f}% variance)',
        labels={'PC1': f'PC1 ({ev[0]*100:.1f}%)', 'PC2': f'PC2 ({ev[1]*100:.1f}%)'},
        color_discrete_sequence=CLUSTER_COLOURS,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return _light(fig, 560)

def plot_archetype_radar(df_cl: pd.DataFrame, archetype: str):
    """Mean radar for a selected archetype vs overall average."""
    metrics = ['Goals','Assists','Key_Passes_per_Game','Dribbles_per_Game',
               'Tackles_per_Game','Pass_Accuracy']
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df_cl[metrics].fillna(0)),
                          columns=metrics)
    scaled['Archetype'] = df_cl['Archetype'].values

    arch_avg   = scaled[scaled['Archetype'] == archetype][metrics].mean().tolist()
    global_avg = scaled[metrics].mean().tolist()
    theta = metrics + [metrics[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=arch_avg + [arch_avg[0]], theta=theta,
        fill='toself', name=archetype,
        line=dict(color='#1d4ed8', width=2.5),
        fillcolor='rgba(29,78,216,0.18)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=global_avg + [global_avg[0]], theta=theta,
        fill='toself', name='League Average',
        line=dict(color='#94a3b8', width=1.5, dash='dash'),
        fillcolor='rgba(148,163,184,0.1)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor='#e2e8f0', tickfont=dict(color='#475569')),
            angularaxis=dict(gridcolor='#e2e8f0'),
            bgcolor='#f8fafc',
        ),
        title=dict(text=f'Archetype Profile: {archetype}', font=dict(color='#0f172a')),
        paper_bgcolor='white', font_color='#0f172a',
        legend=dict(bgcolor='#f1f5f9'),
        height=460,
    )
    return fig

def plot_cluster_composition(df_cl: pd.DataFrame):
    """Stacked bar: position breakdown per archetype."""
    comp = df_cl.groupby(['Archetype','Position']).size().reset_index(name='Count')
    fig = px.bar(comp, x='Archetype', y='Count', color='Position',
                 barmode='stack', color_discrete_map=POS_COLOUR,
                 title='Archetype Composition by Position')
    fig.update_layout(xaxis_tickangle=-20)
    return _light(fig, 440)

def plot_elbow(df: pd.DataFrame, max_k: int = 10):
    """Elbow chart for choosing k."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].fillna(0))
    inertias = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    fig = px.line(x=list(ks), y=inertias, markers=True,
                  title='Elbow Method — Choosing Optimal K',
                  labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
    fig.update_traces(line=dict(color='#1d4ed8', width=2.5),
                      marker=dict(color='#1d4ed8', size=8))
    return _light(fig, 380)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MARKET VALUE PREDICTOR (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
MV_FEATURES = [
    'Age', 'Goals', 'Assists', 'Shots_per_Game', 'Key_Passes_per_Game',
    'Pass_Accuracy', 'Dribbles_per_Game', 'Tackles_per_Game',
    'Interceptions_per_Game', 'Expected_Goals_xG', 'Expected_Assists_xA',
    'Matches_Played', 'Minutes_Played', 'League_Quality_Index', 'Fan_Popularity_Index',
]

def train_market_value_model(df: pd.DataFrame):
    """Train RF + GB regressors, return models + metrics."""
    X = df[MV_FEATURES].fillna(0)
    y = df['Market_Value_Million_Euros']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42,
                                   learning_rate=0.05)

    rf.fit(Xs, y)
    gb.fit(Xs, y)

    # Cross-val metrics
    rf_r2  = cross_val_score(rf, Xs, y, cv=5, scoring='r2').mean()
    rf_mae = -cross_val_score(rf, Xs, y, cv=5, scoring='neg_mean_absolute_error').mean()
    gb_r2  = cross_val_score(gb, Xs, y, cv=5, scoring='r2').mean()
    gb_mae = -cross_val_score(gb, Xs, y, cv=5, scoring='neg_mean_absolute_error').mean()

    # In-sample predictions for plotting
    rf_pred = rf.predict(Xs)
    gb_pred = gb.predict(Xs)

    metrics = {
        'RF':  {'R²': rf_r2,  'MAE': rf_mae,  'preds': rf_pred},
        'GB':  {'R²': gb_r2,  'MAE': gb_mae,  'preds': gb_pred},
    }
    return rf, gb, scaler, metrics

def plot_mv_actual_vs_predicted(df: pd.DataFrame, preds: np.ndarray, model_name: str):
    plot_df = pd.DataFrame({
        'Actual': df['Market_Value_Million_Euros'].values,
        'Predicted': preds,
        'Player': df['Player_Name'].values,
        'Position': df['Position'].values,
    })
    fig = px.scatter(plot_df, x='Actual', y='Predicted',
                     color='Position', color_discrete_map=POS_COLOUR,
                     hover_data=['Player'],
                     title=f'{model_name} — Actual vs Predicted Market Value (€M)',
                     labels={'Actual': 'Actual Value (€M)', 'Predicted': 'Predicted Value (€M)'})
    max_v = max(plot_df['Actual'].max(), plot_df['Predicted'].max()) * 1.05
    fig.add_shape(type='line', x0=0, y0=0, x1=max_v, y1=max_v,
                  line=dict(color='#94a3b8', dash='dash', width=1.5))
    return _light(fig, 500)

def plot_feature_importance(rf_model, feature_names: list):
    imp = pd.DataFrame({'Feature': feature_names,
                        'Importance': rf_model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=True).tail(12)
    fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                 title='Random Forest — Feature Importance',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(coloraxis_showscale=False)
    return _light(fig, 460)

def predict_single_player(rf, gb, scaler, player_row: pd.Series):
    """Return RF and GB predictions for one player."""
    x = player_row[MV_FEATURES].fillna(0).values.reshape(1, -1)
    xs = scaler.transform(x)
    return float(rf.predict(xs)[0]), float(gb.predict(xs)[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUB FIT SCORE (Cosine Similarity)
# ─────────────────────────────────────────────────────────────────────────────
FIT_FEATURES = [
    'Goals', 'Assists', 'Key_Passes_per_Game', 'Pass_Accuracy',
    'Dribbles_per_Game', 'Tackles_per_Game', 'Interceptions_per_Game',
    'Shots_per_Game', 'Expected_Goals_xG',
]

def compute_club_fit(df: pd.DataFrame, player_name: str, top_n: int = 10):
    """
    For a given player, compute cosine similarity to every club's mean profile.
    Returns a ranked DataFrame of clubs + fit scores.
    """
    if player_name not in df['Player_Name'].values:
        return pd.DataFrame()

    X = df[FIT_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=FIT_FEATURES, index=df.index)
    Xs['Club'] = df['Club'].values

    club_profiles = Xs.groupby('Club')[FIT_FEATURES].mean()
    player_vec = Xs.loc[df[df['Player_Name'] == player_name].index[0], FIT_FEATURES].values

    sims = cosine_similarity([player_vec], club_profiles.values)[0]
    result = pd.DataFrame({
        'Club': club_profiles.index,
        'Fit_Score': (sims * 100).round(1),
    }).sort_values('Fit_Score', ascending=False)

    # Add league info
    club_league = df.groupby('Club')['League'].first()
    result['League'] = result['Club'].map(club_league)

    # Exclude current club
    current_club = df[df['Player_Name'] == player_name]['Club'].values[0]
    result = result[result['Club'] != current_club]

    return result.head(top_n).reset_index(drop=True)

def plot_club_fit(fit_df: pd.DataFrame, player_name: str):
    if fit_df.empty:
        return go.Figure()
    fig = px.bar(fit_df, x='Fit_Score', y='Club', orientation='h',
                 color='Fit_Score', color_continuous_scale='Blues',
                 hover_data=['League'],
                 title=f'Best Club Fits for {player_name}',
                 labels={'Fit_Score': 'Fit Score (%)', 'Club': ''})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      coloraxis_showscale=False)
    return _light(fig, 460)
