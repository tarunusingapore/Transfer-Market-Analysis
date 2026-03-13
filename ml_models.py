"""ml_models.py — Machine Learning module for FootballIQ
Four sections:
  1. K-Means Player Archetypes
  2. Strong Signing Classification  (DT / RF / GB / Logistic — Accuracy/Precision/Recall/F1)
  3. Market Value Regression        (Linear / Ridge / Lasso / RF / GB — R²/MAE/RMSE)
  4. Club Fit Score                 (Cosine Similarity)
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, r2_score,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
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
    fig.update_xaxes(color='#0f172a', gridcolor='#e2e8f0')
    fig.update_yaxes(color='#0f172a', gridcolor='#e2e8f0')
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS PLAYER ARCHETYPES
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_FEATURES = [
    'Goals','Assists','Shots_per_Game','Key_Passes_per_Game',
    'Pass_Accuracy','Dribbles_per_Game','Tackles_per_Game',
    'Interceptions_per_Game','Minutes_Played',
]

STAT_MAP = {
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

def run_kmeans(df: pd.DataFrame, n_clusters: int = 6):
    X = df[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    cluster_df = pd.DataFrame(Xs, columns=CLUSTER_FEATURES)
    cluster_df['Cluster'] = labels
    centers = cluster_df.groupby('Cluster')[CLUSTER_FEATURES].mean()
    archetype_map = {}
    for c in range(n_clusters):
        top_stat = centers.loc[c].idxmax()
        archetype_map[c] = STAT_MAP.get(top_stat, f'Profile {c+1}')
    result = df.copy()
    result['Cluster']   = labels
    result['Archetype'] = result['Cluster'].map(archetype_map)
    return result, km, scaler, archetype_map, centers

def plot_kmeans_pca(df_cl: pd.DataFrame):
    X  = df_cl[CLUSTER_FEATURES].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    plot_df = pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],
                             'Archetype':df_cl['Archetype'],'Player':df_cl['Player_Name'],
                             'Position':df_cl['Position'],'Club':df_cl['Club']})
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Archetype',
                     hover_data=['Player','Position','Club'],
                     title=f'Player Archetypes — PCA ({ev[0]*100:.1f}% + {ev[1]*100:.1f}% variance)',
                     labels={'PC1':f'PC1 ({ev[0]*100:.1f}%)','PC2':f'PC2 ({ev[1]*100:.1f}%)'},
                     color_discrete_sequence=CLUSTER_COLOURS)
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return _light(fig, 560)

def plot_archetype_radar(df_cl: pd.DataFrame, archetype: str):
    metrics = ['Goals','Assists','Key_Passes_per_Game','Dribbles_per_Game',
               'Tackles_per_Game','Pass_Accuracy']
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df_cl[metrics].fillna(0)), columns=metrics)
    scaled['Archetype'] = df_cl['Archetype'].values
    arch_avg   = scaled[scaled['Archetype']==archetype][metrics].mean().tolist()
    global_avg = scaled[metrics].mean().tolist()
    theta = metrics + [metrics[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=arch_avg+[arch_avg[0]], theta=theta, fill='toself',
                                   name=archetype, line=dict(color='#1d4ed8',width=2.5),
                                   fillcolor='rgba(29,78,216,0.18)'))
    fig.add_trace(go.Scatterpolar(r=global_avg+[global_avg[0]], theta=theta, fill='toself',
                                   name='League Average', line=dict(color='#94a3b8',width=1.5,dash='dash'),
                                   fillcolor='rgba(148,163,184,0.1)'))
    fig.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,1],gridcolor='#e2e8f0',tickfont=dict(color='#475569')),
        angularaxis=dict(gridcolor='#e2e8f0'), bgcolor='#f8fafc'),
        title=dict(text=f'Archetype Profile: {archetype}',font=dict(color='#0f172a')),
        paper_bgcolor='white', font_color='#0f172a',
        legend=dict(bgcolor='#f1f5f9'), height=460)
    return fig

def plot_cluster_composition(df_cl: pd.DataFrame):
    comp = df_cl.groupby(['Archetype','Position']).size().reset_index(name='Count')
    fig = px.bar(comp, x='Archetype', y='Count', color='Position',
                 barmode='stack', color_discrete_map=POS_COLOUR,
                 title='Archetype Composition by Position')
    fig.update_layout(xaxis_tickangle=-20)
    return _light(fig, 440)

def plot_elbow(df: pd.DataFrame, max_k: int = 10):
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].fillna(0))
    inertias = []
    ks = range(2, max_k+1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    fig = px.line(x=list(ks), y=inertias, markers=True,
                  title='Elbow Method — Choosing Optimal K',
                  labels={'x':'Number of Clusters (k)','y':'Inertia'})
    fig.update_traces(line=dict(color='#1d4ed8',width=2.5), marker=dict(color='#1d4ed8',size=8))
    return _light(fig, 380)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRONG SIGNING CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
CLF_FEATURES = [
    'Age','Goals','Assists','Expected_Goals_xG','Expected_Assists_xA',
    'Key_Passes_per_Game','Dribbles_per_Game','Tackles_per_Game',
    'Interceptions_per_Game','Pass_Accuracy','Minutes_Played',
    'Market_Value_Million_Euros','Injury_Days_Last_Season','League_Quality_Index',
]
STRONG_SIGNING_THRESHOLD = 55   # top ~25% by Transfer Score = "Strong Signing"

CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
}

def train_classifiers(df: pd.DataFrame):
    """
    Train all classifiers, return per-model metrics + fitted models.
    Uses stratified 80/20 split + stores test predictions for confusion matrix.
    """
    df2 = df.copy()
    df2['Strong_Signing'] = (df2['Transfer_Score'] >= STRONG_SIGNING_THRESHOLD).astype(int)

    X = df2[CLF_FEATURES].fillna(0)
    y = df2['Strong_Signing']

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y)

    results  = {}
    fitted   = {}

    for name, clf in CLASSIFIERS.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        results[name] = {
            'Accuracy':  round(accuracy_score(y_te,  y_pred), 3),
            'Precision': round(precision_score(y_te, y_pred, zero_division=0), 3),
            'Recall':    round(recall_score(y_te,    y_pred, zero_division=0), 3),
            'F1 Score':  round(f1_score(y_te,        y_pred, zero_division=0), 3),
            'y_te':      y_te,
            'y_pred':    y_pred,
        }
        fitted[name] = clf

    class_counts = y.value_counts()
    pos_frac = class_counts.get(1, 0) / len(y)

    return results, fitted, scaler, pos_frac, X_tr, X_te, y_tr, y_te

def plot_clf_comparison(results: dict) -> go.Figure:
    """Grouped bar: Accuracy / Precision / Recall / F1 per model."""
    metrics  = ['Accuracy','Precision','Recall','F1 Score']
    models   = list(results.keys())
    colours  = ['#1d4ed8','#16a34a','#f59e0b','#dc2626']

    fig = go.Figure()
    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        fig.add_trace(go.Bar(name=metric, x=models, y=vals,
                             marker_color=colours[i], text=[f'{v:.3f}' for v in vals],
                             textposition='outside'))
    fig.update_layout(barmode='group', title='Classification Model Comparison',
                      yaxis=dict(range=[0,1.12], title='Score'),
                      xaxis_title='Model', legend=dict(bgcolor='#f1f5f9'),
                      bargap=0.18, bargroupgap=0.05)
    return _light(fig, 500)

def plot_confusion_matrix(results: dict, model_name: str) -> go.Figure:
    r   = results[model_name]
    cm  = confusion_matrix(r['y_te'], r['y_pred'])
    labels = ['Not Strong Signing','Strong Signing']
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=labels, y=labels,
                    title=f'Confusion Matrix — {model_name}',
                    labels=dict(x='Predicted',y='Actual',color='Count'))
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(color='#0f172a')
    fig.update_yaxes(color='#0f172a')
    return _light(fig, 400)

def plot_clf_feature_importance(fitted: dict, model_name: str) -> go.Figure:
    clf = fitted[model_name]
    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        imp = np.abs(clf.coef_[0])
    else:
        return go.Figure()
    fi = pd.DataFrame({'Feature': CLF_FEATURES, 'Importance': imp})
    fi = fi.sort_values('Importance', ascending=True)
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 title=f'Feature Importance — {model_name}',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(coloraxis_showscale=False)
    return _light(fig, 520)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET VALUE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
MV_FEATURES = [
    'Age','Goals','Assists','Shots_per_Game','Key_Passes_per_Game',
    'Pass_Accuracy','Dribbles_per_Game','Tackles_per_Game',
    'Interceptions_per_Game','Expected_Goals_xG','Expected_Assists_xA',
    'Matches_Played','Minutes_Played','League_Quality_Index','Fan_Popularity_Index',
]

REGRESSORS = {
    'Linear Regression':       LinearRegression(),
    'Ridge Regression':        Ridge(alpha=10.0),
    'Lasso Regression':        Lasso(alpha=1.0, max_iter=5000),
    'Random Forest':           RandomForestRegressor(n_estimators=200, max_depth=8,
                                                      random_state=42, n_jobs=-1),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                          learning_rate=0.05, random_state=42),
}
REG_COLOURS = {
    'Linear Regression':  '#94a3b8',
    'Ridge Regression':   '#60a5fa',
    'Lasso Regression':   '#a78bfa',
    'Random Forest':      '#1d4ed8',
    'Gradient Boosting':  '#16a34a',
}

def train_regressors(df: pd.DataFrame):
    """
    Train all 5 regressors, evaluate with 80/20 split + 5-fold CV.
    Returns metric table + per-model predictions on test set.
    """
    X = df[MV_FEATURES].fillna(0)
    y = df['Market_Value_Million_Euros']

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

    results = {}
    fitted  = {}

    for name, reg in REGRESSORS.items():
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = r2_score(y_te, y_pred)

        results[name] = {
            'R²':   round(r2,   3),
            'MAE':  round(mae,  2),
            'RMSE': round(rmse, 2),
            'y_te':   y_te,
            'y_pred': y_pred,
        }
        fitted[name] = reg

    return results, fitted, scaler, X_te, y_te

def plot_regression_comparison(results: dict) -> go.Figure:
    """Side-by-side grouped bars: R², MAE, RMSE per model."""
    models = list(results.keys())
    r2s    = [results[m]['R²']   for m in models]
    maes   = [results[m]['MAE']  for m in models]
    rmses  = [results[m]['RMSE'] for m in models]

    fig = make_subplots(rows=1, cols=3, subplot_titles=['R² (higher = better)',
                                                         'MAE — €M (lower = better)',
                                                         'RMSE — €M (lower = better)'])
    colours = [REG_COLOURS[m] for m in models]
    for col_i, (vals, lbl) in enumerate([(r2s,'R²'),(maes,'MAE'),(rmses,'RMSE')], 1):
        fig.add_trace(go.Bar(x=models, y=vals, marker_color=colours,
                             text=[f'{v:.2f}' for v in vals], textposition='outside',
                             name=lbl, showlegend=False), row=1, col=col_i)
    fig.update_layout(title='Regression Model Comparison — All Metrics', height=460)
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(gridcolor='#e2e8f0')
    return _light(fig, 460)

def plot_reg_actual_vs_predicted(results: dict, model_name: str) -> go.Figure:
    r = results[model_name]
    fig = px.scatter(x=r['y_te'], y=r['y_pred'],
                     labels={'x':'Actual Value (€M)','y':'Predicted Value (€M)'},
                     title=f'{model_name} — Actual vs Predicted',
                     opacity=0.75)
    mn = min(float(r['y_te'].min()), float(r['y_pred'].min()))
    mx = max(float(r['y_te'].max()), float(r['y_pred'].max()))
    fig.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                  line=dict(color='#ef4444', dash='dash', width=1.8))
    fig.update_traces(marker=dict(color=REG_COLOURS.get(model_name,'#1d4ed8'), size=8))
    return _light(fig, 480)

def plot_residuals(results: dict, model_name: str) -> go.Figure:
    r = results[model_name]
    resid = np.array(r['y_pred']) - np.array(r['y_te'])
    fig = px.histogram(x=resid, nbins=30, title=f'Residuals — {model_name}',
                       labels={'x':'Residual (Predicted − Actual, €M)','y':'Count'},
                       color_discrete_sequence=['#3b82f6'])
    fig.add_vline(x=0, line_dash='dash', line_color='#ef4444',
                  annotation_text='Zero error', annotation_font_color='#ef4444')
    return _light(fig, 400)

def plot_reg_feature_importance(fitted: dict, model_name: str) -> go.Figure:
    reg = fitted[model_name]
    if hasattr(reg, 'feature_importances_'):
        imp = reg.feature_importances_
    elif hasattr(reg, 'coef_'):
        imp = np.abs(reg.coef_)
    else:
        return go.Figure()
    fi = pd.DataFrame({'Feature': MV_FEATURES, 'Importance': imp})
    fi = fi.sort_values('Importance', ascending=True)
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 title=f'Feature Importance — {model_name}',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(coloraxis_showscale=False)
    return _light(fig, 480)

def predict_player_value(fitted: dict, scaler: StandardScaler,
                          df: pd.DataFrame, player_name: str):
    """Return dict {model_name: predicted_value}."""
    row = df[df['Player_Name'] == player_name].iloc[0]
    x   = row[MV_FEATURES].fillna(0).values.reshape(1,-1)
    xs  = scaler.transform(x)
    preds = {}
    for name, reg in fitted.items():
        preds[name] = round(float(reg.predict(xs)[0]), 1)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLUB FIT SCORE (Cosine Similarity)
# ─────────────────────────────────────────────────────────────────────────────
FIT_FEATURES = [
    'Goals','Assists','Key_Passes_per_Game','Pass_Accuracy',
    'Dribbles_per_Game','Tackles_per_Game','Interceptions_per_Game',
    'Shots_per_Game','Expected_Goals_xG',
]

def compute_club_fit(df: pd.DataFrame, player_name: str, top_n: int = 10):
    if player_name not in df['Player_Name'].values:
        return pd.DataFrame()
    X = df[FIT_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=FIT_FEATURES, index=df.index)
    Xs['Club'] = df['Club'].values
    club_profiles = Xs.groupby('Club')[FIT_FEATURES].mean()
    player_vec = Xs.loc[df[df['Player_Name']==player_name].index[0], FIT_FEATURES].values
    sims = cosine_similarity([player_vec], club_profiles.values)[0]
    result = pd.DataFrame({'Club': club_profiles.index,
                           'Fit_Score': (sims*100).round(1)}).sort_values('Fit_Score', ascending=False)
    club_league = df.groupby('Club')['League'].first()
    result['League'] = result['Club'].map(club_league)
    current_club = df[df['Player_Name']==player_name]['Club'].values[0]
    result = result[result['Club'] != current_club]
    return result.head(top_n).reset_index(drop=True)

def plot_club_fit(fit_df: pd.DataFrame, player_name: str):
    if fit_df.empty:
        return go.Figure()
    fig = px.bar(fit_df, x='Fit_Score', y='Club', orientation='h',
                 color='Fit_Score', color_continuous_scale='Blues',
                 hover_data=['League'],
                 title=f'Best Club Fits for {player_name}',
                 labels={'Fit_Score':'Fit Score (%)','Club':''})
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    return _light(fig, 460)
