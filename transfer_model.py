import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_transfer_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ['Performance_Index','Value_for_Money','Availability_Index','Potential_Index']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[required])
    df['Transfer_Score'] = ((scaled @ np.array([0.35, 0.30, 0.20, 0.15])) * 100).round(1)

    def _tier(s):
        if s >= 80:   return '🏆 Elite Transfer Target'
        elif s >= 60: return '✅ Strong Signing Opportunity'
        elif s >= 40: return '⚠️ Moderate Value'
        else:         return '❌ Low Priority'

    df['Transfer_Tier'] = df['Transfer_Score'].apply(_tier)
    return df


def _base_cols(df):
    """Return whichever of Club/League exist alongside core cols."""
    extra = [c for c in ['Club','League','Club_Size'] if c in df.columns]
    return ['Player_Name','Age','Position','Nationality'] + extra + \
           ['Market_Value_Million_Euros','Performance_Index',
            'Value_for_Money','Transfer_Score','Transfer_Tier']


def top_recommendations(df, n=20, position=None, max_age=None, max_value=None):
    f = df.copy()
    if position and position != 'All':
        f = f[f['Position'] == position]
    if max_age:
        f = f[f['Age'] <= max_age]
    if max_value:
        f = f[f['Market_Value_Million_Euros'] <= max_value]
    return f[_base_cols(f)].sort_values('Transfer_Score', ascending=False).head(n)


def best_young_talents(df, n=10):
    extra = [c for c in ['Club','League'] if c in df.columns]
    cols = ['Player_Name','Age','Position','Nationality'] + extra + \
           ['Market_Value_Million_Euros','Potential_Index','Transfer_Score','Transfer_Tier']
    return (df[df['Age'] <= 23]
            .sort_values('Transfer_Score', ascending=False)
            .head(n)[cols])


def best_value_signings(df, n=10):
    extra = [c for c in ['Club','League'] if c in df.columns]
    cols = ['Player_Name','Age','Position','Nationality'] + extra + \
           ['Market_Value_Million_Euros','Performance_Index',
            'Value_for_Money','Transfer_Score','Transfer_Tier']
    return (df.sort_values('Value_for_Money', ascending=False).head(n)[cols])


def quadrant_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    med_p = df['Performance_Index'].median()
    med_v = df['Market_Value_Million_Euros'].median()

    def _q(row):
        hp = row['Performance_Index']         >= med_p
        hv = row['Market_Value_Million_Euros'] >= med_v
        if hp and not hv:  return 'Undervalued Gems'
        if hp and hv:      return 'Overpriced Stars'
        if not hp and hv:  return 'Declining Veterans'
        return 'Average Players'

    df['Quadrant'] = df.apply(_q, axis=1)
    return df
