import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_transfer_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the AI Transfer Recommendation Score (0-100).

    Components
    ----------
    Performance_Index   — how good the player is (weight 0.35)
    Value_for_Money     — output per € spent (weight 0.30)
    Availability_Index  — fitness / availability (weight 0.20)
    Potential_Index     — future ceiling, age-adjusted (weight 0.15)
    """
    df = df.copy()

    required = ['Performance_Index', 'Value_for_Money',
                'Availability_Index', 'Potential_Index']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Run data_cleaning.py first.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[required])

    weights = np.array([0.35, 0.30, 0.20, 0.15])
    df['Transfer_Score'] = (scaled @ weights) * 100
    df['Transfer_Score'] = df['Transfer_Score'].round(1)

    # Human-readable tier
    def _tier(score):
        if score >= 80:
            return '🏆 Elite Transfer Target'
        elif score >= 60:
            return '✅ Strong Signing Opportunity'
        elif score >= 40:
            return '⚠️ Moderate Value'
        else:
            return '❌ Low Priority'

    df['Transfer_Tier'] = df['Transfer_Score'].apply(_tier)
    return df


def top_recommendations(df: pd.DataFrame,
                         n: int = 20,
                         position: str = None,
                         max_age: int = None,
                         max_value: float = None) -> pd.DataFrame:
    """Return top-n players sorted by Transfer_Score with optional filters."""
    filtered = df.copy()
    if position and position != 'All':
        filtered = filtered[filtered['Position'] == position]
    if max_age:
        filtered = filtered[filtered['Age'] <= max_age]
    if max_value:
        filtered = filtered[filtered['Market_Value_Million_Euros'] <= max_value]

    cols = ['Player_Name', 'Age', 'Position', 'Nationality', 'Club_Size',
            'Market_Value_Million_Euros', 'Performance_Index',
            'Value_for_Money', 'Transfer_Score', 'Transfer_Tier']
    return filtered[cols].sort_values('Transfer_Score', ascending=False).head(n)


def best_young_talents(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (df[df['Age'] <= 23]
            .sort_values('Transfer_Score', ascending=False)
            .head(n)[['Player_Name', 'Age', 'Position', 'Nationality',
                       'Market_Value_Million_Euros', 'Potential_Index',
                       'Transfer_Score', 'Transfer_Tier']])


def best_value_signings(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (df.sort_values('Value_for_Money', ascending=False)
            .head(n)[['Player_Name', 'Age', 'Position', 'Nationality',
                       'Market_Value_Million_Euros', 'Performance_Index',
                       'Value_for_Money', 'Transfer_Score', 'Transfer_Tier']])


def quadrant_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Categorise players into four scouting quadrants."""
    df = df.copy()
    med_perf  = df['Performance_Index'].median()
    med_value = df['Market_Value_Million_Euros'].median()

    def _quad(row):
        hi_perf  = row['Performance_Index']         >= med_perf
        hi_value = row['Market_Value_Million_Euros'] >= med_value
        if hi_perf and not hi_value:
            return 'Undervalued Gems'
        elif hi_perf and hi_value:
            return 'Overpriced Stars'
        elif not hi_perf and hi_value:
            return 'Declining Veterans'
        else:
            return 'Average Players'

    df['Quadrant'] = df.apply(_quad, axis=1)
    return df


if __name__ == '__main__':
    from data_cleaning import full_pipeline
    df = full_pipeline()
    df = compute_transfer_score(df)
    df = quadrant_analysis(df)

    print("\n── Top 10 Transfer Targets ──")
    print(top_recommendations(df, n=10).to_string(index=False))

    print("\n── Best Young Talents ──")
    print(best_young_talents(df).to_string(index=False))

    print("\n── Best Value Signings ──")
    print(best_value_signings(df).to_string(index=False))

    print("\n── Quadrant Distribution ──")
    print(df['Quadrant'].value_counts())
