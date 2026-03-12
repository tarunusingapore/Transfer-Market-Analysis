import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the raw football dataset."""

    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates(subset=['Player_ID']).reset_index(drop=True)

    # 2. Fill missing values
    for col in ['Pass_Accuracy', 'Key_Passes_per_Game', 'Fan_Popularity_Index']:
        median_val = df.groupby('Position')[col].transform('median')
        df[col] = df[col].fillna(median_val)

    # 3. Standardise column names (already snake_case; enforce lowercase)
    df.columns = [c.strip() for c in df.columns]

    # 4. Age group categorisation
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 23, 29, 99],
        labels=['Young Talent (18-23)', 'Prime Player (24-29)', 'Veteran (30+)']
    ).astype(str)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analytical metrics used in scouting and transfer modelling."""

    df = df.copy()

    # ── Performance Index ──────────────────────────────────────────────────────
    # Weighted combination — weights differ by position
    pos = df['Position']

    goal_w   = np.where(pos == 'Forward', 1.5, np.where(pos == 'Midfielder', 1.0, 0.5))
    assist_w = np.where(pos == 'Forward', 1.0, np.where(pos == 'Midfielder', 1.3, 0.7))
    pass_w   = np.where(pos == 'GK',      0.5, np.where(pos == 'Midfielder', 1.2, 0.8))
    tackle_w = np.where(pos == 'Defender',1.5, np.where(pos == 'Midfielder', 1.0, 0.4))

    df['Performance_Index'] = (
        goal_w   * df['Goals'] +
        assist_w * df['Assists'] +
        pass_w   * df['Key_Passes_per_Game'] * df['Matches_Played'] +
        tackle_w * df['Tackles_per_Game']    * df['Matches_Played'] +
        1.0      * df['Dribbles_per_Game']   * df['Matches_Played'] +
        1.2      * df['Expected_Goals_xG']   * df['Matches_Played'] +
        1.2      * df['Expected_Assists_xA'] * df['Matches_Played']
    ).round(2)

    # ── Value for Money ───────────────────────────────────────────────────────
    df['Value_for_Money'] = (
        df['Performance_Index'] / df['Market_Value_Million_Euros'].replace(0, 0.01)
    ).round(3)

    # ── Availability Index ────────────────────────────────────────────────────
    total_possible_minutes = df['Matches_Played'] * 90 + df['Injury_Days_Last_Season']
    df['Availability_Index'] = (
        df['Minutes_Played'] / total_possible_minutes.replace(0, 1)
    ).clip(0, 1).round(3)

    # ── Potential Index ───────────────────────────────────────────────────────
    # Peaks at age 22, decreases linearly after 29
    df['Potential_Index'] = (
        np.where(df['Age'] <= 22, 1.0,
        np.where(df['Age'] <= 29, 1.0 - 0.04 * (df['Age'] - 22),
                                  0.72 - 0.06 * (df['Age'] - 29)))
    ).clip(0.1, 1.0).round(3)

    # ── Scouting Score (0–100) ────────────────────────────────────────────────
    scaler = MinMaxScaler()
    raw = scaler.fit_transform(
        df[['Performance_Index', 'Availability_Index', 'Potential_Index']]
    )
    df['Scouting_Score'] = (
        0.5 * raw[:, 0] + 0.25 * raw[:, 1] + 0.25 * raw[:, 2]
    ) * 100
    df['Scouting_Score'] = df['Scouting_Score'].round(1)

    return df


def scale_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with z-score scaled numeric columns appended (for ML use)."""
    numeric_cols = [
        'Goals', 'Assists', 'Shots_per_Game', 'Key_Passes_per_Game',
        'Dribbles_per_Game', 'Tackles_per_Game', 'Interceptions_per_Game',
        'Expected_Goals_xG', 'Expected_Assists_xA',
        'Market_Value_Million_Euros', 'Minutes_Played',
    ]
    df_scaled = df.copy()
    for col in numeric_cols:
        mu, sigma = df[col].mean(), df[col].std()
        df_scaled[col + '_z'] = ((df[col] - mu) / sigma).round(4)
    return df_scaled


def full_pipeline(csv_path: str = 'players_dataset.csv') -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == '__main__':
    df = full_pipeline()
    print(df[['Player_Name', 'Position', 'Performance_Index',
              'Value_for_Money', 'Availability_Index',
              'Potential_Index', 'Scouting_Score']].head(10))
    print(f"\nShape: {df.shape}")
