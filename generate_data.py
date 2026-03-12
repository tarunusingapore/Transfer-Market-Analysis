import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def generate_football_dataset(n=1000):
    nationalities = [
        'Brazil', 'Argentina', 'France', 'Spain', 'England', 'Germany',
        'Portugal', 'Italy', 'Netherlands', 'Belgium', 'Croatia', 'Uruguay',
        'Colombia', 'Mexico', 'Senegal', 'Nigeria', 'Japan', 'South Korea',
        'Morocco', 'USA', 'Poland', 'Denmark', 'Sweden', 'Switzerland'
    ]

    positions = ['GK', 'Defender', 'Midfielder', 'Forward']
    club_sizes = ['Small', 'Medium', 'Big']
    pos_weights = [0.10, 0.30, 0.35, 0.25]

    pos_list = np.random.choice(positions, size=n, p=pos_weights)
    club_list = np.random.choice(club_sizes, size=n, p=[0.40, 0.35, 0.25])

    ages = np.clip(np.random.normal(26, 4, n).astype(int), 17, 38)
    nat_list = np.random.choice(nationalities, size=n)

    league_quality = np.where(club_list == 'Big', np.random.uniform(7, 10, n),
                     np.where(club_list == 'Medium', np.random.uniform(4, 7, n),
                              np.random.uniform(1, 4, n)))

    matches = np.clip(np.random.normal(28, 8, n).astype(int), 5, 38)
    minutes = matches * np.random.uniform(60, 90, n)
    injury_days = np.clip(np.random.exponential(15, n).astype(int), 0, 90)

    # Position-based stats
    goals, assists, shots_pg, key_passes_pg = [], [], [], []
    dribbles_pg, tackles_pg, interceptions_pg = [], [], []
    xg_list, xa_list = [], []

    for i, pos in enumerate(pos_list):
        m = matches[i]
        if pos == 'GK':
            goals.append(0)
            assists.append(np.random.randint(0, 2))
            shots_pg.append(round(np.random.uniform(0, 0.3), 2))
            key_passes_pg.append(round(np.random.uniform(0.2, 1.0), 2))
            dribbles_pg.append(round(np.random.uniform(0.1, 0.5), 2))
            tackles_pg.append(round(np.random.uniform(0.1, 0.5), 2))
            interceptions_pg.append(round(np.random.uniform(0.1, 0.5), 2))
            xg_list.append(round(np.random.uniform(0, 0.1), 2))
            xa_list.append(round(np.random.uniform(0, 0.1), 2))
        elif pos == 'Defender':
            goals.append(np.random.randint(0, 5))
            assists.append(np.random.randint(0, 6))
            shots_pg.append(round(np.random.uniform(0.3, 1.2), 2))
            key_passes_pg.append(round(np.random.uniform(0.5, 1.8), 2))
            dribbles_pg.append(round(np.random.uniform(0.3, 1.5), 2))
            tackles_pg.append(round(np.random.uniform(1.5, 4.5), 2))
            interceptions_pg.append(round(np.random.uniform(1.0, 4.0), 2))
            xg_list.append(round(np.random.uniform(0.05, 0.4), 2))
            xa_list.append(round(np.random.uniform(0.05, 0.5), 2))
        elif pos == 'Midfielder':
            goals.append(np.random.randint(2, 15))
            assists.append(np.random.randint(3, 18))
            shots_pg.append(round(np.random.uniform(1.0, 3.0), 2))
            key_passes_pg.append(round(np.random.uniform(1.5, 4.0), 2))
            dribbles_pg.append(round(np.random.uniform(0.8, 3.5), 2))
            tackles_pg.append(round(np.random.uniform(0.8, 3.0), 2))
            interceptions_pg.append(round(np.random.uniform(0.5, 2.5), 2))
            xg_list.append(round(np.random.uniform(0.1, 0.8), 2))
            xa_list.append(round(np.random.uniform(0.1, 0.9), 2))
        else:  # Forward
            goals.append(np.random.randint(5, 35))
            assists.append(np.random.randint(1, 15))
            shots_pg.append(round(np.random.uniform(2.0, 5.5), 2))
            key_passes_pg.append(round(np.random.uniform(0.8, 3.0), 2))
            dribbles_pg.append(round(np.random.uniform(1.0, 4.5), 2))
            tackles_pg.append(round(np.random.uniform(0.2, 1.5), 2))
            interceptions_pg.append(round(np.random.uniform(0.1, 1.0), 2))
            xg_list.append(round(np.random.uniform(0.3, 1.5), 2))
            xa_list.append(round(np.random.uniform(0.1, 0.8), 2))

    pass_acc = np.where(pos_list == 'GK', np.random.uniform(75, 92, n),
               np.where(pos_list == 'Defender', np.random.uniform(72, 90, n),
               np.where(pos_list == 'Midfielder', np.random.uniform(78, 93, n),
                        np.random.uniform(68, 85, n))))

    # Market value logic
    base_value = np.where(club_list == 'Big', np.random.uniform(10, 80, n),
                 np.where(club_list == 'Medium', np.random.uniform(2, 25, n),
                          np.random.uniform(0.3, 8, n)))

    age_factor = np.where(ages <= 23, 1.3, np.where(ages <= 29, 1.0, 0.65))
    market_value = np.round(base_value * age_factor, 1)

    wage_base = market_value * np.random.uniform(800, 1500, n)
    weekly_wage = np.round(wage_base, -2)

    fan_pop = np.clip(np.random.beta(2, 5, n) * 100, 1, 100).round(1)
    social_media = (market_value * np.random.uniform(0.5, 3, n) * 10000 + 
                    np.random.randint(1000, 500000, n)).astype(int)
    team_pos = np.where(club_list == 'Big', np.random.randint(1, 8, n),
               np.where(club_list == 'Medium', np.random.randint(5, 16, n),
                        np.random.randint(10, 21, n)))

    first_names = ['Luca', 'Carlos', 'Marcus', 'Andres', 'Thiago', 'Kevin', 'Rui',
                   'Omar', 'Diego', 'Lucas', 'Sergio', 'Marco', 'Ivan', 'Victor',
                   'Felipe', 'João', 'Antoine', 'Leroy', 'Kai', 'Jadon', 'Mason',
                   'Bukayo', 'Declan', 'Jude', 'Erling', 'Kylian', 'Vinicius',
                   'Pedri', 'Gavi', 'Rodri', 'Trent', 'Kieran', 'Ben', 'Reece']
    last_names = ['Silva', 'Martinez', 'Rashford', 'Iniesta', 'Alcantara', 'De Bruyne',
                  'Costa', 'Hassan', 'Torres', 'Hernandez', 'Gomez', 'Verratti',
                  'Perisic', 'Osimhen', 'Coutinho', 'Felix', 'Griezmann', 'Sane',
                  'Havertz', 'Sancho', 'Mount', 'Saka', 'Rice', 'Bellingham',
                  'Haaland', 'Mbappe', 'Junior', 'Gonzalez', 'Lopez', 'Rodrigo',
                  'Alexander', 'Tierney', 'White', 'James']

    names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]

    df = pd.DataFrame({
        'Player_ID': [f'P{str(i+1).zfill(4)}' for i in range(n)],
        'Player_Name': names,
        'Age': ages,
        'Nationality': nat_list,
        'Position': pos_list,
        'Club_Size': club_list,
        'League_Quality_Index': np.round(league_quality, 2),
        'Matches_Played': matches,
        'Minutes_Played': np.round(minutes).astype(int),
        'Goals': goals,
        'Assists': assists,
        'Shots_per_Game': shots_pg,
        'Key_Passes_per_Game': key_passes_pg,
        'Pass_Accuracy': np.round(pass_acc, 1),
        'Dribbles_per_Game': dribbles_pg,
        'Tackles_per_Game': tackles_pg,
        'Interceptions_per_Game': interceptions_pg,
        'Expected_Goals_xG': xg_list,
        'Expected_Assists_xA': xa_list,
        'Market_Value_Million_Euros': market_value,
        'Weekly_Wage_Euros': weekly_wage.astype(int),
        'Injury_Days_Last_Season': injury_days,
        'Team_League_Position': team_pos,
        'Fan_Popularity_Index': fan_pop,
        'Social_Media_Followers': social_media,
    })

    # Introduce ~3% missing values randomly in select columns
    for col in ['Pass_Accuracy', 'Key_Passes_per_Game', 'Fan_Popularity_Index']:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == '__main__':
    df = generate_football_dataset(1000)
    df.to_csv('players_dataset.csv', index=False)
    print(f"Dataset generated: {df.shape[0]} players, {df.shape[1]} columns")
    print(df.head())
