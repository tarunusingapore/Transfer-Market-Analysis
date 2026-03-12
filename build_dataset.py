"""
build_dataset.py
Combines 150 real seed players with ~1050 synthetically generated players
that follow the same statistical distributions per position/league/club_size.
Run:  python build_dataset.py
"""

import pandas as pd
import numpy as np
import os, random

np.random.seed(42)
random.seed(42)

# ── Realistic name pools ──────────────────────────────────────────────────────
FIRST = [
    "Luca","Carlos","Marcus","Andres","Thiago","Kevin","Rui","Omar","Diego","Lucas",
    "Sergio","Marco","Ivan","Victor","Felipe","João","Antoine","Leroy","Kai","Jadon",
    "Mason","Bukayo","Declan","Jude","Erling","Kylian","Vinicius","Pedri","Gavi","Rodri",
    "Trent","Kieran","Ben","Reece","Matthijs","Josko","Theo","Achraf","Alessandro","Jules",
    "William","Florian","Jamal","Phil","Cole","Alexander","Ollie","Brennan","Noni","Nicolas",
    "Ryan","Kobbie","Warren","Xavi","Tijjani","Enzo","Moises","Manuel","Sandro","Nicolo",
    "Adam","Benjamin","Khvicha","Lamine","Nico","Santiago","Jonathan","Serhou","Artem",
    "Michael","Rayan","Evan","Mateus","Randal","Donyell","Bryan","Ivan","Dango","Antoine",
    "Pedro","Gabriel","Emile","Karim","Jamie","Ansgar","Adrien","Pascal","Granit","Thomas",
    "Bernardo","Joshua","Nicolo","Stefan","Callum","Mikel","John","Diogo","Chris","Antoine",
    "Alvaro","Dusan","Kaoru","Emiliano","Jordan","Gregor","Andre","Yann","David","Mike",
    "Ibrahim","Cheick","Castello","Leny","Marc","Ousmane","Nuno","Destiny","Malo","Bradley",
    "Andrew","Edmond","Odilon","Kieran","Mats","Adama","Niclas","Romelu","Olivier","Wilfried",
    "Ruslan","Malang","Takehiro","Tariq","Tino","Alexis","Brenden","Ferran","Goncalo","Rodrygo",
    "Eduardo","Reinier","Fabio","Eder","Henrique","Rafael","Danilo","Alex","Bruno","Andre",
    "Maxime","Terem","Emmanuel","Felix","Seko","Nicolas","Remo","Youri","Charles","Timothy",
    "Christian","Weston","Yunus","Matteo","Simone","Lorenzo","Federico","Stefano","Giacomo",
    "Hiroki","Takumi","Wataru","Daichi","Ritsu","Yuya","Kaoru","Shoya","Ko","Kyogo",
    "Heung-min","Hwang","Lee","Kim","Park","Son","Jung","Choi","Han","Yoon",
    "Riyad","Youcef","Ismail","Sofiane","Nabil","Adem","Ilkay","Serge","Leon","Julian"
]

LAST = [
    "Silva","Martinez","Rashford","Iniesta","Alcantara","De Bruyne","Costa","Hassan",
    "Torres","Hernandez","Gomez","Verratti","Perisic","Osimhen","Coutinho","Felix",
    "Griezmann","Sane","Havertz","Sancho","Mount","Saka","Rice","Bellingham","Haaland",
    "Mbappe","Junior","Gonzalez","Lopez","Rodrigo","Alexander","Tierney","White","James",
    "de Ligt","Gvardiol","Hernandez","Hakimi","Bastoni","Kounde","Saliba","Wirtz","Musiala",
    "Foden","Palmer","Isak","Watkins","Johnson","Madueke","Jackson","Gravenberch","Mainoo",
    "Zaire-Emery","Simons","Reijnders","Fernandez","Caicedo","Ugarte","Tonali","Fagioli",
    "Hlozek","Sesko","Kvaratskhelia","Yamal","Williams","Gimenez","David","Guirassy",
    "Dovbyk","Olise","Cherki","Ferguson","Cunha","Kolo Muani","Malen","Mbeumo","Perez",
    "Ouattara","Semenyo","Neto","Martinelli","Smith Rowe","Adeyemi","Knauff","Marmoush",
    "Rabiot","Gross","Xhaka","Partey","Barella","Mkhitaryan","Trossard","Leao","Giroud",
    "Morata","Vlahovic","Mitoma","Martinez","Pickford","Kobel","Onana","Sommer","Raya","Maignan",
    "Konate","Araujo","Robertson","Mendes","Udogie","Gusto","Bradley","Hummels","Kossounou",
    "Trippier","Branthwaite","Traore","Fullkrug","Lukaku","Zaha","Torres","Konsa","Guehi",
    "Lukeba","Diomande","Porro","Tapsoba","van Dijk","Dias","Rudiger","Marquinhos",
    "Dumfries","Dalot","Magalhaes","Bastoni","Kounde","Hernandez","Davies","Arnold",
    "Modric","Kroos","Valverde","Kimmich","Odegaard","Fernandes","Silva","Bernardo",
    "Musah","Aaronson","Pulisic","McKennie","Reyna","Weah","Dest","Sargent","Mihailovic",
    "Nunez","Rodrigues","Andrade","Araujo","Pellistri","Suarez","Cavani","Forlan",
    "Diallo","Camara","Keita","Camara","Balde","Diaby","Fofana","Coman","Tchouameni",
    "Upamecano","Pavard","Saliba","Camavinga","Guendouzi","Aouar","Fekir","Benzema"
]

# ── League / club configs ─────────────────────────────────────────────────────
LEAGUES = {
    "Premier League":    {"quality": 9.8, "sizes": {"Big":0.4,"Medium":0.45,"Small":0.15}},
    "La Liga":           {"quality": 9.5, "sizes": {"Big":0.35,"Medium":0.45,"Small":0.20}},
    "Bundesliga":        {"quality": 9.2, "sizes": {"Big":0.35,"Medium":0.45,"Small":0.20}},
    "Serie A":           {"quality": 8.8, "sizes": {"Big":0.35,"Medium":0.45,"Small":0.20}},
    "Ligue 1":           {"quality": 8.5, "sizes": {"Big":0.30,"Medium":0.45,"Small":0.25}},
    "Eredivisie":        {"quality": 7.5, "sizes": {"Big":0.20,"Medium":0.50,"Small":0.30}},
    "Primeira Liga":     {"quality": 7.8, "sizes": {"Big":0.25,"Medium":0.45,"Small":0.30}},
    "Jupiler Pro League":{"quality": 7.0, "sizes": {"Big":0.20,"Medium":0.50,"Small":0.30}},
    "Scottish Prem":     {"quality": 6.5, "sizes": {"Big":0.20,"Medium":0.45,"Small":0.35}},
    "Super Lig":         {"quality": 7.4, "sizes": {"Big":0.25,"Medium":0.50,"Small":0.25}},
    "Ekstraklasa":       {"quality": 6.0, "sizes": {"Big":0.15,"Medium":0.45,"Small":0.40}},
    "Eliteserien":       {"quality": 6.2, "sizes": {"Big":0.15,"Medium":0.50,"Small":0.35}},
    "Allsvenskan":       {"quality": 6.3, "sizes": {"Big":0.15,"Medium":0.50,"Small":0.35}},
    "MLS":               {"quality": 6.8, "sizes": {"Big":0.25,"Medium":0.50,"Small":0.25}},
    "Liga MX":           {"quality": 7.0, "sizes": {"Big":0.25,"Medium":0.50,"Small":0.25}},
    "Brasileirao":       {"quality": 7.2, "sizes": {"Big":0.30,"Medium":0.45,"Small":0.25}},
    "Argentine Primera": {"quality": 6.9, "sizes": {"Big":0.25,"Medium":0.45,"Small":0.30}},
    "Saudi Pro League":  {"quality": 5.5, "sizes": {"Big":0.35,"Medium":0.40,"Small":0.25}},
    "J1 League":         {"quality": 6.5, "sizes": {"Big":0.20,"Medium":0.50,"Small":0.30}},
    "K League 1":        {"quality": 6.3, "sizes": {"Big":0.20,"Medium":0.50,"Small":0.30}},
}

CLUB_NAMES = {
    "Premier League": {
        "Big":    ["Manchester City","Liverpool","Arsenal","Chelsea","Manchester United","Tottenham","Newcastle","Aston Villa"],
        "Medium": ["Brighton","West Ham","Brentford","Fulham","Crystal Palace","Wolves","Bournemouth","Everton","Leicester","Nottingham Forest"],
        "Small":  ["Ipswich","Luton","Sheffield United","Burnley","Watford","Middlesbrough"],
    },
    "La Liga": {
        "Big":    ["Real Madrid","Barcelona","Atletico Madrid","Sevilla","Real Betis","Valencia"],
        "Medium": ["Athletic Club","Real Sociedad","Villarreal","Girona","Osasuna","Getafe","Rayo Vallecano"],
        "Small":  ["Deportivo Alaves","Cadiz","Granada","Almeria","Las Palmas"],
    },
    "Bundesliga": {
        "Big":    ["Bayern Munich","Borussia Dortmund","Bayer Leverkusen","RB Leipzig","Eintracht Frankfurt"],
        "Medium": ["Borussia Monchengladbach","Freiburg","Union Berlin","Hoffenheim","Wolfsburg","Mainz","Augsburg"],
        "Small":  ["Darmstadt","Heidenheim","Bochum","Koln","Schalke"],
    },
    "Serie A": {
        "Big":    ["Inter Milan","AC Milan","Juventus","Napoli","Roma","Lazio"],
        "Medium": ["Atalanta","Fiorentina","Bologna","Torino","Monza","Sassuolo","Udinese"],
        "Small":  ["Empoli","Lecce","Salernitana","Frosinone","Verona"],
    },
    "Ligue 1": {
        "Big":    ["PSG","Marseille","Monaco","Lyon","Lille","Nice"],
        "Medium": ["Rennes","Lens","Montpellier","Strasbourg","Nantes","Reims"],
        "Small":  ["Clermont","Metz","Lorient","Toulouse","Le Havre"],
    },
    "Eredivisie": {
        "Big":    ["Ajax","PSV","Feyenoord"],
        "Medium": ["AZ Alkmaar","FC Utrecht","Vitesse","Twente","Groningen"],
        "Small":  ["NEC Nijmegen","Heerenveen","Excelsior","Almere City"],
    },
    "Primeira Liga": {
        "Big":    ["Benfica","Porto","Sporting CP","Braga"],
        "Medium": ["Vitoria Guimaraes","Famalicao","Casa Pia","Boavista"],
        "Small":  ["Rio Ave","Arouca","Estrela Amadora","Moreirense"],
    },
    "Jupiler Pro League": {
        "Big":    ["Club Brugge","Anderlecht","Gent","Antwerp"],
        "Medium": ["Genk","Standard Liege","Mechelen","Westerlo"],
        "Small":  ["Cercle Brugge","Charleroi","Kortrijk"],
    },
    "Scottish Prem": {
        "Big":    ["Celtic","Rangers"],
        "Medium": ["Hearts","Hibernian","Aberdeen","Motherwell"],
        "Small":  ["St Mirren","Livingston","Ross County","Dundee"],
    },
    "Super Lig": {
        "Big":    ["Galatasaray","Fenerbahce","Besiktas","Trabzonspor"],
        "Medium": ["Basaksehir","Konyaspor","Alanyaspor","Sivasspor"],
        "Small":  ["Kasimpasa","Fatih Karagumruk","Rizespor"],
    },
    "Ekstraklasa": {
        "Big":    ["Legia Warsaw","Lech Poznan","Rakow Czestochowa"],
        "Medium": ["Wisla Krakow","Gornik Zabrze","Slask Wroclaw"],
        "Small":  ["Pogon Szczecin","Jagiellonia","Zaglebie Lubin"],
    },
    "Eliteserien": {
        "Big":    ["Bodo/Glimt","Molde","Rosenborg"],
        "Medium": ["Brann","Odd","Viking"],
        "Small":  ["Sandefjord","Stromsgodset","Haugesund"],
    },
    "Allsvenskan": {
        "Big":    ["Malmo","Djurgarden","IFK Goteborg"],
        "Medium": ["AIK","Hammarby","Hacken"],
        "Small":  ["Kalmar","Sirius","Varberg"],
    },
    "MLS": {
        "Big":    ["LA Galaxy","LAFC","Inter Miami","New York City FC","Seattle Sounders"],
        "Medium": ["Portland Timbers","Atlanta United","Toronto FC","New England Rev","DC United"],
        "Small":  ["Colorado Rapids","FC Dallas","Real Salt Lake","San Jose Earthquakes"],
    },
    "Liga MX": {
        "Big":    ["Club America","Chivas","Cruz Azul","Tigres UANL","Monterrey"],
        "Medium": ["Pumas","Leon","Atlas","Santos Laguna","Toluca"],
        "Small":  ["Necaxa","Queretaro","Mazatlan","FC Juarez"],
    },
    "Brasileirao": {
        "Big":    ["Flamengo","Palmeiras","Atletico Mineiro","Corinthians","Santos","Internacional"],
        "Medium": ["Vasco","Botafogo","Fluminense","Gremio","Cruzeiro","Sport"],
        "Small":  ["America Mineiro","Coritiba","Ceara","Goias","Bragantino"],
    },
    "Argentine Primera": {
        "Big":    ["River Plate","Boca Juniors","Racing Club","San Lorenzo","Independiente"],
        "Medium": ["Estudiantes","Lanus","Velez","Huracan","Colon"],
        "Small":  ["Talleres","Belgrano","Godoy Cruz","Central Cordoba"],
    },
    "Saudi Pro League": {
        "Big":    ["Al Hilal","Al Nassr","Al Ittihad","Al Ahli"],
        "Medium": ["Al Qadsiah","Al Taawoun","Al Fayha","Al Wahda"],
        "Small":  ["Al Hazm","Al Khaleej","Al Akhdoud"],
    },
    "J1 League": {
        "Big":    ["Yokohama F. Marinos","Urawa Reds","Kashima Antlers","Gamba Osaka"],
        "Medium": ["Nagoya Grampus","Sanfrecce Hiroshima","Cerezo Osaka","Vissel Kobe"],
        "Small":  ["Sagan Tosu","Shonan Bellmare","Consadole Sapporo"],
    },
    "K League 1": {
        "Big":    ["Jeonbuk Hyundai","Ulsan Hyundai","Seongnam FC"],
        "Medium": ["Incheon United","Suwon Samsung","Gimcheon Sangmu"],
        "Small":  ["Jeju United","Daejeon Citizen","Gangwon FC"],
    },
}

NATIONALITIES = [
    "Brazilian","French","English","Spanish","German","Argentine","Portuguese",
    "Dutch","Belgian","Croatian","Italian","Polish","Norwegian","Danish","Swedish",
    "Swiss","Austrian","Czech","Slovak","Hungarian","Romanian","Serbian","Ukrainian",
    "Russian","Turkish","Moroccan","Senegalese","Nigerian","Ghanaian","Ivorian",
    "Cameroonian","Egyptian","Algerian","South African","Congolese","Malian","Guinean",
    "Burkinabe","Japanese","South Korean","Australian","American","Canadian","Mexican",
    "Colombian","Uruguayan","Chilean","Peruvian","Ecuadorian","Venezulan","Bolivian",
    "Georgian","Armenian","Azerbaijani","Kazakh","Slovenian","Greek","Cypriot","Albanian",
    "Scottish","Welsh","Irish","Finnish","Estonian","Latvian","Lithuanian","Belarusian",
    "Bulgarian","Macedonian","Bosnian","Montenegrin","Kosovan","Moldovan","New Zealander",
]

POSITIONS = ["GK","Defender","Midfielder","Forward"]
POS_WEIGHTS = [0.10, 0.30, 0.35, 0.25]


def rand_name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def pick_club(league, size):
    options = CLUB_NAMES.get(league, {}).get(size, [f"{league} Club"])
    return random.choice(options)


def gen_stats(pos, age, club_size, league_quality):
    """Generate realistic per-position stats with league/club quality scaling."""
    q = league_quality / 10.0  # 0-1 quality scalar

    m  = int(np.clip(np.random.normal(28, 7), 5, 38))
    mp = int(np.clip(np.random.normal(m * 75, m * 12), m * 20, m * 90))
    inj = int(np.clip(np.random.exponential(18), 0, 120))

    if pos == "GK":
        g, a = 0, random.randint(0, 1)
        spg  = round(np.random.uniform(0.0, 0.2), 2)
        kppg = round(np.random.uniform(0.2, 1.0) * q, 2)
        acc  = round(np.random.uniform(78, 93) * q + 78 * (1-q), 1)
        drpg = round(np.random.uniform(0.0, 0.3), 2)
        tpg  = round(np.random.uniform(0.1, 0.4), 2)
        ipg  = round(np.random.uniform(0.1, 0.3), 2)
        xg   = 0.0;  xa = 0.0
        mv_base = {"Big": (5,40), "Medium": (1,10), "Small": (0.2,3)}[club_size]
    elif pos == "Defender":
        g  = random.randint(0, int(4 * q + 1))
        a  = random.randint(0, int(7 * q + 1))
        spg  = round(np.random.uniform(0.3, 1.5) * q, 2)
        kppg = round(np.random.uniform(0.4, 2.0) * q, 2)
        acc  = round(np.random.uniform(74, 93) * q + 74 * (1-q), 1)
        drpg = round(np.random.uniform(0.2, 2.0) * q, 2)
        tpg  = round(np.random.uniform(1.2, 5.0), 2)
        ipg  = round(np.random.uniform(0.8, 4.0), 2)
        xg   = round(np.random.uniform(0.02, 0.5 * q), 2)
        xa   = round(np.random.uniform(0.02, 0.6 * q), 2)
        mv_base = {"Big": (8,80), "Medium": (1,25), "Small": (0.2,6)}[club_size]
    elif pos == "Midfielder":
        g  = random.randint(1, int(18 * q + 2))
        a  = random.randint(2, int(20 * q + 2))
        spg  = round(np.random.uniform(0.8, 3.5) * q, 2)
        kppg = round(np.random.uniform(1.2, 4.5) * q, 2)
        acc  = round(np.random.uniform(78, 94) * q + 78 * (1-q), 1)
        drpg = round(np.random.uniform(0.5, 3.8) * q, 2)
        tpg  = round(np.random.uniform(0.6, 3.5), 2)
        ipg  = round(np.random.uniform(0.4, 2.8), 2)
        xg   = round(np.random.uniform(0.05, 0.9 * q), 2)
        xa   = round(np.random.uniform(0.05, 1.0 * q), 2)
        mv_base = {"Big": (10,100), "Medium": (2,30), "Small": (0.3,8)}[club_size]
    else:  # Forward
        g  = random.randint(3, int(35 * q + 3))
        a  = random.randint(1, int(18 * q + 1))
        spg  = round(np.random.uniform(1.8, 6.0) * q, 2)
        kppg = round(np.random.uniform(0.6, 3.2) * q, 2)
        acc  = round(np.random.uniform(68, 87) * q + 68 * (1-q), 1)
        drpg = round(np.random.uniform(0.8, 4.8) * q, 2)
        tpg  = round(np.random.uniform(0.2, 1.8), 2)
        ipg  = round(np.random.uniform(0.1, 1.2), 2)
        xg   = round(np.random.uniform(0.2, 1.6 * q), 2)
        xa   = round(np.random.uniform(0.05, 0.9 * q), 2)
        mv_base = {"Big": (10,120), "Medium": (1.5,35), "Small": (0.3,10)}[club_size]

    # Market value — peak at 24–27, adjusted by league quality
    age_f = max(0.3, 1.0 - abs(age - 25) * 0.04)
    mv = round(np.random.uniform(*mv_base) * age_f * (0.6 + 0.4 * q), 1)
    mv = max(0.2, mv)

    wage = int(round(mv * np.random.uniform(800, 1800), -2))

    fan  = round(np.clip(np.random.beta(1.5, 5) * 100, 1, 100), 1)
    smf  = int(mv * np.random.uniform(3000, 30000) + np.random.randint(500, 200000))
    tpos = random.randint(1, 20)

    return dict(
        Matches_Played=m, Minutes_Played=mp, Goals=g, Assists=a,
        Shots_per_Game=spg, Key_Passes_per_Game=kppg, Pass_Accuracy=acc,
        Dribbles_per_Game=drpg, Tackles_per_Game=tpg, Interceptions_per_Game=ipg,
        Expected_Goals_xG=xg, Expected_Assists_xA=xa,
        Market_Value_Million_Euros=mv, Weekly_Wage_Euros=wage,
        Injury_Days_Last_Season=inj, Team_League_Position=tpos,
        Fan_Popularity_Index=fan, Social_Media_Followers=smf,
    )


def generate_synthetic_players(n=1050, start_id=151):
    rows = []
    league_names = list(LEAGUES.keys())
    # Weight leagues by realism — top 5 more represented
    league_weights = [4,4,4,4,3, 2,2,2,1,2, 1,1,1,2,2,2,2,1,1,1]
    league_weights = [w/sum(league_weights) for w in league_weights]

    for i in range(n):
        league  = np.random.choice(league_names, p=league_weights)
        cfg     = LEAGUES[league]
        sizes   = cfg["sizes"]
        size    = np.random.choice(list(sizes.keys()), p=list(sizes.values()))
        pos     = np.random.choice(POSITIONS, p=POS_WEIGHTS)
        age     = int(np.clip(np.random.normal(25.5, 4.2), 16, 39))
        nation  = random.choice(NATIONALITIES)
        club    = pick_club(league, size)
        lqi     = round(cfg["quality"] + np.random.uniform(-0.3, 0.3), 2)

        stats = gen_stats(pos, age, size, cfg["quality"])

        rows.append({
            "Player_ID":  f"P{str(start_id + i).zfill(4)}",
            "Player_Name": rand_name(),
            "Age": age,
            "Nationality": nation,
            "Position": pos,
            "Club": club,
            "Club_Size": size,
            "League": league,
            "League_Quality_Index": lqi,
            **stats,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seed_path  = os.path.join(script_dir, "players_seed.csv")
    out_path   = os.path.join(script_dir, "players_dataset.csv")

    seed = pd.read_csv(seed_path)
    print(f"Seed players: {len(seed)}")

    synthetic = generate_synthetic_players(n=1050, start_id=len(seed) + 1)
    print(f"Synthetic players: {len(synthetic)}")

    combined = pd.concat([seed, synthetic], ignore_index=True)
    # Re-index Player_IDs cleanly
    combined["Player_ID"] = [f"P{str(i+1).zfill(4)}" for i in range(len(combined))]
    combined.to_csv(out_path, index=False)
    print(f"Final dataset: {len(combined)} players → {out_path}")
    print(combined["Position"].value_counts().to_dict())
    print(combined["League"].value_counts().head(8).to_dict())
