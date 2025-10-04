# Azerbaijan GP 2025 – Hybrid Race-Prediction Model (Baku v1.7)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v1.7:
#  - FIXED: Corrected a ValueError by implementing a more robust method for
#    fetching fastest laps, ensuring a DataFrame is always returned.
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys, random

warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

# --- Constants & Data Models ---
RACE_YEAR = 2025
RACE_CIRCUIT = "Azerbaijan"
RACE_DATE = "2025-09-21"
SIMILAR_TRACK_CIRCUIT = "Saudi Arabia" # Used for rookie baseline

# Baku: Hybrid track. Qualifying is important but the long straight allows for overtaking.
QUALI_WEIGHT = 0.55
PRACTICE_PACE_WEIGHT = 0.25
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT
SAFETY_CAR_PROBABILITY = 0.80 # Baku has a very high chance of a safety car

GRID_PENALTIES = {}

TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.00, "McLaren": 0.98, "Ferrari": 1.04, "Mercedes": 1.02,
    "Aston Martin": 1.06, "Alpine": 1.08, "RB": 1.05, "Williams": 1.03,
    "Haas": 1.12, "Sauber": 1.10
}

POWER_UNIT_RATING = {
    "Ferrari": -0.20, "Mercedes": -0.15, "Red Bull": -0.10, "Alpine": 0.05
}

# Bonus for drivers who excel at street circuits.
STREET_CIRCUIT_RATING = {
    "VER": -0.25, "LEC": -0.20, "HAM": -0.15, "ALO": -0.15, "ALB": -0.10
}

WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}


def get_qualifying_data_2025():
    # Mock-up quali data for Baku (P1-P20 order)
    return pd.DataFrame({
        "Driver": [
            "VER", "SAI", "LAW", "ANT", "RUS", "TSU", "NOR", "HAD", "PIA", "LEC",
            "ALO", "HAM", "BOR", "STR", "BEA", "COL", "HUL", "OCO", "GAS", "ALB"
        ],
        "QualifyingTime (s)": [
            101.117,  # P1, Q3 Time
            101.595,  # P2, Q3 Time
            101.707,  # P3, Q3 Time
            101.717,  # P4, Q3 Time
            102.070,  # P5, Q3 Time
            102.143,  # P6, Q3 Time
            102.239,  # P7, Q3 Time
            102.372,  # P8, Q3 Time
            101.414,  # P9, Q2 Time (No Q3 time set)
            101.519,  # P10, Q2 Time (No Q3 time set)
            101.857,  # P11, Q2 Time
            102.183,  # P12, Q2 Time
            102.277,  # P13, Q2 Time
            103.061,  # P14, Q2 Time
            102.666,  # P15, Q1 Time (No Q2 time set)
            102.779,  # P16, Q1 Time
            102.916,  # P17, Q1 Time
            103.004,  # P18, Q1 Time
            103.139,  # P19, Q1 Time
            103.778   # P20, Q1 Time
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    # NOTE: This should be updated with the latest points before running.
    pts = { "McLaren": 617, "Ferrari": 280, "Mercedes": 260, "Red Bull": 239, "Williams": 86, "Aston Martin": 62, "Sauber": 55, "RB": 61, "Haas": 44, "Alpine": 20 }
    max_points = max(pts.values()); return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R"); session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        # FIX: Use a more robust method to get fastest lap per driver
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]]
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except: return pd.DataFrame({"Driver": list(get_driver_team_map_2025()), "Pace (s)":[104+i*0.2 for i,_ in enumerate(get_driver_team_map_2025())]})

def get_practice_pace_2025():
    try:
        laps = pd.concat([fastf1.get_session(2024, RACE_CIRCUIT, s).load().laps for s in ["FP1", "FP2", "FP3"]])
        laps.dropna(subset=['LapTime'], inplace=True)
        # FIX: Use a more robust method to get fastest lap per driver
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver","LapTime"]]
        practice["PracticePace (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticePace (s)"]]
    except: return pd.DataFrame({"Driver": list(get_driver_team_map_2025()), "PracticePace (s)":[102+i*0.2 for i,_ in enumerate(get_driver_team_map_2025())]})

def get_similar_track_pace_2025():
    """
    NEW: Fetches race pace from a similar track (Jeddah) in the current year
    to provide a baseline for rookies or drivers with no Baku history.
    """
    try:
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        # FIX: Use a more robust method to get fastest lap per driver
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]]
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except Exception as e:
        print(f"Could not load similar track data: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

def get_driver_track_adjustments():
    # General driver affinity for the Baku circuit.
    # REMOVED: Rookie penalties are no longer needed as pace is now data-driven.
    adjustments = { "VER": -0.15, "LEC": -0.15, "HAM": -0.10, "ALO": -0.10 }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","b16eee47fb847ac07fc76bf44805de5b")
    LAT,LON = 40.3725, 49.8533  # Baku
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.05, 26
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.05, 26

def predict_baku_gp():
    print(f"=== AZERBAIJAN GP {RACE_YEAR} – PREDICTOR v1.7 ===")

    quali = get_qualifying_data_2025()
    quali["QualPos"] = quali.index + 1
    
    # Load all data sources, including the new similar track data
    historical_pace = get_historical_pace()
    practice_pace = get_practice_pace_2025()
    similar_track_pace = get_similar_track_pace_2025()
    
    df = quali.merge(historical_pace,on="Driver",how="left").merge(practice_pace,on="Driver",how="left")
    if not similar_track_pace.empty:
        df = df.merge(similar_track_pace, on="Driver", how="left")

    # ENHANCED: Fill missing historical pace with similar track data for rookies
    if 'SimilarPace (s)' in df.columns:
        print("\n[i] Using similar track pace data to fill gaps for rookies...")
        df['Pace (s)'].fillna(df['SimilarPace (s)'], inplace=True)
    
    # Fill any remaining NaNs with a high-value penalty
    df.fillna({"Pace (s)":999,"PracticePace (s)":999},inplace=True)

    # Normalize pace columns
    for col, new_col in [("PracticePace (s)", "NormalizedPracticePace"), ("Pace (s)", "NormalizedHistoricalPace")]:
        min_val, max_val = df[col][df[col] != 999].min(), df[col][df[col] != 999].max()
        df[new_col] = (df[col] - min_val) / (max_val - min_val)

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    df["StreetRating"] = df["Driver"].map(STREET_CIRCUIT_RATING).fillna(0)
    
    team_to_pu = {team: pu for pu, teams in {"Ferrari":["Ferrari", "Haas", "Sauber"], "Mercedes":["Mercedes", "McLaren", "Aston Martin", "Williams"], "Red Bull":["Red Bull", "RB"], "Alpine":["Alpine"]}.items() for team in teams}
    df["PowerUnitRating"] = df["Team"].map(team_to_pu).map(POWER_UNIT_RATING).fillna(0)

    rain_chance, temp = get_weather_forecast()
    
    # --- Scoring Algorithm ---
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedPracticePace*20 * PRACTICE_PACE_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.20)
        score += d.DriverAdjust + d.PowerUnitRating + d.StreetRating

        if rain_chance > 0.4: score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        predictions.append({"Driver": d.Driver, "Team": d.Team, "Qual": d.QualPos, "InitialScore": score, "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""})

    results = pd.DataFrame(predictions)
    
    # --- ENHANCED: Safety Car Simulation ---
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.")
        mean_score = results["InitialScore"].mean()
        # Compress scores towards the mean to simulate a bunched-up field
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.6
    else:
        results["FinalScore"] = results["InitialScore"]
        
    results["Predicted"] = np.clip(results["FinalScore"], 1, 20)
    results["Win%"] = round((21 - results["Predicted"]) / 20 * 100, 1)
    
    results.sort_values("Predicted", inplace=True)
    results.reset_index(drop=True, inplace=True)
    results["Rank"] = results.index + 1
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank","Driver","Team","Qual","Notes","Predicted","Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100,1)}% | SC Probability: {int(SAFETY_CAR_PROBABILITY*100)}%")
    print(f"Weights: Quali={QUALI_WEIGHT}, Practice={PRACTICE_PACE_WEIGHT}, Historical={HISTORICAL_PACE_WEIGHT}")
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")

if __name__=="__main__":
    predict_baku_gp()

