# Italian GP 2025 – Hybrid Race-Prediction Model (Monza v1.4)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS for Monza:
#  - Reduced quali weight (high-speed track, overtaking is possible).
#  - NEW: Added a POWER_UNIT_RATING for this power-sensitive circuit.
#  - CORRECTED: QualPos uses the official grid order, not raw time rank.
#  - RE-INTEGRATED: Weather model now adjusts scores based on rain probability.
#  - All factors (Tire Deg, Driver Affinity) tuned for Monza.
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
RACE_CIRCUIT = "Italy"   # FastF1 uses "Italy" for Monza
RACE_DATE = "2025-09-07"

# Monza: Overtaking is easier, so race pace is more important than qualifying.
QUALI_WEIGHT = 0.45
PRACTICE_PACE_WEIGHT = 0.30
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT

GRID_PENALTIES = {"VER": 0, "GAS": 0}

# Monza is low-deg, so the spread is smaller.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 0.98, "McLaren": 0.95, "Ferrari": 1.02, "Mercedes": 1.00,
    "Aston Martin": 1.05, "Alpine": 1.06, "RB": 1.04, "Williams": 1.01,
    "Haas": 1.08, "Sauber": 1.07
}

# NEW: Bonus/penalty for engine performance at a power track.
POWER_UNIT_RATING = {
    "Ferrari": -0.20, "Mercedes": -0.15, "Red Bull": -0.10, "Alpine": 0.05
}

# Performance adjustment for wet conditions. Negative is a bonus.
WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}

def get_qualifying_data_2025():
    # Data updated with the qualifying results from image_6aac9c.png
    # The list is in the official P1-P20 qualifying order.
    return pd.DataFrame({
        "Driver": [
            "VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "BOR", "ALO", "TSU",
            "BEA", "HUL", "SAI", "ALB", "OCO", "HAD", "STR", "COL", "GAS", "LAW"
        ], # Driver order is from the qualifying classification
        "QualifyingTime (s)": [
            78.792,  # P1, Q3 Time
            78.869,  # P2, Q3 Time
            78.982,  # P3, Q3 Time
            79.007,  # P4, Q3 Time
            79.124,  # P5, Q3 Time
            79.157,  # P6, Q3 Time
            79.200,  # P7, Q3 Time
            79.390,  # P8, Q3 Time
            79.424,  # P9, Q3 Time
            79.519,  # P10, Q3 Time
            79.446,  # P11, Q2 Time
            79.498,  # P12, Q2 Time
            79.528,  # P13, Q2 Time
            79.593,  # P14, Q2 Time
            79.707,  # P15, Q2 Time
            79.917,  # P16, Q1 Time
            79.949,  # P17, Q1 Time
            79.992,  # P18, Q1 Time
            80.103,  # P19, Q1 Time
            80.279   # P20, Q1 Time
        ] # Times correspond to each driver's best lap in their final session
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    # NOTE: This should be updated with the latest points before running.
    pts = { "McLaren": 584, "Ferrari": 260, "Mercedes": 248, "Red Bull": 214, "Williams": 80, "Aston Martin": 62, "Sauber": 51, "RB": 60, "Haas": 44, "Alpine": 20 }
    max_points = max(pts.values()); return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R"); session.load()
        laps = session.laps.pick_fastest()
        pace = laps[["Driver","LapTime"]].dropna(); pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except: return pd.DataFrame({"Driver": list(get_driver_team_map_2025()), "Pace (s)":[82+i*0.2 for i,_ in enumerate(get_driver_team_map_2025())]})

def get_practice_pace_2025():
    try:
        laps = pd.concat([fastf1.get_session(2024, RACE_CIRCUIT, s).load().laps for s in ["FP1", "FP2", "FP3"]])
        practice = laps.pick_fastest()[["Driver","LapTime"]].dropna()
        practice["PracticePace (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticePace (s)"]]
    except: return pd.DataFrame({"Driver": list(get_driver_team_map_2025()), "PracticePace (s)":[80+i*0.2 for i,_ in enumerate(get_driver_team_map_2025())]})

def get_driver_track_adjustments():
    # Adjustments for Monza, including Ferrari home race factor.
    adjustments = { "LEC": -0.25, "HAM": -0.25, "VER": -0.15, "ALB": -0.10, "GAS": -0.10, "ANT": 0.30, "COL": 0.25, "HAD": 0.20 }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","b16eee47fb847ac07fc76bf44805de5b")
    LAT,LON = 45.6156, 9.2811  # Monza
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.1, 25
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.1, 25

def predict_monza_gp():
    print(f"=== ITALIAN GP {RACE_YEAR} – PREDICTOR v1.4 ===")

    quali = get_qualifying_data_2025()
    # CORRECTED: Use the correct P1-P20 grid order, not a re-rank of times.
    quali["QualPos"] = quali.index + 1
    
    df = quali.merge(get_historical_pace(),on="Driver",how="left").merge(get_practice_pace_2025(),on="Driver",how="left")
    df.fillna({"Pace (s)":999,"PracticePace (s)":999},inplace=True)

    # Normalize Practice Pace
    p_min, p_max = df["PracticePace (s)"][df["PracticePace (s)"] != 999].min(), df["PracticePace (s)"][df["PracticePace (s)"] != 999].max()
    df["NormalizedPracticePace"] = (df["PracticePace (s)"] - p_min) / (p_max - p_min)

    # Normalize Historical Pace (with the correct column name)
    h_min, h_max = df["Pace (s)"][df["Pace (s)"] != 999].min(), df["Pace (s)"][df["Pace (s)"] != 999].max()
    df["NormalizedHistoricalPace"] = (df["Pace (s)"] - h_min) / (h_max - h_min)

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    
    # Map the Power Unit rating for each driver based on their team
    team_to_pu = {team: pu for pu, teams in {"Ferrari":["Ferrari", "Haas"], "Mercedes":["Mercedes", "McLaren", "Aston Martin", "Williams"], "Red Bull":["Red Bull", "RB"], "Alpine":["Alpine", "Sauber"]}.items() for team in teams}
    df["PowerUnitRating"] = df["Team"].map(team_to_pu).map(POWER_UNIT_RATING).fillna(0)

    rain_chance, temp = get_weather_forecast()
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedPracticePace*20 * PRACTICE_PACE_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.15) # Lower impact for low-deg track
        score += d.DriverAdjust + d.PowerUnitRating # Add PU rating

        if rain_chance > 0.4: score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        score += random.uniform(-0.3, 0.3) # Safety car randomness

        final_score = np.clip(score,1,20)
        win_chance = round((21-final_score)/20*100,1)

        predictions.append({"Driver":d.Driver,"Team":d.Team,"Qual":d.QualPos,
                             "Predicted":round(final_score,2),"Win%":win_chance,
                             "Notes":f"+{int(d.Penalty)}p" if d.Penalty>0 else ""})

    results = pd.DataFrame(predictions).sort_values("Predicted").reset_index(drop=True)
    results["Rank"] = results.index+1
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank","Driver","Team","Qual","Notes","Predicted","Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100,1)}% | Stochastic factors active")
    print(f"Weights: Quali={QUALI_WEIGHT}, Practice={PRACTICE_PACE_WEIGHT}, Historical={HISTORICAL_PACE_WEIGHT}")
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")

if __name__=="__main__":
    predict_monza_gp()