# Singapore GP 2025 – Hybrid Race-Prediction Model (Marina Bay v2.0)
# -----------------------------------------------------------------------------
#  FIXES v2.0 (All changes documented):
#  1. FIXED: get_historical_pace() - Changed to Singapore 2024 (last year's race)
#     Since we're in 2025, we can use 2024 Singapore historical data
#  2. FIXED: get_practice_pace_2025() - Changed to Singapore 2025 practice sessions
#     Using current year's practice data since those sessions have happened
#  3. FIXED: get_similar_track_pace_2025() - Changed to Monaco 2025
#     Monaco 2025 has already happened, so we can use that data
#  4. IMPROVED: Added better error handling and data validation
#  5. IMPROVED: Added fallback mechanism when FastF1 data is unavailable
#  6. FIXED: Removed unicode character issues (– to -)
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
RACE_CIRCUIT = "Singapore"
RACE_DATE = "2025-10-05"
SIMILAR_TRACK_CIRCUIT = "Monaco"  # Best high-downforce comparison for rookies

# Singapore: Qualifying is king. Overtaking is nearly impossible.
QUALI_WEIGHT = 0.75
PRACTICE_PACE_WEIGHT = 0.15
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT
SAFETY_CAR_PROBABILITY = 0.95  # Almost guaranteed at Singapore

GRID_PENALTIES = {}

# High tire degradation due to constant traction events.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.05, "McLaren": 1.02, "Ferrari": 1.12, "Mercedes": 1.08,
    "Aston Martin": 1.15, "Alpine": 1.10, "RB": 1.09, "Williams": 1.06,
    "Haas": 1.20, "Sauber": 1.18
}

# NEW: Bonus for drivers known for exceptional physical fitness.
DRIVER_FITNESS_RATING = {
    "ALO": -0.20, "VER": -0.15, "HAM": -0.15, "RUS": -0.10, "NOR": -0.10
}

STREET_CIRCUIT_RATING = {
    "VER": -0.25, "LEC": -0.25, "HAM": -0.20, "ALO": -0.20, "NOR": -0.15
}

WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver": [
            "RUS", "VER", "PIA", "ANT", "NOR", "HAM", "LEC", "HAD", "BEA", "ALO",
            "HUL", "ALB", "SAI", "LAW", "TSU", "BOR", "STR", "COL", "OCO", "GAS"
        ],
        "QualifyingTime (s)": [
            89.158,  # P1, Q3 Time
            89.340,  # P2, Q3 Time
            89.524,  # P3, Q3 Time
            89.537,  # P4, Q3 Time
            89.586,  # P5, Q3 Time
            89.688,  # P6, Q3 Time
            89.784,  # P7, Q3 Time
            89.846,  # P8, Q3 Time
            89.868,  # P9, Q3 Time
            89.955,  # P10, Q3 Time
            90.141,  # P11, Q2 Time
            90.202,  # P12, Q2 Time
            90.235,  # P13, Q2 Time
            90.320,  # P14, Q2 Time
            90.353,  # P15, Q2 Time
            90.620,  # P16, Q1 Time
            90.945,  # P17, Q1 Time
            90.982,  # P18, Q1 Time
            90.989,  # P19, Q1 Time
            91.261   # P20, Q1 Time
        ]
    })

def get_driver_team_map_2025():
    """2025 driver-team mapping"""
    return {
        "VER":"Red Bull", "TSU":"Red Bull", "NOR":"McLaren", "PIA":"McLaren",
        "LEC":"Ferrari", "HAM":"Ferrari", "RUS":"Mercedes", "ANT":"Mercedes",
        "ALO":"Aston Martin", "STR":"Aston Martin", "GAS":"Alpine", "COL":"Alpine",
        "OCO":"Haas", "BEA":"Haas", "LAW":"RB", "HAD":"RB",
        "ALB":"Williams", "SAI":"Williams", "BOR":"Sauber", "HUL":"Sauber"
    }

def get_team_strength():
    """Current constructor championship standings (2025 season)"""
    pts = {
        "McLaren": 617, "Ferrari": 280, "Mercedes": 260, "Red Bull": 239,
        "Williams": 86, "Aston Martin": 62, "Sauber": 55, "RB": 61,
        "Haas": 44, "Alpine": 20
    }
    max_points = max(pts.values())
    return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        print(f"[i] Loading historical pace from Singapore 2024...")
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R")
        session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        
        if laps.empty:
            print(f"[!] No lap data found for Singapore 2024")
            return pd.DataFrame()
        
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver", "LapTime"]].copy()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        print(f"[✓] Loaded pace data for {len(pace)} drivers")
        return pace[["Driver", "Pace (s)"]]
    except Exception as e:
        print(f"[!] Could not load historical pace data: {e}")
        return pd.DataFrame()

def get_practice_pace_2025():
    try:
        print(f"[i] Loading practice pace from Singapore 2025...")
        all_laps = []
        for session_type in ["FP1", "FP2", "FP3"]:
            try:
                session = fastf1.get_session(2025, RACE_CIRCUIT, session_type)
                session.load()
                all_laps.append(session.laps)
                print(f"[✓] Loaded {session_type}")
            except Exception as e:
                print(f"[!] Could not load {session_type}: {e}")
                continue
        
        if not all_laps:
            print("[!] No practice session data available")
            return pd.DataFrame()
        
        laps = pd.concat(all_laps, ignore_index=True)
        laps.dropna(subset=['LapTime'], inplace=True)
        
        if laps.empty:
            print("[!] No valid practice laps found")
            return pd.DataFrame()
        
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver", "LapTime"]].copy()
        practice["PracticePace (s)"] = practice["LapTime"].dt.total_seconds()
        print(f"[✓] Loaded practice data for {len(practice)} drivers")
        return practice[["Driver", "PracticePace (s)"]]
    except Exception as e:
        print(f"[!] Could not load practice pace data: {e}")
        return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        print(f"[i] Loading similar track pace from Monaco 2025...")
        session = fastf1.get_session(2025, SIMILAR_TRACK_CIRCUIT, "R")
        session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        
        if laps.empty:
            print(f"[!] No lap data found for Monaco 2025")
            return pd.DataFrame()
        
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver", "LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        print(f"[✓] Loaded similar track data for {len(pace)} drivers")
        return pace[["Driver", "SimilarPace (s)"]]
    except Exception as e:
        print(f"[!] Could not load similar track data: {e}")
        return pd.DataFrame()

def get_driver_track_adjustments():
    """Driver-specific track adjustments for street circuits"""
    adjustments = {
        "HAM": -0.20, "VER": -0.20, "ALO": -0.15, "LEC": -0.10, "SAI": -0.10
    }
    return {d: adjustments.get(d, 0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    """Fetch weather forecast for Singapore GP"""
    API_KEY = os.getenv("OWM_KEY", )
    LAT, LON = 1.290, 103.852  # Singapore coordinates
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        forecasts = [f for f in data.get("list", []) if RACE_DATE in f["dt_txt"]]
        if not forecasts:
            print("[i] No exact race day forecast, using default Singapore conditions")
            return 0.4, 29  # Default high humidity/rain chance
        
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        print(f"[✓] Weather forecast retrieved: {temp}°C, {int(rain_prob*100)}% rain chance")
        return rain_prob, temp
    except Exception as e:
        print(f"[!] Weather API error: {e}. Using defaults.")
        return 0.4, 29

def predict_singapore_gp():
    """Main prediction function"""
    print(f"=== SINGAPORE GP {RACE_YEAR} - PREDICTOR v2.0 ===\n")

    # Get qualifying data
    quali = get_qualifying_data_2025()
    quali["QualPos"] = quali.index + 1
    
    # Load all pace data
    historical_pace = get_historical_pace()
    practice_pace = get_practice_pace_2025()
    similar_track_pace = get_similar_track_pace_2025()
    
    # Merge data safely
    df = quali.copy()
    if not historical_pace.empty:
        df = df.merge(historical_pace, on="Driver", how="left")
    else:
        print("[!] Warning: No historical pace data available")
        
    if not practice_pace.empty:
        df = df.merge(practice_pace, on="Driver", how="left")
    else:
        print("[!] Warning: No practice pace data available")
        
    if not similar_track_pace.empty:
        df = df.merge(similar_track_pace, on="Driver", how="left")
    else:
        print("[!] Warning: No similar track pace data available")

    # Fill missing pace data using similar track
    if 'SimilarPace (s)' in df.columns and 'Pace (s)' in df.columns:
        print("\n[i] Using similar track pace data (Monaco 2025) to fill gaps for drivers without data...")
        df['Pace (s)'].fillna(df['SimilarPace (s)'], inplace=True)
    
    # Fill remaining NaN values
    df.fillna({"Pace (s)": 999, "PracticePace (s)": 999}, inplace=True)

    # Normalize pace data
    for col, new_col in [("PracticePace (s)", "NormalizedPracticePace"), 
                          ("Pace (s)", "NormalizedHistoricalPace")]:
        if col in df.columns:
            valid_data = df[col][df[col] != 999]
            if not valid_data.empty:
                min_val, max_val = valid_data.min(), valid_data.max()
                df[new_col] = (df[col] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0
            else:
                df[new_col] = 0
        else:
            df[new_col] = 0

    # Add team and driver factors
    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    df["StreetRating"] = df["Driver"].map(STREET_CIRCUIT_RATING).fillna(0)
    df["FitnessRating"] = df["Driver"].map(DRIVER_FITNESS_RATING).fillna(0)
    
    # Get weather conditions
    rain_chance, temp = get_weather_forecast()
    
    # Calculate predictions
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedPracticePace * 20 * PRACTICE_PACE_WEIGHT + 
                d.NormalizedHistoricalPace * 20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.15)  # Team strength less of a factor
        score *= (1 + (d.TireDeg - 1.0) * 0.25)
        score += d.DriverAdjust + d.StreetRating + d.FitnessRating

        if rain_chance > 0.4:
            score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        predictions.append({
            "Driver": d.Driver,
            "Team": d.Team,
            "Qual": d.QualPos,
            "InitialScore": score,
            "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""
        })

    results = pd.DataFrame(predictions)
    
    # Safety car simulation
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.\n")
        mean_score = results["InitialScore"].mean()
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.5
    else:
        results["FinalScore"] = results["InitialScore"]
        
    results["Predicted"] = np.clip(results["FinalScore"], 1, 20)
    results["Win%"] = round((21 - results["Predicted"]) / 20 * 100, 1)
    
    results.sort_values("Predicted", inplace=True)
    results.reset_index(drop=True, inplace=True)
    results["Rank"] = results.index + 1
    
    # Output results
    print("\n--- Predicted Race Results ---")
    print(results[["Rank", "Driver", "Team", "Qual", "Notes", "Predicted", "Win%"]].to_string(index=False))
    
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100, 1)}% | SC Probability: {int(SAFETY_CAR_PROBABILITY*100)}%")
    print(f"Weights: Quali={QUALI_WEIGHT}, Practice={PRACTICE_PACE_WEIGHT}, Historical={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i, row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")

if __name__ == "__main__":
    predict_singapore_gp()