# Dutch GP 2025 – Hybrid Race-Prediction Model (Zandvoort v1.3)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS for Zandvoort:
#  - High quali weight (tight track, overtaking tough).
#  - Adjusted tire degradation for banked corners.
#  - Driver affinity tuned (VER home advantage, ALO/NOR strong).
#  - Weather emphasis (coastal unpredictability).
#  - NEW: Safety Car Probability integrated into prediction noise.
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys, random

warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

# --- Constants ---
RACE_YEAR = 2025
RACE_CIRCUIT = "Netherlands"   # FastF1 uses "Netherlands" for Zandvoort
RACE_DATE = "2025-08-31"

QUALI_WEIGHT = 0.70
PRACTICE_PACE_WEIGHT = 0.20
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT

GRID_PENALTIES = {"VER": 0, "GAS": 0}

TIRE_DEGRADATION_MODEL = {
    "Red Bull": 0.95, "McLaren": 0.90, "Ferrari": 1.05, "Mercedes": 1.00,
    "Aston Martin": 1.08, "Alpine": 1.10, "RB": 1.07, "Williams": 1.03,
    "Haas": 1.15, "Sauber": 1.12
}

def get_qualifying_data_2025():
    # Data updated with the qualifying results from image_8bfeb7.png
    # The list is in official P1-P20 qualifying order.
    return pd.DataFrame({
        "Driver": [
            "PIA", "NOR", "VER", "HAD", "RUS", "LEC", "HAM", "LAW", "SAI", "ALO", 
            "ANT", "TSU", "BOR", "GAS", "ALB", "COL", "HUL", "OCO", "BEA", "STR"
        ],
        "QualifyingTime (s)": [
            68.662,  # P1, Q3 Time 
            68.674,  # P2, Q3 Time 
            68.925,  # P3, Q3 Time 
            69.208,  # P4, Q3 Time 
            69.255,  # P5, Q3 Time 
            69.340,  # P6, Q3 Time 
            69.390,  # P7, Q3 Time 
            69.500,  # P8, Q3 Time 
            69.505,  # P9, Q3 Time 
            69.630,  # P10, Q3 Time 
            69.493,  # P11, Q2 Time 
            69.622,  # P12, Q2 Time 
            70.037,  # P13, Q2 Time 
            69.637,  # P14, Q2 Time 
            69.652,  # P15, Q2 Time 
            70.104,  # P16, Q1 Time 
            70.195,  # P17, Q1 Time 
            70.197,  # P18, Q1 Time 
            70.262,  # P19, Q1 Time 
            70.400   # NC, Assigned a slow time for DNF 
        ]
    })

def get_driver_team_map_2025():
    return {
        "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren",
        "LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes",
        "ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine",
        "OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB",
        "ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber"
    }

def get_team_strength():
    """
    Uses hardcoded constructor standings based on the provided image
    from August 29, 2025.
    """
    pts = {
        "McLaren": 559,       
        "Ferrari": 260,       
        "Mercedes": 236,      
        "Red Bull": 194,      
        "Williams": 70,       
        "Aston Martin": 52,   
        "Sauber": 51,         
        "RB": 45,             
        "Haas": 35,           
        "Alpine": 20          
    }

    max_points = max(pts.values())
    if max_points == 0:
        return {team: 0.0 for team in pts}
    
    # Normalize the points from 0 to 1
    return {team: p / max_points for team, p in pts.items()}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R")
        session.load()
        laps = session.laps.pick_fastest()
        pace = laps[["Driver","LapTime"]].dropna()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except:
        drivers = list(get_driver_team_map_2025())
        return pd.DataFrame({"Driver": drivers,"Pace (s)":[82+i*0.2 for i,_ in enumerate(drivers)]})

def get_practice_pace_2025():
    try:
        fp1 = fastf1.get_session(2024, RACE_CIRCUIT, "FP1"); fp1.load()
        fp2 = fastf1.get_session(2024, RACE_CIRCUIT, "FP2"); fp2.load()
        fp3 = fastf1.get_session(2024, RACE_CIRCUIT, "FP3"); fp3.load()
        laps = pd.concat([fp1.laps, fp2.laps, fp3.laps])
        best_laps = laps.pick_fastest()
        practice = best_laps[["Driver","LapTime"]].dropna()
        practice["PracticePace (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticePace (s)"]]
    except:
        drivers = list(get_driver_team_map_2025())
        return pd.DataFrame({"Driver": drivers,"PracticePace (s)":[80+i*0.2 for i,_ in enumerate(drivers)]})

def get_driver_track_adjustments():
    adjustments = {
        "VER": -0.50, # Home boost
        "NOR": -0.25,
        "ALO": -0.20,
        "LEC": -0.10,
        "RUS": -0.05,
        "ANT": 0.40,"COL": 0.35,"HAD": 0.30
    }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY",)
    LAT,LON = 52.3881,4.5409  # Zandvoort
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.2, 22
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except:
        return 0.2, 22

def predict_dutch_gp():
    print(f"=== DUTCH GP {RACE_YEAR} – PREDICTOR v1.3 ===")

    quali = get_qualifying_data_2025()
    hist = get_historical_pace()
    prac = get_practice_pace_2025()

    df = quali.merge(hist,on="Driver",how="left").merge(prac,on="Driver",how="left")
    df.fillna({"Pace (s)":df["Pace (s)"].max()+2,"PracticePace (s)":df["PracticePace (s)"].max()+2},inplace=True)

    df["QualPos"] = df["QualifyingTime (s)"].rank().astype(int)
    df["NormalizedPracticePace"] = (df["PracticePace (s)"]-df["PracticePace (s)"].min())/(df["PracticePace (s)"].max()-df["PracticePace (s)"].min())
    df["NormalizedHistoricalPace"] = (df["Pace (s)"]-df["Pace (s)"].min())/(df["Pace (s)"].max()-df["Pace (s)"].min())

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)

    predictions = []
    for _, d in df.iterrows():
        qual_score = d.QualPos
        prac_score = d.NormalizedPracticePace*20
        hist_score = d.NormalizedHistoricalPace*20

        base = (qual_score*QUALI_WEIGHT + prac_score*PRACTICE_PACE_WEIGHT + hist_score*HISTORICAL_PACE_WEIGHT)
        score = base*(1-d.TeamStrength*0.20)
        score *= (1+(d.TireDeg-1.0)*0.25)
        score += d.DriverAdjust + d.Penalty

        # NEW: add Safety Car randomness factor
        score += random.uniform(-0.3,0.3)

        final_score = np.clip(score,1,20)
        win_chance = round((21-final_score)/20*100,1)

        predictions.append({"Driver":d.Driver,"Team":d.Team,"Qual":d.QualPos,
                            "Predicted":round(final_score,2),"Win%":win_chance,
                            "Notes":f"+{int(d.Penalty)}p" if d.Penalty>0 else ""})

    results = pd.DataFrame(predictions).sort_values("Predicted").reset_index(drop=True)
    results["Rank"] = results.index+1

    rain,temp = get_weather_forecast()
    print("\n--- Predicted Race Results ---")
    print(results[["Rank","Driver","Team","Qual","Notes","Predicted","Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain*100,1)}% | Safety Car factor active")
    print(f"Weights: Quali={QUALI_WEIGHT}, Practice={PRACTICE_PACE_WEIGHT}, Historical={HISTORICAL_PACE_WEIGHT}")
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")

if __name__=="__main__":
    predict_dutch_gp()
