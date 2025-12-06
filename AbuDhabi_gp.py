# Abu Dhabi GP 2025 – Hybrid Race-Prediction Model (Yas Marina v3.0)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v3.0 (Season Finale):
#  - NEW: `TWILIGHT_RACE_RATING`. Rewards adaptation to falling track temps.
#  - NEW: `SECTOR_3_TECHNICAL_RATING`. Critical for the hotel section traction.
#  - SIMILAR TRACK: Using Bahrain (Sakhir) as the best desert/night comparison.
#  - TUNING: Balanced standard weekend weights.
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys, random
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

# --- Constants & Data Models ---
RACE_YEAR = 2025
RACE_CIRCUIT = "Yas Marina" # FastF1 name
RACE_DATE = "2025-12-07"
SIMILAR_TRACK_CIRCUIT = "Bahrain" # Best comparison for track surface/layout

# Standard Weekend Weights (No Sprint)
QUALI_WEIGHT = 0.50
LONG_RUN_PACE_WEIGHT = 0.25 # FP2 Long runs are key
PRACTICE_FAST_LAP_WEIGHT = 0.10
HISTORICAL_PACE_WEIGHT = 0.15

SAFETY_CAR_PROBABILITY = 0.40 # Relatively low, lots of runoff

GRID_PENALTIES = {}

# Smooth surface but thermal deg in Sector 3 is high.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.04, "McLaren": 1.05, 
    "Ferrari": 1.03, # Good mechanical grip in S3
    "Mercedes": 1.07, "Aston Martin": 1.10, "Alpine": 1.11,
    "RB": 1.08, "Williams": 1.12, # Struggles in twisty S3
    "Haas": 1.15, "Sauber": 1.14
}

# Rewards mechanical grip and traction for the hotel section.
SECTOR_3_TECHNICAL_RATING = {
    "Ferrari": -0.20,  # Historically strong in traction zones
    "McLaren": -0.15,
    "Red Bull": -0.15,
    "Mercedes": -0.05,
    "Williams": 0.15   # Penalty for lack of grip
}

# Ability to adapt setup as track temps drop 10-15C during the session.
TWILIGHT_RACE_RATING = {
    "VER": -0.20, "HAM": -0.20, # Masters of adaptation
    "ALO": -0.15, "LEC": -0.10, "NOR": -0.10
}

def get_qualifying_data_2025():
    # UPDATED: Data from image_4411ba.png (Abu Dhabi GP Quali)
    # Drivers are in the exact P1-P20 order from the image.
    return pd.DataFrame({
        "Driver": [
            "VER", "NOR", "PIA", "RUS", "LEC", "ALO", "BOR", "OCO", "HAD", "TSU",
            "BEA", "SAI", "LAW", "ANT", "STR", "HAM", "ALB", "HUL", "GAS", "COL"
        ],
        "QualifyingTime (s)": [
            82.207,  # P1, Q3 Time (1:22.207)
            82.408,  # P2, Q3 Time (1:22.408)
            82.437,  # P3, Q3 Time (1:22.437)
            82.645,  # P4, Q3 Time (1:22.645)
            82.730,  # P5, Q3 Time (1:22.730)
            82.902,  # P6, Q3 Time (1:22.902)
            82.904,  # P7, Q3 Time (1:22.904)
            82.913,  # P8, Q3 Time (1:22.913)
            83.072,  # P9, Q3 Time (1:23.072)
            83.034,  # P10, Q2 Time (1:23.034)
            83.041,  # P11, Q2 Time (1:23.041)
            83.042,  # P12, Q2 Time (1:23.042)
            83.077,  # P13, Q2 Time (1:23.077)
            83.080,  # P14, Q2 Time (1:23.080)
            83.097,  # P15, Q2 Time (1:23.097)
            83.394,  # P16, Q1 Time (1:23.394)
            83.416,  # P17, Q1 Time (1:23.416)
            83.450,  # P18, Q1 Time (1:23.450)
            83.468,  # P19, Q1 Time (1:23.468)
            83.890   # P20, Q1 Time (1:23.890)
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    pts = { "McLaren": 800, "Ferrari": 382, "Mercedes": 459, "Red Bull": 426, "Williams": 137, "Aston Martin": 80, "Sauber": 68, "RB": 92, "Haas": 73, "Alpine": 22 }
    max_points = max(pts.values()); return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2023, RACE_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except: return pd.DataFrame()

def get_practice_fast_lap_2025():
    try:
        # Standard weekend: Check all sessions
        print(f"[i] Loading fastest practice laps from Abu Dhabi 2025...")
        all_laps = []
        for session_type in ["FP1", "FP2", "FP3"]:
            try:
                session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, session_type)
                session.load(telemetry=False, laps=True)
                all_laps.append(session.laps)
            except Exception: continue
        
        if not all_laps: return pd.DataFrame()
        laps = pd.concat(all_laps, ignore_index=True).dropna(subset=['LapTime'])
        if laps.empty: return pd.DataFrame()
            
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver","LapTime"]].copy()
        practice["PracticeFastLap (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticeFastLap (s)"]]
    except: return pd.DataFrame()

def get_long_run_pace_2025():
    try:
        # Standard weekend: FP2 is crucial (same time as race)
        print(f"[i] Loading LONG RUN pace from Abu Dhabi 2025 (FP2)...")
        session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, "FP2")
        session.load(laps=True, telemetry=False)
        laps = session.laps.pick_track_status('20').pick_quicklaps()
        stint_laps = []
        for driver in laps['Driver'].unique():
            driver_laps = laps.pick_driver(driver)
            stints = driver_laps.get_stints()
            for stint in stints:
                if stint['StintLength'] >= 5:
                    stint_laps.append(stint['Laps'])
        
        if not stint_laps: return pd.DataFrame()
        long_runs = pd.concat(stint_laps).dropna(subset=['LapTime'])
        median_pace = long_runs.groupby('Driver')['LapTime'].median().dt.total_seconds()
        pace_df = pd.DataFrame(median_pace).reset_index()
        pace_df.columns = ['Driver', 'LongRunPace (s)']
        return pace_df
    except: return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        print(f"\n[i] Using similar track pace data (Bahrain) to fill gaps...")
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except: return pd.DataFrame()

def get_driver_track_adjustments():
    # Yas Marina specialists
    adjustments = { "VER": -0.30, "HAM": -0.25, "LEC": -0.15, "NOR": -0.10 }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","")
    LAT,LON = 24.4672, 54.6031 # Yas Island
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.0, 26 # Standard mild evening
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.0, 26

def visualize_data(results_df, team_factors_df):
    plt.style.use('dark_background')

    # --- Chart 1: Win Chance Percentage ---
    top_10 = results_df.head(10).sort_values("Win%", ascending=True)
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, 10))
    bars1 = ax1.barh(top_10['Driver'], top_10['Win%'], color=colors)
    ax1.set_xlabel('Win Chance %', color='white')
    ax1.set_title('Predicted Win Chance for Top 10 Contenders', color='white', fontsize=16)
    ax1.tick_params(axis='x', colors='white'); ax1.tick_params(axis='y', colors='white')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}%', va='center', ha='left', color='white')
    fig1.tight_layout()

    # --- Chart 2: Team Technical Factors ---
    team_factors_df = team_factors_df.drop_duplicates(subset=['Team']).set_index('Team')
    team_factors_df['TireDegPenalty'] = team_factors_df['TireDeg'] - 1.0
    plot_data = team_factors_df[['TireDegPenalty', 'Sector3TechRating']]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_data.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_ylabel('Performance Adjustment (Higher is Worse)', color='white')
    ax2.set_title('Team Technical Factors (Tire Deg & Sector 3 Grip)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white', rotation=45); ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.3); ax2.legend(title='Factors')
    fig2.tight_layout()
    plt.show()

def predict_abu_dhabi_gp():
    print(f"=== ABU DHABI GP {RACE_YEAR} – PREDICTOR v3.0 ===")

    quali = get_qualifying_data_2025(); quali["QualPos"] = quali.index + 1
    
    historical_pace = get_historical_pace()
    practice_fast_lap = get_practice_fast_lap_2025()
    long_run_pace = get_long_run_pace_2025()
    similar_track_pace = get_similar_track_pace_2025()
    
    df = quali
    if not historical_pace.empty: df = df.merge(historical_pace, on="Driver", how="left")
    if not practice_fast_lap.empty: df = df.merge(practice_fast_lap, on="Driver", how="left")
    if not long_run_pace.empty: df = df.merge(long_run_pace, on="Driver", how="left")
    if not similar_track_pace.empty: df = df.merge(similar_track_pace, on="Driver", how="left")

    if 'SimilarPace (s)' in df.columns:
        if 'Pace (s)' in df.columns: df['Pace (s)'].fillna(df['SimilarPace (s)'], inplace=True)
        if 'LongRunPace (s)' in df.columns: df['LongRunPace (s)'].fillna(df['SimilarPace (s)'], inplace=True)
        if 'PracticeFastLap (s)' in df.columns: df['PracticeFastLap (s)'].fillna(df['SimilarPace (s)'], inplace=True)
    
    df.fillna({"Pace (s)":999, "PracticeFastLap (s)":999, "LongRunPace (s)":999}, inplace=True)

    normalization_map = {
        "PracticeFastLap (s)": "NormalizedPracticeFastLap",
        "LongRunPace (s)": "NormalizedLongRunPace",
        "Pace (s)": "NormalizedHistoricalPace"
    }
    for col, new_col in normalization_map.items():
        if col in df.columns:
            valid_data = df[col][df[col] != 999]
            if not valid_data.empty:
                min_val, max_val = valid_data.min(), valid_data.max()
                df[new_col] = (df[col] - min_val) / (max_val - min_val) if (max_val > min_val) else 0
            else: df[new_col] = 0
        else: df[new_col] = 0

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["Sector3TechRating"] = df["Team"].map(SECTOR_3_TECHNICAL_RATING).fillna(0) # NEW
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    df["TwilightRating"] = df["Driver"].map(TWILIGHT_RACE_RATING).fillna(0)

    rain_chance, temp = get_weather_forecast()
    team_factors_for_viz = df[['Team', 'TireDeg', 'Sector3TechRating']].copy()
    
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedLongRunPace*20 * LONG_RUN_PACE_WEIGHT +
                d.NormalizedPracticeFastLap*20 * PRACTICE_FAST_LAP_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.25)
        score += d.DriverAdjust + d.Sector3TechRating + d.TwilightRating
        
        predictions.append({"Driver": d.Driver, "Team": d.Team, "Qual": d.QualPos, "InitialScore": score, "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""})

    results = pd.DataFrame(predictions)
    
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.")
        mean_score = results["InitialScore"].mean()
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.6
    else:
        results["FinalScore"] = results["InitialScore"]
        
    results["Predicted"] = np.clip(results["FinalScore"], 1, 20)
    results["Win%"] = round((21 - results["Predicted"]) / 20 * 100, 1)
    
    results.sort_values("Predicted", inplace=True)
    results.reset_index(drop=True, inplace=True); results["Rank"] = results.index + 1
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank","Driver","Team","Qual","Notes","Predicted","Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100,1)}% | SC Probability: {int(SAFETY_CAR_PROBABILITY*100)}%")
    print(f"Weights: Quali={QUALI_WEIGHT}, LongRun={LONG_RUN_PACE_WEIGHT}, FastLap={PRACTICE_FAST_LAP_WEIGHT}, History={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    visualize_data(results, team_factors_for_viz)

if __name__=="__main__":
    predict_abu_dhabi_gp()