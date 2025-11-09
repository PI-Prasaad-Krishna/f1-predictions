# São Paulo GP 2025 – Hybrid Race-Prediction Model (Interlagos v2.6)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v2.6 (Brazil):
#  - NEW: Re-implemented Sprint Weekend logic. Weights are now split between
#    Quali, Sprint Result, Long Run, Fast Lap, and History.
#  - NEW: "Interlagos Weather Lottery" simulation. Adds a significant
#    random factor if rain chance is high, modeling the track's chaotic nature.
#  - NEW: Replaced extreme Mexico altitude factors with a single, more moderate
#    TEAM_ALTITUDE_RATING suitable for Interlagos (~760m).
#  - UPDATED: Long run pace analysis is now correctly limited to FP1.
#  - RETAINED: Visualizations, Safety Car Sim, and Long Run Pace analysis.
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
RACE_CIRCUIT = "São Paulo" # Official FastF1 name for Interlagos
RACE_DATE = "2025-11-16"
SIMILAR_TRACK_CIRCUIT = "Austria" # Good high-altitude, short-lap comparison

# UPDATED: Weights for a Sprint weekend, prioritizing quali, sprint, and long run.
QUALI_WEIGHT = 0.35
SPRINT_RESULT_WEIGHT = 0.35 # NEW: Sprint race is a strong indicator
LONG_RUN_PACE_WEIGHT = 0.20 # Best indicator of pure race pace
PRACTICE_FAST_LAP_WEIGHT = 0.05 # Low importance
HISTORICAL_PACE_WEIGHT = 0.05 # Low importance
# TOTAL = 1.0

SAFETY_CAR_PROBABILITY = 0.85 # Very high chance at Interlagos

GRID_PENALTIES = {}

TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.06, "McLaren": 1.08, "Ferrari": 1.14, "Mercedes": 1.10,
    "Aston Martin": 1.15, "Alpine": 1.12, "RB": 1.10, "Williams": 1.07,
    "Haas": 1.18, "Sauber": 1.16
}

# NEW: Tuned for moderate altitude (~760m). Less extreme than Mexico.
TEAM_ALTITUDE_RATING = {
    "Red Bull": -0.15, # Car concept still strong
    "McLaren": -0.05,
    "Williams": 0.0,
    "RB": 0.0,
    "Aston Martin": 0.05,
    "Alpine": 0.10,
    "Sauber": 0.10,
    "Haas": 0.15,
    "Ferrari": 0.15,
    "Mercedes": -0.05 # Still a weakness
}

# NEW: Mockup for Sprint Race. To be filled in.
SPRINT_RACE_RESULTS = {
    "NOR": 1, "ANT": 2, "RUS": 3, "VER": 4, "LEC": 5, "ALO": 6, "HAM": 7,
    "GAS": 8, "STR": 9, "HAD": 10, "OCO": 11, "BEA": 12, "TSU": 13,
    "SAI": 14, "HUL": 15, "LAW": 16, "ALB": 17, "BOR": 18, "PIA": 19, "COL": 20
}

WET_WEATHER_RATING = { # Standard rating for light/medium rain
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}

def get_qualifying_data_2025():
    # UPDATED: Data from image_71ec1d.png (Brazil GP Quali)
    # Drivers are in the exact P1-P20 order from the image.
    return pd.DataFrame({
        "Driver": [
            "NOR", "ANT", "LEC", "PIA", "HAD", "RUS", "LAW", "BEA", "GAS", "HUL",
            "ALO", "ALB", "HAM", "STR", "SAI", "VER", "OCO", "COL", "TSU", "BOR"
        ],
        "QualifyingTime (s)": [
            69.511,  # P1, Q3 Time (1:09.511)
            69.685,  # P2, Q3 Time (1:09.685)
            69.805,  # P3, Q3 Time (1:09.805)
            69.886,  # P4, Q3 Time (1:09.886)
            69.931,  # P5, Q3 Time (1:09.931)
            69.942,  # P6, Q3 Time (1:09.942)
            69.962,  # P7, Q3 Time (1:09.962)
            69.977,  # P8, Q3 Time (1:09.977)
            70.002,  # P9, Q3 Time (1:10.002)
            70.039,  # P10, Q3 Time (1:10.039)
            70.001,  # P11, Q2 Time (1:10.001)
            70.053,  # P12, Q2 Time (1:10.053)
            70.100,  # P13, Q2 Time (1:10.100)
            70.161,  # P14, Q2 Time (1:10.161)
            70.472,  # P15, Q2 Time (1:10.472)
            70.403,  # P16, Q1 Time (1:10.403)
            70.438,  # P17, Q1 Time (1:10.438)
            70.632,  # P18, Q1 Time (1:10.632)
            70.711,  # P19, Q1 Time (1:10.711)
            71.000    # P20, No time set
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    pts = { "McLaren": 680, "Ferrari": 320, "Mercedes": 300, "Red Bull": 290, "Williams": 90, "Aston Martin": 70, "Sauber": 60, "RB": 65, "Haas": 50, "Alpine": 25 }
    max_points = max(pts.values()); return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except: return pd.DataFrame()

def get_practice_fast_lap_2025():
    # Sprint weekend: Only FP1
    try:
        print(f"[i] Loading fastest practice laps from São Paulo 2025 (FP1)...")
        session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, "FP1")
        session.load(telemetry=False, laps=True)
        laps = session.laps.dropna(subset=['LapTime'])
        if laps.empty: return pd.DataFrame()
            
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver","LapTime"]].copy()
        practice["PracticeFastLap (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticeFastLap (s)"]]
    except: return pd.DataFrame()

def get_long_run_pace_2025():
    # Sprint weekend: Only FP1 data is available for long runs
    try:
        print(f"[i] Loading LONG RUN pace from São Paulo 2025 (FP1)...")
        stint_laps = []
        session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, "FP1")
        session.load(laps=True, telemetry=False)
        laps = session.laps.pick_track_status('20').pick_quicklaps() # Green flag laps
        
        for driver in laps['Driver'].unique():
            driver_laps = laps.pick_driver(driver)
            stints = driver_laps.get_stints()
            for stint in stints:
                if stint['StintLength'] >= 5: # A long run
                    stint_laps.append(stint['Laps'])
        
        if not stint_laps:
            print("[!] No long run data found in FP1.")
            return pd.DataFrame()

        long_runs = pd.concat(stint_laps).dropna(subset=['LapTime'])
        if long_runs.empty:
            print("[!] No valid long run laps found in FP1.")
            return pd.DataFrame()
            
        median_pace = long_runs.groupby('Driver')['LapTime'].median().dt.total_seconds()
        pace_df = pd.DataFrame(median_pace).reset_index()
        pace_df.columns = ['Driver', 'LongRunPace (s)']
        print(f"[✓] Calculated long run pace for {len(pace_df)} drivers from FP1.")
        return pace_df
        
    except Exception as e:
        print(f"[!] Error processing long run data: {e}")
        return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        print(f"\n[i] Using similar track pace data (Austria) to fill gaps...")
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except: 
        print(f"[!] Could not load similar track data from {SIMILAR_TRACK_CIRCUIT}")
        return pd.DataFrame()

def get_driver_track_adjustments():
    # Driver affinity for Interlagos.
    adjustments = { "HAM": -0.30, "VER": -0.20, "NOR": -0.10, "ALO": -0.10, "RUS": -0.10, "ANT": -0.15}
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","")
    LAT,LON = -23.7036, -46.6997 # Interlagos
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.5, 22 # Default high rain chance
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.5, 22

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

    # --- Chart 2: Team Technical Factors (Altitude & Tire) ---
    team_factors_df = team_factors_df.drop_duplicates(subset=['Team']).set_index('Team')
    team_factors_df['TireDegPenalty'] = team_factors_df['TireDeg'] - 1.0
    # UPDATED: Plot new altitude rating
    plot_data = team_factors_df[['TireDegPenalty', 'AltitudeRating']]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_data.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_ylabel('Performance Adjustment (Higher is Worse)', color='white')
    ax2.set_title('Team Technical Factors (Altitude & Tire Impact)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white', rotation=45); ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.3); ax2.legend(title='Factors')
    fig2.tight_layout()
    plt.show()

def predict_brazilian_gp():
    print(f"=== SÃO PAULO GP {RACE_YEAR} – PREDICTOR v2.6 (Sprint Weekend) ===")

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
    
    df.fillna({"Pace (s)":999, "PracticeFastLap (s)":999, "LongRunPace (s)":999, "SprintResult": 20}, inplace=True)

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
    driver_adjustments = get_driver_track_adjustments()
    df["DriverAdjust"] = df["Driver"].map(driver_adjustments)
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    df["SprintResult"] = df["Driver"].map(SPRINT_RACE_RESULTS)
    df["AltitudeRating"] = df["Team"].map(TEAM_ALTITUDE_RATING) # NEW

    rain_chance, temp = get_weather_forecast()
    team_factors_for_viz = df[['Team', 'TireDeg', 'AltitudeRating']].copy()
    
    predictions = []
    weather_lottery_triggered = False
    
    for _, d in df.iterrows():
        sprint_score = d.SprintResult
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.SprintResult * SPRINT_RESULT_WEIGHT +
                d.NormalizedLongRunPace*20 * LONG_RUN_PACE_WEIGHT +
                d.NormalizedPracticeFastLap*20 * PRACTICE_FAST_LAP_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.25)
        score += d.DriverAdjust
        score += d.AltitudeRating # Add altitude penalty

        # NEW: Interlagos Weather Lottery
        if rain_chance > 0.5: # High chance of rain = chaos
            weather_lottery_triggered = True
            weather_chaos = random.uniform(-0.5, 0.5) # Add significant chaos
            score += weather_chaos
        elif rain_chance > 0.3: # Medium chance of rain = standard wet rating
            score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        predictions.append({"Driver": d.Driver, "Team": d.Team, "Qual": d.QualPos, "InitialScore": score, "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""})

    results = pd.DataFrame(predictions)
    
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.")
        mean_score = results["InitialScore"].mean()
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.7
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
    if weather_lottery_triggered: print("[!] INTERLAGOS WEATHER LOTTERY WAS ACTIVATED")
    print(f"Weights: Quali={QUALI_WEIGHT}, Sprint={SPRINT_RESULT_WEIGHT}, LongRun={LONG_RUN_PACE_WEIGHT}, FastLap={PRACTICE_FAST_LAP_WEIGHT}, History={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    visualize_data(results, team_factors_for_viz)

if __name__=="__main__":
    predict_brazilian_gp()