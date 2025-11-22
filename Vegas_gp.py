# Las Vegas GP 2025 – Hybrid Race-Prediction Model (v2.8)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v2.8 (Las Vegas):
#  - NEW: Added `WET_WEATHER_RATING` and logic to apply it if rain is forecast.
#  - RETAINED: `COLD_TIRE_WARMUP_RATING` is still used (always active due to cold temps).
#  - LOGIC UPDATE: If rain > 40%, Wet Weather ratings are ADDED to the score.
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
RACE_CIRCUIT = "Las Vegas"
RACE_DATE = "2025-11-22"
SIMILAR_TRACK_CIRCUIT = "Baku" # Best high-speed street comparison

# Las Vegas: Track position is key, but straights allow overtaking.
QUALI_WEIGHT = 0.55
LONG_RUN_PACE_WEIGHT = 0.25
PRACTICE_FAST_LAP_WEIGHT = 0.10
HISTORICAL_PACE_WEIGHT = 0.10
SAFETY_CAR_PROBABILITY = 0.85 # High chance of SC due to walls + cold tires

GRID_PENALTIES = {}

# Low grip, smooth asphalt, very cold. Graining is the main enemy.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.05, "McLaren": 1.06, 
    "Ferrari": 1.02, # Ferrari historically good at warm-up
    "Mercedes": 1.08, "Aston Martin": 1.10, "Alpine": 1.12,
    "RB": 1.09, "Williams": 1.10, 
    "Haas": 1.03, # Haas often eats tires, which helps in qualifying/cold
    "Sauber": 1.14
}

# Power and Drag Efficiency are king on The Strip.
POWER_UNIT_RATING = {
    "Red Bull": -0.25, # Honda engine + Red Bull Aero efficiency
    "Ferrari": -0.20,  # Strong engine
    "Mercedes": -0.15,
    "Alpine": 0.10,    # Often struggles with pure power/drag
}

STREET_CIRCUIT_RATING = {
    "VER": -0.25, "LEC": -0.25, "PER": -0.15, "SAI": -0.15, "HAM": -0.10
}

# Ability to switch tires on in cold/low-grip conditions.
COLD_TIRE_WARMUP_RATING = {
    "LEC": -0.30, # Unbeatable one-lap pace in low grip
    "SAI": -0.25,
    "HUL": -0.20, # Good at qualifying heater laps
    "VER": -0.20,
    "NOR": -0.10
}

# NEW: Wet Weather Rating (Added for v2.8)
WET_WEATHER_RATING = {
    "VER": -0.50, # Rain master
    "HAM": -0.40, # Excellent in wet
    "ALO": -0.35, # Adaptive genius
    "NOR": -0.30, # Very strong in wet
    "STR": -0.30, # Good in wet conditions
    "RUS": -0.25,
    "OCO": -0.20,
    "HUL": -0.15
}

def get_qualifying_data_2025():
    # UPDATED: Data from image_2a3603.png (Las Vegas GP Quali)
    # Drivers are in the exact P1-P20 order from the image.
    return pd.DataFrame({
        "Driver": [
            "NOR", "VER", "SAI", "RUS", "PIA", "LAW", "ALO", "HAD", "LEC", "GAS",
            "HUL", "STR", "OCO", "BEA", "COL", "ALB", "ANT", "BOR", "TSU", "HAM"
        ],
        "QualifyingTime (s)": [
            107.934,  # P1, Q3 Time (1:47.934)
            108.257,  # P2, Q3 Time (1:48.257)
            108.296,  # P3, Q3 Time (1:48.296)
            108.803,  # P4, Q3 Time (1:48.803)
            108.961,  # P5, Q3 Time (1:48.961)
            109.062,  # P6, Q3 Time (1:49.062)
            109.466,  # P7, Q3 Time (1:49.466)
            109.554,  # P8, Q3 Time (1:49.554)
            109.872,  # P9, Q3 Time (1:49.872)
            111.540,  # P10, Q3 Time (1:51.540)
            112.781,  # P11, Q2 Time (1:52.781)
            112.850,  # P12, Q2 Time (1:52.850)
            112.987,  # P13, Q2 Time (1:52.987)
            113.094,  # P14, Q2 Time (1:53.094)
            113.683,  # P15, Q2 Time (1:53.683)
            116.220,  # P16, Q1 Time (1:56.220)
            116.314,  # P17, Q1 Time (1:56.314)
            116.674,  # P18, Q1 Time (1:56.674)
            116.798,  # P19, Q1 Time (1:56.798)
            117.115   # P20, Q1 Time (1:57.115)
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    pts = { "McLaren": 756, "Ferrari": 362, "Mercedes": 398, "Red Bull": 366, "Williams": 111, "Aston Martin": 72, "Sauber": 62, "RB": 82, "Haas": 70, "Alpine": 22 }
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
    try:
        print(f"[i] Loading fastest practice laps from Las Vegas 2025...")
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
        print(f"[i] Loading LONG RUN pace from Las Vegas 2025...")
        stint_laps = []
        for session_type in ["FP2", "FP3"]:
            try:
                session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, session_type)
                session.load(laps=True, telemetry=False)
                laps = session.laps.pick_track_status('20').pick_quicklaps()
                for driver in laps['Driver'].unique():
                    driver_laps = laps.pick_driver(driver)
                    stints = driver_laps.get_stints()
                    for stint in stints:
                        if stint['StintLength'] >= 5:
                            stint_laps.append(stint['Laps'])
            except Exception: continue
        
        if not stint_laps: return pd.DataFrame()
        long_runs = pd.concat(stint_laps).dropna(subset=['LapTime'])
        if long_runs.empty: return pd.DataFrame()
            
        median_pace = long_runs.groupby('Driver')['LapTime'].median().dt.total_seconds()
        pace_df = pd.DataFrame(median_pace).reset_index()
        pace_df.columns = ['Driver', 'LongRunPace (s)']
        return pace_df
    except: return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        print(f"\n[i] Using similar track pace data (Baku) to fill gaps...")
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except: return pd.DataFrame()

def get_driver_track_adjustments():
    # Base affinity + Cold Warmup bonus
    street_adj = STREET_CIRCUIT_RATING
    warmup_adj = COLD_TIRE_WARMUP_RATING
    
    combined = {}
    for d in get_driver_team_map_2025():
        combined[d] = street_adj.get(d, 0.0) + warmup_adj.get(d, 0.0)
    return combined

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","")
    LAT,LON = 36.1147, -115.1728 # Las Vegas
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.2, 10 # Default 20% rain, 10 degrees C
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.2, 10

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

    # --- Chart 2: Team Technical Factors (PU & Tire) ---
    team_factors_df = team_factors_df.drop_duplicates(subset=['Team']).set_index('Team')
    team_factors_df['TireDegPenalty'] = team_factors_df['TireDeg'] - 1.0
    plot_data = team_factors_df[['TireDegPenalty', 'PowerUnitRating']]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_data.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_ylabel('Performance Adjustment (Higher is Worse)', color='white')
    ax2.set_title('Team Technical Factors (Power Unit & Tire Impact)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white', rotation=45); ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.3); ax2.legend(title='Factors')
    fig2.tight_layout()
    plt.show()

def predict_las_vegas_gp():
    print(f"=== LAS VEGAS GP {RACE_YEAR} – PREDICTOR v2.8 ===")

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

    # Safe filling logic
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
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    
    team_to_pu = {team: pu for pu, teams in {"Ferrari":["Ferrari", "Haas", "Sauber"], "Mercedes":["Mercedes", "McLaren", "Aston Martin", "Williams"], "Red Bull":["Red Bull", "RB"], "Alpine":["Alpine"]}.items() for team in teams}
    df["PowerUnitRating"] = df["Team"].map(team_to_pu).map(POWER_UNIT_RATING).fillna(0)

    rain_chance, temp = get_weather_forecast()
    team_factors_for_viz = df[['Team', 'TireDeg', 'PowerUnitRating']].copy()
    
    predictions = []
    for _, d in df.iterrows():
        # Standard weekend logic
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedLongRunPace*20 * LONG_RUN_PACE_WEIGHT +
                d.NormalizedPracticeFastLap*20 * PRACTICE_FAST_LAP_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.20)
        score += d.DriverAdjust + d.PowerUnitRating

        # UPDATED: Check if rain chance > 40%, then add wet rating
        if rain_chance > 0.4: 
            score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        predictions.append({"Driver": d.Driver, "Team": d.Team, "Qual": d.QualPos, "InitialScore": score, "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""})

    results = pd.DataFrame(predictions)
    
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.")
        mean_score = results["InitialScore"].mean()
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.65
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
    if rain_chance > 0.4:
        print(f"[!] WET WEATHER CONDITIONS DETECTED ({round(rain_chance*100,1)}%). Applying Wet Weather Ratings.")
    print(f"Weights: Quali={QUALI_WEIGHT}, LongRun={LONG_RUN_PACE_WEIGHT}, FastLap={PRACTICE_FAST_LAP_WEIGHT}, History={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    visualize_data(results, team_factors_for_viz)

if __name__=="__main__":
    predict_las_vegas_gp()