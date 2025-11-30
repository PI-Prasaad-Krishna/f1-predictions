# Qatar GP 2025 – Hybrid Race-Prediction Model (Lusail v2.9)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v2.9 (Qatar):
#  - SPRINT WEEKEND: Integrated Sprint results from user image (PIA P1).
#  - NEW: `HIGH_G_CORNERING_RATING`. Rewards cars/drivers good in high-speed sweeps.
#  - NEW: `TIRE_STRESS_PENALTY`. Lusail destroys tires; high deg is punished heavily.
#  - RE-INTRODUCED: `DRIVER_FITNESS_RATING`. Qatar is physically grueling.
#  - SIMILAR TRACK: Using Suzuka (Japan) as the best high-load cornering baseline.
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
RACE_CIRCUIT = "Lusail" # FastF1 name for Qatar
RACE_DATE = "2025-11-30"
SIMILAR_TRACK_CIRCUIT = "Japan" # Suzuka is the best high-speed/flowing match

# Sprint Weekend Weights
QUALI_WEIGHT = 0.35
SPRINT_RESULT_WEIGHT = 0.35 # Sprint is a massive indicator here
LONG_RUN_PACE_WEIGHT = 0.20 # FP1 Long runs
PRACTICE_FAST_LAP_WEIGHT = 0.05
HISTORICAL_PACE_WEIGHT = 0.05

SAFETY_CAR_PROBABILITY = 0.60 # Moderate (Gravel traps, but wide run-offs)

GRID_PENALTIES = {"BOR":5}

# High lateral load = High Deg. Cars that slide pay a huge price.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.05, "McLaren": 1.03, # McLaren loves high speed corners
    "Ferrari": 1.10, "Mercedes": 1.08,
    "Aston Martin": 1.12, "Alpine": 1.11,
    "RB": 1.09, "Williams": 1.14, # Often struggles with high downforce
    "Haas": 1.18, "Sauber": 1.16
}

# Rewards cars with efficient high-downforce packages.
HIGH_G_CORNERING_RATING = {
    "McLaren": -0.25, # Best in class for high speed
    "Red Bull": -0.20,
    "Ferrari": -0.05,
    "Mercedes": -0.10,
    "Williams": 0.15 # Draggy or lacks load
}

# Physical endurance bonus for the heat/G-force.
DRIVER_FITNESS_RATING = {
    "ALO": -0.15, "HAM": -0.10, "VER": -0.10, "NOR": -0.10, "RUS": -0.05
}

# Sprint Results from Image (P1 Piastri -> P20 Colapinto)
SPRINT_RACE_RESULTS = {
    "PIA": 1, "RUS": 2, "NOR": 3, "VER": 4, "TSU": 5, "ANT": 6, "ALO": 7,
    "SAI": 8, "HAD": 9, "ALB": 10, "BOR": 11, "BEA": 12, "LEC": 13,
    "LAW": 14, "OCO": 15, "HUL": 16, "HAM": 17, "GAS": 18, "STR": 19, "COL": 20
}

WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.30
}

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver": [
            "PIA", "NOR", "VER", "RUS", "ANT", "HAD", "SAI", "ALO", "GAS", "LEC",
            "HUL", "LAW", "BEA", "BOR", "ALB", "TSU", "OCO", "HAM", "STR", "COL"
        ],
        "QualifyingTime (s)": [
            79.387,  # P1, Q3 Time (1:19.387)
            79.495,  # P2, Q3 Time (1:19.495)
            79.651,  # P3, Q3 Time (1:19.651)
            79.662,  # P4, Q3 Time (1:19.662)
            79.846,  # P5, Q3 Time (1:19.846)
            80.114,  # P6, Q3 Time (1:20.114)
            80.287,  # P7, Q3 Time (1:20.287)
            80.418,  # P8, Q3 Time (1:20.418)
            80.477,  # P9, Q3 Time (1:20.477)
            80.561,  # P10, Q3 Time (1:20.561)
            80.353,  # P11, Q2 Time (1:20.353)
            80.433,  # P12, Q2 Time (1:20.433)
            80.438,  # P13, Q2 Time (1:20.438)
            80.534,  # P14, Q2 Time (1:20.534)
            80.629,  # P15, Q2 Time (1:20.629)
            80.761,  # P16, Q1 Time (1:20.761)
            80.864,  # P17, Q1 Time (1:20.864)
            80.907,  # P18, Q1 Time (1:20.907)
            81.058,  # P19, Q1 Time (1:21.058)
            81.137   # P20, Q1 Time (1:21.137)
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    pts = { "McLaren": 770, "Ferrari": 378, "Mercedes": 441, "Red Bull": 400, "Williams": 122, "Aston Martin": 74, "Sauber": 68, "RB": 90, "Haas": 73, "Alpine": 22 }
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
        # Sprint weekend = FP1 only
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
    try:
        # Sprint weekend = FP1 only for long runs
        print(f"[i] Loading LONG RUN pace from Qatar 2025 (FP1)...")
        stint_laps = []
        session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, "FP1")
        session.load(laps=True, telemetry=False)
        laps = session.laps.pick_track_status('20').pick_quicklaps()
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
        print(f"\n[i] Using similar track pace data (Suzuka) to fill gaps...")
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except: return pd.DataFrame()

def get_driver_track_adjustments():
    # Affinity for high-speed, flowing tracks
    adjustments = { "VER": -0.25, "NOR": -0.20, "PIA": -0.20, "HAM": -0.15, "RUS": -0.10 }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","")
    LAT,LON = 25.4888, 51.4542 # Lusail
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.0, 28 # Warm desert night
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.0, 28

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
    plot_data = team_factors_df[['TireDegPenalty', 'HighGCornering']]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_data.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_ylabel('Performance Adjustment (Higher is Worse)', color='white')
    ax2.set_title('Team Technical Factors (High G-Force & Tire Stress)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white', rotation=45); ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.3); ax2.legend(title='Factors')
    fig2.tight_layout()
    plt.show()

def predict_qatar_gp():
    print(f"=== QATAR GP {RACE_YEAR} – PREDICTOR v2.9 ===")

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
    df["HighGCornering"] = df["Team"].map(HIGH_G_CORNERING_RATING) # NEW
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    df["SprintResult"] = df["Driver"].map(SPRINT_RACE_RESULTS)
    df["FitnessRating"] = df["Driver"].map(DRIVER_FITNESS_RATING).fillna(0)

    rain_chance, temp = get_weather_forecast()
    team_factors_for_viz = df[['Team', 'TireDeg', 'HighGCornering']].copy()
    
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.SprintResult * SPRINT_RESULT_WEIGHT + 
                d.NormalizedLongRunPace*20 * LONG_RUN_PACE_WEIGHT +
                d.NormalizedPracticeFastLap*20 * PRACTICE_FAST_LAP_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.30) # High impact of tire deg
        score += d.DriverAdjust + d.HighGCornering + d.FitnessRating

        if rain_chance > 0.4: score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
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
    print(f"Weights: Quali={QUALI_WEIGHT}, Sprint={SPRINT_RESULT_WEIGHT}, LongRun={LONG_RUN_PACE_WEIGHT}, FastLap={PRACTICE_FAST_LAP_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    visualize_data(results, team_factors_for_viz)

if __name__=="__main__":
    predict_qatar_gp()