# US GP 2025 – Hybrid Race-Prediction Model (Austin v2.2)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS v2.2:
#  - UPDATED: The qualifying data now reflects the FINAL starting grid,
#    including all applied penalties.
#  - All other features from v2.1 (Sprint Results, Visualizations) are retained.
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
RACE_CIRCUIT = "United States"
RACE_DATE = "2025-10-26"
SIMILAR_TRACK_CIRCUIT = "Japan" # Suzuka is a good high-speed comparison

# COTA: A balanced track. Qualifying and race pace are both important.
QUALI_WEIGHT = 0.50
PRACTICE_PACE_WEIGHT = 0.20
HISTORICAL_PACE_WEIGHT = 0.15
# NEW: Sprint result is a strong indicator of current form.
SPRINT_RESULT_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT - HISTORICAL_PACE_WEIGHT

# Penalties are now baked into the final starting grid order.
GRID_PENALTIES = {}

# COTA is notoriously bumpy and high-energy, leading to high tire deg.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.08, "McLaren": 1.05, "Ferrari": 1.15, "Mercedes": 1.10,
    "Aston Martin": 1.18, "Alpine": 1.12, "RB": 1.11, "Williams": 1.07,
    "Haas": 1.22, "Sauber": 1.20
}

# Sprint Race results from Saturday. Lower score is better.
SPRINT_RACE_RESULTS = {
    "VER": 1, "RUS": 2, "SAI": 3, "HAM": 4, "LEC": 5, "ALB": 6, "TSU": 7,
    "ANT": 8, "LAW": 9, "GAS": 10, "BOR": 11, "HAD": 12, "HUL": 13,
    "COL": 14, "BEA": 15, "OCO": 16, "STR": 17, "NOR": 18, "PIA": 19, "ALO": 20
}

WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}

def get_qualifying_data_2025():
    # UPDATED: Reflects the FINAL starting grid for the race.
    return pd.DataFrame({
        "Driver": [
            "VER", "NOR", "LEC", "RUS", "HAM", "PIA", "ANT", "BEA", "SAI", "ALO",
            "HUL", "LAW", "TSU", "GAS", "COL", "BOR", "OCO", "ALB", "STR", "HAD"
        ],
        "QualifyingTime (s)": [
            92.510,  # P1
            92.801,  # P2
            92.807,  # P3
            92.826,  # P4
            92.912,  # P5
            93.084,  # P6
            93.114,  # P7
            93.139,  # P8
            93.150,  # P9
            93.160,  # P10
            93.334,  # P11
            93.360,  # P12
            93.466,  # P13
            93.651,  # P14
            94.044,  # P15
            94.125,  # P16
            94.136,  # P17
            94.690,  # P18 (original P19)
            94.540,  # P19 (original P18, with penalty)
            95.000   # P20
        ]
    })

def get_driver_team_map_2025():
    return { "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber" }

def get_team_strength():
    pts = { "McLaren": 650, "Ferrari": 300, "Mercedes": 280, "Red Bull": 260, "Williams": 90, "Aston Martin": 70, "Sauber": 60, "RB": 65, "Haas": 50, "Alpine": 25 }
    max_points = max(pts.values()); return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R"); session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","Pace (s)"]]
    except: return pd.DataFrame()

def get_practice_pace_2025():
    # On a sprint weekend, there is only one practice session.
    try:
        session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, "FP1"); session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver","LapTime"]].copy()
        practice["PracticePace (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver","PracticePace (s)"]]
    except: return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R"); session.load()
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver","LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver","SimilarPace (s)"]]
    except: return pd.DataFrame()

def get_driver_track_adjustments():
    # Driver affinity for COTA.
    adjustments = { "HAM": -0.30, "VER": -0.25, "LEC": -0.15, "NOR": -0.10 }
    return {d: adjustments.get(d,0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY","")
    LAT,LON = 30.1328, -97.6411 # COTA
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        data = requests.get(url,timeout=10).json()
        forecasts = [f for f in data.get("list",[]) if RACE_DATE in f["dt_txt"]]
        if not forecasts: return 0.1, 24
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts)//2]["main"]["temp"]
        return rain_prob, temp
    except: return 0.1, 24

def visualize_results(results_df, driver_adjustments):
    """NEW: Generate and display charts using Matplotlib."""
    # --- Chart 1: Win Chance Percentage ---
    top_10 = results_df.head(10).sort_values("Win%", ascending=True)
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, 10))
    bars = ax1.barh(top_10['Driver'], top_10['Win%'], color=colors)
    ax1.set_xlabel('Win Chance %', color='white')
    ax1.set_title('Predicted Win Chance for Top 10 Contenders', color='white', fontsize=16)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)

    for bar in bars:
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.1f}%', va='center', ha='left', color='white')

    # --- Chart 2: Driver Track Affinity ---
    affinity_df = pd.DataFrame(list(driver_adjustments.items()), columns=['Driver', 'Affinity'])
    affinity_df = affinity_df.sort_values('Affinity', ascending=False)
    
    fig, ax2 = plt.subplots(figsize=(12, 8))
    colors = ['#FF5733' if x > 0 else '#33C1FF' for x in affinity_df['Affinity']]
    bars = ax2.barh(affinity_df['Driver'], affinity_df['Affinity'], color=colors)
    ax2.set_xlabel('Performance Adjustment (Lower is Better)', color='white')
    ax2.set_title('Driver Track Affinity Scores (COTA)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.axvline(0, color='white', linewidth=0.8)

    plt.tight_layout()
    plt.show()


def predict_austin_gp():
    print(f"=== US GP {RACE_YEAR} – PREDICTOR v2.2 (Sprint Weekend) ===")

    quali = get_qualifying_data_2025(); quali["QualPos"] = quali.index + 1
    
    historical_pace = get_historical_pace()
    practice_pace = get_practice_pace_2025()
    similar_track_pace = get_similar_track_pace_2025()
    
    df = quali
    if not historical_pace.empty: df = df.merge(historical_pace, on="Driver", how="left")
    if not practice_pace.empty: df = df.merge(practice_pace, on="Driver", how="left")
    if not similar_track_pace.empty: df = df.merge(similar_track_pace, on="Driver", how="left")

    if 'SimilarPace (s)' in df.columns:
        print("\n[i] Using similar track pace data (Suzuka) to fill gaps for rookies...")
        df['Pace (s)'].fillna(df['SimilarPace (s)'], inplace=True)
    
    df.fillna({"Pace (s)":999,"PracticePace (s)":999},inplace=True)

    for col, new_col in [("PracticePace (s)", "NormalizedPracticePace"), ("Pace (s)", "NormalizedHistoricalPace")]:
        valid_data = df[col][df[col] != 999]
        min_val, max_val = (valid_data.min(), valid_data.max()) if not valid_data.empty else (0,1)
        df[new_col] = (df[col] - min_val) / (max_val - min_val) if (max_val > min_val) else 0

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["SprintResult"] = df["Driver"].map(SPRINT_RACE_RESULTS)
    driver_adjustments = get_driver_track_adjustments()
    df["DriverAdjust"] = df["Driver"].map(driver_adjustments)
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)

    rain_chance, temp = get_weather_forecast()
    
    predictions = []
    for _, d in df.iterrows():
        sprint_score = d.SprintResult
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedPracticePace*20 * PRACTICE_PACE_WEIGHT + 
                d.NormalizedHistoricalPace*20 * HISTORICAL_PACE_WEIGHT +
                sprint_score * SPRINT_RESULT_WEIGHT) # Add sprint result to score
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.25)
        score += d.DriverAdjust

        if rain_chance > 0.4: score += WET_WEATHER_RATING.get(d.Driver, 0.0)
        
        predictions.append({"Driver": d.Driver, "Team": d.Team, "Qual": d.QualPos, "InitialScore": score, "Notes": f"+{int(d.Penalty)}p" if d.Penalty > 0 else ""})

    results = pd.DataFrame(predictions)
    results["Predicted"] = np.clip(results["InitialScore"], 1, 20)
    results["Win%"] = round((21 - results["Predicted"]) / 20 * 100, 1)
    
    results.sort_values("Predicted", inplace=True)
    results.reset_index(drop=True, inplace=True); results["Rank"] = results.index + 1
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank","Driver","Team","Qual","Notes","Predicted","Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100,1)}%")
    print(f"Weights: Quali={QUALI_WEIGHT}, Sprint={SPRINT_RESULT_WEIGHT}, Practice={PRACTICE_PACE_WEIGHT}, Historical={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i,row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    # NEW: Call the visualization function
    visualize_results(results, driver_adjustments)

if __name__=="__main__":
    predict_austin_gp()