# Mexico City GP 2025 – Hybrid Race-Prediction Model (v2.5 - FIXED)
# -----------------------------------------------------------------------------
#  FIXES:
#  - Fixed KeyError when columns don't exist
#  - Added safe column filling and normalization
#  - Improved error handling for data loading
#  - Fixed division by zero in normalization
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
RACE_CIRCUIT = "Mexico City"
RACE_DATE = "2025-11-02"
SIMILAR_TRACK_CIRCUIT = "Austria"

QUALI_WEIGHT = 0.55
LONG_RUN_PACE_WEIGHT = 0.25
PRACTICE_FAST_LAP_WEIGHT = 0.10
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - LONG_RUN_PACE_WEIGHT - PRACTICE_FAST_LAP_WEIGHT
SAFETY_CAR_PROBABILITY = 0.75

GRID_PENALTIES = {}

TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.05, "McLaren": 1.07, "Ferrari": 1.12, "Mercedes": 1.10,
    "Aston Martin": 1.16, "Alpine": 1.11, "RB": 1.09, "Williams": 1.06,
    "Haas": 1.18, "Sauber": 1.15
}

ALTITUDE_COOLING_RATING = {
    "Red Bull": -0.20, "McLaren": -0.10, "Williams": -0.05, "RB": 0.0,
    "Aston Martin": 0.05, "Alpine": 0.10, "Sauber": 0.15, "Haas": 0.20,
    "Ferrari": 0.25, "Mercedes": 0.30
}

POWER_UNIT_RELIABILITY = {
    "Red Bull": -0.25,
    "Ferrari": 0.10,
    "Mercedes": 0.15,
    "Alpine": 0.20
}

WET_WEATHER_RATING = {
    "VER": -0.50, "HAM": -0.40, "ALO": -0.35, "NOR": -0.20, "OCO": -0.20
}

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver": [
            "NOR", "LEC", "HAM", "RUS", "VER", "ANT", "SAI", "PIA", "HAD", "BEA",
            "TSU", "OCO", "HUL", "ALO", "LAW", "BOR", "ALB", "GAS", "STR", "COL"
        ],
        "QualifyingTime (s)": [
            75.586, 75.848, 75.938, 76.034, 76.070, 76.118, 76.172, 76.174,
            76.252, 76.460, 76.816, 76.837, 77.016, 77.103, 78.072, 77.412,
            77.490, 77.546, 77.606, 77.670
        ]
    })

def get_driver_team_map_2025():
    return {
        "VER": "Red Bull", "TSU": "Red Bull", "NOR": "McLaren", "PIA": "McLaren",
        "LEC": "Ferrari", "HAM": "Ferrari", "RUS": "Mercedes", "ANT": "Mercedes",
        "ALO": "Aston Martin", "STR": "Aston Martin", "GAS": "Alpine", "COL": "Alpine",
        "OCO": "Haas", "BEA": "Haas", "LAW": "RB", "HAD": "RB",
        "ALB": "Williams", "SAI": "Williams", "BOR": "Sauber", "HUL": "Sauber"
    }

def get_team_strength():
    pts = {
        "McLaren": 670, "Ferrari": 310, "Mercedes": 290, "Red Bull": 280,
        "Williams": 90, "Aston Martin": 70, "Sauber": 60, "RB": 65,
        "Haas": 50, "Alpine": 25
    }
    max_points = max(pts.values())
    return {team: p / max_points for team, p in pts.items()} if max_points > 0 else {team: 0 for team in pts}

def get_historical_pace():
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R")
        session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver", "LapTime"]].copy()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver", "Pace (s)"]]
    except Exception as e:
        print(f"[!] Could not load historical pace: {e}")
        return pd.DataFrame()

def get_practice_fast_lap_2025():
    try:
        print(f"[i] Loading fastest practice laps from Mexico 2025 (FP1, FP2, FP3)...")
        all_laps = []
        for session_type in ["FP1", "FP2", "FP3"]:
            try:
                session = fastf1.get_session(RACE_YEAR, RACE_CIRCUIT, session_type)
                session.load(telemetry=False, laps=True)
                all_laps.append(session.laps)
            except Exception:
                continue
        
        if not all_laps:
            print("[!] No practice data found.")
            return pd.DataFrame()
        
        laps = pd.concat(all_laps, ignore_index=True).dropna(subset=['LapTime'])
        if laps.empty:
            return pd.DataFrame()
            
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        practice = fastest_laps[["Driver", "LapTime"]].copy()
        practice["PracticeFastLap (s)"] = practice["LapTime"].dt.total_seconds()
        return practice[["Driver", "PracticeFastLap (s)"]]
    except Exception as e:
        print(f"[!] Error loading practice data: {e}")
        return pd.DataFrame()

def get_long_run_pace_2025():
    try:
        print(f"[i] Loading LONG RUN pace from Mexico 2025 (FP2, FP3)...")
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
            except Exception as e:
                print(f"[!] Could not load {session_type} for long runs: {e}")
                continue
        
        if not stint_laps:
            print("[!] No long run data found.")
            return pd.DataFrame()

        long_runs = pd.concat(stint_laps).dropna(subset=['LapTime'])
        if long_runs.empty:
            print("[!] No valid long run laps found.")
            return pd.DataFrame()
            
        median_pace = long_runs.groupby('Driver')['LapTime'].median().dt.total_seconds()
        pace_df = pd.DataFrame(median_pace).reset_index()
        pace_df.columns = ['Driver', 'LongRunPace (s)']
        print(f"[✓] Calculated long run pace for {len(pace_df)} drivers.")
        return pace_df
        
    except Exception as e:
        print(f"[!] Error processing long run data: {e}")
        return pd.DataFrame()

def get_similar_track_pace_2025():
    try:
        print(f"\n[i] Using similar track pace data (Austria) to fill gaps...")
        session = fastf1.get_session(RACE_YEAR, SIMILAR_TRACK_CIRCUIT, "R")
        session.load(laps=True, telemetry=False, weather=False)
        laps = session.laps.dropna(subset=['LapTime'])
        fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
        pace = fastest_laps[["Driver", "LapTime"]].copy()
        pace["SimilarPace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver", "SimilarPace (s)"]]
    except Exception as e:
        print(f"[!] Could not load similar track data from {SIMILAR_TRACK_CIRCUIT}: {e}")
        return pd.DataFrame()

def get_driver_track_adjustments():
    adjustments = {"VER": -0.30, "HAM": -0.15, "LEC": -0.10}
    return {d: adjustments.get(d, 0.0) for d in get_driver_team_map_2025()}

def get_weather_forecast():
    API_KEY = os.getenv("OWM_KEY", "")
    LAT, LON = 19.4042, -99.0911
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        forecasts = [f for f in data.get("list", []) if RACE_DATE in f["dt_txt"]]
        if not forecasts:
            return 0.1, 22
        rain_prob = max(f["pop"] for f in forecasts)
        temp = forecasts[len(forecasts) // 2]["main"]["temp"]
        return rain_prob, temp
    except Exception as e:
        print(f"[!] Weather API error: {e}")
        return 0.1, 22

def visualize_data(results_df, team_factors_df):
    plt.style.use('dark_background')

    # Chart 1: Win Chance Percentage
    top_10 = results_df.head(10).sort_values("Win%", ascending=True)
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, 10))
    bars1 = ax1.barh(top_10['Driver'], top_10['Win%'], color=colors)
    ax1.set_xlabel('Win Chance %', color='white')
    ax1.set_title('Predicted Win Chance for Top 10 Contenders', color='white', fontsize=16)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.1f}%', va='center', ha='left', color='white')
    fig1.tight_layout()

    # Chart 2: Team Technical Factors
    team_factors_df = team_factors_df.drop_duplicates(subset=['Team']).set_index('Team')
    team_factors_df['TireDegPenalty'] = team_factors_df['TireDeg'] - 1.0
    plot_data = team_factors_df[['TireDegPenalty', 'CoolingRating', 'PUReliability']]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    plot_data.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_ylabel('Performance Adjustment (Higher is Worse)', color='white')
    ax2.set_title('Team Technical Factors (Altitude & Tire Impact)', color='white', fontsize=16)
    ax2.tick_params(axis='x', colors='white', rotation=45)
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(title='Factors')
    fig2.tight_layout()
    plt.show()

def predict_mexico_city_gp():
    print(f"=== MEXICO CITY GP {RACE_YEAR} – PREDICTOR v2.5 ===")

    quali = get_qualifying_data_2025()
    quali["QualPos"] = quali.index + 1
    
    historical_pace = get_historical_pace()
    practice_fast_lap = get_practice_fast_lap_2025()
    long_run_pace = get_long_run_pace_2025()
    similar_track_pace = get_similar_track_pace_2025()
    
    df = quali
    if not historical_pace.empty:
        df = df.merge(historical_pace, on="Driver", how="left")
    if not practice_fast_lap.empty:
        df = df.merge(practice_fast_lap, on="Driver", how="left")
    if not long_run_pace.empty:
        df = df.merge(long_run_pace, on="Driver", how="left")
    if not similar_track_pace.empty:
        df = df.merge(similar_track_pace, on="Driver", how="left")

    # FIXED: Safe filling with similar pace data
    if 'SimilarPace (s)' in df.columns:
        for col in ['Pace (s)', 'PracticeFastLap (s)', 'LongRunPace (s)']:
            if col in df.columns:
                df[col].fillna(df['SimilarPace (s)'], inplace=True)
    
    # FIXED: Ensure all pace columns exist before filling with default
    for col in ['Pace (s)', 'PracticeFastLap (s)', 'LongRunPace (s)']:
        if col not in df.columns:
            df[col] = 999
        else:
            df[col].fillna(999, inplace=True)

    # FIXED: Safe normalization with proper checks
    normalization_map = {
        "PracticeFastLap (s)": "NormalizedPracticeFastLap",
        "LongRunPace (s)": "NormalizedLongRunPace",
        "Pace (s)": "NormalizedHistoricalPace"
    }
    
    for col, new_col in normalization_map.items():
        if col in df.columns:
            valid_data = df[col][df[col] != 999]
            if not valid_data.empty and len(valid_data) > 0:
                min_val, max_val = valid_data.min(), valid_data.max()
                if max_val > min_val:
                    df[new_col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[new_col] = 0
            else:
                df[new_col] = 0
        else:
            df[new_col] = 0

    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    driver_adjustments = get_driver_track_adjustments()
    df["DriverAdjust"] = df["Driver"].map(driver_adjustments)
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)
    
    df["CoolingRating"] = df["Team"].map(ALTITUDE_COOLING_RATING)
    team_to_pu = {
        team: pu for pu, teams in {
            "Ferrari": ["Ferrari", "Haas", "Sauber"],
            "Mercedes": ["Mercedes", "McLaren", "Aston Martin", "Williams"],
            "Red Bull": ["Red Bull", "RB"],
            "Alpine": ["Alpine"]
        }.items() for team in teams
    }
    df["PUReliability"] = df["Team"].map(team_to_pu).map(POWER_UNIT_RELIABILITY).fillna(0)

    rain_chance, temp = get_weather_forecast()
    team_factors_for_viz = df[['Team', 'TireDeg', 'CoolingRating', 'PUReliability']].copy()
    
    predictions = []
    for _, d in df.iterrows():
        base = ((d.QualPos + d.Penalty) * QUALI_WEIGHT + 
                d.NormalizedLongRunPace * 20 * LONG_RUN_PACE_WEIGHT +
                d.NormalizedPracticeFastLap * 20 * PRACTICE_FAST_LAP_WEIGHT + 
                d.NormalizedHistoricalPace * 20 * HISTORICAL_PACE_WEIGHT)
        
        score = base * (1 - d.TeamStrength * 0.20)
        score *= (1 + (d.TireDeg - 1.0) * 0.20)
        score += d.DriverAdjust
        score += d.CoolingRating + d.PUReliability

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
    
    # Safety Car Simulation
    if random.random() < SAFETY_CAR_PROBABILITY:
        print("\n[!] SAFETY CAR SIMULATION ACTIVATED: Compressing race gaps.")
        mean_score = results["InitialScore"].mean()
        results["FinalScore"] = mean_score + (results["InitialScore"] - mean_score) * 0.7
    else:
        results["FinalScore"] = results["InitialScore"]
        
    results["Predicted"] = np.clip(results["FinalScore"], 1, 20)
    results["Win%"] = round((21 - results["Predicted"]) / 20 * 100, 1)
    
    results.sort_values("Predicted", inplace=True)
    results.reset_index(drop=True, inplace=True)
    results["Rank"] = results.index + 1
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank", "Driver", "Team", "Qual", "Notes", "Predicted", "Win%"]].to_string(index=False))
    print("\n--- Model Insights ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance * 100, 1)}% | SC Probability: {int(SAFETY_CAR_PROBABILITY * 100)}%")
    print(f"Weights: Quali={QUALI_WEIGHT}, LongRun={LONG_RUN_PACE_WEIGHT}, FastLap={PRACTICE_FAST_LAP_WEIGHT}, History={HISTORICAL_PACE_WEIGHT}")
    
    print("\n--- Predicted Podium ---")
    for i, row in results.head(3).iterrows():
        print(f"P{i + 1}: {row['Driver']} ({row['Team']}) - Win% {row['Win%']}")
        
    visualize_data(results, team_factors_for_viz)

if __name__ == "__main__":
    predict_mexico_city_gp()