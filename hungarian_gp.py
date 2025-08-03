# Hungarian GP 2025 – Hybrid Race-Prediction Model (Hungaroring v1.2)
# -----------------------------------------------------------------------------
#  IMPROVEMENTS:
#  - Increased weight for Qualifying, reflecting Hungaroring's nature.
#  - Added current weekend Practice Pace as a key performance indicator.
#  - Introduced grid penalties for real-world accuracy.
#  - Refined driver adjustments for track-specific history (Driver Affinity).
#  - Tuned tire degradation model for Hungary's high-stress environment.
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")

# Set up console output for all platforms
if sys.platform.startswith("win"):
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

# Enable FastF1 cache to speed up data loading
if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

# --- Model Weights & Constants ---
RACE_YEAR = 2025
RACE_CIRCUIT = "Hungary"
RACE_DATE = "2025-08-03"  # Hungarian GP Sunday

# Hungaroring: Overtaking is difficult, so qualifying is critical.
QUALI_WEIGHT = 0.65
# Pace is a combination of current practice and historical performance.
PRACTICE_PACE_WEIGHT = 0.25
HISTORICAL_PACE_WEIGHT = 1 - QUALI_WEIGHT - PRACTICE_PACE_WEIGHT


# --- Data Models ---

# NEW: Grid penalties for component changes. Key real-world factor.
GRID_PENALTIES = {
    "VER": 0,   # Example: 5-place grid penalty for a new gearbox
    "GAS": 0,  # Example: 10-place penalty for new power unit elements
}

# REFINED: Tire degradation model tuned for Hungaroring's high temps and corners.
TIRE_DEGRADATION_MODEL = {
    "Red Bull": 1.02, "McLaren": 0.88, "Ferrari": 1.08, "Mercedes": 1.00,
    "Aston Martin": 1.12, "Alpine": 1.09, "RB": 1.07, "Williams": 1.04,
    "Haas": 1.16, "Sauber": 1.13
}

# --- Data Loading Functions ---

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver": [
            "LEC", "PIA", "NOR", "RUS", "ALO", "STR", "BOR", "VER", "LAW", "HAD", 
            "BEA", "HAM", "SAI", "COL", "ANT", "TSU", "GAS", "OCO", "HUL", "ALB"
        ], 
        "QualifyingTime (s)": [
            75.372, 75.398, 75.413, 75.425, 75.481, 75.498, 75.725, 75.728, 75.821, 75.915, 
            75.694, 75.702, 75.781, 76.159, 76.386, 75.899, 75.966, 76.023, 76.081, 76.223
        ] 
    })

def get_driver_team_map_2025():
    """ Map of drivers to their 2025 teams. """
    return {
        "VER":"Red Bull", "TSU":"Red Bull", "NOR":"McLaren", "PIA":"McLaren",
        "LEC":"Ferrari", "HAM":"Ferrari", "RUS":"Mercedes", "ANT":"Mercedes",
        "ALO":"Aston Martin", "STR":"Aston Martin", "GAS":"Alpine", "COL":"Alpine",
        "OCO":"Haas", "BEA":"Haas", "LAW":"RB", "HAD":"RB",
        "ALB":"Williams", "SAI":"Williams", "BOR":"Sauber", "HUL":"Sauber"
    }

def get_team_strength():
    """ Normalized team scores based on hypothetical 2025 constructor points. """
    pts = {"Red Bull": 750, "McLaren": 880, "Ferrari": 650, "Mercedes": 580,
           "Aston Martin": 120, "Alpine": 115, "RB": 112, "Williams": 140,
           "Haas": 95, "Sauber": 45}
    max_points = max(pts.values())
    return {team: points / max_points for team, points in pts.items()}

def get_historical_pace():
    """
    Loads historical race pace from the 2024 Hungarian GP.
    This serves as a baseline for a driver's performance at this track.
    """
    try:
        session = fastf1.get_session(2024, RACE_CIRCUIT, "R")
        session.load()
        laps = session.laps.pick_fastest()
        pace = laps[["Driver", "LapTime"]].dropna()
        pace["Pace (s)"] = pace["LapTime"].dt.total_seconds()
        return pace[["Driver", "Pace (s)"]]
    except Exception:
        # Fallback data if API fails
        drivers = list(get_driver_team_map_2025())
        return pd.DataFrame({
            "Driver": drivers,
            "Pace (s)": [88.0 + i * 0.2 for i, _ in enumerate(drivers)]
        })

# NEW: Function to get current weekend practice pace.
def get_practice_pace_2025():
    """
    Simulates fetching 2025 practice pace by using 2024 data.
    In a real scenario, this would use the current year's data.
    This shows how well a driver/car is dialed-in for the weekend.
    """
    try:
        # We combine all practice sessions and find the best lap for each driver.
        fp1 = fastf1.get_session(2024, RACE_CIRCUIT, "FP1"); fp1.load()
        fp2 = fastf1.get_session(2024, RACE_CIRCUIT, "FP2"); fp2.load()
        fp3 = fastf1.get_session(2024, RACE_CIRCUIT, "FP3"); fp3.load()
        
        laps = pd.concat([fp1.laps, fp2.laps, fp3.laps])
        best_laps = laps.pick_fastest()
        practice_pace = best_laps[["Driver", "LapTime"]].dropna()
        practice_pace["PracticePace (s)"] = practice_pace["LapTime"].dt.total_seconds()
        return practice_pace[["Driver", "PracticePace (s)"]]
    except Exception as e:
        print(f"Could not load practice data: {e}. Using fallback.")
        drivers = list(get_driver_team_map_2025())
        return pd.DataFrame({
            "Driver": drivers,
            "PracticePace (s)": [84.5 + i * 0.2 for i, _ in enumerate(drivers)]
        })


# --- Adjustments & Weather ---

# REFINED: Driver adjustments based on historical performance at the Hungaroring.
def get_driver_track_adjustments():
    """
    Applies a performance adjustment based on a driver's known affinity
    for the Hungaroring circuit. Positive is a penalty, negative is a bonus.
    """
    adjustments = {
        "HAM": -0.45,  # Historically exceptional at this track
        "VER": -0.35,  # Very strong record here
        "ALO": -0.25,  # Veteran with strong performances and a win
        "NOR": -0.20,  # Consistently performs well here
        "OCO": -0.15,  # Former winner
        "PIA": -0.10,
        "RUS": -0.05,
        "ANT": 0.40,   # Rookie penalty for a complex track
        "COL": 0.35,   # Rookie penalty
        "HAD": 0.30    # Rookie penalty
    }
    # Default adjustment for drivers not in the list
    return {driver: adjustments.get(driver, 0.0) for driver in get_driver_team_map_2025()}

def get_weather_forecast():
    """
    Fetches weather from OpenWeatherMap API for the Hungaroring.
    """
    API_KEY = os.getenv("OWM_KEY", "")
    LAT, LON = 47.5789, 19.2486  # Hungaroring coordinates
    try:
        url = (f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}"
               f"&appid={API_KEY}&units=metric")
        data = requests.get(url, timeout=10).json()
        
        race_day_forecasts = [f for f in data.get("list", []) if RACE_DATE in f["dt_txt"]]
        if not race_day_forecasts: return 0.1, 28  # Default: low rain, high temp

        rain_prob = max(slot["pop"] for slot in race_day_forecasts)
        # Get temperature from the forecast closest to 15:00 local time
        mid_day_slot = race_day_forecasts[len(race_day_forecasts) // 2]
        temperature = mid_day_slot["main"]["temp"]
        return rain_prob, temperature
    except Exception:
        return 0.1, 28 # Return default values on failure

# --- Prediction Routine ---

def predict_hungarian_gp():
    """ Main function to run the prediction model. """
    print(f"=== HUNGARIAN GP {RACE_YEAR} – PREDICTOR v1.2 ===")
    
    # 1. Load all data sources
    qualifying = get_qualifying_data_2025()
    historical_pace = get_historical_pace()
    practice_pace = get_practice_pace_2025()

    # 2. Combine data into a single DataFrame
    df = qualifying.merge(historical_pace, on="Driver", how="left")
    df = df.merge(practice_pace, on="Driver", how="left")

    # Fill missing pace data with a penalty (e.g., driver didn't set a time)
    df.fillna({
        "Pace (s)": df["Pace (s)"].max() + 2,
        "PracticePace (s)": df["PracticePace (s)"].max() + 2
    }, inplace=True)

    # 3. Normalize scores for fair comparison
    # Qualifying position is a direct rank
    df["QualPos"] = df["QualifyingTime (s)"].rank().astype(int)
    
    # For pace, a lower time is better. We normalize it 0-1.
    p_min, p_max = df["PracticePace (s)"].min(), df["PracticePace (s)"].max()
    df["NormalizedPracticePace"] = (df["PracticePace (s)"] - p_min) / (p_max - p_min)

    h_min, h_max = df["Pace (s)"].min(), df["Pace (s)"].max()
    df["NormalizedHistoricalPace"] = (df["Pace (s)"] - h_min) / (h_max - h_min)

    # 4. Map team, driver, and car-specific factors
    df["Team"] = df["Driver"].map(get_driver_team_map_2025())
    df["TeamStrength"] = df["Team"].map(get_team_strength())
    df["TireDeg"] = df["Team"].map(TIRE_DEGRADATION_MODEL)
    df["DriverAdjust"] = df["Driver"].map(get_driver_track_adjustments())
    df["Penalty"] = df["Driver"].map(GRID_PENALTIES).fillna(0)

    # 5. Core scoring algorithm
    predictions = []
    for _, driver in df.iterrows():
        # Combine weighted scores from quali, practice, and historical pace
        qualifying_score = driver.QualPos
        practice_score = driver.NormalizedPracticePace * 20 # Scale to grid size
        historical_score = driver.NormalizedHistoricalPace * 20

        # Calculate a base performance score based on weighted inputs
        base_score = (qualifying_score * QUALI_WEIGHT +
                      practice_score * PRACTICE_PACE_WEIGHT +
                      historical_score * HISTORICAL_PACE_WEIGHT)

        # Apply adjustments
        # Team Strength: Better teams get a reduction in score (better position)
        score = base_score * (1 - driver.TeamStrength * 0.20)
        
        # Tire Degradation: Higher degradation adds a penalty to the score
        score *= (1 + (driver.TireDeg - 1.0) * 0.25)
        
        # Driver/Track Affinity: Apply driver-specific bonus/penalty
        score += driver.DriverAdjust
        
        # Grid Penalty: Add the grid drop directly to the score
        score += driver.Penalty

        # Clip score to be within the bounds of race positions (1-20)
        final_score = np.clip(score, 1, 20)

        # Calculate a hypothetical "Win Percentage"
        win_chance = round((21 - final_score) / 20 * 100, 1)

        predictions.append({
            "Driver": driver.Driver,
            "Team": driver.Team,
            "Qual": driver.QualPos,
            "Predicted": round(final_score, 2),
            "Win%": win_chance,
            "Notes": f"+{int(driver.Penalty)}p" if driver.Penalty > 0 else ""
        })

    # 6. Finalize and display results
    results = pd.DataFrame(predictions).sort_values("Predicted").reset_index(drop=True)
    results["Rank"] = results.index + 1
    
    rain_chance, temp = get_weather_forecast()
    
    print("\n--- Predicted Race Results ---")
    print(results[["Rank", "Driver", "Team", "Qual", "Notes", "Predicted", "Win%"]].to_string(index=False))
    
    print("\n--- Model Insights & Conditions ---")
    print(f"Weather: {temp}°C | Rain Probability: {round(rain_chance*100, 1)}%")
    print(f"Model Weights: Quali={QUALI_WEIGHT}, Practice Pace={PRACTICE_PACE_WEIGHT}, Historical Pace={HISTORICAL_PACE_WEIGHT}")
    if GRID_PENALTIES:
        print(f"Grid Penalties Applied: {GRID_PENALTIES}")
        
    print("\n--- Predicted Podium ---")
    for i, row in results.head(3).iterrows():
        print(f"P{i+1}: {row['Driver']} ({row['Team']}) - (Est. Win Chance: {row['Win%']}%)")

# --- Run the predictor ---
if __name__ == "__main__":
    predict_hungarian_gp()