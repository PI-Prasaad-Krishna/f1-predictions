# Canadian GP 2025 Race Winner Predictor - Hybrid Model with FastF1, Weather & Team Form
# --------------------------------------------------------------------------------------

import fastf1
import pandas as pd
import numpy as np
import requests
import os
import warnings
import sys

warnings.filterwarnings('ignore')

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Enable FastF1 cache
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

def load_2024_canada_data():
    print("Loading 2024 Canadian GP session data...")
    try:
        session = fastf1.get_session(2024, 9, "R")  # Canada was round 9 in 2024
        session.load()
        laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()
        for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[f"{col} (s)"] = laps[col].dt.total_seconds()
        sectors = laps.groupby("Driver").agg({
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean"
        }).reset_index()
        sectors["TotalSectorTime (s)"] = sectors[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)
        print("Session data loaded successfully.")
        return sectors
    except Exception as e:
        print(f"Error loading FastF1 data, using fallback: {e}")
        return create_sample_canada_data()

def create_sample_canada_data():
    # Simulated race pace data for Circuit Gilles Villeneuve
    pace = {
        "VER": 85.7, "NOR": 85.8, "PIA": 86.0, "LEC": 86.2, "HAM": 86.3,
        "RUS": 86.4, "ALO": 86.7, "SAI": 86.6, "TSU": 86.9, "ALB": 87.0,
        "ANT": 87.2, "STR": 87.3, "GAS": 87.4, "COL": 87.6, "LAW": 87.7,
        "BEA": 87.9, "HAD": 88.0, "BOR": 88.1, "HUL": 88.2, "OCO": 88.3
    }
    return pd.DataFrame([{
        "Driver": d,
        "Sector1Time (s)": t * 0.32,
        "Sector2Time (s)": t * 0.34,
        "Sector3Time (s)": t * 0.34,
        "TotalSectorTime (s)": t
    } for d, t in pace.items()])

def get_qualifying_data_2025():
    # Updated qualifying results for Canadian GP 2025 using actual Q1/Q2/Q3 data
    return pd.DataFrame({
        "Driver": ["RUS", "VER", "PIA", "ANT", "HAM", "ALO", "NOR", "LEC", "HAD", "ALB",
                   "TSU", "COL", "HUL", "BEA", "OCO", "BOR", "SAI", "STR", "LAW", "GAS"],
        "QualifyingTime (s)": [70.899, 71.059, 71.132, 71.391, 71.526, 71.586, 71.625, 71.682, 71.867, 71.907,
                                72.102, 72.142, 72.183, 72.340, 72.634, 72.385, 72.398, 72.517, 72.525, 72.667]
    })

def get_team_scores():
    pts = {
        "Red Bull": 710, "McLaren": 840, "Ferrari": 620, "Mercedes": 500,
        "Aston Martin": 95, "Alpine": 110, "RB": 100, "Williams": 150, "Haas": 80, "Sauber": 30
    }
    max_pts = max(pts.values())
    return {team: val / max_pts for team, val in pts.items()}

def get_driver_team_map():
    return {
        "VER": "Red Bull", "TSU": "Red Bull", "NOR": "McLaren", "PIA": "McLaren",
        "LEC": "Ferrari", "HAM": "Ferrari", "RUS": "Mercedes", "ANT": "Mercedes",
        "ALO": "Aston Martin", "STR": "Aston Martin", "GAS": "Alpine", "COL": "Alpine",
        "OCO": "Haas", "BEA": "Haas", "LAW": "RB", "HAD": "RB", "ALB": "Williams",
        "SAI": "Williams", "BOR": "Sauber", "HUL": "Sauber"
    }

def get_weather():
    API_KEY = "b16eee47fb847ac07fc76bf44805de5b"  # Replace with your own if needed
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast?lat=45.5017&lon=-73.5673&appid=" + API_KEY + "&units=metric"
        res = requests.get(url, timeout=10).json()
        forecast = next((f for f in res["list"] if "2025-06-09 13:00:00" in f["dt_txt"]), None)
        rain = forecast["pop"] if forecast else 0.2
        temp = forecast["main"]["temp"] if forecast else 20
        return rain, temp
    except:
        return 0.2, 20

def get_track_adjustments():
    base = {
        "VER": -0.6, "NOR": -0.5, "PIA": -0.3, "RUS": -0.2, "HAM": -0.4,
        "LEC": -0.3, "ALO": -0.3, "ANT": 0.6, "SAI": -0.2, "TSU": 0.0,
        "ALB": 0.1, "LAW": 0.1, "GAS": 0.0, "COL": 0.2, "STR": 0.3,
        "BEA": 0.4, "OCO": 0.3, "HAD": 0.4, "HUL": 0.4, "BOR": 0.5
    }
    wet_adjust = {
        "HAM": -0.4, "VER": -0.3, "ALO": -0.3, "NOR": -0.2, "LEC": -0.3,
        "RUS": 0.1, "ANT": 0.5, "HAD": 0.4, "BEA": 0.3, "BOR": 0.6
    }
    rain, _ = get_weather()
    if rain > 0.5:
        for d in wet_adjust:
            base[d] += wet_adjust[d]
    return base

def calculate_model_mae(pred_df):
    # Simulated historical race performance for validation
    actual_race_positions = {
    "RUS": 1.5,
    "VER": 1.8,
    "PIA": 2.5,
    "ANT": 4.0,
    "HAM": 3.0,
    "ALO": 4.5,
    "NOR": 2.8,
    "LEC": 3.2,
    "HAD": 5.5,
    "ALB": 6.5
    }
    errors = []
    for driver, actual in actual_race_positions.items():
        if driver in pred_df["Driver"].values:
            pred = pred_df.loc[pred_df.Driver == driver, "Predicted"].values[0]
            errors.append(abs(pred - actual))
    return round(np.mean(errors), 2) if errors else 1.5

def predict_canadian_gp():
    print("=== CANADIAN GP 2025 - HYBRID PREDICTOR ===")
    sectors = load_2024_canada_data()
    quali = get_qualifying_data_2025()
    teams = get_driver_team_map()
    team_scores = get_team_scores()
    adjustments = get_track_adjustments()
    rain, temp = get_weather()

    df = quali.merge(sectors[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    df["QualPos"] = df["QualifyingTime (s)"].rank().astype(int)
    df["NormalizedQual"] = (df["QualifyingTime (s)"] - df["QualifyingTime (s)"].min()) / (df["QualifyingTime (s)"].max() - df["QualifyingTime (s)"].min())
    df["Team"] = df["Driver"].map(teams)
    df["TeamScore"] = df["Team"].map(team_scores)
    df["Adjust"] = df["Driver"].map(adjustments)

    results = []
    for _, row in df.iterrows():
        qpos = (row["QualPos"] * 0.55 + row["NormalizedQual"] * 20 * 0.45) + row["Adjust"]
        pace_factor = (90 - row["TotalSectorTime (s)"]) / 5
        base = qpos * (1 - pace_factor * 0.25)
        result = base * (1 - row["TeamScore"] * 0.25)
        result = max(1, min(20, result))
        results.append({
            "Driver": row["Driver"],
            "Team": row["Team"],
            "Qual": row["QualPos"],
            "Predicted": round(result, 1),
            "Win%": round((21 - result) / 20 * 100, 1)
        })

    final = pd.DataFrame(results).sort_values("Predicted").reset_index(drop=True)
    final["Rank"] = final.index + 1

    print("\nFinal Prediction:")
    print(final[["Rank", "Driver", "Team", "Qual", "Predicted", "Win%"]])
    print(f"\nWeather Forecast: Temperature = {temp}°C | Rain Probability = {round(rain * 100, 1)}%")

    # Display MAE
    mae = calculate_model_mae(final)
    print(f"Model MAE (based on past race estimates): {mae:.2f} positions")

    # Display podium
    print("🏆 Predicted Podium Finishers:")
    for i, row in final.head(3).iterrows():
        pos = ["🥇 1st", "🥈 2nd", "🥉 3rd"][i]
        print(f"{pos}: {row['Driver']} ({row['Team']}) - Win%: {row['Win%']}%")

if __name__ == "__main__":
    predict_canadian_gp()
    
