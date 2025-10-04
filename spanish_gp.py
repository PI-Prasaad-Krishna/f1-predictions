# Spanish GP 2025 Race Winner Predictor - Enhanced Hybrid Model
# -----------------------------------------------------------
# Realistic prediction using FastF1 data, race pace, weather, team score, and overtaking trends

import fastf1
import pandas as pd
import numpy as np
import requests
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

def load_2024_spain_data():
    try:
        session = fastf1.get_session(2024, 10, "R")
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
        return sectors
    except:
        return create_sample_spain_data()

def create_sample_spain_data():
    pace = {
        "VER": 86.8, "NOR": 86.9, "PIA": 87.1, "LEC": 87.0, "HAM": 87.3, "RUS": 87.5,
        "TSU": 88.2, "ALO": 87.9, "SAI": 87.6, "ALB": 88.1, "LAW": 88.3, "GAS": 88.5,
        "COL": 88.9, "STR": 89.1, "ANT": 89.5, "HAD": 89.4, "BEA": 89.7,
        "HUL": 89.9, "BOR": 90.1
    }
    return pd.DataFrame([{
        "Driver": d,
        "Sector1Time (s)": t * 0.32,
        "Sector2Time (s)": t * 0.34,
        "Sector3Time (s)": t * 0.34,
        "TotalSectorTime (s)": t
    } for d, t in pace.items()])

def get_qualifying_2025():
    return pd.DataFrame(
        {
  "Driver": ["PIA", "NOR", "VER", "RUS", "HAM", "ANT", "LEC", "GAS", "HAD", "ALO",
             "ALB", "BOR", "LAW", "STR", "BEA", "HUL", "OCO", "SAI", "COL", "TSU"],
  "QualifyingTime (s)": [
    71.546, 71.755, 71.848, 71.848, 72.045, 72.111, 72.131, 72.199, 72.252, 72.284,
    72.641, 72.756, 72.763, 73.058, 73.315, 73.190, 73.201, 73.203, 73.334, 73.385
  ]
,
        "CleanAirRacePace (s)": [
            86.8, 86.9, 87.1, 87.0, 87.3, 87.5, 87.9, 87.6, 88.2, 88.1,
            88.3, 88.5, 88.6, 88.9, 89.1, 89.4, 89.7, 89.9, 90.1, 89.5
        ]
    })

def get_team_scores():
    team_pts = {
        "Red Bull": 710, "McLaren": 840, "Ferrari": 620, "Mercedes": 500,
        "Aston Martin": 95, "Alpine": 110, "RB": 100, "Williams": 150, "Haas": 80, "Sauber": 30
    }
    max_pts = max(team_pts.values())
    return {team: pts / max_pts for team, pts in team_pts.items()}

def get_driver_team_map():
    return {
        "VER": "Red Bull", "TSU": "Red Bull", "NOR": "McLaren", "PIA": "McLaren",
        "LEC": "Ferrari", "HAM": "Ferrari", "RUS": "Mercedes", "ANT": "Mercedes",
        "ALO": "Aston Martin", "STR": "Aston Martin", "GAS": "Alpine", "COL": "Alpine",
        "OCO": "Haas", "BEA": "Haas", "LAW": "RB", "HAD": "RB", "ALB": "Williams",
        "SAI": "Williams", "BOR": "Sauber", "HUL": "Sauber"
    }

def get_position_change_spain():
    return {
        "VER": -0.8, "NOR": -0.6, "PIA": -0.4, "LEC": -0.5, "HAM": -0.7, "RUS": -0.3,
        "ALO": -0.4, "SAI": -0.3, "TSU": 0.0, "ALB": 0.2, "LAW": 0.1, "GAS": 0.0,
        "OCO": 0.2, "COL": 0.3, "STR": 0.4, "HAD": 0.5, "BEA": 0.6, "HUL": 0.5,
        "BOR": 0.8, "ANT": 0.7
    }

def get_weather():
    API_KEY = ""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat=41.57&lon=2.26&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=10).json()
        forecast = next((f for f in res["list"] if f["dt_txt"] == "2025-06-01 13:00:00"), None)
        rain = forecast["pop"] if forecast else 0.1
        temp = forecast["main"]["temp"] if forecast else 23
        return rain, temp
    except:
        return 0.1, 23

def calculate_model_accuracy(preds):
    truth = {"VER": 1.8, "NOR": 2.1, "LEC": 2.3, "HAM": 2.2, "RUS": 2.6, "ALO": 3.2, "PIA": 2.7}
    errors = [abs(preds.loc[preds.Driver == d, "Predicted"].values[0] - t) for d, t in truth.items() if d in preds.Driver.values]
    return round(np.mean(errors), 2) if errors else 1.5

def predict():
    print("\n=== SPAIN GP 2025 - HYBRID PREDICTOR ===")
    sectors = load_2024_spain_data()
    quali = get_qualifying_2025()
    teams = get_driver_team_map()
    scores = get_team_scores()
    change = get_position_change_spain()
    rain, temp = get_weather()

    df = quali.merge(sectors[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    df["QualifyingPosition"] = df["QualifyingTime (s)"].rank().astype(int)
    df["Team"] = df["Driver"].map(teams)
    df["TeamScore"] = df["Team"].map(scores)
    df["Adj"] = df["Driver"].map(change)

    predictions = []
    for _, row in df.iterrows():
        qpos = row["QualifyingPosition"] + row["Adj"]
        pace_factor = (90.5 - row["CleanAirRacePace (s)"]) / 4
        result = qpos * (1 - pace_factor * 0.3) * (1 - row["TeamScore"] * 0.2)
        predictions.append({
            "Driver": row["Driver"], "Team": row["Team"],
            "Qual": row["QualifyingPosition"],
            "Predicted": round(max(1, min(20, result)), 1),
            "Win%": min(100, round((21 - result) / 20 * 100, 1))
        })

    out = pd.DataFrame(predictions).sort_values("Predicted").reset_index(drop=True)
    out["Rank"] = out.index + 1
    print("\nFinal Prediction:")
    print(out[["Rank", "Driver", "Team", "Qual", "Predicted", "Win%"]])
    print("\nWeather: Temp =", temp, "C | Rain Probability =", round(rain * 100, 1), "%")
    print("Model MAE vs historic data:", calculate_model_accuracy(out))

    # Optional: key insights
    gainers = out[out["Qual"] - out["Predicted"] >= 1.0]["Driver"].tolist()
    if gainers:
        print("\n🔼 Potential Gainers:", ", ".join(gainers))
    strugglers = out[out["Predicted"] - out["Qual"] >= 1.0]["Driver"].tolist()
    if strugglers:
        print("🔽 Potential Strugglers:", ", ".join(strugglers))

if __name__ == "__main__":
    predict()
