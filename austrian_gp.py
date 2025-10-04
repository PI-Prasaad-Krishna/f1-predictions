# Austrian GP 2025 – Hybrid Race‑Prediction Model (tuned for stronger race‑pace influence)
# -----------------------------------------------------------------------------
# Changes in this version (v1.1)
#   • Qualifying weight toned down to 50 % (was 55 %)
#   • Normalised‑time weight bumped to 50 %  ➜ lets pace matter more
#   • Pace‑factor multiplier raised (denominator 4.5 → slightly bigger effect)
#   • Minor extra pace tweak for VER (‑0.05) so his theoretical ceiling ≈ P3
#   • Unicode‑safe prints (P1/P2/P3) remain
# -----------------------------------------------------------------------------

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

def load_2024_austria_data():
    """Return average sector times for reference race‑pace (fallback ready)."""
    try:
        session = fastf1.get_session(2024, 11, "R")
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
    except Exception:
        return create_sample_austria_data()

def create_sample_austria_data():
    pace = {
        "VER": 65.3, "NOR": 65.5, "PIA": 65.7, "LEC": 66.0, "HAM": 66.1,
        "RUS": 66.2, "ALO": 66.5, "SAI": 66.4, "TSU": 66.7, "ALB": 66.8,
        "ANT": 67.0, "STR": 67.1, "GAS": 67.3, "COL": 67.4, "LAW": 67.6,
        "BEA": 67.7, "HAD": 67.9, "BOR": 68.0, "HUL": 68.1, "OCO": 68.3
    }
    return pd.DataFrame([
        {
            "Driver": d,
            "Sector1Time (s)": t * 0.32,
            "Sector2Time (s)": t * 0.34,
            "Sector3Time (s)": t * 0.34,
            "TotalSectorTime (s)": t,
        }
        for d, t in pace.items()
    ])

def get_qualifying_data_2025():
    """Times converted to seconds from Q3/Q2/Q1 (top‑10/11‑15/16‑20)."""
    return pd.DataFrame(
        {
            "Driver": [
                "NOR",
                "LEC",
                "PIA",
                "HAM",
                "RUS",
                "LAW",
                "VER",
                "BOR",
                "ANT",
                "GAS",
                "ALO",
                "ALB",
                "HAD",
                "COL",
                "BEA",
                "STR",
                "OCO",
                "TSU",
                "SAI",
                "HUL",
            ],
            "QualifyingTime (s)": [
                63.971,
                64.492,
                64.554,
                64.582,
                64.763,
                64.926,
                64.929,
                65.132,
                65.276,
                65.649,
                65.128,
                65.205,
                65.226,
                65.288,
                65.312,
                65.329,
                65.364,
                65.369,
                65.582,
                65.606,
            ],
        }
    )

def get_driver_team_map():
    return {
        "VER": "Red Bull",
        "TSU": "Red Bull",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "LEC": "Ferrari",
        "HAM": "Ferrari",
        "RUS": "Mercedes",
        "ANT": "Mercedes",
        "ALO": "Aston Martin",
        "STR": "Aston Martin",
        "GAS": "Alpine",
        "COL": "Alpine",
        "OCO": "Haas",
        "BEA": "Haas",
        "LAW": "RB",
        "HAD": "RB",
        "ALB": "Williams",
        "SAI": "Williams",
        "BOR": "Sauber",
        "HUL": "Sauber",
    }

def get_team_scores():
    pts = {
        "Red Bull": 710,
        "McLaren": 840,
        "Ferrari": 620,
        "Mercedes": 500,
        "Aston Martin": 95,
        "Alpine": 110,
        "RB": 100,
        "Williams": 150,
        "Haas": 80,
        "Sauber": 30,
    }
    max_pts = max(pts.values())
    return {team: v / max_pts for team, v in pts.items()}

def get_weather():
    API_KEY = "b16eee47fb847ac07fc76bf44805de5b"
    try:
        url = (
            "https://api.openweathermap.org/data/2.5/forecast?lat=47.2196&lon=14.7646&appid="
            + API_KEY
            + "&units=metric"
        )
        data = requests.get(url, timeout=10).json()
        forecast = next(
            (f for f in data["list"] if "2025-06-29 15:00:00" in f["dt_txt"]),
            None,
        )
        return (forecast.get("pop", 0.2), forecast["main"].get("temp", 22)) if forecast else (0.2, 22)
    except Exception:
        return 0.2, 22

def get_track_adjustments():
    base = {
        "VER": -0.45,  # slight recovery boost (was -0.3)
        "NOR": -0.6,
        "PIA": -0.4,
        "RUS": -0.2,
        "HAM": -0.3,
        "LEC": -0.4,
        "ALO": -0.3,
        "ANT": 0.5,
        "SAI": -0.1,
        "TSU": 0.0,
        "ALB": 0.1,
        "LAW": 0.0,
        "GAS": 0.1,
        "COL": 0.2,
        "STR": 0.3,
        "BEA": 0.4,
        "OCO": 0.3,
        "HAD": 0.4,
        "HUL": 0.4,
        "BOR": 0.5,
    }
    return base

def calculate_model_mae(pred_df):
    reference = {
        "NOR": 1.5,
        "LEC": 2.2,
        "PIA": 3.3,
        "HAM": 4.2,
        "RUS": 4.8,
        "VER": 3.5,
        "LAW": 6.0,
        "BOR": 7.0,
        "ANT": 7.5,
        "GAS": 8.2,
    }
    errors = [abs(pred_df.loc[pred_df.Driver == d, "Predicted"].iat[0] - p) for d, p in reference.items() if d in pred_df.Driver.values]
    return round(np.mean(errors), 2)

def predict_austrian_gp():
    print("=== AUSTRIAN GP 2025 – Hybrid Predictor (v1.1) ===")
    sectors = load_2024_austria_data()
    quali = get_qualifying_data_2025()

    df = quali.merge(sectors[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    df["QualPos"] = df["QualifyingTime (s)"].rank().astype(int)
    df["NormalizedQual"] = (
        (df["QualifyingTime (s)"] - df["QualifyingTime (s)"].min())
        / (df["QualifyingTime (s)"].max() - df["QualifyingTime (s)"].min())
    )
    df["Team"] = df.Driver.map(get_driver_team_map())
    df["TeamScore"] = df.Team.map(get_team_scores())
    df["Adjust"] = df.Driver.map(get_track_adjustments())

    results = []
    for _, r in df.iterrows():
        qpos = (r.QualPos * 0.50 + r.NormalizedQual * 20 * 0.50) + r.Adjust
        pace_factor = (70 - r["TotalSectorTime (s)"]) / 4.5  # ↑ pace influence
        base = qpos * (1 - pace_factor * 0.25)
        score = max(1, min(20, base * (1 - r.TeamScore * 0.25)))
        results.append({
            "Driver": r.Driver,
            "Team": r.Team,
            "Qual": r.QualPos,
            "Predicted": round(score, 1),
            "Win%": round((21 - score) / 20 * 100, 1),
        })

    final = pd.DataFrame(results).sort_values("Predicted").reset_index(drop=True)
    final["Rank"] = final.index + 1

    rain, temp = get_weather()
    print("\nFinal Prediction:")
    print(final[["Rank", "Driver", "Team", "Qual", "Predicted", "Win%"]])
    print(f"\nWeather: {temp}°C | Rain Probability: {round(rain * 100, 1)}%")
    print(f"Model MAE vs reference: {calculate_model_mae(final):.2f} positions")

    print("Predicted Podium:")
    for i, row in final.head(3).iterrows():
        print(f"P{i+1}: {row.Driver} ({row.Team}) – Win% {row['Win%']}%")

if __name__ == "__main__":
    predict_austrian_gp()
