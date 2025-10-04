# Spa GP 2025 – Hybrid Race‑Prediction Model (Spa v1.1)
# -----------------------------------------------------------------------------
#  Added tire degradation model per team
#  2025‑07‑27 qualifying + dynamic weather + pace/qual hybrid scoring
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys
warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import codecs; sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

if not os.path.exists("f1_cache"): os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

QUALI_WEIGHT = 0.45
PACE_WEIGHT  = 1 - QUALI_WEIGHT
RACE_DATE    = "2025-07-27"  # Spa GP Sunday

# ----------------------------- Tire degradation ------------------------------

TIRE_DEG_SENSITIVITY = {
    "Red Bull": 1.00, "McLaren": 0.85, "Ferrari": 1.05, "Mercedes": 1.02,
    "Aston Martin": 1.10, "Alpine": 1.08, "RB": 1.06, "Williams": 1.03,
    "Haas": 1.15, "Sauber": 1.12
}

# ----------------------------- Sector data -----------------------------------

def load_2024_sectors():
    try:
        s = fastf1.get_session(2024, 14, "R"); s.load()
        laps = s.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].dropna()
        for c in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
            laps[f"{c} (s)"] = laps[c].dt.total_seconds()
        d = laps.groupby("Driver").agg({
            "Sector1Time (s)":"mean","Sector2Time (s)":"mean","Sector3Time (s)":"mean"}).reset_index()
        d["TotalSectorTime (s)"] = d[["Sector1Time (s)","Sector2Time (s)","Sector3Time (s)"]].sum(axis=1)
        return d
    except Exception:
        drivers = list(get_driver_team_map()); base = 95.0
        return pd.DataFrame({"Driver":drivers, "TotalSectorTime (s)":[base + i*0.15 for i,_ in enumerate(drivers)]})

# ----------------------------- Weekend inputs --------------------------------

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver": [
            "NOR", "PIA", "LEC", "VER", "ALB", "RUS", "TSU", "HAD", "LAW", "BOR",
            "OCO", "BEA", "GAS", "HUL", "SAI", "HAM", "COL", "ANT", "ALO", "STR"
        ],
        "QualifyingTime (s)":[
            100.562, 100.647, 100.900, 100.903, 101.201, 101.260, 101.284, 101.310, 101.328, 102.387,
            101.525, 101.617, 101.633, 101.707, 101.758, 101.939, 102.022, 102.139, 102.385, 102.502
        ]
    })

def get_driver_team_map():
    return {
        "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber"}

def get_team_scores():
    pts={"Red Bull":710,"McLaren":860,"Ferrari":620,"Mercedes":520,"Aston Martin":110,"Alpine":105,"RB":102,"Williams":130,"Haas":85,"Sauber":35}
    m = max(pts.values()); return {k:v/m for k,v in pts.items()}

# -------------------------- Weather + adjustments ---------------------------

def get_weather():
    key=os.getenv("OWM_KEY",)
    try:
        url=("https://api.openweathermap.org/data/2.5/forecast?lat=50.4372&lon=5.9714"
             f"&appid={key}&units=metric")
        data = requests.get(url,timeout=10).json()
        slots = [x for x in data.get("list",[]) if RACE_DATE in x["dt_txt"]]
        if not slots: return 0.2, 20
        rain = max(s["pop"] for s in slots)
        mid = slots[len(slots)//2]
        temp = mid["main"]["temp"]
        return rain, temp
    except Exception:
        return 0.2, 20

def get_track_adjustments():
    base = {"VER":-0.40,"NOR":-0.47,"PIA":-0.45,"HAM":-0.43,"LEC":-0.31,"RUS":-0.22,"ANT":0.45,"ALO":-0.20,"BEA":0.32,"TSU":0.0,"STR":0.18,"GAS":0.05,"OCO":0.25,"ALB":0.08,"HAD":0.38,"LAW":0.0,"SAI":-0.10,"COL":0.22,"BOR":0.48,"HUL":0.40}
    rain,_ = get_weather()
    if rain > 0.4:
        for d in ("HAM","VER","ALO"): base[d] -= 0.1
    return base

# ---------------------------- MAE reference ---------------------------------

def calculate_mae(pred):
    ref={"VER":3.0,"NOR":2.7,"PIA":3.4,"HAM":4.0,"RUS":4.2}
    return round(np.mean([abs(pred.loc[pred.Driver==d,"Predicted"].iat[0]-p) for d,p in ref.items() if d in pred.Driver.values]),2)

# --------------------------- Predictor routine ------------------------------

def predict_spa_gp():
    print("=== BELGIAN GP 2025 – Predictor v1.1 ===")
    sectors = load_2024_sectors(); quali = get_qualifying_data_2025()
    df = quali.merge(sectors[["Driver","TotalSectorTime (s)"]],on="Driver",how="left")
    df["QualPos"] = df["QualifyingTime (s)"].rank().astype(int)
    qmin, qmax = df["QualifyingTime (s)"].min(), df["QualifyingTime (s)"].max()
    df["NormalizedQual"] = (df["QualifyingTime (s)"] - qmin)/(qmax - qmin)
    df["Team"] = df.Driver.map(get_driver_team_map())
    df["TeamScore"] = df.Team.map(get_team_scores())
    df["TireDeg"] = df.Team.map(TIRE_DEG_SENSITIVITY)
    df["Adjust"] = df.Driver.map(get_track_adjustments())

    out = []
    for _, r in df.iterrows():
        qcomp = r.QualPos * QUALI_WEIGHT + r.NormalizedQual * 20 * PACE_WEIGHT
        qpos = qcomp + r.Adjust
        pace_factor = (130 - r["TotalSectorTime (s)"]) / 6.0
        base = qpos * (1 - pace_factor * 0.25)
        score = base * (1 - r.TeamScore * 0.25)
        score *= (1 - (r.TireDeg - 1.0) * 0.15)
        score = np.clip(score, 1, 20)
        out.append({"Driver":r.Driver,"Team":r.Team,"Qual":r.QualPos,"Predicted":round(score,1),"Win%":round((21-score)/20*100,1)})

    final = pd.DataFrame(out).sort_values("Predicted").reset_index(drop=True); final["Rank"] = final.index + 1
    rain, temp = get_weather()
    print(final[["Rank","Driver","Team","Qual","Predicted","Win%"]])
    print(f"\nWeather {temp}°C | Rain {round(rain*100,1)}%  | Quali/Pace {QUALI_WEIGHT}/{PACE_WEIGHT}")
    print("MAE est:", calculate_mae(final))
    print("\nPodium:")
    for i, row in final.head(3).iterrows():
        print(f"P{i+1}: {row.Driver} ({row.Team}) – Win% {row['Win%']}%")

if __name__ == "__main__":
    predict_spa_gp()
