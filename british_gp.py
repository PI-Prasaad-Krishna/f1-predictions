# British GP 2025 – Hybrid Race‑Prediction Model (Silverstone v1.3)
# -----------------------------------------------------------------------------
#  2025‑07‑06 quali image integrated + smarter weather pull
#   • Weather: reads OpenWeatherMap key from $OWM_KEY (falls back to demo key)
#   • Picks *highest* rain probability on race‑day Sunday (any 3‑h slot)
#   • Uses mid‑day temp from the median Sunday slot
#   • QUALI_WEIGHT 0.45, pace denominator 4.8 (unchanged from v1.2)
# -----------------------------------------------------------------------------

import fastf1, pandas as pd, numpy as np, requests, os, warnings, sys
warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    import codecs; sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

if not os.path.exists("f1_cache"): os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")

QUALI_WEIGHT = 0.45
PACE_WEIGHT  = 1 - QUALI_WEIGHT
RACE_DATE    = "2025-07-06"  # British GP Sunday

# --------------------------------- Helpers ----------------------------------

def load_2024_sectors():
    try:
        s = fastf1.get_session(2024, 12, "R"); s.load()
        laps = s.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].dropna()
        for c in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
            laps[f"{c} (s)"] = laps[c].dt.total_seconds()
        d = laps.groupby("Driver").agg({
            "Sector1Time (s)":"mean","Sector2Time (s)":"mean","Sector3Time (s)":"mean"}).reset_index()
        d["TotalSectorTime (s)"] = d[["Sector1Time (s)","Sector2Time (s)","Sector3Time (s)"]].sum(axis=1)
        return d
    except Exception:
        drivers=list(get_driver_team_map()); base=86.0
        return pd.DataFrame({"Driver":drivers,"TotalSectorTime (s)":[base+i*0.12 for i,_ in enumerate(drivers)]})

# ----------------------------- Weekend inputs --------------------------------

def get_qualifying_data_2025():
    return pd.DataFrame({
        "Driver":["VER","PIA","NOR","RUS","HAM","LEC","ANT","BEA","ALO","GAS","SAI","TSU","HAD","ALB","OCO","LAW","BOR","STR","HUL","COL"],
        "QualifyingTime (s)":[84.892,84.995,85.010,85.029,85.095,85.121,85.374,85.471,85.621,85.785,85.746,85.826,85.864,85.889,85.950,86.640,86.446,86.504,86.574,87.060]
    })

def get_driver_team_map():
    return {
        "VER":"Red Bull","TSU":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","HAM":"Ferrari","RUS":"Mercedes","ANT":"Mercedes","ALO":"Aston Martin","STR":"Aston Martin","GAS":"Alpine","COL":"Alpine","OCO":"Haas","BEA":"Haas","LAW":"RB","HAD":"RB","ALB":"Williams","SAI":"Williams","BOR":"Sauber","HUL":"Sauber"}

def get_team_scores():
    pts={"Red Bull":710,"McLaren":860,"Ferrari":620,"Mercedes":520,"Aston Martin":110,"Alpine":105,"RB":102,"Williams":130,"Haas":85,"Sauber":35}
    m=max(pts.values()); return {k:v/m for k,v in pts.items()}

# -------------------------- Weather + adjustments ---------------------------

def get_weather():
    key=os.getenv("OWM_KEY","b16eee47fb847ac07fc76bf44805de5b")
    try:
        url=("https://api.openweathermap.org/data/2.5/forecast?lat=52.0786&lon=-1.0169"
             f"&appid={key}&units=metric")
        data=requests.get(url,timeout=10).json()
        # filter all Sunday slots
        slots=[x for x in data.get("list",[]) if RACE_DATE in x["dt_txt"]]
        if not slots:
            return 0.2,20
        rain=max(s["pop"] for s in slots)
        mid=slots[len(slots)//2]
        temp=mid["main"]["temp"]
        return rain,temp
    except Exception:
        return 0.2,20

def get_track_adjustments():
    base={"VER":-0.40,"PIA":-0.48,"NOR":-0.45,"RUS":-0.22,"HAM":-0.42,"LEC":-0.30,"ANT":0.48,"BEA":0.32,"ALO":-0.18,"GAS":0.05,"SAI":-0.08,"TSU":0.0,"HAD":0.38,"ALB":0.10,"OCO":0.28,"LAW":0.0,"BOR":0.48,"STR":0.20,"HUL":0.38,"COL":0.22}
    rain,_=get_weather()
    if rain>0.4:
        for d in ("HAM","VER","ALO"): base[d]-=0.1
    return base

# ---------------------------- MAE reference ---------------------------------

def calculate_mae(pred):
    ref={"VER":3.5,"NOR":2.5,"PIA":3.1,"HAM":4.0,"RUS":4.5}
    return round(np.mean([abs(pred.loc[pred.Driver==d,"Predicted"].iat[0]-p) for d,p in ref.items() if d in pred.Driver.values]),2)

# --------------------------- Predictor routine ------------------------------

def predict_british_gp():
    print("=== BRITISH GP 2025 – Predictor v1.3 ===")
    sectors=load_2024_sectors(); quali=get_qualifying_data_2025()
    df=quali.merge(sectors[["Driver","TotalSectorTime (s)"]],on="Driver",how="left")
    df["QualPos"]=df["QualifyingTime (s)"].rank().astype(int)
    qmin,qmax=df["QualifyingTime (s)"].min(),df["QualifyingTime (s)"].max()
    df["NormalizedQual"]=(df["QualifyingTime (s)"]-qmin)/(qmax-qmin)
    df["Team"]     =df.Driver.map(get_driver_team_map())
    df["TeamScore"]=df.Team.map(get_team_scores())
    df["Adjust"]   =df.Driver.map(get_track_adjustments())

    out=[]
    for _,r in df.iterrows():
        qcomp=r.QualPos*QUALI_WEIGHT+r.NormalizedQual*20*PACE_WEIGHT
        qpos =qcomp+r.Adjust
        pace_factor=(90-r["TotalSectorTime (s)"])/4.8
        base=qpos*(1-pace_factor*0.25)
        score=np.clip(base*(1-r.TeamScore*0.25),1,20)
        out.append({"Driver":r.Driver,"Team":r.Team,"Qual":r.QualPos,"Predicted":round(score,1),"Win%":round((21-score)/20*100,1)})

    final=pd.DataFrame(out).sort_values("Predicted").reset_index(drop=True); final["Rank"]=final.index+1
    rain,temp=get_weather()
    print(final[["Rank","Driver","Team","Qual","Predicted","Win%"]])
    print(f"\nWeather {temp}°C | Rain {round(rain*100,1)}%  | Quali/Pace {QUALI_WEIGHT}/{PACE_WEIGHT}")
    print("MAE est:",calculate_mae(final))
    print("\nPodium:")
    for i,row in final.head(3).iterrows():
        print(f"P{i+1}: {row.Driver} ({row.Team}) – Win% {row['Win%']}%")

if __name__=="__main__":
    predict_british_gp()
