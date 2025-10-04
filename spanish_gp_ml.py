import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

def get_weather_data():
    """Get weather data for Barcelona (latitude: 41.5719, longitude: 2.2619)"""
    API_KEY = ""  # Replace with your OpenWeatherMap API key
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=41.5719&lon=2.2619&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(weather_url, timeout=10)
        if response.status_code == 200:
            weather_data = response.json()
            
            # Get forecast for race day (assuming race is at 15:00 CEST local time) 
            forecast_time = "2025-06-02 15:00:00"  # 15:00 CEST local time (Spain GP race day)
            forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
            
            if forecast_data:
                rain_probability = forecast_data.get("pop", 0)  # probability of precipitation
                temperature = forecast_data["main"]["temp"]
                
                print(f"Weather forecast for Spain GP:")
                print(f"Temperature: {temperature}°C")
                print(f"Rain probability: {rain_probability*100:.1f}%")
                
                return rain_probability, temperature
            else:
                print("Using default weather conditions")
                return 0.1, 22  # Default: 10% rain chance, 22°C
        else:
            print("Weather API request failed, using defaults")
            return 0.1, 22
            
    except Exception as e:
        print(f"Weather API error: {e}")
        return 0.1, 22

# Load the 2024 Spanish GP race session (Spain is GP #10 in 2024 calendar)
session_2024 = fastf1.get_session(2024, 10, "R")
session_2024.load()

# Extract laps and relevant data
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Clean air race pace (example values, you can adjust based on actual data)
clean_air_race_pace = {
    "VER": 92.5, "HAM": 93.2, "LEC": 92.8, "NOR": 92.7, "ALO": 93.5,
    "PIA": 92.6, "RUS": 93.0, "SAI": 93.3, "STR": 94.1, "HUL": 94.0,
    "OCO": 94.3
}

# Qualifying data for Spanish GP 2025 (dummy example data)
qualifying_2025 = pd.DataFrame({
  "Driver": ["PIA", "NOR", "VER", "RUS", "HAM", "ANT", "LEC", "GAS", "HAD", "ALO",
             "ALB", "BOR", "LAW", "STR", "BEA", "HUL", "OCO", "SAI", "COL", "TSU"],
  "QualifyingTime (s)": [
    71.546, 71.755, 71.848, 71.848, 72.045, 72.111, 72.131, 72.199, 72.252, 72.284,
    72.641, 72.756, 72.763, 73.058, 73.315, 73.190, 73.201, 73.203, 73.334, 73.385
  ]
}
)

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Fetch weather data for Spanish GP day
rain_probability, temperature = get_weather_data()

# Adjust qualifying times if rain probability is high (simulate slower qualifying lap times due to wet conditions)
if rain_probability >= 0.75:
    print("High rain probability detected — adjusting qualifying times by +10% to simulate wet conditions.")
    qualifying_2025["QualifyingTime (s)"] *= 1.10
else:
    print("Rain probability low — no qualifying time adjustment.")

# Add weather data to qualifying dataframe
qualifying_2025["RainProbability"] = rain_probability
qualifying_2025["Temperature"] = temperature

# Add constructor's data (example points, adjust as necessary)
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Average position change at Spain (dummy data, replace with real if available)
average_position_change_spain = {
    "VER": -0.8, "NOR": -0.6, "PIA": -0.4, "LEC": -0.5, "HAM": -0.7, "RUS": -0.3,
    "ALO": -0.4, "SAI": -0.3, "TSU": 0.0, "ALB": 0.2, "LAW": 0.1, "GAS": 0.0,
    "OCO": 0.2, "COL": 0.3, "STR": 0.4, "HAD": 0.5, "BEA": 0.6, "HUL": 0.5,
    "BOR": 0.8, "ANT": 0.7
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_spain)

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")

# Filter to only drivers with lap data
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)

merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Sort results to find predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")

print("\n Predicted 2025 Spanish GP Winner \n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Clean Air Race Pace on Predicted Race Results")
plt.tight_layout()
plt.show()

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

# Top 3 predicted podium
final_results = final_results.reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n*** Predicted Top 3 ***")
print(f"P1: {podium.iloc[0]['Driver']}")
print(f"P2: {podium.iloc[1]['Driver']}")
print(f"P3: {podium.iloc[2]['Driver']}")