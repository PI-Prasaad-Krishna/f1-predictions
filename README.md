[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# 🏁 F1 Race Winner Predictor 🏎️

A Python-based machine learning project that predicts the **winner of an upcoming Formula 1 race** based on historical performance, team stats, qualifying results, and track characteristics.

## 📌 Project Highlights

- 🔍 Analyzes driver performance, team consistency, qualifying positions, and track type
- 📊 Uses statistical modeling and machine learning to make winner predictions
- 🧠 Built with `scikit-learn`, `pandas`, and `matplotlib` for insights and visualization
- 🗂️ Modular code structure for easy updates and adding new seasons/tracks

## 🧠 How It Works

1. **Data Collection**: Gathers past F1 race data (driver standings, constructors, race results)
2. **Feature Engineering**: Extracts meaningful inputs (driver form, car reliability, track type, etc.)
3. **Model Training**: Trains classifiers to identify trends and predict winners
4. **Prediction**: User inputs current track and qualifying data → returns predicted winner(s)

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** – for data manipulation
- **NumPy** – for numerical operations
- **scikit-learn** – for machine learning models
- **Matplotlib & Seaborn** – for data visualization
- **FastF1** – for accessing F1 telemetry and session data
- **Requests** – for API calls and data fetching
- **Datetime** – for date and time operations
- **OS & Sys** – for environment handling and system-level tasks
- **Warnings** – to manage runtime warnings
