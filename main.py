"""
F1 Race Position Predictor - 2025 Updated Version (FIXED)
---------------------------------------------------------
This script predicts the finishing positions for an upcoming F1 Grand Prix based on:
- Historical driver performance
- Current season form
- Track-specific performance
- Weather conditions (using OpenWeatherMap API)

Updated for 2025 with:
- Correct 2025 driver lineup
- Improved error handling for 2024 data issues
- Fallback mechanisms when API data is unavailable
- Fixed syntax errors and missing function implementations

Requirements:
- pip install fastf1 pandas numpy scikit-learn matplotlib requests
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import requests
import datetime
import os
import sys
from typing import Dict, List, Tuple, Any
import warnings

# Set console encoding to UTF-8 if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

class F1RacePredictor:
    def __init__(self, api_key="b16eee47fb847ac07fc76bf44805de5b", years_to_analyze=3):
        """
        Initialize the F1 race predictor.
        
        Args:
            api_key: OpenWeatherMap API key
            years_to_analyze: Number of past years to collect data from
        """
        self.api_key = api_key
        self.years_to_analyze = years_to_analyze
        self.current_year = datetime.datetime.now().year
        self.model = None
        self.drivers_df = None
        self.track_data = {}
        self.weather_conditions = ['dry', 'wet', 'mixed']
        
        # 2025 Driver-Constructor mapping for reference
        self.driver_2025_mapping = {
            'VER': 'Red Bull Racing',
            'TSU': 'Red Bull Racing',  # Tsunoda moved to Red Bull
            'LEC': 'Ferrari',
            'HAM': 'Ferrari',  # Hamilton moved to Ferrari
            'RUS': 'Mercedes',
            'ANT': 'Mercedes',  # Kimi Antonelli
            'NOR': 'McLaren',
            'PIA': 'McLaren',
            'ALO': 'Aston Martin',
            'STR': 'Aston Martin',
            'GAS': 'Alpine',
            'COL': 'Alpine',  # Colapinto moved to Alpine
            'SAI': 'Williams',  # Sainz moved to Williams
            'ALB': 'Williams',
            'OCO': 'Haas F1 Team',  # Ocon moved to Haas
            'BEA': 'Haas F1 Team',  # Oliver Bearman
            'HUL': 'Stake F1',  # Hulkenberg moved to Stake
            'BOR': 'Stake F1',  # Gabriel Bortoleto
            'HAD': 'RB',  # Isac Hadjar
            'LAW': 'RB',  # Liam Lawson
        }
        
        # Dictionary mapping of weather condition codes to our categories
        self.weather_mapping = {
            # Clear
            800: 'dry', 801: 'dry', 802: 'dry', 803: 'dry', 804: 'dry',
            
            # Drizzle or light rain - could be mixed conditions
            300: 'mixed', 301: 'mixed', 302: 'mixed', 310: 'mixed', 
            311: 'mixed', 312: 'mixed', 313: 'mixed', 314: 'mixed', 321: 'mixed',
            
            # Rain - wet conditions
            500: 'mixed', 501: 'wet', 502: 'wet', 503: 'wet', 504: 'wet',
            511: 'wet', 520: 'mixed', 521: 'wet', 522: 'wet', 531: 'wet',
            
            # Thunderstorm - definitely wet
            200: 'wet', 201: 'wet', 202: 'wet', 210: 'wet', 
            211: 'wet', 212: 'wet', 221: 'wet', 230: 'wet', 
            231: 'wet', 232: 'wet',
        }
        
        # Track coordinates for weather API
        self.track_coordinates = {
            'bahrain': {'lat': 26.0325, 'lon': 50.5106},
            'jeddah': {'lat': 21.6319, 'lon': 39.1044},
            'albert_park': {'lat': -37.8497, 'lon': 144.9681},
            'suzuka': {'lat': 34.8431, 'lon': 136.5414},
            'shanghai': {'lat': 31.3381, 'lon': 121.2208},
            'miami': {'lat': 25.9581, 'lon': -80.2389},
            'imola': {'lat': 44.3439, 'lon': 11.7167},
            'monaco': {'lat': 43.7347, 'lon': 7.4206},
            'catalunya': {'lat': 41.5638, 'lon': 2.2606},
            'montreal': {'lat': 45.5017, 'lon': -73.5673},
            'barcelona': {'lat': 41.5638, 'lon': 2.2606},
            'spielberg': {'lat': 47.2197, 'lon': 14.7647},
            'silverstone': {'lat': 52.0706, 'lon': -1.0174},
            'hungaroring': {'lat': 47.5830, 'lon': 19.2526},
            'spa': {'lat': 50.4372, 'lon': 5.9719},
            'zandvoort': {'lat': 52.3888, 'lon': 4.5474},
            'monza': {'lat': 45.6156, 'lon': 9.2811},
            'baku': {'lat': 40.3725, 'lon': 49.8533},
            'marina_bay': {'lat': 1.2914, 'lon': 103.8638},
            'americas': {'lat': 30.1328, 'lon': -97.6411},
            'rodriguez': {'lat': 19.4042, 'lon': -99.0907},
            'interlagos': {'lat': -23.7036, 'lon': -46.6997},
            'las_vegas': {'lat': 36.1699, 'lon': -115.1398},
            'losail': {'lat': 25.4719, 'lon': 51.4548},
            'yas_marina': {'lat': 24.4672, 'lon': 54.6031},
        }
        
        print("F1 Race Predictor initialized with 2025 driver lineup")

    def safe_print(self, text):
        """Safely print text, handling encoding issues"""
        try:
            print(text)
        except UnicodeEncodeError:
            safe_text = text.encode('ascii', 'ignore').decode('ascii')
            print(safe_text)

    def collect_historical_data(self) -> pd.DataFrame:
        """Collect historical race data with improved error handling for 2024 issues"""
        print("Collecting historical race data...")

        all_race_data = []
        # Focus on 2022-2023 data since 2024 has API issues
        start_year = max(2021, self.current_year - self.years_to_analyze)
        end_year = min(2024, self.current_year)  # Don't go beyond 2024 due to API issues
        
        for year in range(start_year, end_year):  # Fixed: start_Year -> start_year
            try:
                print(f"Processing {year} season...")
                
                # Load the season schedule with timeout
                try:
                    season = fastf1.get_event_schedule(year)
                except Exception as e:
                    print(f"Failed to load {year} season schedule: {e}")
                    continue
                
                successful_races = 0
                
                # Process each race in the season
                for idx, race in season.iterrows():
                    # Skip future races and problematic 2024 races
                    if race['EventDate'] > datetime.datetime.now():
                        continue
                        
                    race_name = race['EventName']
                    circuit_name = self._get_circuit_name(race)
                    
                    try:
                        print(f"Attempting to load {race_name} {year}...")
                        
                        # Load race session with additional error handling
                        race_session = fastf1.get_session(year, race_name, 'R')
                        
                        # Add timeout and retry logic
                        try:
                            race_session.load()
                        except Exception as load_error:
                            print(f"Failed to load session data for {race_name}: {load_error}")
                            continue
                        
                        # Check if results are available
                        if race_session.results is None or race_session.results.empty:
                            print(f"No results available for {race_name} {year}")
                            continue
                        
                        results = race_session.results
                        
                        # Determine weather condition
                        weather_condition = self._determine_weather_from_session(race_session)
                        
                        # Process each driver's result
                        for _, result in results.iterrows():
                            try:
                                driver_code = result.get('Abbreviation', 'UNK')
                                constructor = result.get('TeamName', 'Unknown Team')
                                
                                # Skip if we don't have basic data
                                if driver_code == 'UNK' or constructor == 'Unknown Team':
                                    continue
                                
                                # Handle grid position
                                grid_pos = result.get('GridPosition', result.get('StartingGridPosition', 20))
                                if pd.isna(grid_pos) or grid_pos == 0:
                                    grid_pos = 20
                                
                                # Handle finish position - be more careful with NaN values
                                finish_pos = result.get('Position')
                                if pd.isna(finish_pos) or finish_pos == 'NC' or finish_pos is None:
                                    # For DNF, assign position based on laps completed or last position
                                    finish_pos = len(results) + 1
                                else:
                                    try:
                                        finish_pos = int(finish_pos)
                                    except (ValueError, TypeError):
                                        finish_pos = len(results) + 1
                                
                                # Check for DNF status
                                status = result.get('Status', 'Finished')
                                dnf = 1 if status != 'Finished' and 'Lap' not in str(status) else 0
                                
                                race_data = {
                                    'Year': year,
                                    'Round': race.get('RoundNumber', idx + 1),
                                    'Circuit': circuit_name,
                                    'Driver': driver_code,
                                    'Constructor': constructor,
                                    'GridPosition': int(grid_pos),
                                    'FinishPosition': int(finish_pos),
                                    'Weather': weather_condition,
                                    'DNF': dnf,
                                    'Points': result.get('Points', 0) or 0
                                }
                                
                                all_race_data.append(race_data)
                                
                            except Exception as e:
                                print(f"Error processing driver result in {race_name}: {e}")
                                continue
                        
                        successful_races += 1
                        print(f"[OK] Processed {race_name} ({len(results)} drivers)")
                        
                    except Exception as e:
                        print(f"[ERROR] Skipping {race_name} {year}: {e}")
                        continue
                
                print(f"Successfully processed {successful_races} races from {year}")
                
            except Exception as e:
                print(f"Error processing {year} season: {e}")
                continue

        # If we have insufficient data, supplement with enhanced sample data
        if len(all_race_data) < 100:  # Minimum threshold for meaningful predictions
            print(f"Only collected {len(all_race_data)} race entries. Supplementing with enhanced sample data...")
            sample_data = self._create_enhanced_sample_data()
            all_race_data.extend(sample_data)
        
        # Convert to DataFrame
        self.drivers_df = pd.DataFrame(all_race_data)
        
        # Data cleaning
        self.drivers_df['FinishPosition'] = pd.to_numeric(self.drivers_df['FinishPosition'], errors='coerce')
        self.drivers_df = self.drivers_df.dropna(subset=['FinishPosition'])
        
        print(f"Successfully processed {len(self.drivers_df)} total race entries")
        return self.drivers_df

    def _get_circuit_name(self, race) -> str:
        """Extract circuit name from race data"""
        # Try different possible field names
        for field in ['CircuitShortName', 'Circuit.short_name', 'CircuitName']:
            if field in race and pd.notna(race[field]):
                return str(race[field]).lower().replace(' ', '_')
        
        # Fallback to event name processing
        event_name = str(race.get('EventName', 'unknown')).lower()
        event_name = event_name.replace('grand prix', '').replace('gp', '').strip()
        
        # Map common event names to circuit names
        name_mapping = {
            'monaco': 'monaco',
            'spanish': 'catalunya', 
            'british': 'silverstone',
            'italian': 'monza',
            'belgian': 'spa',
            'dutch': 'zandvoort',
            'hungarian': 'hungaroring',
            'austrian': 'spielberg',
            'canadian': 'montreal',
            'australian': 'albert_park',
            'japanese': 'suzuka',
            'singapore': 'marina_bay',
            'abu dhabi': 'yas_marina',
            'brazilian': 'interlagos',
            'mexican': 'rodriguez',
            'united states': 'americas',
            'las vegas': 'las_vegas',
            'qatar': 'losail',
            'saudi arabian': 'jeddah',
            'bahrain': 'bahrain',
            'chinese': 'shanghai',
            'miami': 'miami',
            'emilia romagna': 'imola',
            'azerbaijan': 'baku'
        }
        
        for key, value in name_mapping.items():
            if key in event_name:
                return value
                
        return event_name.replace(' ', '_')

    def _create_enhanced_sample_data(self) -> List[Dict]:
        """Create enhanced sample data based on realistic F1 performance patterns"""
        print("Creating enhanced sample data based on recent F1 patterns...")
        
        # Performance tiers based on recent seasons (2022-2024 patterns)
        top_tier = [
            ('VER', 'Red Bull Racing', 1.5),  # (Driver, Team, Avg Position)
            ('LEC', 'Ferrari', 4.2),
            ('PER', 'Red Bull Racing', 4.8),
            ('SAI', 'Ferrari', 5.1)
        ]
        
        mid_tier = [
            ('RUS', 'Mercedes', 6.2),
            ('HAM', 'Mercedes', 6.8),
            ('NOR', 'McLaren', 7.1),
            ('PIA', 'McLaren', 8.3),
            ('ALO', 'Aston Martin', 8.7),
            ('STR', 'Aston Martin', 11.2)
        ]
        
        lower_tier = [
            ('GAS', 'Alpine', 12.1),
            ('OCO', 'Alpine', 12.8),
            ('ALB', 'Williams', 13.5),
            ('TSU', 'AlphaTauri', 14.2),
            ('BOT', 'Alfa Romeo', 14.8),
            ('ZHO', 'Alfa Romeo', 15.3),
            ('HUL', 'Haas F1 Team', 15.7),
            ('MAG', 'Haas F1 Team', 16.2),
            ('RIC', 'AlphaTauri', 16.8),
            ('LAW', 'Williams', 17.5)
        ]
        
        all_drivers = top_tier + mid_tier + lower_tier
        circuits = ['monaco', 'silverstone', 'monza', 'spa', 'barcelona', 'hungaroring', 
                   'spielberg', 'zandvoort', 'singapore', 'suzuka']
        
        sample_data = []
        
        # Generate data for 2022-2023
        for year in [2022, 2023]:
            for round_num, circuit in enumerate(circuits, 1):
                for driver, constructor, avg_pos in all_drivers:
                    # Add realistic variance around average position
                    position_variance = np.random.normal(0, 2.5)  # Standard deviation of 2.5 positions
                    grid_variance = np.random.normal(0, 1.5)     # Less variance for qualifying
                    
                    finish_pos = max(1, min(20, avg_pos + position_variance))
                    grid_pos = max(1, min(20, avg_pos + grid_variance))
                    
                    # Circuit-specific adjustments
                    if circuit == 'monaco':
                        # Monaco rewards qualifying position more
                        finish_pos = 0.7 * grid_pos + 0.3 * finish_pos
                    elif circuit in ['monza', 'spa']:
                        # More overtaking possible
                        finish_pos = 0.4 * grid_pos + 0.6 * finish_pos
                    
                    finish_pos = max(1, min(20, round(finish_pos)))
                    
                    # DNF probability based on reliability patterns
                    dnf_prob = 0.15 if constructor in ['Haas F1 Team', 'Williams'] else 0.08
                    dnf = 1 if np.random.random() < dnf_prob else 0
                    
                    # Weather patterns
                    if circuit in ['silverstone', 'spa', 'hungaroring']:
                        weather_probs = [0.6, 0.3, 0.1]  # More chance of rain
                    else:
                        weather_probs = [0.8, 0.15, 0.05]  # Mostly dry
                    
                    weather = np.random.choice(['dry', 'wet', 'mixed'], p=weather_probs)
                    
                    sample_data.append({
                        'Year': year,
                        'Round': round_num,
                        'Circuit': circuit,
                        'Driver': driver,
                        'Constructor': constructor,
                        'GridPosition': int(grid_pos),
                        'FinishPosition': int(finish_pos),
                        'Weather': weather,
                        'DNF': dnf,
                        'Points': max(0, 26 - finish_pos) if finish_pos <= 10 else (1 if finish_pos == 11 else 0)
                    })
        
        print(f"Generated {len(sample_data)} enhanced sample data entries")
        return sample_data

    def _determine_weather_from_session(self, session) -> str:
        """Determine weather condition from FastF1 session data"""
        try:
            if hasattr(session, 'weather_data') and not session.weather_data.empty:
                if 'Rainfall' in session.weather_data.columns:
                    rainfall = session.weather_data['Rainfall'].max()
                    if rainfall > 0.5:
                        return 'wet'
                    elif rainfall > 0:
                        return 'mixed'
        except:
            pass
        
        # Default to dry if we can't determine
        return 'dry'

    def prepare_features(self):
        """Prepare features for the prediction model"""
        if self.drivers_df is None or self.drivers_df.empty:
            raise ValueError("No historical data available. Run collect_historical_data first.")
        
        print("Preparing features for model training...")
        
        # Create feature dataframe
        features_df = self.drivers_df.copy()
        
        # Calculate additional features
        features_df['YearsSinceRace'] = self.current_year - features_df['Year']
        
        # Calculate rolling averages for each driver
        print("Calculating driver form metrics...")
        features_df['Last3Avg'] = features_df['FinishPosition']  # Initialize
        
        for driver in features_df['Driver'].unique():
            driver_data = features_df[features_df['Driver'] == driver].sort_values(by=['Year', 'Round'])
            
            if len(driver_data) > 1:
                # Calculate rolling average of last 3 races
                rolling_avg = driver_data['FinishPosition'].rolling(window=3, min_periods=1).mean()
                
                # Map back to feature dataframe
                driver_indices = features_df[features_df['Driver'] == driver].sort_values(by=['Year', 'Round']).index
                
                for i, idx in enumerate(driver_indices):
                    if i > 0:  # Skip first race as there's no previous data
                        features_df.loc[idx, 'Last3Avg'] = rolling_avg.iloc[i-1]
        
        # Add weather encoding
        features_df['IsWet'] = (features_df['Weather'] == 'wet').astype(int)
        features_df['IsMixed'] = (features_df['Weather'] == 'mixed').astype(int)
        
        # Define features for ML
        categorical_features = ['Circuit', 'Driver', 'Constructor']
        numeric_features = ['GridPosition', 'YearsSinceRace', 'Last3Avg', 'DNF', 'IsWet', 'IsMixed']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create pipeline with preprocessing and model
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=15))
        ])
        
        # Prepare X and y
        X = features_df[categorical_features + numeric_features]
        y = features_df['FinishPosition']
        
        # Final check for NaN values
        if y.isna().any():
            print("Removing remaining NaN values from target variable...")
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
        
        # Train the model
        print("Training prediction model...")
        self.model.fit(X, y)
        print("Model training complete")
        
        return self.model

    def get_weather_forecast(self, circuit: str, race_date: str) -> Dict:
        """Get weather forecast for a circuit on race day"""
        circuit = circuit.lower()
        
        # Try to get actual weather data from API if available
        if circuit in self.track_coordinates and self.api_key:
            try:
                coords = self.track_coordinates[circuit]
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    weather_code = data['weather'][0]['id']
                    temp = data['main']['temp']
                    description = data['weather'][0]['description']
                    
                    condition = self.weather_mapping.get(weather_code, 'dry')
                    
                    return {
                        'condition': condition,
                        'temp': temp,
                        'description': f"Current: {description} ({temp}°C)"
                    }
            except Exception as e:
                print(f"Could not fetch weather data: {e}")
        
        # Monaco historically has mild weather in May
        if circuit == 'monaco':
            return {'condition': 'dry', 'temp': 22, 'description': 'Monaco in May - typically dry and mild'}
        
        # Default weather patterns for other circuits
        weather_defaults = {
            'silverstone': {'condition': 'mixed', 'temp': 18, 'description': 'British weather - mixed conditions likely'},
            'spa': {'condition': 'mixed', 'temp': 16, 'description': 'Belgian Ardennes - changeable weather'},
            'hungaroring': {'condition': 'dry', 'temp': 28, 'description': 'Hungarian summer - typically hot and dry'},
            'singapore': {'condition': 'mixed', 'temp': 30, 'description': 'Tropical climate - risk of rain'},
        }
        
        return weather_defaults.get(circuit, {'condition': 'dry', 'temp': 25, 'description': 'Default dry conditions'})

    def predict_race_results(self, next_race: Dict) -> pd.DataFrame:
        """Predict race results for the upcoming Grand Prix"""
        if self.model is None:
            raise ValueError("Model not trained. Run prepare_features first.")
        
        print(f"Predicting results for {next_race['circuit']} on {next_race['date']}...")
        
        # Get weather forecast
        weather = self.get_weather_forecast(next_race['circuit'], next_race['date'])
        print(f"Weather forecast: {weather['description']} ({weather['condition']})")
        
        # Create prediction dataframe
        prediction_data = []
        
        for driver_info in next_race['drivers']:
            # Calculate recent form based on historical data or defaults
            recent_form = 10.0  # Default middle-field position
            
            if self.drivers_df is not None:
                driver_data = self.drivers_df[self.drivers_df['Driver'] == driver_info['code']]
                if len(driver_data) > 0:
                    recent_form = driver_data['FinishPosition'].tail(5).mean()
                else:
                    # For new drivers, estimate based on team strength
                    team_data = self.drivers_df[self.drivers_df['Constructor'] == driver_info['constructor']]
                    if len(team_data) > 0:
                        recent_form = team_data['FinishPosition'].mean()
            
            prediction_entry = {
                'Circuit': next_race['circuit'],
                'Driver': driver_info['code'],
                'Constructor': driver_info['constructor'],
                'GridPosition': driver_info['qualifying'],
                'YearsSinceRace': 0,
                'Last3Avg': recent_form,
                'DNF': 0,
                'IsWet': 1 if weather['condition'] == 'wet' else 0,
                'IsMixed': 1 if weather['condition'] == 'mixed' else 0
            }
            
            prediction_data.append(prediction_entry)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame(prediction_data)
        
        # Make predictions
        predicted_positions = self.model.predict(pred_df)
        
        # Add predictions to dataframe
        pred_df['PredictedPosition'] = predicted_positions
        pred_df['PredictedPosition'] = pred_df['PredictedPosition'].round(1)
        
        # Sort by predicted position
        results = pred_df.sort_values('PredictedPosition').reset_index(drop=True)
        
        # Add rank column
        results['PredictedRank'] = range(1, len(results) + 1)
        
        # Select columns for output
        final_results = results[['PredictedRank', 'Driver', 'Constructor', 
                               'GridPosition', 'PredictedPosition']].copy()
        
        return final_results

    def plot_predictions(self, predictions: pd.DataFrame, race_name: str):
        """Create visualization of race predictions"""
        plt.figure(figsize=(12, 8))
        
        # Create the plot
        bars = plt.barh(predictions['Driver'], predictions['PredictedPosition'], 
                       color=['#1f77b4' if i <= 3 else '#ff7f0e' if i <= 10 else '#d62728' 
                              for i in predictions['PredictedRank']])
        
        # Customize the plot
        plt.xlabel('Predicted Finish Position')
        plt.ylabel('Driver')
        plt.title(f'{race_name} - Race Position Predictions')
        plt.gca().invert_yaxis()  # Invert y-axis so position 1 is at top
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, pos) in enumerate(zip(bars, predictions['PredictedPosition'])):
            plt.text(pos + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{pos:.1f}', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Podium (1-3)'),
            Patch(facecolor='#ff7f0e', label='Points (4-10)'),
            Patch(facecolor='#d62728', label='No Points (11+)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()

    def analyze_model_performance(self):
        """Analyze and display model performance metrics"""
        if self.model is None or self.drivers_df is None:
            print("Model not trained yet. Cannot analyze performance.")
            return
        
        print("\n=== MODEL PERFORMANCE ANALYSIS ===")
        
        # Prepare features for evaluation
        features_df = self.drivers_df.copy()
        features_df['YearsSinceRace'] = self.current_year - features_df['Year']
        features_df['Last3Avg'] = features_df['FinishPosition']
        features_df['IsWet'] = (features_df['Weather'] == 'wet').astype(int)
        features_df['IsMixed'] = (features_df['Weather'] == 'mixed').astype(int)
        categorical_features = ['Circuit', 'Driver', 'Constructor']
        numeric_features = ['GridPosition', 'YearsSinceRace', 'Last3Avg', 'DNF', 'IsWet', 'IsMixed']
        
        X = features_df[categorical_features + numeric_features]
        y = features_df['FinishPosition']
        
        # Remove NaN values
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Make predictions on training data
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"Mean Absolute Error: {mae:.2f} positions")
        print(f"Root Mean Square Error: {rmse:.2f} positions")
        print(f"R² Score: {r2:.3f}")
        
        # Calculate accuracy within different position ranges
        position_diff = np.abs(y - y_pred)
        within_1 = np.mean(position_diff <= 1) * 100
        within_2 = np.mean(position_diff <= 2) * 100
        within_3 = np.mean(position_diff <= 3) * 100
        
        print(f"\nPrediction Accuracy:")
        print(f"Within 1 position: {within_1:.1f}%")
        print(f"Within 2 positions: {within_2:.1f}%")
        print(f"Within 3 positions: {within_3:.1f}%")
        
        # Feature importance
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            print(f"\nTotal features in model: {len(self.model.named_steps['regressor'].feature_importances_)}")
            print("Top feature categories by importance:")
            
            # Get feature names after preprocessing
            try:
                feature_names = (self.model.named_steps['preprocessor']
                               .named_transformers_['num'].get_feature_names_out(numeric_features).tolist() +
                               self.model.named_steps['preprocessor']
                               .named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
                
                importances = self.model.named_steps['regressor'].feature_importances_
                
                # Group by feature category
                importance_dict = {}
                for name, importance in zip(feature_names, importances):
                    category = name.split('_')[0] if '_' in name else name
                    if category not in importance_dict:
                        importance_dict[category] = 0
                    importance_dict[category] += importance
                
                # Sort and display
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                for category, importance in sorted_importance[:8]:
                    print(f"  {category}: {importance:.3f}")
                    
            except Exception as e:
                print(f"Could not extract detailed feature importance: {e}")

def main():
    """Main function to run the F1 race predictor"""
    print("=== F1 RACE POSITION PREDICTOR - 2025 ===\n")
    
    try:
        # Initialize predictor
        predictor = F1RacePredictor()
        
        # Collect historical data
        historical_data = predictor.collect_historical_data()
        
        if historical_data.empty:
            print("Failed to collect sufficient historical data. Exiting.")
            return
        
        # Train model
        model = predictor.prepare_features()
        
        # Analyze model performance
        predictor.analyze_model_performance()
        
        # Example race prediction for Monaco 2025
        print("\n=== EXAMPLE PREDICTION: MONACO GRAND PRIX 2025 ===")
        
        # Define next race with 2025 drivers
        monaco_2025 = {
            'circuit': 'monaco',
            'date': '2025-05-25',
            'drivers': [
                {'code': 'VER', 'constructor': 'Red Bull Racing', 'qualifying': 1},
                {'code': 'TSU', 'constructor': 'Red Bull Racing', 'qualifying': 4},
                {'code': 'LEC', 'constructor': 'Ferrari', 'qualifying': 2},
                {'code': 'HAM', 'constructor': 'Ferrari', 'qualifying': 6},
                {'code': 'RUS', 'constructor': 'Mercedes', 'qualifying': 3},
                {'code': 'ANT', 'constructor': 'Mercedes', 'qualifying': 8},
                {'code': 'NOR', 'constructor': 'McLaren', 'qualifying': 5},
                {'code': 'PIA', 'constructor': 'McLaren', 'qualifying': 7},
                {'code': 'ALO', 'constructor': 'Aston Martin', 'qualifying': 9},
                {'code': 'STR', 'constructor': 'Aston Martin', 'qualifying': 12},
                {'code': 'GAS', 'constructor': 'Alpine', 'qualifying': 10},
                {'code': 'COL', 'constructor': 'Alpine', 'qualifying': 14},
                {'code': 'SAI', 'constructor': 'Williams', 'qualifying': 11},
                {'code': 'ALB', 'constructor': 'Williams', 'qualifying': 15},
                {'code': 'OCO', 'constructor': 'Haas F1 Team', 'qualifying': 13},
                {'code': 'BEA', 'constructor': 'Haas F1 Team', 'qualifying': 17},
                {'code': 'HUL', 'constructor': 'Stake F1', 'qualifying': 16},
                {'code': 'BOR', 'constructor': 'Stake F1', 'qualifying': 19},
                {'code': 'HAD', 'constructor': 'RB', 'qualifying': 18},
                {'code': 'LAW', 'constructor': 'RB', 'qualifying': 20}
            ]
        }
        
        # Make predictions
        predictions = predictor.predict_race_results(monaco_2025)
        
        # Display results
        print("\nPREDICTED RACE RESULTS:")
        print("=" * 70)
        print(f"{'Pos':<4} {'Driver':<6} {'Team':<20} {'Grid':<6} {'Pred':<6}")
        print("=" * 70)
        
        for _, row in predictions.iterrows():
            print(f"{row['PredictedRank']:<4} {row['Driver']:<6} {row['Constructor']:<20} "
                  f"{row['GridPosition']:<6} {row['PredictedPosition']:<6.1f}")
        
        # Create visualization
        try:
            predictor.plot_predictions(predictions, "Monaco Grand Prix 2025")
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        print("\n=== PREDICTION INSIGHTS ===")
        
        # Calculate some insights
        biggest_gainer = predictions.loc[predictions['GridPosition'] - predictions['PredictedRank'] > 0]
        biggest_loser = predictions.loc[predictions['PredictedRank'] - predictions['GridPosition'] > 0]
        
        if not biggest_gainer.empty:
            best_gain = biggest_gainer.loc[biggest_gainer['GridPosition'] - biggest_gainer['PredictedRank'].idxmax()]
            gain = best_gain['GridPosition'] - best_gain['PredictedRank']
            print(f"Biggest predicted gainer: {best_gain['Driver']} (+{gain} positions)")
        
        if not biggest_loser.empty:
            worst_loss = biggest_loser.loc[biggest_loser['PredictedRank'] - biggest_loser['GridPosition'].idxmax()]  
            loss = worst_loss['PredictedRank'] - worst_loss['GridPosition']
            print(f"Biggest predicted loser: {worst_loss['Driver']} (-{loss} positions)")
        
        # Podium prediction
        podium = predictions.head(3)
        print(f"\nPredicted Podium:")
        print(f"🥇 {podium.iloc[0]['Driver']} ({podium.iloc[0]['Constructor']})")
        print(f"🥈 {podium.iloc[1]['Driver']} ({podium.iloc[1]['Constructor']})")
        print(f"🥉 {podium.iloc[2]['Driver']} ({podium.iloc[2]['Constructor']})")
        
        print(f"\nNote: Predictions based on {len(historical_data)} historical race entries")
        print("Weather and track conditions can significantly impact actual results!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()