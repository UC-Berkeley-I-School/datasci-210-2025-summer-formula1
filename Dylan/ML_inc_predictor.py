#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
import ast
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class MLEnhancedIncidentPredictor:
    def __init__(self):
        self.vsc_model = None
        self.sc_model = None
        self.red_flag_model = None
        self.track_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.track_profiles = {}
        
    def parse_lap_data(self, lap_string):
        """Parse lap data from various formats in the CSV"""
        if pd.isna(lap_string) or lap_string == '' or lap_string == '0':
            return []
        
        # Handle string representation of arrays
        try:
            # Try to parse as Python literal (array format)
            if '[' in str(lap_string) and ']' in str(lap_string):
                # Remove extra spaces and parse as array
                cleaned = re.sub(r'\s+', ' ', str(lap_string).strip())
                laps = ast.literal_eval(cleaned)
                if isinstance(laps, (list, np.ndarray)):
                    return [int(x) for x in laps if not np.isnan(float(x))]
                else:
                    return [int(laps)] if not np.isnan(float(laps)) else []
            else:
                # Single number or comma-separated
                if ',' in str(lap_string):
                    return [int(x.strip()) for x in str(lap_string).split(',') if x.strip().isdigit()]
                else:
                    return [int(lap_string)] if str(lap_string).isdigit() else []
        except:
            # Fallback: try to extract numbers
            numbers = re.findall(r'\d+', str(lap_string))
            return [int(x) for x in numbers]
    
    def load_and_preprocess_data(self, csv_file: str) -> pd.DataFrame:
        """Load and preprocess the historical F1 incident data"""
        print("📊 Loading historical F1 incident data...")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} race records from {df['Season'].min()}-{df['Season'].max()}")
        
        # Process incident data
        processed_data = []
        
        for _, row in df.iterrows():
            # Parse incident data
            vsc_laps = self.parse_lap_data(row['VSCLaps'])
            sc_laps = self.parse_lap_data(row['SCLaps'])
            red_flag_occurred = len(self.parse_lap_data(row['RedFlag'])) > 0
            
            # Create features for this race
            race_data = {
                'season': row['Season'],
                'track': row['Track'],
                'has_vsc': len(vsc_laps) > 0,
                'vsc_count': len(vsc_laps),
                'vsc_laps_total': len(vsc_laps),
                'has_sc': len(sc_laps) > 0,
                'sc_count': len(sc_laps),
                'sc_laps_total': len(sc_laps),
                'has_red_flag': red_flag_occurred,
                'total_incidents': len(vsc_laps) + len(sc_laps) + (1 if red_flag_occurred else 0),
                'first_vsc_lap': min(vsc_laps) if vsc_laps else 999,
                'first_sc_lap': min(sc_laps) if sc_laps else 999,
                'early_incidents': len([x for x in vsc_laps + sc_laps if x <= 10]),
                'late_incidents': len([x for x in vsc_laps + sc_laps if x >= 50])
            }
            
            processed_data.append(race_data)
        
        processed_df = pd.DataFrame(processed_data)
        
        # Add track characteristics
        processed_df = self.add_track_features(processed_df)
        
        print(f"✅ Processed data: {len(processed_df)} races")
        print(f"   VSC incidents: {processed_df['has_vsc'].sum()} races ({processed_df['has_vsc'].mean()*100:.1f}%)")
        print(f"   SC incidents: {processed_df['has_sc'].sum()} races ({processed_df['has_sc'].mean()*100:.1f}%)")
        print(f"   Red flags: {processed_df['has_red_flag'].sum()} races ({processed_df['has_red_flag'].mean()*100:.1f}%)")
        
        return processed_df
    
    def add_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track-specific features based on circuit characteristics"""
        
        # Define track characteristics
        track_types = {
            'Monaco': {'type': 'street', 'length': 3337, 'corners': 19, 'overtaking': 'very_hard'},
            'Baku': {'type': 'street', 'length': 6003, 'corners': 20, 'overtaking': 'hard'}, 
            'Marina Bay': {'type': 'street', 'length': 5063, 'corners': 23, 'overtaking': 'hard'},
            'Jeddah': {'type': 'street', 'length': 6174, 'corners': 27, 'overtaking': 'medium'},
            'Las Vegas': {'type': 'street', 'length': 6201, 'corners': 17, 'overtaking': 'medium'},
            
            'Silverstone': {'type': 'permanent', 'length': 5891, 'corners': 18, 'overtaking': 'easy'},
            'Spa-Francorchamps': {'type': 'permanent', 'length': 7004, 'corners': 20, 'overtaking': 'easy'},
            'Monza': {'type': 'permanent', 'length': 5793, 'corners': 11, 'overtaking': 'very_easy'},
            'Interlagos': {'type': 'permanent', 'length': 4309, 'corners': 15, 'overtaking': 'medium'},
            'Suzuka': {'type': 'permanent', 'length': 5807, 'corners': 18, 'overtaking': 'medium'},
            
            'Barcelona': {'type': 'permanent', 'length': 4675, 'corners': 16, 'overtaking': 'hard'},
            'Budapest': {'type': 'permanent', 'length': 4381, 'corners': 14, 'overtaking': 'very_hard'},
            'Sakhir': {'type': 'permanent', 'length': 5412, 'corners': 15, 'overtaking': 'easy'},
            'Spielberg': {'type': 'permanent', 'length': 4318, 'corners': 10, 'overtaking': 'easy'},
            'Zandvoort': {'type': 'permanent', 'length': 4259, 'corners': 14, 'overtaking': 'hard'},
        }
        
        # Add features
        df['track_type'] = df['track'].map(lambda x: track_types.get(x, {}).get('type', 'permanent'))
        df['track_length'] = df['track'].map(lambda x: track_types.get(x, {}).get('length', 5000))
        df['track_corners'] = df['track'].map(lambda x: track_types.get(x, {}).get('corners', 15))
        df['overtaking_difficulty'] = df['track'].map(lambda x: track_types.get(x, {}).get('overtaking', 'medium'))
        
        # Encode categorical variables
        df['is_street_circuit'] = (df['track_type'] == 'street').astype(int)
        df['overtaking_score'] = df['overtaking_difficulty'].map({
            'very_easy': 1, 'easy': 2, 'medium': 3, 'hard': 4, 'very_hard': 5
        })
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare features for ML models"""
        
        # Select features for modeling
        feature_columns = [
            'track_length', 'track_corners', 'is_street_circuit', 'overtaking_score'
        ]
        
        track_encoded = self.track_encoder.fit_transform(df['track'])
        
        X = df[feature_columns].copy()
        X['track_encoded'] = track_encoded
        
        X_scaled = self.scaler.fit_transform(X)

        targets = {
            'vsc': df['has_vsc'].astype(int),
            'sc': df['has_sc'].astype(int), 
            'red_flag': df['has_red_flag'].astype(int),
            'vsc_count': df['vsc_count'],
            'sc_count': df['sc_count']
        }
        
        return X_scaled, targets
    
    def train_models(self, X_train: np.ndarray, y_train: Dict, X_val: np.ndarray, y_val: Dict):
        """Train ML models on historical data"""
        print("🤖 Training ML models on historical data...")
        
        # Train VSC prediction model
        self.vsc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vsc_model.fit(X_train, y_train['vsc'])
        vsc_accuracy = accuracy_score(y_val['vsc'], self.vsc_model.predict(X_val))
        
        # Train SC prediction model  
        self.sc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sc_model.fit(X_train, y_train['sc'])
        sc_accuracy = accuracy_score(y_val['sc'], self.sc_model.predict(X_val))
        
        # Train Red Flag prediction model
        self.red_flag_model = LogisticRegression(random_state=42)
        self.red_flag_model.fit(X_train, y_train['red_flag'])
        rf_accuracy = accuracy_score(y_val['red_flag'], self.red_flag_model.predict(X_val))
        
        print(f"✅ Model training complete:")
        print(f"   VSC Model Accuracy: {vsc_accuracy:.3f}")
        print(f"   SC Model Accuracy: {sc_accuracy:.3f}")
        print(f"   Red Flag Model Accuracy: {rf_accuracy:.3f}")
        
        return {
            'vsc_accuracy': vsc_accuracy,
            'sc_accuracy': sc_accuracy,
            'red_flag_accuracy': rf_accuracy
        }
    
    def predict_race_incidents(self, track_name: str) -> Dict:
        """Predict incidents for a specific race"""
        
        # Create feature vector for this race
        track_features = self.create_track_features(track_name)
        feature_cols = ['track_length','track_corners','is_street_circuit','overtaking_score','track_encoded']
        df_pred = pd.DataFrame([track_features], columns=feature_cols)
        X_pred_scaled = self.scaler.transform(df_pred)
        
        # Get predictions
        vsc_prob = self.vsc_model.predict_proba(X_pred_scaled)[0][1]
        sc_prob = self.sc_model.predict_proba(X_pred_scaled)[0][1]
        rf_prob = self.red_flag_model.predict_proba(X_pred_scaled)[0][1]
        
        return {
            'track': track_name,
            'vsc_probability': vsc_prob,
            'sc_probability': sc_prob,
            'red_flag_probability': rf_prob,
            'any_incident_probability': 1 - (1-vsc_prob) * (1-sc_prob) * (1-rf_prob)
        }
    
    def create_track_features(self, track_name: str) -> List[float]:
        """Create feature vector for a track"""
        
        # Track characteristics (same as in add_track_features)
        track_data = {
            'Monaco': [3337, 19, 1, 5],
            'Baku': [6003, 20, 1, 4],
            'Marina Bay': [5063, 23, 1, 4],
            'Silverstone': [5891, 18, 0, 2],
            'Spa-Francorchamps': [7004, 20, 0, 2],
            'Monza': [5793, 11, 0, 1],
        }
        
        features = track_data.get(track_name, [5000, 15, 0, 3])  # Default values
        
        # Add encoded track name
        try:
            track_encoded = self.track_encoder.transform([track_name])[0]
        except:
            track_encoded = 0  # Unknown track
        
        features.append(track_encoded)
        return features
    
    def validate_2024_predictions(self, df_2024: pd.DataFrame) -> Dict:
        """Validate model predictions against 2024 actual results"""
        print("🎯 Validating predictions against 2024 season...")
        
        predictions = []
        actuals = []
        
        for _, race in df_2024.iterrows():
            # Get prediction
            pred = self.predict_race_incidents(race['track'])
            
            # Get actual results
            actual = {
                'track': race['track'],
                'vsc_actual': race['has_vsc'],
                'sc_actual': race['has_sc'],
                'red_flag_actual': race['has_red_flag']
            }
            
            predictions.append(pred)
            actuals.append(actual)
        
        return self.compare_predictions_vs_actuals(predictions, actuals)
    
    def compare_predictions_vs_actuals(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Compare predictions vs actual results"""
        
        print("\n📊 PREDICTION vs ACTUAL COMPARISON")
        print("="*70)
        
        results = {
            'races': [],
            'overall_accuracy': {},
            'track_analysis': {}
        }
        
        vsc_correct = 0
        sc_correct = 0
        rf_correct = 0
        total_races = len(predictions)
        
        print(f"{'Track':<20} {'VSC Pred':<10} {'VSC Act':<10} {'SC Pred':<10} {'SC Act':<10} {'RF Pred':<10} {'RF Act':<10}")
        print("-"*70)
        
        for pred, actual in zip(predictions, actuals):
            track = actual['track']
            
            # Convert probabilities to binary predictions (>50% threshold)
            vsc_pred_binary = 1 if pred['vsc_probability'] > 0.5 else 0
            sc_pred_binary = 1 if pred['sc_probability'] > 0.5 else 0
            rf_pred_binary = 1 if pred['red_flag_probability'] > 0.5 else 0
            
            # Check accuracy
            vsc_match = vsc_pred_binary == actual['vsc_actual']
            sc_match = sc_pred_binary == actual['sc_actual']  
            rf_match = rf_pred_binary == actual['red_flag_actual']
            
            if vsc_match: vsc_correct += 1
            if sc_match: sc_correct += 1
            if rf_match: rf_correct += 1
            
            print(f"{track:<20} {pred['vsc_probability']:<10.3f} {actual['vsc_actual']:<10} "
                  f"{pred['sc_probability']:<10.3f} {actual['sc_actual']:<10} "
                  f"{pred['red_flag_probability']:<10.3f} {actual['red_flag_actual']:<10}")
            
            results['races'].append({
                'track': track,
                'predictions': pred,
                'actual': actual,
                'accuracy': {
                    'vsc': vsc_match,
                    'sc': sc_match,
                    'red_flag': rf_match
                }
            })
        
        results['overall_accuracy'] = {
            'vsc_accuracy': vsc_correct / total_races,
            'sc_accuracy': sc_correct / total_races,
            'red_flag_accuracy': rf_correct / total_races,
            'total_races': total_races
        }
        
        print(f"\n📈 OVERALL ACCURACY:")
        print(f"   VSC Predictions: {vsc_correct}/{total_races} ({vsc_correct/total_races*100:.1f}%)")
        print(f"   SC Predictions: {sc_correct}/{total_races} ({sc_correct/total_races*100:.1f}%)")
        print(f"   Red Flag Predictions: {rf_correct}/{total_races} ({rf_correct/total_races*100:.1f}%)")
        
        return results
    
    def run_full_training_and_validation(self, csv_file: str):
        """Complete training and validation pipeline"""
        
        print("🏁 F1 INCIDENT PREDICTION - ML TRAINING & VALIDATION")
        print("="*60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_file)
        
        # Split data by season
        df_2022 = df[df['season'] == 2022]
        df_2023 = df[df['season'] == 2023] 
        df_2024 = df[df['season'] == 2024]
        
        # Combine 2022-2023 for training
        df_train = pd.concat([df_2022, df_2023])
        
        print(f"\n📚 Training on {len(df_train)} races (2022-2023)")
        print(f"🎯 Testing on {len(df_2024)} races (2024)")
        
        # Prepare features
        X_train, y_train = self.prepare_features(df_train)
        X_test, y_test = self.prepare_features(df_2024)
        
        # Train models
        training_metrics = self.train_models(X_train, y_train, X_test, y_test)
        
        # Validate on 2024 season
        validation_results = self.validate_2024_predictions(df_2024)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'training_metrics': training_metrics,
            'validation_results': validation_results,
            'model_features': ['track_length', 'track_corners', 'is_street_circuit', 'overtaking_score', 'track_encoded'],
            'timestamp': timestamp
        }
        
        filename = f"f1_ml_incident_validation_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
        print("🎉 ML Training and Validation Complete!")
        
        return results

def quick_incident_prediction(track_name: str, csv_file: str = 'output.csv'):
    """Quick prediction for a specific track"""
    
    predictor = MLEnhancedIncidentPredictor()
    
    # Train on historical data
    print(f"🔮 Training ML model and predicting incidents for {track_name}...")
    predictor.run_full_training_and_validation(csv_file)
    
    # Get prediction
    prediction = predictor.predict_race_incidents(track_name)
    
    print(f"\n🎯 INCIDENT PREDICTION FOR {track_name}:")
    print(f"   VSC Probability: {prediction['vsc_probability']*100:.1f}%")
    print(f"   Safety Car Probability: {prediction['sc_probability']*100:.1f}%") 
    print(f"   Red Flag Probability: {prediction['red_flag_probability']*100:.1f}%")
    print(f"   Any Incident Probability: {prediction['any_incident_probability']*100:.1f}%")
    
    return prediction

if __name__ == "__main__":
    # Run the complete ML training and validation
    predictor = MLEnhancedIncidentPredictor()
    results = predictor.run_full_training_and_validation('output.csv')
    
    # Example predictions for upcoming races
    print("\n🏁 Example predictions for key tracks:")
    for track in ['Monaco', 'Silverstone', 'Spa-Francorchamps', 'Monza']:
        pred = predictor.predict_race_incidents(track)
        print(f"{track}: VSC {pred['vsc_probability']*100:.0f}%, SC {pred['sc_probability']*100:.0f}%, RF {pred['red_flag_probability']*100:.0f}%")
    
