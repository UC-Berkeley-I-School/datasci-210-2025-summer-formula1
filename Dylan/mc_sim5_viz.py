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
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

class MonteCarloF1Predictor:
    def __init__(self):
        self.vsc_model = None
        self.sc_model = None
        self.red_flag_model = None
        self.track_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.track_profiles = {}
        self.mc_results = {}
        
    def parse_lap_data(self, lap_string):
        """Parse lap data from various formats in the CSV"""
        if pd.isna(lap_string) or lap_string == '' or lap_string == '0':
            return []
        
        try:
            if '[' in str(lap_string) and ']' in str(lap_string):
                cleaned = re.sub(r'\s+', ' ', str(lap_string).strip())
                laps = ast.literal_eval(cleaned)
                if isinstance(laps, (list, np.ndarray)):
                    return [int(x) for x in laps if not np.isnan(float(x))]
                else:
                    return [int(laps)] if not np.isnan(float(laps)) else []
            else:
                if ',' in str(lap_string):
                    return [int(x.strip()) for x in str(lap_string).split(',') if x.strip().isdigit()]
                else:
                    return [int(lap_string)] if str(lap_string).isdigit() else []
        except:
            numbers = re.findall(r'\d+', str(lap_string))
            return [int(x) for x in numbers]
    
    def load_and_preprocess_data(self, csv_file: str) -> pd.DataFrame:
        """Load and preprocess the historical F1 incident data"""
        print("📊 Loading historical F1 incident data...")
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} race records from {df['Season'].min()}-{df['Season'].max()}")
        
        processed_data = []
        
        for _, row in df.iterrows():
            vsc_laps = self.parse_lap_data(row['VSCLaps'])
            sc_laps = self.parse_lap_data(row['SCLaps'])
            red_flag_occurred = len(self.parse_lap_data(row['RedFlag'])) > 0
            
            race_data = {
                'season': row['Season'],
                'track': row['Track'],
                'has_vsc': len(vsc_laps) > 0,
                'vsc_count': len(vsc_laps),
                'has_sc': len(sc_laps) > 0,
                'sc_count': len(sc_laps),
                'has_red_flag': red_flag_occurred,
                'total_incidents': len(vsc_laps) + len(sc_laps) + (1 if red_flag_occurred else 0),
                'first_vsc_lap': min(vsc_laps) if vsc_laps else 999,
                'first_sc_lap': min(sc_laps) if sc_laps else 999,
                'early_incidents': len([x for x in vsc_laps + sc_laps if x <= 10]),
                'late_incidents': len([x for x in vsc_laps + sc_laps if x >= 50])
            }
            processed_data.append(race_data)
        
        processed_df = pd.DataFrame(processed_data)
        processed_df = self.add_track_features(processed_df)
        
        print(f"✅ Processed data: {len(processed_df)} races")
        print(f"   VSC incidents: {processed_df['has_vsc'].sum()} races ({processed_df['has_vsc'].mean()*100:.1f}%)")
        print(f"   SC incidents: {processed_df['has_sc'].sum()} races ({processed_df['has_sc'].mean()*100:.1f}%)")
        print(f"   Red flags: {processed_df['has_red_flag'].sum()} races ({processed_df['has_red_flag'].mean()*100:.1f}%)")
        
        return processed_df
    
    def add_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track-specific features based on circuit characteristics"""
        
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
        
        df['track_type'] = df['track'].map(lambda x: track_types.get(x, {}).get('type', 'permanent'))
        df['track_length'] = df['track'].map(lambda x: track_types.get(x, {}).get('length', 5000))
        df['track_corners'] = df['track'].map(lambda x: track_types.get(x, {}).get('corners', 15))
        df['overtaking_difficulty'] = df['track'].map(lambda x: track_types.get(x, {}).get('overtaking', 'medium'))
        
        df['is_street_circuit'] = (df['track_type'] == 'street').astype(int)
        df['overtaking_score'] = df['overtaking_difficulty'].map({
            'very_easy': 1, 'easy': 2, 'medium': 3, 'hard': 4, 'very_hard': 5
        })
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare features for ML models"""
        
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
        
        self.vsc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vsc_model.fit(X_train, y_train['vsc'])
        vsc_accuracy = accuracy_score(y_val['vsc'], self.vsc_model.predict(X_val))
        
        self.sc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sc_model.fit(X_train, y_train['sc'])
        sc_accuracy = accuracy_score(y_val['sc'], self.sc_model.predict(X_val))
        
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
    
    def monte_carlo_simulation(self, track_name: str, n_simulations: int = 10000, 
                             weather_uncertainty: float = 0.1,
                             driver_uncertainty: float = 0.05,
                             track_condition_uncertainty: float = 0.08) -> Dict:
        """
        Run Monte Carlo simulation for race incident prediction
        
        Parameters:
        - track_name: Name of the track
        - n_simulations: Number of Monte Carlo iterations
        - weather_uncertainty: Uncertainty factor for weather conditions
        - driver_uncertainty: Uncertainty factor for driver performance
        - track_condition_uncertainty: Uncertainty factor for track conditions
        """
        
        print(f"🎲 Running Monte Carlo simulation for {track_name} ({n_simulations:,} iterations)...")
        
        # Base track features
        base_features = self.create_track_features(track_name)
        
        # Storage for simulation results
        vsc_probs = []
        sc_probs = []
        rf_probs = []
        any_incident_probs = []
        
        # Store feature variations for sensitivity analysis
        feature_variations = {
            'track_length': [],
            'track_corners': [],
            'is_street_circuit': [],
            'overtaking_score': [],
            'weather_factor': [],
            'driver_factor': [],
            'track_condition_factor': []
        }
        
        for i in range(n_simulations):
            # Add random variations to simulate uncertainties
            
            # Weather uncertainty affects overtaking difficulty
            weather_factor = np.random.normal(1.0, weather_uncertainty)
            
            # Driver uncertainty affects incident probability
            driver_factor = np.random.normal(1.0, driver_uncertainty)
            
            # Track condition uncertainty
            track_condition_factor = np.random.normal(1.0, track_condition_uncertainty)
            
            # Modify features based on uncertainty factors
            modified_features = base_features.copy()
            
            # Weather affects overtaking score
            modified_features[3] = np.clip(modified_features[3] * weather_factor, 1, 5)
            
            # Track conditions affect effective track length (grip, surface)
            modified_features[0] = modified_features[0] * track_condition_factor
            
            # Store variations
            feature_variations['track_length'].append(modified_features[0])
            feature_variations['track_corners'].append(modified_features[1])
            feature_variations['is_street_circuit'].append(modified_features[2])
            feature_variations['overtaking_score'].append(modified_features[3])
            feature_variations['weather_factor'].append(weather_factor)
            feature_variations['driver_factor'].append(driver_factor)
            feature_variations['track_condition_factor'].append(track_condition_factor)
            
            # Scale features
            X_sim = np.array([modified_features])
            X_sim_scaled = self.scaler.transform(X_sim)
            
            # Get base predictions
            vsc_prob = self.vsc_model.predict_proba(X_sim_scaled)[0][1]
            sc_prob = self.sc_model.predict_proba(X_sim_scaled)[0][1]
            rf_prob = self.red_flag_model.predict_proba(X_sim_scaled)[0][1]
            
            # Apply driver factor to incident probabilities
            vsc_prob = np.clip(vsc_prob * driver_factor, 0, 1)
            sc_prob = np.clip(sc_prob * driver_factor, 0, 1)
            rf_prob = np.clip(rf_prob * driver_factor, 0, 1)
            
            # Calculate any incident probability
            any_incident_prob = 1 - (1-vsc_prob) * (1-sc_prob) * (1-rf_prob)
            
            # Store results
            vsc_probs.append(vsc_prob)
            sc_probs.append(sc_prob)
            rf_probs.append(rf_prob)
            any_incident_probs.append(any_incident_prob)
        
        # Convert to numpy arrays
        results = {
            'track': track_name,
            'n_simulations': n_simulations,
            'vsc_probabilities': np.array(vsc_probs),
            'sc_probabilities': np.array(sc_probs),
            'red_flag_probabilities': np.array(rf_probs),
            'any_incident_probabilities': np.array(any_incident_probs),
            'feature_variations': feature_variations,
            'parameters': {
                'weather_uncertainty': weather_uncertainty,
                'driver_uncertainty': driver_uncertainty,
                'track_condition_uncertainty': track_condition_uncertainty
            }
        }
        
        # Calculate statistics
        results['statistics'] = self.calculate_mc_statistics(results)
        
        # Store results
        self.mc_results[track_name] = results
        
        print(f"✅ Monte Carlo simulation complete for {track_name}")
        
        return results
    
    def calculate_mc_statistics(self, results: Dict) -> Dict:
        """Calculate statistics from Monte Carlo results"""
        
        mc_stats = {}
        
        for prob_type in ['vsc_probabilities', 'sc_probabilities', 'red_flag_probabilities', 'any_incident_probabilities']:
            probs = results[prob_type]
            
            mc_stats[prob_type] = {
                'mean': np.mean(probs),
                'std': np.std(probs),
                'median': np.median(probs),
                'percentile_5': np.percentile(probs, 5),
                'percentile_25': np.percentile(probs, 25),
                'percentile_75': np.percentile(probs, 75),
                'percentile_95': np.percentile(probs, 95),
                'min': np.min(probs),
                'max': np.max(probs)
            }
        
        return mc_stats
    
    def create_comprehensive_visualizations(self, track_name: str, save_plots: bool = True):
        """Create comprehensive Monte Carlo visualizations"""
        
        if track_name not in self.mc_results:
            print(f"No Monte Carlo results found for {track_name}")
            return
        
        results = self.mc_results[track_name]
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
                
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribution Histograms (2x2 grid)
        incident_types = [
            ('vsc_probabilities', 'VSC Probability', 'lightblue'),
            ('sc_probabilities', 'Safety Car Probability', 'lightgreen'),
            ('red_flag_probabilities', 'Red Flag Probability', 'lightcoral'),
            ('any_incident_probabilities', 'Any Incident Probability', 'gold')
        ]
        
        for i, (prob_key, title, color) in enumerate(incident_types):
            ax = plt.subplot(4, 4, i+1)
            data = results[prob_key]
            
            # Histogram with density curve
            plt.hist(data, bins=50, alpha=0.7, density=True, color=color, edgecolor='black')
            
            # Add density curve
            x = np.linspace(data.min(), data.max(), 100)
            kde = scipy_stats.gaussian_kde(data)
            plt.plot(x, kde(x), 'r-', linewidth=2)
            
            # Add statistics
            mean_val = np.mean(data)
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            plt.axvline(np.percentile(data, 5), color='orange', linestyle=':', label='5th percentile')
            plt.axvline(np.percentile(data, 95), color='orange', linestyle=':', label='95th percentile')
            
            plt.title(f'{title}\n{track_name}', fontsize=10)
            plt.xlabel('Probability')
            plt.ylabel('Density')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 2. Cumulative Distribution Functions
        for i, (prob_key, title, color) in enumerate(incident_types):
            ax = plt.subplot(4, 4, i+5)
            data = results[prob_key]
            
            sorted_data = np.sort(data)
            p = np.arange(len(data)) / len(data)
            
            plt.plot(sorted_data, p, color=color, linewidth=2)
            plt.fill_between(sorted_data, p, alpha=0.3, color=color)
            
            # Add percentile lines
            plt.axvline(np.percentile(data, 50), color='red', linestyle='--', label='Median')
            plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
            
            plt.title(f'CDF - {title}', fontsize=10)
            plt.xlabel('Probability')
            plt.ylabel('Cumulative Probability')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 3. Box Plots Comparison
        ax = plt.subplot(4, 4, 9)
        box_data = [results[prob_key] for prob_key, _, _ in incident_types]
        box_labels = [title.replace(' Probability', '') for _, title, _ in incident_types]
        
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Probability Distributions Comparison', fontsize=10)
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Convergence Plot
        ax = plt.subplot(4, 4, 10)
        n_sims = len(results['vsc_probabilities'])
        
        for prob_key, title, color in incident_types:
            data = results[prob_key]
            running_mean = np.cumsum(data) / np.arange(1, len(data) + 1)
            plt.plot(running_mean, color=color, label=title.replace(' Probability', ''), alpha=0.8)
        
        plt.title('Convergence of Mean Estimates', fontsize=10)
        plt.xlabel('Simulation Number')
        plt.ylabel('Running Mean Probability')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 5. Sensitivity Analysis - Correlation Heatmap
        ax = plt.subplot(4, 4, 11)
        
        # Create correlation matrix
        sensitivity_data = pd.DataFrame({
            'VSC_Prob': results['vsc_probabilities'],
            'SC_Prob': results['sc_probabilities'],
            'RF_Prob': results['red_flag_probabilities'],
            'Weather': results['feature_variations']['weather_factor'],
            'Driver': results['feature_variations']['driver_factor'],
            'Track_Cond': results['feature_variations']['track_condition_factor']
        })
        
        corr_matrix = sensitivity_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        plt.title('Parameter Sensitivity Analysis', fontsize=10)
        
        # 6. Risk Curves
        ax = plt.subplot(4, 4, 12)
        
        for prob_key, title, color in incident_types:
            data = results[prob_key]
            
            # Calculate exceedance probability (1 - CDF)
            sorted_data = np.sort(data)
            exceedance_prob = 1 - np.arange(len(data)) / len(data)
            
            plt.plot(sorted_data, exceedance_prob, color=color, label=title.replace(' Probability', ''), linewidth=2)
        
        plt.yscale('log')
        plt.title('Risk Curves (Exceedance Probability)', fontsize=10)
        plt.xlabel('Incident Probability')
        plt.ylabel('Exceedance Probability')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 7. Feature Impact Tornado Plot
        ax = plt.subplot(4, 4, 13)
        
        # Calculate correlations between features and any incident probability
        feature_impacts = []
        feature_names = []
        
        for feature, values in results['feature_variations'].items():
            if feature in ['weather_factor', 'driver_factor', 'track_condition_factor']:
                corr = np.corrcoef(values, results['any_incident_probabilities'])[0, 1]
                feature_impacts.append(abs(corr))
                feature_names.append(feature.replace('_factor', '').replace('_', ' ').title())
        
        # Sort by impact
        sorted_indices = np.argsort(feature_impacts)
        feature_impacts = np.array(feature_impacts)[sorted_indices]
        feature_names = np.array(feature_names)[sorted_indices]
        
        colors_tornado = ['red' if x > 0 else 'blue' for x in feature_impacts]
        bars = plt.barh(range(len(feature_impacts)), feature_impacts, color=colors_tornado, alpha=0.7)
        
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Absolute Correlation with Incident Probability')
        plt.title('Parameter Impact Analysis', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 8. Confidence Intervals
        ax = plt.subplot(4, 4, 14)
        
        means = []
        ci_lower = []
        ci_upper = []
        labels = []
        
        for prob_key, title, color in incident_types:
            data = results[prob_key]
            mean_val = np.mean(data)
            ci_low = np.percentile(data, 2.5)
            ci_high = np.percentile(data, 97.5)
            
            means.append(mean_val)
            ci_lower.append(ci_low)
            ci_upper.append(ci_high)
            labels.append(title.replace(' Probability', ''))
        
        x_pos = np.arange(len(labels))
        plt.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                        np.array(ci_upper) - np.array(means)], 
                    fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
        
        plt.xticks(x_pos, labels, rotation=45)
        plt.ylabel('Probability')
        plt.title('95% Confidence Intervals', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 9. Monte Carlo Statistics Summary
        ax = plt.subplot(4, 4, 15)
        ax.axis('off')
        
        stats_text = f"Monte Carlo Summary - {track_name}\n"
        stats_text += f"Simulations: {results['n_simulations']:,}\n\n"
        
        for prob_key, title, _ in incident_types:
            result_stats = results['statistics'][prob_key]
            stats_text += f"{title}:\n"
            stats_text += f"  Mean: {result_stats['mean']:.3f} ± {result_stats['std']:.3f}\n"
            stats_text += f"  95% CI: [{result_stats['percentile_5']:.3f}, {result_stats['percentile_95']:.3f}]\n"
            stats_text += f"  Range: [{result_stats['min']:.3f}, {result_stats['max']:.3f}]\n\n"
        
        plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 10. Probability Distribution Comparison (Violin Plot)
        ax = plt.subplot(4, 4, 16)
        
        violin_data = [results[prob_key] for prob_key, _, _ in incident_types]
        violin_labels = [title.replace(' Probability', '') for _, title, _ in incident_types]
        
        parts = plt.violinplot(violin_data, positions=range(len(violin_labels)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        plt.xticks(range(len(violin_labels)), violin_labels, rotation=45)
        plt.ylabel('Probability')
        plt.title('Probability Distribution Shapes', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monte_carlo_analysis_{track_name.replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive analysis saved to {filename}")
        
        plt.show()
        
        return fig
    
    def create_track_features(self, track_name: str) -> List[float]:
        """Create feature vector for a track"""
        
        track_data = {
            'Monaco': [3337, 19, 1, 5],
            'Baku': [6003, 20, 1, 4],
            'Marina Bay': [5063, 23, 1, 4],
            'Silverstone': [5891, 18, 0, 2],
            'Spa-Francorchamps': [7004, 20, 0, 2],
            'Monza': [5793, 11, 0, 1],
        }
        
        features = track_data.get(track_name, [5000, 15, 0, 3])
        
        try:
            track_encoded = self.track_encoder.transform([track_name])[0]
        except:
            track_encoded = 0
        
        features.append(track_encoded)
        return features
    
    def run_monte_carlo_analysis(self, csv_file: str, tracks: List[str] = None, 
                                n_simulations: int = 10000):
        """Complete Monte Carlo analysis pipeline"""
        
        print("🎲 F1 MONTE CARLO INCIDENT ANALYSIS")
        print("="*60)
        
        # Load and train models
        df = self.load_and_preprocess_data(csv_file)
        
        df_2022 = df[df['season'] == 2022]
        df_2023 = df[df['season'] == 2023] 
        df_2024 = df[df['season'] == 2024]
        
        df_train = pd.concat([df_2022, df_2023])
        
        X_train, y_train = self.prepare_features(df_train)
        X_test, y_test = self.prepare_features(df_2024)
        
        self.train_models(X_train, y_train, X_test, y_test)
        
        # Default tracks if none specified
        if tracks is None:
            tracks = ['Monaco', 'Silverstone', 'Spa-Francorchamps', 'Monza']
        
        # Run Monte Carlo simulations for each track
        for track in tracks:
            print(f"\n🏁 Analyzing {track}...")
            
            # Run simulation
            mc_results = self.monte_carlo_simulation(track, n_simulations)
            
            # Create visualizations
            self.create_comprehensive_visualizations(track, save_plots=True)
            
            # Print summary
            self.print_monte_carlo_summary(track)
        
        print(f"\n🎉 Monte Carlo Analysis Complete!")
        print(f"📊 Generated comprehensive visualizations for {len(tracks)} tracks")
        
        return self.mc_results
    
    def print_monte_carlo_summary(self, track_name: str):
        """Print Monte Carlo simulation summary"""
        
        if track_name not in self.mc_results:
            return
        
        results = self.mc_results[track_name]
        result_stats = results['statistics']
        
        print(f"\n📊 MONTE CARLO SUMMARY - {track_name}")
        print("="*50)
        print(f"Simulations: {results['n_simulations']:,}")
        print(f"Track: {track_name}")
        print()
        
        incident_types = [
            ('vsc_probabilities', 'VSC'),
            ('sc_probabilities', 'Safety Car'),
            ('red_flag_probabilities', 'Red Flag'),
            ('any_incident_probabilities', 'Any Incident')
        ]
        
        for prob_key, title in incident_types:
            s = result_stats[prob_key]
            print(f"{title} Probability:")
            print(f"  Mean: {s['mean']:.3f} ± {s['std']:.3f}")
            print(f"  Median: {s['median']:.3f}")
            print(f"  95% CI: [{s['percentile_5']:.3f}, {s['percentile_95']:.3f}]")
            print(f"  Range: [{s['min']:.3f}, {s['max']:.3f}]")
            print()

# Example usage and quick functions
def quick_monte_carlo_analysis(track_name: str, csv_file: str = 'output.csv', 
                              n_simulations: int = 10000):
    """Quick Monte Carlo analysis for a single track"""
    
    predictor = MonteCarloF1Predictor()
    
    print(f"🎲 Running Monte Carlo analysis for {track_name}...")
    
    # Train models
    df = predictor.load_and_preprocess_data(csv_file)
    df_train = df[df['season'].isin([2022, 2023])]
    X_train, y_train = predictor.prepare_features(df_train)
    X_test, y_test = predictor.prepare_features(df[df['season'] == 2024])
    
    predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Run Monte Carlo simulation
    results = predictor.monte_carlo_simulation(track_name, n_simulations)
    
    # Create visualizations
    predictor.create_comprehensive_visualizations(track_name)
    
    # Print summary
    predictor.print_monte_carlo_summary(track_name)
    
    return results

if __name__ == "__main__":
    # Run comprehensive Monte Carlo analysis
    predictor = MonteCarloF1Predictor()
    
    # Analyze multiple tracks
    tracks_to_analyze = ['Monaco', 'Silverstone', 'Spa-Francorchamps', 'Monza']
    
    results = predictor.run_monte_carlo_analysis(
        csv_file='output.csv',
        tracks=tracks_to_analyze,
        n_simulations=10000
    )
    
    print("\n🏁 Analysis complete! Check the generated visualization files.")