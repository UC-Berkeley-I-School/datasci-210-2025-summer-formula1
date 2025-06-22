"""
Temporal Safety Car Prediction System for Formula 1

This system uses temporal sliding windows to predict safety car deployments
based on individual driver telemetry data. The approach simulates real-time
prediction by sliding forward in time increments and asking "will a safety
car be deployed in the next prediction window?"

Author: Your Name  
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SafetyCarEvent:
    """Represents a safety car deployment with precise timing."""
    deployment_time: pd.Timestamp
    lap_context: Optional[int] = None  # For reference, but not used in prediction

@dataclass
class TemporalWindow:
    """Represents a temporal prediction window for one driver."""
    driver: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    prediction_target_time: pd.Timestamp  # When we're predicting SC deployment
    prediction_window_end: pd.Timestamp   # End of prediction window
    laps_data: pd.DataFrame               # Driver's laps in this window
    telemetry_data: Optional[Dict] = None # Additional telemetry if available

@dataclass
class TemporalPrediction:
    """Represents a temporal safety car prediction."""
    driver: str
    prediction_time: pd.Timestamp
    target_time: pd.Timestamp
    probability: float
    prediction: bool
    features_used: List[str]
    actual_outcome: Optional[bool] = None
    is_correct: Optional[bool] = None
    prediction_lead_time: Optional[pd.Timedelta] = None  # How far ahead we predicted

# =============================================================================
# Core Temporal Processing
# =============================================================================

class TemporalDataProcessor:
    """
    Processes F1 session data into temporal windows for individual drivers.
    
    This processor creates sliding time windows that simulate real-time prediction,
    where at each time step we ask "will a safety car be deployed in the next N minutes?"
    """
    
    def __init__(self, 
                 time_step: pd.Timedelta = pd.Timedelta(minutes=1),
                 lookback_window: pd.Timedelta = pd.Timedelta(minutes=5),
                 prediction_horizon: pd.Timedelta = pd.Timedelta(minutes=2)):
        """
        Initialize temporal processor.
        
        Args:
            time_step: How frequently to make predictions (e.g., every 1 minute)
            lookback_window: How much historical data to use for each prediction
            prediction_horizon: How far ahead to predict safety car deployment
        """
        self.time_step = time_step
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
    
    def find_safety_car_events(self, session) -> List[SafetyCarEvent]:
        """Find safety car events with precise timestamps."""
        track_status_df = session.track_status
        
        # Find safety car deployments
        sc_deployments = track_status_df[track_status_df['Status'] == '4'].copy()
        
        events = []
        for _, sc_row in sc_deployments.iterrows():
            event = SafetyCarEvent(
                deployment_time=sc_row['Time']
            )
            events.append(event)
        
        return events
    
    def create_temporal_windows(self, session, events: List[SafetyCarEvent]) -> List[TemporalWindow]:
        """
        Create temporal sliding windows for each driver.
        
        This method slides through time, creating prediction windows at regular intervals.
        For each window, we collect the driver's recent lap data and ask whether
        a safety car will be deployed in the near future.
        """
        laps_df = session.laps
        
        if laps_df.empty:
            return []
        
        # Get session time bounds
        session_start = laps_df['LapStartTime'].min()
        session_end = laps_df['Time'].max()
        
        # Get all drivers
        drivers = laps_df['Driver'].unique()
        
        windows = []
        
        # For each driver, create sliding temporal windows
        for driver in drivers:
            driver_laps = laps_df[laps_df['Driver'] == driver].copy()
            driver_laps = driver_laps.sort_values('LapStartTime')
            
            windows.extend(self._create_driver_windows(driver, driver_laps, events, session_start, session_end))
        
        return windows
    
    def _create_driver_windows(self, driver: str, driver_laps: pd.DataFrame, 
                              events: List[SafetyCarEvent], 
                              session_start: pd.Timestamp, 
                              session_end: pd.Timestamp) -> List[TemporalWindow]:
        """Create temporal windows for a single driver."""
        
        windows = []
        
        # Start sliding from when this driver has enough historical data
        current_time = session_start + self.lookback_window
        
        # Slide through time until we can't make meaningful predictions
        while current_time + self.prediction_horizon <= session_end:
            
            # Define the lookback window for this prediction
            window_start = current_time - self.lookback_window
            window_end = current_time
            
            # Define the prediction target window
            prediction_target_time = current_time
            prediction_window_end = current_time + self.prediction_horizon
            
            # Get driver's laps in the lookback window
            window_laps = driver_laps[
                (driver_laps['LapStartTime'] >= window_start) & 
                (driver_laps['LapStartTime'] <= window_end)
            ].copy()
            
            # Only create window if we have meaningful data
            if len(window_laps) >= 1:  # At least one lap of data
                
                # Determine if this window should predict a safety car
                # (i.e., is there a safety car deployment in the prediction window?)
                has_sc_in_prediction_window = any(
                    prediction_target_time <= event.deployment_time <= prediction_window_end
                    for event in events
                )
                
                window = TemporalWindow(
                    driver=driver,
                    window_start=window_start,
                    window_end=window_end,
                    prediction_target_time=prediction_target_time,
                    prediction_window_end=prediction_window_end,
                    laps_data=window_laps
                )
                
                # Add metadata about whether this window should predict SC
                window.should_predict_sc = has_sc_in_prediction_window
                
                windows.append(window)
            
            # Slide forward by time step
            current_time += self.time_step
        
        return windows

# =============================================================================
# Temporal Feature Engineering
# =============================================================================

class TemporalFeatureEngineer:
    """
    Extracts features from temporal windows focused on real-time observables.
    
    This engineer creates features that would be available in real-time,
    focusing on recent trends and sudden changes in driver behavior.
    """
    
    def __init__(self, include_telemetry: bool = True):
        """
        Initialize temporal feature engineer.
        
        Args:
            include_telemetry: Whether to extract detailed telemetry features
        """
        self.include_telemetry = include_telemetry
    
    def extract_features(self, window: TemporalWindow) -> pd.DataFrame:
        """
        Extract features from a temporal window.
        
        Returns a single-row DataFrame with features for this prediction moment.
        """
        
        features = {
            'driver': window.driver,
            'prediction_time': window.prediction_target_time,
            'window_start': window.window_start,
            'window_end': window.window_end
        }
        
        # Basic lap-based features
        features.update(self._extract_lap_features(window.laps_data))
        
        # Temporal trend features
        features.update(self._extract_temporal_trends(window.laps_data))
        
        # Recent change detection features
        features.update(self._extract_recent_changes(window.laps_data))
        
        # Telemetry features (if available and enabled)
        if self.include_telemetry:
            features.update(self._extract_telemetry_features(window.laps_data))
        
        # Target variable
        features['sc_in_prediction_window'] = getattr(window, 'should_predict_sc', False)
        
        return pd.DataFrame([features])
    
    def _extract_lap_features(self, laps_data: pd.DataFrame) -> Dict:
        """Extract basic features from recent laps."""
        
        features = {}
        
        if laps_data.empty:
            return features
        
        # Number of laps in window
        features['num_laps_in_window'] = len(laps_data)
        
        # Lap time analysis
        if 'LapTime' in laps_data.columns:
            lap_times = laps_data['LapTime'].dropna()
            if not lap_times.empty:
                # Convert to seconds for analysis
                if pd.api.types.is_timedelta64_dtype(lap_times):
                    lap_times_seconds = lap_times.dt.total_seconds()
                else:
                    try:
                        lap_times_seconds = pd.to_timedelta(lap_times).dt.total_seconds()
                    except:
                        lap_times_seconds = pd.to_numeric(lap_times, errors='coerce')
                
                lap_times_clean = lap_times_seconds.dropna()
                if not lap_times_clean.empty:
                    features.update({
                        'recent_laptime_mean': lap_times_clean.mean(),
                        'recent_laptime_std': lap_times_clean.std(),
                        'recent_laptime_min': lap_times_clean.min(),
                        'recent_laptime_max': lap_times_clean.max(),
                        'laptime_consistency': lap_times_clean.std() / lap_times_clean.mean() if lap_times_clean.mean() > 0 else 0
                    })
        
        # Position and competitive context
        if 'Position' in laps_data.columns:
            positions = laps_data['Position'].dropna()
            if not positions.empty:
                features.update({
                    'current_position': positions.iloc[-1] if len(positions) > 0 else np.nan,
                    'position_changes': abs(positions.diff()).sum() if len(positions) > 1 else 0,
                    'avg_position': positions.mean()
                })
        
        # Tire strategy context
        if 'TyreLife' in laps_data.columns:
            tyre_life = laps_data['TyreLife'].dropna()
            if not tyre_life.empty:
                features.update({
                    'current_tyre_life': tyre_life.iloc[-1] if len(tyre_life) > 0 else np.nan,
                    'avg_tyre_life': tyre_life.mean(),
                    'max_tyre_life': tyre_life.max()
                })
        
        # Stint information
        if 'Stint' in laps_data.columns:
            stint_data = laps_data['Stint'].dropna()
            if not stint_data.empty:
                features.update({
                    'current_stint': stint_data.iloc[-1] if len(stint_data) > 0 else np.nan,
                    'stint_changes': stint_data.nunique() - 1  # Number of stint changes in window
                })
        
        return features
    
    def _extract_temporal_trends(self, laps_data: pd.DataFrame) -> Dict:
        """Extract temporal trend features."""
        
        features = {}
        
        if len(laps_data) < 2:
            return features
        
        # Sort by lap start time to ensure temporal order
        laps_sorted = laps_data.sort_values('LapStartTime')
        
        # Lap time trends
        if 'LapTime' in laps_sorted.columns:
            lap_times = laps_sorted['LapTime'].dropna()
            if len(lap_times) >= 2:
                # Convert to seconds
                if pd.api.types.is_timedelta64_dtype(lap_times):
                    lap_times_seconds = lap_times.dt.total_seconds()
                else:
                    try:
                        lap_times_seconds = pd.to_timedelta(lap_times).dt.total_seconds()
                    except:
                        lap_times_seconds = pd.to_numeric(lap_times, errors='coerce')
                
                lap_times_clean = lap_times_seconds.dropna()
                if len(lap_times_clean) >= 2:
                    # Linear trend in lap times
                    x = np.arange(len(lap_times_clean))
                    if len(x) > 1 and np.var(x) > 0:
                        correlation = np.corrcoef(x, lap_times_clean)[0, 1]
                        features['laptime_trend'] = correlation if not np.isnan(correlation) else 0
                    
                    # Recent vs early performance
                    if len(lap_times_clean) >= 4:
                        early_times = lap_times_clean[:len(lap_times_clean)//2]
                        recent_times = lap_times_clean[len(lap_times_clean)//2:]
                        features['recent_vs_early_laptime'] = recent_times.mean() - early_times.mean()
        
        return features
    
    def _extract_recent_changes(self, laps_data: pd.DataFrame) -> Dict:
        """Extract recent change detection features."""
        
        features = {}
        
        if len(laps_data) < 2:
            return features
        
        # Sort by lap start time
        laps_sorted = laps_data.sort_values('LapStartTime')
        
        # Recent lap time changes
        if 'LapTime' in laps_sorted.columns:
            lap_times = laps_sorted['LapTime'].dropna()
            if len(lap_times) >= 2:
                # Convert to seconds
                if pd.api.types.is_timedelta64_dtype(lap_times):
                    lap_times_seconds = lap_times.dt.total_seconds()
                else:
                    try:
                        lap_times_seconds = pd.to_timedelta(lap_times).dt.total_seconds()
                    except:
                        lap_times_seconds = pd.to_numeric(lap_times, errors='coerce')
                
                lap_times_clean = lap_times_seconds.dropna()
                if len(lap_times_clean) >= 2:
                    # Most recent change
                    features['most_recent_laptime_change'] = lap_times_clean.iloc[-1] - lap_times_clean.iloc[-2]
                    
                    # Sudden deterioration indicator
                    if len(lap_times_clean) >= 3:
                        recent_avg = lap_times_clean.iloc[-2:].mean()
                        earlier_avg = lap_times_clean.iloc[:-2].mean()
                        features['sudden_laptime_deterioration'] = recent_avg - earlier_avg
        
        # Position changes
        if 'Position' in laps_sorted.columns:
            positions = laps_sorted['Position'].dropna()
            if len(positions) >= 2:
                features['recent_position_change'] = positions.iloc[-1] - positions.iloc[0]
                features['position_volatility'] = positions.std() if len(positions) > 1 else 0
        
        return features
    
    def _extract_telemetry_features(self, laps_data: pd.DataFrame) -> Dict:
        """Extract detailed telemetry features if available."""
        
        features = {}
        
        # This would be expanded to extract detailed telemetry
        # For now, we'll extract what's available in the basic lap data
        
        # Speed-related features
        speed_columns = [col for col in laps_data.columns if 'Speed' in col]
        for speed_col in speed_columns:
            speed_data = laps_data[speed_col].dropna()
            if not speed_data.empty:
                col_name = speed_col.lower().replace('speed', '')
                features.update({
                    f'recent_{col_name}speed_mean': speed_data.mean(),
                    f'recent_{col_name}speed_std': speed_data.std(),
                    f'{col_name}speed_trend': np.corrcoef(np.arange(len(speed_data)), speed_data)[0, 1] if len(speed_data) > 1 else 0
                })
        
        # Sector time analysis
        sector_columns = [col for col in laps_data.columns if 'Sector' in col and 'Time' in col]
        for sector_col in sector_columns:
            sector_data = laps_data[sector_col].dropna()
            if not sector_data.empty and pd.api.types.is_timedelta64_dtype(sector_data):
                sector_seconds = sector_data.dt.total_seconds()
                sector_name = sector_col.lower().replace('time', '').replace('sector', 's')
                features.update({
                    f'recent_{sector_name}_mean': sector_seconds.mean(),
                    f'recent_{sector_name}_consistency': sector_seconds.std() / sector_seconds.mean() if sector_seconds.mean() > 0 else 0
                })
        
        return features

# =============================================================================
# Temporal Model Training
# =============================================================================

class TemporalModelTrainer:
    """
    Trains models on temporal prediction windows.
    
    This trainer handles the unique aspects of temporal prediction,
    including proper time-based validation and handling of imbalanced targets.
    """
    
    def __init__(self, 
                 model_type: str = 'logistic_regression',
                 class_weight: str = 'balanced',
                 random_state: int = 42):
        """
        Initialize temporal model trainer.
        
        Args:
            model_type: Type of model to train
            class_weight: How to handle class imbalance
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
    
    def train(self, temporal_features: List[pd.DataFrame]) -> 'TemporalSafetyCarModel':
        """
        Train a temporal safety car prediction model.
        
        Args:
            temporal_features: List of feature DataFrames from temporal windows
            
        Returns:
            Trained TemporalSafetyCarModel
        """
        if not temporal_features:
            raise ValueError("No temporal features provided for training")
        
        # Combine all temporal features
        combined_features = pd.concat(temporal_features, ignore_index=True)
        
        # Prepare features and target
        exclude_cols = ['driver', 'prediction_time', 'window_start', 'window_end', 'sc_in_prediction_window']
        feature_cols = [col for col in combined_features.columns if col not in exclude_cols]
        
        X = combined_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = combined_features['sc_in_prediction_window'].astype(int)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        if self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model.fit(X_scaled, y)
        
        return TemporalSafetyCarModel(
            model=model,
            scaler=scaler,
            feature_names=feature_cols
        )

# =============================================================================
# Temporal Model and Prediction
# =============================================================================

class TemporalSafetyCarModel:
    """
    Temporal safety car prediction model.
    
    This model makes predictions based on temporal windows and provides
    information about prediction timing and lead times.
    """
    
    def __init__(self, model, scaler, feature_names: List[str]):
        """Initialize temporal model."""
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict(self, features_df: pd.DataFrame, 
                actual_events: Optional[List[SafetyCarEvent]] = None) -> List[TemporalPrediction]:
        """
        Make temporal safety car predictions.
        
        Args:
            features_df: DataFrame with temporal features
            actual_events: Actual safety car events for evaluation
            
        Returns:
            List of TemporalPrediction objects
        """
        
        # Prepare features
        exclude_cols = ['driver', 'prediction_time', 'window_start', 'window_end', 'sc_in_prediction_window']
        available_features = [col for col in self.feature_names if col in features_df.columns]
        
        if not available_features:
            raise ValueError("No matching features found")
        
        X = features_df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Handle missing features
        if len(available_features) < len(self.feature_names):
            missing_features = [col for col in self.feature_names if col not in available_features]
            for col in missing_features:
                X[col] = 0
            X = X[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)
        
        # Create prediction results
        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            
            prediction_time = row['prediction_time']
            driver = row['driver']
            
            # Determine actual outcome and correctness
            actual_outcome = None
            is_correct = None
            lead_time = None
            
            if actual_events:
                # Check if any safety car was deployed in the prediction window
                window_start = prediction_time
                window_end = prediction_time + pd.Timedelta(minutes=2)  # Assuming 2-minute prediction horizon
                
                actual_outcome = any(
                    window_start <= event.deployment_time <= window_end
                    for event in actual_events
                )
                
                is_correct = bool(predictions[i]) == actual_outcome
                
                # Calculate lead time if we correctly predicted a safety car
                if predictions[i] and actual_outcome:
                    matching_events = [
                        event for event in actual_events
                        if window_start <= event.deployment_time <= window_end
                    ]
                    if matching_events:
                        closest_event = min(matching_events, key=lambda e: e.deployment_time)
                        lead_time = closest_event.deployment_time - prediction_time
            
            result = TemporalPrediction(
                driver=driver,
                prediction_time=prediction_time,
                target_time=prediction_time + pd.Timedelta(minutes=2),  # Assuming horizon
                probability=probabilities[i],
                prediction=bool(predictions[i]),
                features_used=available_features,
                actual_outcome=actual_outcome,
                is_correct=is_correct,
                prediction_lead_time=lead_time
            )
            
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        return importance_df

# =============================================================================
# Main Orchestrator
# =============================================================================

class TemporalSafetyCarSystem:
    """
    Main orchestrator for temporal safety car prediction.
    
    This system coordinates temporal data processing, feature engineering,
    and model training to create a realistic safety car prediction system.
    """
    
    def __init__(self,
                 time_step: pd.Timedelta = pd.Timedelta(minutes=1),
                 lookback_window: pd.Timedelta = pd.Timedelta(minutes=5),
                 prediction_horizon: pd.Timedelta = pd.Timedelta(minutes=2)):
        """
        Initialize temporal prediction system.
        
        Args:
            time_step: How frequently to make predictions
            lookback_window: How much historical data to use
            prediction_horizon: How far ahead to predict
        """
        self.data_processor = TemporalDataProcessor(
            time_step=time_step,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon
        )
        self.feature_engineer = TemporalFeatureEngineer()
        self.model_trainer = TemporalModelTrainer()
        self.trained_model = None
    
    def train_on_session(self, session) -> Dict:
        """
        Train the temporal prediction system on a session.
        
        Args:
            session: FastF1 Session object
            
        Returns:
            Dictionary with training results and analysis
        """
        
        print("=== TEMPORAL SAFETY CAR PREDICTION TRAINING ===")
        
        # Find safety car events
        events = self.data_processor.find_safety_car_events(session)
        print(f"Found {len(events)} safety car events")
        
        if not events:
            return {"error": "No safety car events found"}
        
        # Create temporal windows
        windows = self.data_processor.create_temporal_windows(session, events)
        print(f"Created {len(windows)} temporal windows across all drivers")
        
        if not windows:
            return {"error": "No temporal windows created"}
        
        # Extract features
        temporal_features = []
        for window in windows:
            features = self.feature_engineer.extract_features(window)
            if not features.empty:
                temporal_features.append(features)
        
        if not temporal_features:
            return {"error": "No features extracted"}
        
        print(f"Extracted features from {len(temporal_features)} windows")
        
        # Train model
        self.trained_model = self.model_trainer.train(temporal_features)
        
        # Evaluate
        all_features = pd.concat(temporal_features, ignore_index=True)
        predictions = self.trained_model.predict(all_features, events)
        
        # Calculate metrics
        correct = sum(1 for p in predictions if p.is_correct)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Driver-level analysis
        driver_results = {}
        for driver in all_features['driver'].unique():
            driver_preds = [p for p in predictions if p.driver == driver]
            driver_correct = sum(1 for p in driver_preds if p.is_correct)
            driver_total = len(driver_preds)
            driver_results[driver] = {
                'accuracy': driver_correct / driver_total if driver_total > 0 else 0,
                'predictions': driver_total,
                'correct': driver_correct
            }
        
        return {
            "success": True,
            "events_found": len(events),
            "temporal_windows": len(windows),
            "total_predictions": total,
            "overall_accuracy": accuracy,
            "driver_results": driver_results,
            "feature_importance": self.trained_model.get_feature_importance(),
            "predictions": predictions
        }

def display_temporal_results(results: Dict, show_driver_details: bool = True, 
                           show_predictions: bool = False, max_predictions: int = 20):
    """
    Display temporal safety car prediction results in a comprehensive, readable format.
    
    Args:
        results: Results dictionary from analyze_session_temporal()
        show_driver_details: Whether to show per-driver analysis
        show_predictions: Whether to show individual predictions
        max_predictions: Maximum number of individual predictions to display
    """
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("=" * 80)
    print("🏎️  TEMPORAL SAFETY CAR PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Overview section
    print("\n📊 OVERVIEW")
    print("-" * 40)
    print(f"Safety car events found:     {results['events_found']}")
    print(f"Temporal windows created:    {results['temporal_windows']}")
    print(f"Total predictions made:      {results['total_predictions']}")
    print(f"Overall accuracy:            {results['overall_accuracy']:.1%}")
    
    # Driver performance analysis
    if show_driver_details and 'driver_results' in results:
        print("\n👥 DRIVER-SPECIFIC PERFORMANCE")
        print("-" * 40)
        
        driver_results = results['driver_results']
        
        # Sort drivers by accuracy for better readability
        sorted_drivers = sorted(driver_results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
        
        print(f"{'Driver':<8} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Performance'}")
        print("-" * 50)
        
        for driver, stats in sorted_drivers:
            accuracy = stats['accuracy']
            correct = stats['correct']
            total = stats['predictions']
            
            # Performance indicator
            if accuracy >= 0.8:
                performance = "🟢 Excellent"
            elif accuracy >= 0.6:
                performance = "🟡 Good"
            elif accuracy >= 0.4:
                performance = "🟠 Fair"
            else:
                performance = "🔴 Poor"
            
            print(f"{driver:<8} {accuracy:<10.1%} {correct:<8} {total:<8} {performance}")
        
        # Identify top and bottom performers
        if len(sorted_drivers) > 0:
            best_driver, best_stats = sorted_drivers[0]
            worst_driver, worst_stats = sorted_drivers[-1]
            
            print(f"\n🏆 Best predictor:  {best_driver} ({best_stats['accuracy']:.1%} accuracy)")
            print(f"⚠️  Worst predictor: {worst_driver} ({worst_stats['accuracy']:.1%} accuracy)")
    
    # Prediction timing analysis
    if 'predictions' in results:
        predictions = results['predictions']
        
        # Analyze prediction lead times
        successful_predictions = [p for p in predictions if p.is_correct and p.prediction and p.prediction_lead_time]
        
        if successful_predictions:
            lead_times = [p.prediction_lead_time.total_seconds() / 60 for p in successful_predictions]  # Convert to minutes
            avg_lead_time = np.mean(lead_times)
            max_lead_time = np.max(lead_times)
            min_lead_time = np.min(lead_times)
            
            print(f"\n⏰ PREDICTION TIMING ANALYSIS")
            print("-" * 40)
            print(f"Successful early warnings:   {len(successful_predictions)}")
            print(f"Average lead time:           {avg_lead_time:.1f} minutes")
            print(f"Maximum lead time:           {max_lead_time:.1f} minutes")
            print(f"Minimum lead time:           {min_lead_time:.1f} minutes")
        
        # Analyze prediction types
        true_positives = sum(1 for p in predictions if p.prediction and p.actual_outcome)
        false_positives = sum(1 for p in predictions if p.prediction and not p.actual_outcome)
        true_negatives = sum(1 for p in predictions if not p.prediction and not p.actual_outcome)
        false_negatives = sum(1 for p in predictions if not p.prediction and p.actual_outcome)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 DETAILED METRICS")
        print("-" * 40)
        print(f"Precision (accuracy of warnings):  {precision:.1%}")
        print(f"Recall (% of events predicted):    {recall:.1%}")
        print(f"F1 Score:                          {f1_score:.1%}")
        
        print(f"\n🎯 CONFUSION MATRIX")
        print("-" * 40)
        print(f"True Positives (correct warnings):     {true_positives:4d}")
        print(f"False Positives (false alarms):        {false_positives:4d}")
        print(f"True Negatives (correct no-warning):   {true_negatives:4d}")
        print(f"False Negatives (missed events):       {false_negatives:4d}")
    
    # Feature importance
    if 'feature_importance' in results:
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES")
        print("-" * 40)
        
        feature_importance = results['feature_importance']
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            feature_name = row['feature']
            importance = row['importance']
            
            # Truncate long feature names
            if len(feature_name) > 35:
                feature_name = feature_name[:32] + "..."
            
            print(f"{i:2d}. {feature_name:<35} | {importance:.3f}")
    
    # Individual predictions (if requested)
    if show_predictions and 'predictions' in results:
        print(f"\n🔮 INDIVIDUAL PREDICTIONS (showing up to {max_predictions})")
        print("-" * 80)
        
        predictions = results['predictions']
        
        # Sort predictions by time for chronological display
        sorted_predictions = sorted(predictions, key=lambda p: p.prediction_time)
        
        # Show only the most interesting predictions (high probability or incorrect)
        interesting_predictions = []
        
        # Always include correct positive predictions and incorrect predictions
        for pred in sorted_predictions:
            if (pred.prediction and pred.actual_outcome) or not pred.is_correct or pred.probability > 0.7:
                interesting_predictions.append(pred)
        
        # If we don't have enough interesting ones, add some high-probability negatives
        if len(interesting_predictions) < max_predictions:
            remaining_predictions = [p for p in sorted_predictions if p not in interesting_predictions]
            remaining_predictions.sort(key=lambda p: p.probability, reverse=True)
            interesting_predictions.extend(remaining_predictions[:max_predictions - len(interesting_predictions)])
        
        # Display predictions
        print(f"{'Time':<12} {'Driver':<6} {'Prob':<6} {'Pred':<6} {'Actual':<8} {'Result':<8} {'Lead Time'}")
        print("-" * 80)
        
        for pred in interesting_predictions[:max_predictions]:
            # Handle different time formats
            if hasattr(pred.prediction_time, 'strftime'):
                time_str = pred.prediction_time.strftime('%H:%M:%S')
            elif hasattr(pred.prediction_time, 'total_seconds'):
                # Handle Timedelta - convert to hours:minutes:seconds
                total_seconds = int(pred.prediction_time.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = str(pred.prediction_time)[:12]
            
            driver = pred.driver[:6]  # Truncate driver names
            prob_str = f"{pred.probability:.3f}"
            pred_str = "🚨SC" if pred.prediction else "✓OK"
            
            if pred.actual_outcome is not None:
                actual_str = "SC!" if pred.actual_outcome else "Normal"
            else:
                actual_str = "Unknown"
            
            if pred.is_correct is not None:
                result_str = "✅" if pred.is_correct else "❌"
            else:
                result_str = "?"
            
            lead_time_str = ""
            if pred.prediction_lead_time:
                lead_time_str = f"{pred.prediction_lead_time.total_seconds()/60:.1f}min"
            
            print(f"{time_str:<12} {driver:<6} {prob_str:<6} {pred_str:<6} {actual_str:<8} {result_str:<8} {lead_time_str}")
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 40)
    
    if 'driver_results' in results:
        driver_results = results['driver_results']
        
        # Find drivers with notably different performance
        accuracies = [stats['accuracy'] for stats in driver_results.values()]
        if len(accuracies) > 1:
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            acc_spread = max_acc - min_acc
            
            if acc_spread > 0.3:  # Significant difference
                print(f"• Large variation in driver predictability ({acc_spread:.1%} spread)")
                print(f"  This suggests some drivers have more predictable risk patterns")
            else:
                print(f"• Consistent predictability across drivers ({acc_spread:.1%} spread)")
                print(f"  This suggests universal risk factors rather than driver-specific patterns")
    
    if 'predictions' in results:
        predictions = results['predictions']
        
        # Analyze false alarm rate
        total_predictions_made = sum(1 for p in predictions if p.prediction)
        false_alarms = sum(1 for p in predictions if p.prediction and not p.actual_outcome)
        
        if total_predictions_made > 0:
            false_alarm_rate = false_alarms / total_predictions_made
            if false_alarm_rate < 0.2:
                print(f"• Low false alarm rate ({false_alarm_rate:.1%}) - system is conservative")
            elif false_alarm_rate > 0.5:
                print(f"• High false alarm rate ({false_alarm_rate:.1%}) - system may be too sensitive")
            else:
                print(f"• Moderate false alarm rate ({false_alarm_rate:.1%}) - balanced sensitivity")
        
        # Check if we're getting advance warning
        if successful_predictions:
            avg_lead_time = np.mean([p.prediction_lead_time.total_seconds() / 60 for p in successful_predictions])
            if avg_lead_time > 1.5:
                print(f"• Good advance warning capability ({avg_lead_time:.1f} min average lead time)")
            else:
                print(f"• Limited advance warning ({avg_lead_time:.1f} min average lead time)")
    
    print("\n" + "=" * 80)

def display_prediction_timeline(results: Dict, driver: Optional[str] = None):
    """
    Display a timeline view of predictions for a specific driver or all drivers.
    
    Args:
        results: Results dictionary from analyze_session_temporal()
        driver: Specific driver to show (None for all drivers)
    """
    
    if "error" in results or 'predictions' not in results:
        print("No predictions available to display")
        return
    
    predictions = results['predictions']
    
    # Filter by driver if specified
    if driver:
        predictions = [p for p in predictions if p.driver == driver]
        print(f"\n🏁 PREDICTION TIMELINE FOR {driver}")
    else:
        print(f"\n🏁 PREDICTION TIMELINE (ALL DRIVERS)")
    
    print("=" * 100)
    
    if not predictions:
        print("No predictions found for the specified criteria")
        return
    
    # Sort by time
    sorted_predictions = sorted(predictions, key=lambda p: p.prediction_time)
    
    # Group by time to show simultaneous predictions
    from itertools import groupby
    
    for time, time_group in groupby(sorted_predictions, key=lambda p: p.prediction_time):
        time_predictions = list(time_group)
        
        # Handle different time formats
        if hasattr(time, 'strftime'):
            time_str = time.strftime('%H:%M:%S')
        elif hasattr(time, 'total_seconds'):
            # Handle Timedelta - convert to hours:minutes:seconds
            total_seconds = int(time.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = str(time)[:12]
        
        print(f"\n⏰ {time_str}")
        print("-" * 60)
        
        for pred in time_predictions:
            # Create visual indicator
            if pred.prediction:
                if pred.actual_outcome:
                    indicator = "🚨✅"  # Correct safety car prediction
                elif pred.actual_outcome is False:
                    indicator = "🚨❌"  # False alarm
                else:
                    indicator = "🚨❓"  # Unknown outcome
            else:
                if pred.actual_outcome is False or pred.actual_outcome is None:
                    indicator = "✅"    # Correct normal prediction
                else:
                    indicator = "❌"    # Missed safety car
            
            prob_bar = "█" * int(pred.probability * 20)  # Visual probability bar
            prob_bar = prob_bar.ljust(20, "░")
            
            lead_time_str = ""
            if pred.prediction_lead_time:
                lead_time_str = f" (Lead: {pred.prediction_lead_time.total_seconds()/60:.1f}min)"
            
            print(f"  {indicator} {pred.driver:<6} [{prob_bar}] {pred.probability:.3f}{lead_time_str}")

# Add to the end of the file
def analyze_session_temporal(session, 
                           time_step_minutes: int = 1,
                           lookback_minutes: int = 5,
                           prediction_horizon_minutes: int = 2) -> Dict:
    """
    Convenience function for temporal safety car analysis.
    
    Args:
        session: FastF1 Session object
        time_step_minutes: Prediction frequency in minutes
        lookback_minutes: Historical data window in minutes  
        prediction_horizon_minutes: Prediction horizon in minutes
        
    Returns:
        Analysis results dictionary
    """
    
    system = TemporalSafetyCarSystem(
        time_step=pd.Timedelta(minutes=time_step_minutes),
        lookback_window=pd.Timedelta(minutes=lookback_minutes),
        prediction_horizon=pd.Timedelta(minutes=prediction_horizon_minutes)
    )
    
    return system.train_on_session(session)


def test():
    import fastf1 as f1

    # session = f1.get_session(2024, 'São Paulo Grand Prix', 'R')
    session = f1.get_session(2024, 'Saudi Arabian Grand Prix', 'R')

    session.load()

    # Analyze with 1-minute predictions, 5-minute lookback, 2-minute horizon
    results = analyze_session_temporal(
        session, 
        time_step_minutes=1,
        lookback_minutes=5, 
        prediction_horizon_minutes=2
    )

    display_temporal_results(results, 
                            show_driver_details=True,
                            show_predictions=True,
                            max_predictions=15)


    display_prediction_timeline(results)


if __name__ == "__main__":
    test()