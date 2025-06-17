"""
Modular Safety Car Prediction System for Formula 1

This module provides a clean, well-structured system for predicting safety car deployments
in Formula 1 races. The design follows separation of concerns principles with distinct
components for data processing, feature engineering, model training, and prediction.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Data Structures and Protocols
# =============================================================================

@dataclass
class SafetyCarEvent:
    """Represents a safety car deployment event."""
    lap_number: int
    deployment_time: pd.Timestamp
    confidence: float = 1.0
    
@dataclass 
class TrainingWindow:
    """Represents a training window for model development."""
    sc_event: SafetyCarEvent
    window_start: int
    window_end: int
    laps_data: pd.DataFrame
    num_drivers: int
    total_laps: int

@dataclass
class PredictionResult:
    """Represents a safety car prediction for a specific lap."""
    lap_number: int
    probability: float
    prediction: bool
    features_used: List[str]
    actual_outcome: Optional[bool] = None  # True if SC actually deployed next lap
    is_correct: Optional[bool] = None      # True if prediction matches actual outcome
    confidence_interval: Optional[Tuple[float, float]] = None

class DataProcessor(Protocol):
    """Protocol for data processing components."""
    
    def find_safety_car_events(self, session) -> List[SafetyCarEvent]:
        """Find safety car events in session data."""
        ...
    
    def create_training_windows(self, laps_df: pd.DataFrame, 
                               events: List[SafetyCarEvent],
                               window_size: int) -> List[TrainingWindow]:
        """Create training windows from safety car events."""
        ...

class FeatureEngineer(Protocol):
    """Protocol for feature engineering components."""
    
    def extract_features(self, window: TrainingWindow) -> pd.DataFrame:
        """Extract features from a training window."""
        ...

class ModelTrainer(Protocol):
    """Protocol for model training components."""
    
    def train(self, training_data: List[pd.DataFrame]) -> 'SafetyCarModel':
        """Train a safety car prediction model."""
        ...

# =============================================================================
# Data Processing Components
# =============================================================================

class FastF1DataProcessor:
    """
    Handles processing of FastF1 session data to extract safety car events
    and create training windows.
    
    This class encapsulates all the complexity of working with F1 timing data,
    including timestamp alignment and lap number mapping.
    """
    
    def __init__(self, min_window_size: int = 3):
        """
        Initialize the data processor.
        
        Args:
            min_window_size: Minimum number of laps required for a valid training window
        """
        self.min_window_size = min_window_size
    
    def find_safety_car_events(self, session) -> List[SafetyCarEvent]:
        """
        Find all safety car deployment events in a session.
        
        Args:
            session: FastF1 Session object
            
        Returns:
            List of SafetyCarEvent objects
        """
        track_status_df = session.track_status
        
        # Find safety car deployments (Status == '4')
        sc_deployments = track_status_df[track_status_df['Status'] == '4'].copy()
        
        if sc_deployments.empty:
            return []
        
        events = []
        
        for _, sc_row in sc_deployments.iterrows():
            sc_time = sc_row['Time']
            
            # Find the lap number directly from the track status or use a simple mapping
            lap_number = self._find_lap_for_safety_car(sc_time, session.laps)
            
            if lap_number is not None:
                event = SafetyCarEvent(
                    lap_number=lap_number,
                    deployment_time=sc_time,
                    confidence=1.0
                )
                events.append(event)
        
        return events
    
    def _find_lap_for_safety_car(self, sc_time: pd.Timestamp, laps_df: pd.DataFrame) -> Optional[int]:
        """
        Find which lap the safety car was deployed on using the LapNumber column.
        
        This method finds the lap number by looking at which laps were being
        completed around the time of safety car deployment.
        """
        
        # Find the lap completion times closest to the safety car deployment
        if 'Time' not in laps_df.columns:
            return None
        
        # Calculate time differences between lap completions and safety car deployment
        laps_with_diff = laps_df.copy()
        laps_with_diff['time_diff'] = abs(laps_with_diff['Time'] - sc_time)
        
        # Find laps completed within a reasonable window (e.g., 3 minutes)
        time_window = pd.Timedelta(minutes=3)
        nearby_laps = laps_with_diff[laps_with_diff['time_diff'] <= time_window]
        
        if nearby_laps.empty:
            return None
        
        # Get the most common lap number among drivers near the safety car time
        lap_numbers = nearby_laps['LapNumber'].dropna()
        
        if lap_numbers.empty:
            return None
        
        # Return the most frequent lap number (handles drivers completing laps at different times)
        from collections import Counter
        lap_counts = Counter(lap_numbers)
        most_common_lap = lap_counts.most_common(1)[0][0]
        
        return int(most_common_lap)
    
    def create_training_windows(self, laps_df: pd.DataFrame, 
                               events: List[SafetyCarEvent],
                               window_size: int) -> List[TrainingWindow]:
        """
        Create training windows for each safety car event.
        
        Args:
            laps_df: DataFrame containing lap data
            events: List of safety car events
            window_size: Number of laps to include before each event
            
        Returns:
            List of TrainingWindow objects
        """
        windows = []
        
        for event in events:
            window_start = max(1, event.lap_number - window_size)
            window_end = event.lap_number - 1
            
            # Validate window
            if window_end - window_start + 1 < self.min_window_size:
                continue
            
            # Check for conflicts with other events
            conflicting_events = [
                other_event for other_event in events
                if other_event != event and 
                window_start <= other_event.lap_number <= window_end
            ]
            
            if conflicting_events:
                continue
            
            # Extract lap data
            window_laps = laps_df[
                (laps_df['LapNumber'] >= window_start) & 
                (laps_df['LapNumber'] <= window_end)
            ].copy()
            
            if len(window_laps) == 0:
                continue
            
            # Create training window
            window = TrainingWindow(
                sc_event=event,
                window_start=window_start,
                window_end=window_end,
                laps_data=window_laps,
                num_drivers=window_laps['Driver'].nunique() if 'Driver' in window_laps.columns else 0,
                total_laps=len(window_laps)
            )
            
            windows.append(window)
        
        return windows

# =============================================================================
# Feature Engineering Components
# =============================================================================

class StandardFeatureEngineer:
    """
    Standard feature engineering for safety car prediction.
    
    This class implements our proven approach of sudden change detection
    combined with threshold crossing indicators and multi-feature risk assessment.
    """
    
    def __init__(self, 
                 include_basic_features: bool = True,
                 include_change_features: bool = True,
                 include_threshold_features: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            include_basic_features: Whether to include basic statistical features
            include_change_features: Whether to include sudden change detection features
            include_threshold_features: Whether to include threshold crossing features
        """
        self.include_basic_features = include_basic_features
        self.include_change_features = include_change_features
        self.include_threshold_features = include_threshold_features
    
    def extract_features(self, window: TrainingWindow) -> pd.DataFrame:
        """
        Extract comprehensive features from a training window.
        
        Args:
            window: TrainingWindow object containing lap data
            
        Returns:
            DataFrame with engineered features per lap
        """
        # Start with basic features
        features_df = self._extract_basic_features(window.laps_data)
        
        if features_df.empty:
            return features_df
        
        # Add sudden change detection features
        if self.include_change_features:
            features_df = self._add_change_detection_features(features_df)
        
        # Add threshold crossing features
        if self.include_threshold_features:
            features_df = self._add_threshold_features(features_df)
        
        # Add target variable
        features_df = self._add_target_variable(features_df, window.sc_event.lap_number)
        
        return features_df
    
    def _extract_basic_features(self, window_laps: pd.DataFrame) -> pd.DataFrame:
        """Extract basic statistical features from lap data."""
        window_laps = window_laps.copy()
        
        # Handle lap time conversion
        if 'LapTime' in window_laps.columns:
            if pd.api.types.is_timedelta64_dtype(window_laps['LapTime']):
                window_laps['LapTime_seconds'] = window_laps['LapTime'].dt.total_seconds()
            else:
                try:
                    window_laps['LapTime'] = pd.to_timedelta(window_laps['LapTime'])
                    window_laps['LapTime_seconds'] = window_laps['LapTime'].dt.total_seconds()
                except:
                    try:
                        window_laps['LapTime_seconds'] = pd.to_numeric(window_laps['LapTime'], errors='coerce')
                    except:
                        window_laps['LapTime_seconds'] = np.nan
        
        # Group by lap number and aggregate
        lap_features = []
        
        for lap_num in sorted(window_laps['LapNumber'].unique()):
            lap_data = window_laps[window_laps['LapNumber'] == lap_num]
            features = self._compute_lap_features(lap_data, lap_num)
            lap_features.append(features)
        
        return pd.DataFrame(lap_features)
    
    def _compute_lap_features(self, lap_data: pd.DataFrame, lap_num: float) -> Dict:
        """Compute aggregated features for a single lap."""
        features = {'LapNumber': lap_num}
        
        # Lap time statistics
        if 'LapTime_seconds' in lap_data.columns:
            laptime_clean = lap_data['LapTime_seconds'].dropna()
            if len(laptime_clean) > 0:
                features.update({
                    'laptime_mean': laptime_clean.mean(),
                    'laptime_std': laptime_clean.std() if len(laptime_clean) > 1 else 0,
                    'laptime_min': laptime_clean.min(),
                    'laptime_max': laptime_clean.max(),
                    'laptime_range': laptime_clean.max() - laptime_clean.min()
                })
        
        # Field characteristics
        features['num_drivers'] = len(lap_data)
        features['drivers_with_valid_times'] = lap_data.get('LapTime_seconds', pd.Series()).notna().sum()
        
        # Position features
        if 'Position' in lap_data.columns:
            positions = lap_data['Position'].dropna()
            if len(positions) > 0:
                features.update({
                    'position_spread': positions.max() - positions.min(),
                    'avg_position': positions.mean()
                })
        
        # Tire and stint features
        for col_name, prefix in [('TyreLife', 'tyre_life'), ('Stint', 'stint')]:
            if col_name in lap_data.columns:
                data = lap_data[col_name].dropna()
                if len(data) > 0:
                    features.update({
                        f'avg_{prefix}': data.mean(),
                        f'max_{prefix}': data.max(),
                        f'{prefix}_spread': data.max() - data.min() if len(data) > 1 else 0
                    })
        
        # Speed features
        speed_columns = [col for col in lap_data.columns if 'Speed' in col]
        for speed_col in speed_columns:
            speed_data = lap_data[speed_col].dropna()
            if len(speed_data) > 0:
                col_prefix = speed_col.lower()
                features.update({
                    f'{col_prefix}_mean': speed_data.mean(),
                    f'{col_prefix}_std': speed_data.std() if len(speed_data) > 1 else 0,
                    f'{col_prefix}_range': speed_data.max() - speed_data.min()
                })
        
        return features
    
    def _add_change_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sudden change detection features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        risk_features = [col for col in numeric_cols if col not in ['LapNumber']]
        
        for feature in risk_features:
            if feature in df.columns:
                values = df[feature].fillna(0)
                
                # Spike detection
                recent_baseline = values.rolling(window=3, min_periods=1).mean().shift(1)
                df[f'{feature}_spike_ratio'] = values / (recent_baseline + 0.001)
                
                # Jump detection
                df[f'{feature}_jump'] = values.diff()
                df[f'{feature}_jump_pct'] = values.pct_change().fillna(0)
                
                # Z-score
                if values.std() > 0:
                    df[f'{feature}_zscore'] = (values - values.mean()) / values.std()
                else:
                    df[f'{feature}_zscore'] = 0
        
        return df
    
    def _add_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add threshold crossing indicator features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        risk_features = [col for col in numeric_cols 
                        if col not in ['LapNumber'] and not any(suffix in col for suffix in ['_spike_ratio', '_jump', '_zscore'])]
        
        for feature in risk_features:
            if feature in df.columns:
                values = df[feature].fillna(0)
                
                # Percentile ranking
                df[f'{feature}_percentile'] = values.rank(pct=True)
                
                # Threshold flags
                if len(values) > 1:
                    p75 = values.quantile(0.75)
                    p90 = values.quantile(0.90)
                    df[f'{feature}_above_p75'] = (values > p75).astype(int)
                    df[f'{feature}_above_p90'] = (values > p90).astype(int)
        
        # Multi-feature indicators
        spike_features = [col for col in df.columns if '_spike_ratio' in col]
        if spike_features:
            spike_flags = df[spike_features] > 1.5
            df['simultaneous_spikes'] = spike_flags.sum(axis=1)
        
        threshold_features = [col for col in df.columns if '_above_p75' in col]
        if threshold_features:
            df['concurrent_risk_factors'] = df[threshold_features].sum(axis=1)
        
        return df
    
    def _add_target_variable(self, df: pd.DataFrame, sc_lap: int) -> pd.DataFrame:
        """Add the target variable indicating if safety car comes next lap."""
        df['sc_next_lap'] = 0
        final_lap = sc_lap - 1
        df.loc[df['LapNumber'] == final_lap, 'sc_next_lap'] = 1
        return df

# =============================================================================
# Model Training Components
# =============================================================================

class LogisticRegressionTrainer:
    """
    Trains logistic regression models for safety car prediction.
    
    This trainer implements best practices for binary classification with
    imbalanced data, including proper scaling and class weight handling.
    """
    
    def __init__(self, 
                 class_weight: str = 'balanced',
                 random_state: int = 42,
                 max_iter: int = 1000):
        """
        Initialize the model trainer.
        
        Args:
            class_weight: How to handle class imbalance ('balanced' recommended)
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
    
    def train(self, training_data: List[pd.DataFrame]) -> 'SafetyCarModel':
        """
        Train a safety car prediction model.
        
        Args:
            training_data: List of feature DataFrames from training windows
            
        Returns:
            Trained SafetyCarModel instance
        """
        if not training_data:
            raise ValueError("No training data provided")
        
        # Combine all training data
        combined_data = pd.concat(training_data, ignore_index=True)
        
        # Prepare features and target
        exclude_cols = ['LapNumber', 'sc_next_lap']
        feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
        
        X = combined_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = combined_data['sc_next_lap']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        
        model.fit(X_scaled, y)
        
        # Create trained model wrapper
        trained_model = SafetyCarModel(
            model=model,
            scaler=scaler,
            feature_names=feature_cols
        )
        
        return trained_model

# =============================================================================
# Model and Prediction Components
# =============================================================================

class SafetyCarModel:
    """
    Wrapper for trained safety car prediction models.
    
    This class encapsulates a trained model along with its preprocessing
    components and provides a clean interface for making predictions.
    """
    
    def __init__(self, model, scaler, feature_names: List[str]):
        """
        Initialize a trained safety car model.
        
        Args:
            model: Trained scikit-learn model
            scaler: Fitted feature scaler
            feature_names: List of feature names used during training
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict(self, features_df: pd.DataFrame, 
                actual_events: Optional[List[SafetyCarEvent]] = None) -> List[PredictionResult]:
        """
        Make safety car predictions for given features.
        
        Args:
            features_df: DataFrame with features (must include LapNumber)
            actual_events: Optional list of actual safety car events for accuracy evaluation
            
        Returns:
            List of PredictionResult objects
        """
        # Prepare features
        available_features = [col for col in self.feature_names if col in features_df.columns]
        
        if not available_features:
            raise ValueError("No matching features found in input data")
        
        X = features_df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Handle missing features by padding with zeros
        if len(available_features) < len(self.feature_names):
            missing_features = [col for col in self.feature_names if col not in available_features]
            for col in missing_features:
                X[col] = 0
            X = X[self.feature_names]  # Reorder to match training
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)
        
        # Create actual outcomes lookup for accuracy evaluation
        actual_sc_laps = set()
        if actual_events:
            actual_sc_laps = {event.lap_number for event in actual_events}
        
        # Create results
        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            lap_number = row['LapNumber']
            prediction_bool = bool(predictions[i])
            
            # Determine actual outcome (did SC actually deploy next lap?)
            actual_outcome = None
            is_correct = None
            
            if actual_events is not None:
                # Check if safety car deployed on the lap following this one
                next_lap = lap_number + 1
                actual_outcome = next_lap in actual_sc_laps
                is_correct = prediction_bool == actual_outcome
            
            result = PredictionResult(
                lap_number=lap_number,
                probability=probabilities[i],
                prediction=prediction_bool,
                features_used=available_features,
                actual_outcome=actual_outcome,
                is_correct=is_correct
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

class SafetyCarPredictionSystem:
    """
    Main orchestrator for the safety car prediction system.
    
    This class provides a high-level interface that coordinates all components
    to provide end-to-end safety car prediction functionality.
    """
    
    def __init__(self, 
                 data_processor: Optional[DataProcessor] = None,
                 feature_engineer: Optional[FeatureEngineer] = None,
                 model_trainer: Optional[ModelTrainer] = None,
                 window_size: int = 5):
        """
        Initialize the prediction system.
        
        Args:
            data_processor: Component for processing F1 data (default: FastF1DataProcessor)
            feature_engineer: Component for feature engineering (default: StandardFeatureEngineer)
            model_trainer: Component for model training (default: LogisticRegressionTrainer)
            window_size: Number of laps to analyze before potential safety cars
        """
        self.data_processor = data_processor or FastF1DataProcessor()
        self.feature_engineer = feature_engineer or StandardFeatureEngineer()
        self.model_trainer = model_trainer or LogisticRegressionTrainer()
        self.window_size = window_size
        self.trained_model = None
    
    def train_on_session(self, session) -> Dict:
        """
        Train the prediction system on a single F1 session.
        
        Args:
            session: FastF1 Session object
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Find safety car events
        events = self.data_processor.find_safety_car_events(session)
        
        if not events:
            return {"error": "No safety car events found in session"}
        
        # Create training windows
        windows = self.data_processor.create_training_windows(
            session.laps, events, self.window_size
        )
        
        if not windows:
            return {"error": "No valid training windows created"}
        
        # Extract features
        training_data = []
        for window in windows:
            features = self.feature_engineer.extract_features(window)
            if not features.empty:
                training_data.append(features)
        
        if not training_data:
            return {"error": "No features could be extracted"}
        
        # Train model
        self.trained_model = self.model_trainer.train(training_data)
        
        # Evaluate on training data with actual outcomes
        all_results = []
        correct_predictions = 0
        total_predictions = 0
        
        for features_df in training_data:
            # Pass the actual events to get accuracy evaluation
            results = self.trained_model.predict(features_df, events)
            all_results.extend(results)
            
            # Count correct predictions
            for result in results:
                if result.is_correct is not None:
                    total_predictions += 1
                    if result.is_correct:
                        correct_predictions += 1
        
        # Calculate detailed metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Separate predictions by type for detailed analysis
        true_positives = sum(1 for r in all_results if r.prediction and r.actual_outcome)
        false_positives = sum(1 for r in all_results if r.prediction and not r.actual_outcome)
        true_negatives = sum(1 for r in all_results if not r.prediction and not r.actual_outcome)
        false_negatives = sum(1 for r in all_results if not r.prediction and r.actual_outcome)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "success": True,
            "events_found": len(events),
            "windows_created": len(windows),
            "training_samples": len(all_results),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            },
            "feature_importance": self.trained_model.get_feature_importance(),
            "predictions": all_results
        }
    
    def predict_session(self, session) -> List[PredictionResult]:
        """
        Make safety car predictions for an entire session.
        
        Args:
            session: FastF1 Session object
            
        Returns:
            List of predictions for each analyzable lap
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features for the entire session
        # This would need to be implemented based on your specific requirements
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Session-wide prediction not yet implemented")

def analyze_session(session, window_size: int = 5) -> Dict:
    """
    Convenience function to analyze a single F1 session.
    
    Args:
        session: FastF1 Session object
        window_size: Number of laps to analyze before potential safety cars
        
    Returns:
        Dictionary containing analysis results
    """
    system = SafetyCarPredictionSystem(window_size=window_size)
    return system.train_on_session(session)

def display_prediction_results(results: Dict, show_details: bool = True):
    """
    Display prediction results in a formatted, easy-to-read way.
    
    Args:
        results: Results dictionary from analyze_session()
        show_details: Whether to show individual prediction details
    """
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("=== SAFETY CAR PREDICTION ANALYSIS RESULTS ===")
    print(f"Events found: {results['events_found']}")
    print(f"Training windows: {results['windows_created']}")
    print(f"Total predictions: {results['training_samples']}")
    print()
    
    print("=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:  {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1 Score:  {results['f1_score']:.3f}")
    print()
    
    print("=== CONFUSION MATRIX ===")
    cm = results['confusion_matrix']
    print(f"True Positives:  {cm['true_positives']:2d} (Correctly predicted safety cars)")
    print(f"False Positives: {cm['false_positives']:2d} (False alarms)")
    print(f"True Negatives:  {cm['true_negatives']:2d} (Correctly predicted normal laps)")
    print(f"False Negatives: {cm['false_negatives']:2d} (Missed safety cars)")
    print()
    
    if show_details:
        print("=== DETAILED PREDICTIONS BY LAP ===")
        
        # Group predictions by safety car event
        predictions = results['predictions']
        
        # Find which laps had actual safety cars
        sc_laps = set()
        for pred in predictions:
            if pred.actual_outcome:
                sc_laps.add(pred.lap_number + 1)
        
        current_window = None
        for pred in predictions:
            # Detect new window (when lap numbers reset or jump significantly)
            if current_window is None or pred.lap_number < current_window - 1:
                # Find the safety car lap for this window
                next_sc_lap = None
                for potential_sc in sc_laps:
                    if potential_sc > pred.lap_number:
                        next_sc_lap = potential_sc
                        break
                
                if next_sc_lap:
                    print(f"\n--- Window: Safety Car on Lap {next_sc_lap} ---")
                else:
                    print(f"\n--- Window: Starting from Lap {pred.lap_number} ---")
            
            current_window = pred.lap_number
            
            # Format prediction result
            status_icon = "🚨" if pred.prediction else "✓"
            correctness_icon = ""
            
            if pred.is_correct is not None:
                correctness_icon = " ✅" if pred.is_correct else " ❌"
            
            actual_text = ""
            if pred.actual_outcome is not None:
                actual_text = " (SC NEXT LAP)" if pred.actual_outcome else ""
            
            print(f"  Lap {int(pred.lap_number):2d}: {status_icon} {pred.probability:.3f}{actual_text}{correctness_icon}")
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    feature_importance = results['feature_importance']
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature'][:40]:40} | {row['importance']:.3f}")
    
    print("\n" + "="*60)


# Basic usage with default components
import fastf1 as f1

season = 2024
# session = f1.get_session(season, 'Saudi Arabian Grand Prix', 'R')
session = f1.get_session(season, 'São Paulo Grand Prix', 'R')
session.load()
results = analyze_session(session, window_size=5)

# Advanced usage with custom components
custom_engineer = StandardFeatureEngineer(
    include_basic_features=True,
    include_change_features=True,
    include_threshold_features=False
)

system = SafetyCarPredictionSystem(
    feature_engineer=custom_engineer,
    window_size=7
)

training_results = system.train_on_session(session)

display_prediction_results(training_results)