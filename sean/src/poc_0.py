import fastf1 as f1
from fastf1.core import Session
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


season = 2024
event_name = 'São Paulo Grand Prix'
event_type = "R"

session = f1.get_session(season, event_name, event_type)
session.load()


class SafetyCarPredictor:
    """
    A clean, robust system for predicting safety car deployments in Formula 1 races.

    This class handles the entire pipeline from raw lap data to trained models,
    with careful attention to the real-world complexities of F1 data including
    timing inconsistencies, missing telemetry, and varying data formats.
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize the safety car prediction system.

        Args:
            window_size: Number of laps to analyze before each potential safety car deployment
        """
        self.window_size = window_size
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.training_windows = []

    def find_safety_car_laps(self, laps_df: pd.DataFrame, track_status_df: pd.DataFrame) -> List[int]:
        """
        Identify the lap numbers when safety cars were deployed.

        This method carefully maps safety car deployment times to lap numbers,
        handling the complexity that different drivers complete laps at different times.

        Args:
            laps_df: DataFrame containing lap data with LapNumber, Driver, and timing columns
            track_status_df: DataFrame containing track status with Time and Status columns

        Returns:
            List of lap numbers when safety cars were deployed
        """
        print("=== FINDING SAFETY CAR DEPLOYMENTS ===")

        # Find all safety car deployments (Status == '4' indicates safety car)
        sc_deployments = track_status_df[track_status_df['Status'] == '4'].copy()

        if sc_deployments.empty:
            print("No safety car deployments found")
            return []

        print(f"Found {len(sc_deployments)} safety car deployment(s)")

        sc_laps = []

        for idx, sc_row in sc_deployments.iterrows():
            sc_time = sc_row['Time']
            print(f"\nAnalyzing safety car deployment at time: {sc_time}")

            # Strategy: Find the lap that most drivers were on when SC was deployed
            # We'll look at all laps that were "active" during the SC deployment time

            # Convert LapTime to timedelta if it isn't already
            if 'LapTime' in laps_df.columns:
                if not pd.api.types.is_timedelta64_dtype(laps_df['LapTime']):
                    # Try to convert if it's not already a timedelta
                    try:
                        laps_with_time = laps_df.copy()
                        laps_with_time['LapTime'] = pd.to_timedelta(laps_with_time['LapTime'])
                    except:
                        laps_with_time = laps_df.copy()
                        laps_with_time['LapTime'] = pd.to_timedelta(laps_with_time['LapTime'], errors='coerce')
                else:
                    laps_with_time = laps_df.copy()
            else:
                print("Warning: No LapTime column found, using approximation")
                laps_with_time = laps_df.copy()

            # Find laps that were potentially active during SC deployment
            candidate_laps = []

            # Method 1: If we have LapStartTime, use it directly
            if 'LapStartTime' in laps_with_time.columns:
                # Find laps where SC time falls between lap start and estimated lap end
                for _, lap_row in laps_with_time.iterrows():
                    lap_start = lap_row['LapStartTime']

                    # Estimate lap end time
                    if pd.notna(lap_row.get('LapTime')):
                        try:
                            lap_end = lap_start + lap_row['LapTime']
                            if lap_start <= sc_time <= lap_end:
                                candidate_laps.append(lap_row['LapNumber'])
                        except:
                            # Fallback: just check if SC time is close to lap start
                            time_diff = abs((sc_time - lap_start).total_seconds())
                            if time_diff < 180:  # Within 3 minutes (reasonable lap time)
                                candidate_laps.append(lap_row['LapNumber'])
                    else:
                        # No lap time available, use proximity to start time
                        time_diff = abs((sc_time - lap_start).total_seconds())
                        if time_diff < 180:
                            candidate_laps.append(lap_row['LapNumber'])

            # Method 2: Fallback - find the most common lap number around the SC time
            if not candidate_laps:
                print("Using fallback method: closest lap by time")
                # Find laps with timestamps close to SC deployment
                if 'LapStartTime' in laps_with_time.columns:
                    time_diffs = abs(laps_with_time['LapStartTime'] - sc_time).dt.total_seconds()
                    closest_indices = time_diffs.nsmallest(10).index  # Take 10 closest laps
                    candidate_laps = laps_with_time.loc[closest_indices, 'LapNumber'].tolist()

            # Determine the most likely lap number
            if candidate_laps:
                # Take the most common lap number among candidates
                from collections import Counter
                lap_counts = Counter(candidate_laps)
                most_common_lap = lap_counts.most_common(1)[0][0]
                sc_laps.append(most_common_lap)
                print(f"  Safety car mapped to lap {most_common_lap}")
                print(f"  Candidate laps: {sorted(set(candidate_laps))}")
            else:
                print("  Warning: Could not map safety car to a specific lap")

        # Remove duplicates and sort
        unique_sc_laps = sorted(list(set(sc_laps)))
        print(f"\nFinal safety car laps: {unique_sc_laps}")
        return unique_sc_laps

    def create_training_windows(self, laps_df: pd.DataFrame, sc_laps: List[int]) -> List[Dict]:
        """
        Create training windows for each safety car deployment.

        Each window contains the laps leading up to a safety car deployment,
        providing the data needed to learn predictive patterns.

        Args:
            laps_df: DataFrame containing lap data
            sc_laps: List of lap numbers where safety cars were deployed

        Returns:
            List of training window dictionaries
        """
        print("\n=== CREATING TRAINING WINDOWS ===")

        windows = []

        for sc_lap in sc_laps:
            # Define the analysis window
            window_start = max(1, sc_lap - self.window_size)
            window_end = sc_lap - 1  # Last lap before safety car

            print(f"\nCreating window for safety car on lap {sc_lap}")
            print(f"  Analyzing laps {window_start} to {window_end}")

            # Check if we have enough laps
            if window_end < window_start:
                print(f"  Skipping: Not enough preceding laps")
                continue

            # Extract laps in this window
            window_laps = laps_df[
                (laps_df['LapNumber'] >= window_start) &
                (laps_df['LapNumber'] <= window_end)
            ].copy()

            if len(window_laps) == 0:
                print(f"  Skipping: No lap data found in window")
                continue

            # Check for conflicts with other safety car deployments
            conflicting_sc = [other_sc for other_sc in sc_laps
                             if other_sc != sc_lap and window_start <= other_sc <= window_end]

            if conflicting_sc:
                print(f"  Skipping: Conflicts with other safety cars at laps {conflicting_sc}")
                continue

            # Create window metadata
            window_info = {
                'sc_lap': sc_lap,
                'window_start': window_start,
                'window_end': window_end,
                'laps_data': window_laps,
                'num_drivers': window_laps['Driver'].nunique() if 'Driver' in window_laps.columns else 0,
                'total_laps': len(window_laps)
            }

            windows.append(window_info)
            print(f"  Created window: {len(window_laps)} lap records from {window_info['num_drivers']} drivers")

        self.training_windows = windows
        print(f"\nSuccessfully created {len(windows)} training windows")
        return windows

    def extract_basic_features(self, window_laps: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features from a window of lap data.

        This method creates robust features that work with the core data available
        in all F1 datasets, handling missing columns and data type issues gracefully.

        Args:
            window_laps: DataFrame containing laps in the analysis window

        Returns:
            DataFrame with features aggregated by lap number
        """

        # Convert LapTime to seconds for calculations
        window_laps = window_laps.copy()

        if 'LapTime' in window_laps.columns:
            # Handle different LapTime formats
            if pd.api.types.is_timedelta64_dtype(window_laps['LapTime']):
                window_laps['LapTime_seconds'] = window_laps['LapTime'].dt.total_seconds()
            else:
                # Try to convert to timedelta first
                try:
                    window_laps['LapTime'] = pd.to_timedelta(window_laps['LapTime'])
                    window_laps['LapTime_seconds'] = window_laps['LapTime'].dt.total_seconds()
                except:
                    # If conversion fails, try to extract numeric value
                    try:
                        window_laps['LapTime_seconds'] = pd.to_numeric(window_laps['LapTime'], errors='coerce')
                    except:
                        window_laps['LapTime_seconds'] = np.nan

        # Group by lap number to create aggregated features
        lap_features = []

        for lap_num in sorted(window_laps['LapNumber'].unique()):
            lap_data = window_laps[window_laps['LapNumber'] == lap_num]

            features = {'LapNumber': lap_num}

            # Basic lap time statistics
            if 'LapTime_seconds' in lap_data.columns:
                laptime_clean = lap_data['LapTime_seconds'].dropna()
                if len(laptime_clean) > 0:
                    features['laptime_mean'] = laptime_clean.mean()
                    features['laptime_std'] = laptime_clean.std() if len(laptime_clean) > 1 else 0
                    features['laptime_min'] = laptime_clean.min()
                    features['laptime_max'] = laptime_clean.max()
                    features['laptime_range'] = laptime_clean.max() - laptime_clean.min()

            # Driver count and field characteristics
            features['num_drivers'] = len(lap_data)
            features['drivers_with_valid_times'] = lap_data['LapTime_seconds'].notna().sum() if 'LapTime_seconds' in lap_data.columns else 0

            # Position-related features (if available)
            if 'Position' in lap_data.columns:
                positions = lap_data['Position'].dropna()
                if len(positions) > 0:
                    features['position_spread'] = positions.max() - positions.min()
                    features['avg_position'] = positions.mean()

            # Tire age features (if available)
            if 'TyreLife' in lap_data.columns:
                tyre_life = lap_data['TyreLife'].dropna()
                if len(tyre_life) > 0:
                    features['avg_tyre_life'] = tyre_life.mean()
                    features['max_tyre_life'] = tyre_life.max()
                    features['tyre_life_spread'] = tyre_life.max() - tyre_life.min()

            # Stint-related features (if available)
            if 'Stint' in lap_data.columns:
                stint_data = lap_data['Stint'].dropna()
                if len(stint_data) > 0:
                    features['avg_stint'] = stint_data.mean()
                    features['stint_variety'] = stint_data.nunique()

            # Speed-related features (if available)
            speed_columns = [col for col in lap_data.columns if 'Speed' in col]
            for speed_col in speed_columns:
                speed_data = lap_data[speed_col].dropna()
                if len(speed_data) > 0:
                    features[f'{speed_col.lower()}_mean'] = speed_data.mean()
                    features[f'{speed_col.lower()}_std'] = speed_data.std() if len(speed_data) > 1 else 0
                    features[f'{speed_col.lower()}_range'] = speed_data.max() - speed_data.min()

            lap_features.append(features)

        return pd.DataFrame(lap_features)

    def create_predictive_features(self, basic_features_df: pd.DataFrame, target_lap: int) -> pd.DataFrame:
        """
        Create predictive features focused on sudden change detection.

        This method applies our key insight that racing incidents are more likely
        caused by sudden, unexpected changes rather than gradual escalation.

        Args:
            basic_features_df: DataFrame with basic features per lap
            target_lap: The lap number that we're trying to predict (safety car lap)

        Returns:
            DataFrame with sudden change detection features
        """

        df = basic_features_df.copy()

        # Get numeric columns for feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        risk_features = [col for col in numeric_cols if col not in ['LapNumber']]

        # 1. Sudden spike detection
        for feature in risk_features:
            if feature in df.columns:
                values = df[feature].fillna(0)

                # Spike ratio: current value vs recent 3-lap average
                recent_baseline = values.rolling(window=3, min_periods=1).mean().shift(1)
                df[f'{feature}_spike_ratio'] = values / (recent_baseline + 0.001)

                # Lap-to-lap jump
                df[f'{feature}_jump'] = values.diff()
                df[f'{feature}_jump_pct'] = values.pct_change().fillna(0)

                # Z-score (how extreme is current value?)
                if values.std() > 0:
                    df[f'{feature}_zscore'] = (values - values.mean()) / values.std()
                else:
                    df[f'{feature}_zscore'] = 0

        # 2. Threshold crossing indicators
        for feature in risk_features:
            if feature in df.columns:
                values = df[feature].fillna(0)

                # Percentile ranking
                df[f'{feature}_percentile'] = values.rank(pct=True)

                # Above historical thresholds
                if len(values) > 1:
                    p75 = values.quantile(0.75)
                    p90 = values.quantile(0.90)
                    df[f'{feature}_above_p75'] = (values > p75).astype(int)
                    df[f'{feature}_above_p90'] = (values > p90).astype(int)

        # 3. Multi-feature risk indicators
        spike_features = [col for col in df.columns if '_spike_ratio' in col]
        if spike_features:
            # Count simultaneous spikes (>1.5x recent average)
            spike_flags = df[spike_features] > 1.5
            df['simultaneous_spikes'] = spike_flags.sum(axis=1)

        threshold_features = [col for col in df.columns if '_above_p75' in col]
        if threshold_features:
            # Count concurrent risk factors
            df['concurrent_risk_factors'] = df[threshold_features].sum(axis=1)

        # 4. Create target variable
        df['sc_next_lap'] = 0
        final_lap = target_lap - 1  # Last lap before safety car
        df.loc[df['LapNumber'] == final_lap, 'sc_next_lap'] = 1

        return df

    def train_model(self, training_data: List[pd.DataFrame]) -> None:
        """
        Train a safety car prediction model on multiple training windows.

        Args:
            training_data: List of DataFrames, each containing features for one window
        """
        print("\n=== TRAINING SAFETY CAR PREDICTION MODEL ===")

        if not training_data:
            print("No training data available")
            return

        # Combine all training windows
        all_features = pd.concat(training_data, ignore_index=True)

        # Prepare features and target
        exclude_cols = ['LapNumber', 'sc_next_lap']
        feature_cols = [col for col in all_features.columns if col not in exclude_cols]

        X = all_features[feature_cols].fillna(0)
        y = all_features['sc_next_lap']

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Features: {len(feature_cols)}")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )

        self.model.fit(X_scaled, y)
        self.feature_names = feature_cols

        # Evaluate training performance
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        print(f"\nTraining Performance:")
        print(f"Accuracy: {(y_pred == y).mean():.3f}")

        if len(np.unique(y)) > 1:
            print(f"ROC AUC: {roc_auc_score(y, y_prob):.3f}")

        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature'][:40]:40} | {row['importance']:.3f}")

    def analyze_predictions(self, training_data: List[pd.DataFrame]) -> None:
        """
        Analyze model predictions on training data to understand what it learned.

        Args:
            training_data: List of DataFrames containing training windows
        """
        print("\n=== PREDICTION ANALYSIS ===")

        if self.model is None:
            print("No trained model available")
            return

        for i, window_df in enumerate(training_data):
            sc_lap = window_df[window_df['sc_next_lap'] == 1]['LapNumber'].iloc[0] + 1 if (window_df['sc_next_lap'] == 1).any() else "Unknown"

            print(f"\nWindow {i+1} (Safety Car on lap {sc_lap}):")
            print("-" * 40)

            # Prepare features
            exclude_cols = ['LapNumber', 'sc_next_lap']
            feature_cols = [col for col in window_df.columns if col not in exclude_cols and col in self.feature_names]

            if not feature_cols:
                print("  No matching features found")
                continue

            X_window = window_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = self.scaler.transform(X_window)

            # Get predictions
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = self.model.predict(X_scaled)

            # Show results by lap
            for j, (_, row) in enumerate(window_df.iterrows()):
                lap_num = row['LapNumber']
                prob = probabilities[j]
                pred = predictions[j]
                actual = row['sc_next_lap']

                status = "🚨 SC PREDICTED" if pred == 1 else "✓ Normal"
                actual_status = "(ACTUAL SC NEXT)" if actual == 1 else ""

                print(f"  Lap {int(lap_num):2d}: {status} | Probability: {prob:.3f} {actual_status}")

def run_complete_analysis(session: Session, window_size: int = 5):
    """
    Run the complete safety car prediction analysis pipeline.

    This function orchestrates the entire process from finding safety car deployments
    to training and evaluating prediction models.

    Args:
        session: FastF1 Session object
        window_size: Number of laps to analyze before each safety car

    Returns:
        Trained SafetyCarPredictor instance
    """

    print("Starting complete safety car prediction analysis...")
    print("="*60)

    # Initialize predictor
    predictor = SafetyCarPredictor(window_size=window_size)

    laps_df = session.laps
    track_status_df = session.track_status

    # Step 1: Find safety car deployments
    sc_laps = predictor.find_safety_car_laps(laps_df, track_status_df)

    if not sc_laps:
        print("No safety car deployments found. Analysis cannot continue.")
        return predictor

    # Step 2: Create training windows
    windows = predictor.create_training_windows(laps_df, sc_laps)

    if not windows:
        print("No valid training windows created. Analysis cannot continue.")
        return predictor

    # Step 3: Extract features for each window
    training_data = []

    for window in windows:
        print(f"\nProcessing window for safety car on lap {window['sc_lap']}...")

        # Extract basic features
        basic_features = predictor.extract_basic_features(window['laps_data'])

        if basic_features.empty:
            print("  No features extracted, skipping window")
            continue

        # Create predictive features
        predictive_features = predictor.create_predictive_features(basic_features, window['sc_lap'])

        if not predictive_features.empty:
            training_data.append(predictive_features)
            print(f"  Created {len(predictive_features)} feature rows")
        else:
            print("  No predictive features created, skipping window")

    if not training_data:
        print("No training data available. Analysis cannot continue.")
        return predictor

    # Step 4: Train model
    predictor.train_model(training_data)

    # Step 5: Analyze predictions
    predictor.analyze_predictions(training_data)

    print("\n" + "="*60)
    print("Analysis complete!")

    return predictor

# Example usage:
# predictor = run_complete_analysis(session, window_size=5)