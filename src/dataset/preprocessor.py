import fastf1.core
from pandas import DataFrame
from typing import Tuple
from enum import Enum
from dataclasses import dataclass
import fastf1

class BaseFeatures(Enum):
    DRIVER = 'Driver'
    SAFETY_CAR = 'SafetyCar'
    LOCATION = 'Location'
    COMPOUND = 'Compound'


def create_high_freq_dataset(session: fastf1.core.Session, interval_seconds: int = 1.0) -> DataFrame:
    """
    Create high-frequency dataset with fixed time intervals using session time
    """
    laps_df = session.laps

    # work with session timedeltas
    session_start = laps_df['LapStartTime'].min()
    session_end = laps_df['Time'].max()

    # create time grid as seconds from session start
    start_seconds = session_start.total_seconds()
    end_seconds = session_end.total_seconds()

    time_grid_seconds = np.arange(start_seconds, end_seconds, interval_seconds)

    all_driver_data = []

    for driver in laps_df['Driver'].unique():
        driver_laps = laps_df[laps_df['Driver'] == driver].copy()
        driver_telemetry = []

        for _, lap in driver_laps.iterrows():
            try:
                # get telemtry for this lap
                car_data = lap.get_car_data()
                pos_data = lap.get_pos_data()
                weather_data = lap.get_weather_data()
                
                # Convert SessionTime to seconds for easier indexing
                if not car_data.empty:
                    car_data = car_data.copy()
                    car_data['SessionSeconds'] = car_data['SessionTime'].dt.total_seconds()
                    
                if not pos_data.empty:
                    pos_data = pos_data.copy()
                    pos_data['SessionSeconds'] = pos_data['SessionTime'].dt.total_seconds()
                
                lap_static = {
                    'Driver': driver,
                    'LapNumber': lap['LapNumber'],
                    'Compound': lap['Compound'],
                    'TyreLife': lap['TyreLife'],
                    'AirTemp': weather_data['AirTemp'] if pd.notna(weather_data['AirTemp']) else None,
                    'TrackTemp': weather_data['TrackTemp'] if pd.notna(weather_data['TrackTemp']) else None,
                    'Humidity': weather_data['Humidity'] if pd.notna(weather_data['Humidity']) else None,
                }
                
                driver_telemetry.append({
                    'car_data': car_data,
                    'pos_data': pos_data,
                    'lap_static': lap_static,
                    'lap_start_seconds': lap['LapStartTime'].total_seconds(),
                    'lap_end_seconds': lap['Time'].total_seconds()
                })
            
            except Exception as e:
                print(f"Skipping lap {lap['LapNumber']} for {driver}: {e}")
                continue
        
        # Align to time grid for this driver
        driver_grid_data = align_to_time_grid(driver_telemetry, time_grid_seconds)
        all_driver_data.append(driver_grid_data)
    
    # Combine all drivers
    final_df = pd.concat(all_driver_data, ignore_index=True)
    return final_df


def balance_dataset_by_feature(dataset: DataFrame, features: list[BaseFeatures], method='remove_insufficient', min_samples: int = 1000) -> DataFrame:
    """
    Balance dataset by multiple features using specified strategy.
    
    Args:
        dataset: Input DataFrame
        features: List of BaseFeatures to balance on
        method: Balancing strategy ('remove_insufficient', 'undersample_to_min', 'undersample_to_target', 'keep_all')
        min_samples: Minimum samples threshold for balancing decisions
    
    Returns:
        Balanced DataFrame
    """
    if not features:
        return dataset.copy()
    
    # Convert enum values to column names
    feature_cols = [feature.value for feature in features]
    
    # Create combination groups for all specified features
    if len(feature_cols) == 1:
        combo_col = feature_cols[0]
        dataset_copy = dataset.copy()
    else:
        # Create combined feature column for multi-feature balancing
        combo_col = 'FeatureCombination'
        dataset_copy = dataset.copy()
        dataset_copy[combo_col] = dataset_copy[feature_cols].apply(
            lambda row: '_'.join(str(val) for val in row), axis=1
        )
    
    # Get combination counts
    combo_counts = dataset_copy[combo_col].value_counts()
    print(f"Original {'/'.join(feature_cols)} distribution:")
    print(combo_counts)
    
    if method == 'remove_insufficient':
        # Remove combinations with insufficient data
        sufficient_combos = combo_counts[combo_counts >= min_samples].index
        balanced_df = dataset_copy[dataset_copy[combo_col].isin(sufficient_combos)].copy()
        removed_combos = set(combo_counts.index) - set(sufficient_combos)
        if removed_combos:
            print(f"\nRemoved combinations with < {min_samples} samples: {removed_combos}")
        
    elif method == 'undersample_to_min':
        # Undersample all combinations to match smallest viable class
        viable_combos = combo_counts[combo_counts >= min_samples]
        if viable_combos.empty:
            print(f"No combinations have >= {min_samples} samples. Returning empty DataFrame.")
            return DataFrame()
        
        min_viable_count = viable_combos.min()
        balanced_dfs = []
        
        for combo in viable_combos.index:
            combo_data = dataset_copy[dataset_copy[combo_col] == combo]
            sampled_data = combo_data.sample(n=min_viable_count, random_state=42)
            balanced_dfs.append(sampled_data)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"\nUndersampled all viable combinations to {min_viable_count} samples each")
        
    elif method == 'undersample_to_target':
        # Undersample to target amount (default 3000)
        target_samples = 3000
        viable_combos = combo_counts[combo_counts >= min_samples]
        
        balanced_dfs = []
        for combo in viable_combos.index:
            combo_data = dataset_copy[dataset_copy[combo_col] == combo]
            sample_size = min(len(combo_data), target_samples)
            sampled_data = combo_data.sample(n=sample_size, random_state=42)
            balanced_dfs.append(sampled_data)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"\nSampled combinations to max {target_samples} samples each")
        
    elif method == 'keep_all':
        # Keep all combinations but flag the imbalance
        balanced_df = dataset_copy.copy()
        print(f"\nKept all combinations (imbalance preserved)")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Clean up temporary combination column if created
    if len(feature_cols) > 1 and combo_col in balanced_df.columns:
        balanced_df = balanced_df.drop(columns=[combo_col])
    
    print(f"\nFinal dataset shape: {balanced_df.shape}")
    print("Final distribution:")
    
    # Show final distribution for each feature
    for feature_col in feature_cols:
        if feature_col in balanced_df.columns:
            print(f"{feature_col}:")
            print(balanced_df[feature_col].value_counts().sort_values(ascending=False))
            print()
    
    return balanced_df


def add_track_status_labels(dataset: DataFrame, session: fastf1.core.Session) -> DataFrame:
    """
    Add track status labels to the dataset based on session track status
    """
    # Get track status data
    track_status_df = session.track_status.copy()
    
    # Convert track status times to seconds
    track_status_df['StatusTimeSeconds'] = track_status_df['Time'].dt.total_seconds()
    
    # Sort by time to ensure proper forward-fill logic
    track_status_df = track_status_df.sort_values('StatusTimeSeconds').reset_index(drop=True)
    
    print("Track status changes:")
    print(track_status_df[['StatusTimeSeconds', 'Status', 'Message']])
    
    # Add track status to dataset
    dataset_with_status = dataset.copy()
    dataset_with_status['TrackStatus'] = None
    dataset_with_status['TrackStatusMessage'] = None
    
    # For each row, find the most recent track status
    for idx, row in dataset_with_status.iterrows():
        session_time = row['SessionTimeSeconds']
        
        # Find the most recent status change before or at this time
        valid_statuses = track_status_df[track_status_df['StatusTimeSeconds'] <= session_time]
        
        if not valid_statuses.empty:
            latest_status = valid_statuses.iloc[-1]
            dataset_with_status.at[idx, 'TrackStatus'] = latest_status['Status']
            dataset_with_status.at[idx, 'TrackStatusMessage'] = latest_status['Message']
        else:
            # If no status change has occurred yet, assume normal conditions
            dataset_with_status.at[idx, 'TrackStatus'] = '1'  # AllClear
            dataset_with_status.at[idx, 'TrackStatusMessage'] = 'AllClear'
    
    # Create binary safety car label
    dataset_with_status['SafetyCar'] = (dataset_with_status['TrackStatus'] == '4').astype(int)
    
    print(f"\nTrack status distribution:")
    print(dataset_with_status['TrackStatusMessage'].value_counts())
    print(f"\nSafety car samples: {dataset_with_status['SafetyCar'].sum()}")
    print(f"Total samples: {len(dataset_with_status)}")
    print(f"Safety car percentage: {dataset_with_status['SafetyCar'].mean():.2%}")
    
    return dataset_with_status

def add_event_info_columns(dataset: DataFrame, session: fastf1.core.Session) -> DataFrame:
    """
    Add event identification columns to the dataset for multi-race aggregation.
    
    Args:
        labeled_dataset (pd.DataFrame): F1 telemetry dataset
        session: FastF1 session object with session_info attribute
    
    Returns:
        pd.DataFrame: Dataset with added event columns
    """
    # Extract session info
    session_info = session.session_info
    
    # Extract the required fields
    session_name = session_info.get('Meeting', {}).get('Name', 'Unknown')  # e.g., 'Saudi Arabian Grand Prix'
    country_name = session_info.get('Meeting', {}).get('Country', {}).get('Name', 'Unknown')  # e.g., 'Saudi Arabia'
    session_type = session_info.get('Type', 'Unknown')  # e.g., 'Race'
    
    # Create a copy of the dataset to avoid modifying original
    enhanced_dataset = labeled_dataset.copy()
    
    # Add event identification columns
    enhanced_dataset['SessionName'] = session_name  
    enhanced_dataset['Country'] = country_name
    enhanced_dataset['SessionType'] = session_type
    
    # Optional: Add additional useful fields
    enhanced_dataset['StartDate'] = session_info.get('StartDate')
    enhanced_dataset['Location'] = session_info.get('Meeting', {}).get('Location', 'Unknown')
    
    print(f"Added event info columns:")
    print(f"  Session: {session_name}")
    print(f"  Country: {country_name}")
    print(f"  Session Type: {session_type}")
    print(f"  Location: {enhanced_dataset['Location'].iloc[0]}")
    
    return enhanced_dataset


def process_dataset(session: fastf1.core.Session, 
                    interval_seconds: int = 1.0,
                    balance_by: list[BaseFeatures] = [BaseFeatures.DRIVER, BaseFeatures.SAFETY_CAR, BaseFeatures.LOCATION, BaseFeatures.COMPOUND]) -> DataFrame:
    
    if not session:
        raise Exception("``session`` must be defined")
    
    if interval_seconds < 0.0:
        raise Exception("``interval_seconds`` must be a positive number")

    # Slice the dataset up into temporal windows (including getting telemetry for each driver)
    temporal_df = create_high_freq_dataset(session, interval_seconds)
    
    # Balance the dataset
    if len(balance_by) >= 1:
        balanced_df = balance_dataset_by_feature(temporal_df, balance_by)
    else:
        balanced_df = temporal_df
    
    # Add clf labels
    labeled_df = add_track_status_labels(balanced_df, session)

    return add_event_info_columns(labeled_df, session)