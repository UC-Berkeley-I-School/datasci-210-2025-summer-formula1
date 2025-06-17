"""
Simple Season Data Collector - Clean implementation following user's pseudo code
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from poc_3 import TemporalSafetyCarSystem
import warnings
warnings.filterwarnings('ignore')

def collect_season_temporal_blocks(year: int, max_races: int = None, grand_prix_only: bool = True) -> pd.DataFrame:
    """
    Collect temporal blocks from all races in a season.
    Following the user's pseudo code structure.
    
    Args:
        year: F1 season year
        max_races: Limit number of races (for testing)
        grand_prix_only: Only include Grand Prix events (excludes Sprint races, etc.)
        
    Returns:
        DataFrame with all temporal blocks from the season
    """
    try:
        import fastf1 as f1
    except ImportError:
        print("FastF1 not available")
        return pd.DataFrame()
    
    entire_season_temporal_blocks = pd.DataFrame()
    
    # Get season schedule
    schedule = f1.get_event_schedule(year)
    races_processed = 0
    
    # Loop over all races in season
    for _, event in schedule.iterrows():
        if max_races and races_processed >= max_races:
            break
            
        race_name = event['EventName']
        
        # Filter for Grand Prix events only if requested
        if grand_prix_only and 'Grand Prix' not in race_name:
            print(f"   ⏭️ Skipping {race_name} (not a Grand Prix)")
            continue
            
        print(f"\n🏁 Processing {race_name}...")
        
        try:
            # Load race session
            race = f1.get_session(year, race_name, 'R')
            race.load()
            
            # Get temporal splits for this race
            race_splits_df = temporal_split(race)
            
            if race_splits_df.empty:
                print(f"   ⚠️ No temporal splits for {race_name}")
                continue
            
            # Add SC event markers
            race_splits_with_sc_markers_df = locate_sc_event_in_splits(race_splits_df, race)
            
            race_temporal_blocks = pd.DataFrame()
            
            # Loop over all drivers in race
            drivers = race_splits_with_sc_markers_df['driver'].unique()
            for driver in drivers:
                driver_data = race_splits_with_sc_markers_df[
                    race_splits_with_sc_markers_df['driver'] == driver
                ]
                
                # Get ±5 temporal windows surrounding SC events
                curr_driver_temporal_blocks = get_plus_or_minus_five_temporal_windows_surrounding_the_sc_event(
                    driver_data
                )
                
                if not curr_driver_temporal_blocks.empty:
                    # Add race indicator columns
                    curr_driver_temporal_blocks = add_race_indicator_columns(
                        curr_driver_temporal_blocks, race_name, year
                    )
                    
                    # Append to race blocks
                    race_temporal_blocks = pd.concat([race_temporal_blocks, curr_driver_temporal_blocks], 
                                                   ignore_index=True)
            
            # Append to season blocks
            entire_season_temporal_blocks = pd.concat([entire_season_temporal_blocks, race_temporal_blocks], 
                                                    ignore_index=True)
            
            races_processed += 1
            print(f"   ✅ Added {len(race_temporal_blocks)} temporal blocks from {race_name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {race_name}: {e}")
            continue
    
    print(f"\n🏆 Season collection complete!")
    print(f"   Races processed: {races_processed}")
    print(f"   Total temporal blocks: {len(entire_season_temporal_blocks)}")
    print(f"   SC events: {entire_season_temporal_blocks['has_sc_event'].sum() if 'has_sc_event' in entire_season_temporal_blocks.columns else 'Unknown'}")
    
    return entire_season_temporal_blocks

def temporal_split(race) -> pd.DataFrame:
    """
    Create temporal splits for a race using the existing system.
    
    Args:
        race: FastF1 Session object
        
    Returns:
        DataFrame with temporal feature windows
    """
    # Use existing system to create temporal windows
    system = TemporalSafetyCarSystem()
    
    # Find safety car events
    events = system.data_processor.find_safety_car_events(race)
    
    # Create temporal windows
    windows = system.data_processor.create_temporal_windows(race, events)
    
    # Extract features
    temporal_features = []
    for window in windows:
        features = system.feature_engineer.extract_features(window)
        if not features.empty:
            temporal_features.append(features)
    
    if temporal_features:
        return pd.concat(temporal_features, ignore_index=True)
    else:
        return pd.DataFrame()

def locate_sc_event_in_splits(race_splits_df: pd.DataFrame, race) -> pd.DataFrame:
    """
    Add SC event markers to temporal splits.
    
    Args:
        race_splits_df: DataFrame with temporal splits
        race: FastF1 Session object
        
    Returns:
        DataFrame with SC event markers added
    """
    # Copy the dataframe
    marked_df = race_splits_df.copy()
    
    # The sc_in_prediction_window column already exists from the temporal system
    # Just rename it for clarity
    if 'sc_in_prediction_window' in marked_df.columns:
        marked_df['has_sc_event'] = marked_df['sc_in_prediction_window']
    else:
        marked_df['has_sc_event'] = False
    
    return marked_df

def get_plus_or_minus_five_temporal_windows_surrounding_the_sc_event(driver_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get ±5 temporal windows around SC events for a driver.
    
    Args:
        driver_data: DataFrame with temporal windows for one driver
        
    Returns:
        DataFrame with windows around SC events plus some normal windows
    """
    if driver_data.empty:
        return pd.DataFrame()
    
    # Sort by prediction time
    driver_data = driver_data.sort_values('prediction_time').reset_index(drop=True)
    
    selected_windows = []
    
    # Find SC events
    sc_indices = driver_data[driver_data['has_sc_event'] == True].index.tolist()
    
    # For each SC event, get ±5 windows
    for sc_idx in sc_indices:
        start_idx = max(0, sc_idx - 5)
        end_idx = min(len(driver_data), sc_idx + 6)  # +6 to include sc_idx and 5 after
        
        # Get the window
        window_data = driver_data.iloc[start_idx:end_idx].copy()
        selected_windows.append(window_data)
    
    # Also add some random normal windows for balance
    normal_windows = driver_data[driver_data['has_sc_event'] == False]
    if len(normal_windows) > 0:
        # Sample 10 random normal windows (or all if fewer than 10)
        n_sample = min(10, len(normal_windows))
        random_normal = normal_windows.sample(n=n_sample, random_state=42)
        selected_windows.append(random_normal)
    
    # Combine all selected windows
    if selected_windows:
        result = pd.concat(selected_windows, ignore_index=True)
        # Remove duplicates (in case windows overlap)
        result = result.drop_duplicates(subset=['driver', 'prediction_time']).reset_index(drop=True)
        return result
    else:
        return pd.DataFrame()

def add_race_indicator_columns(temporal_blocks: pd.DataFrame, race_name: str, year: int) -> pd.DataFrame:
    """
    Add race identifier columns to temporal blocks.
    
    Args:
        temporal_blocks: DataFrame with temporal blocks
        race_name: Name of the race
        year: Year of the season
        
    Returns:
        DataFrame with race indicators added
    """
    temporal_blocks = temporal_blocks.copy()
    temporal_blocks['race_name'] = race_name
    temporal_blocks['season_year'] = year
    temporal_blocks['race_id'] = f"{year}_{race_name.replace(' ', '_')}"
    
    return temporal_blocks

def simple_train_test_split(season_data: pd.DataFrame, test_race_ratio: float = 0.3) -> tuple:
    """
    Simple train/test split by races.
    
    Args:
        season_data: DataFrame with season temporal blocks
        test_race_ratio: Proportion of races to use for testing
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if season_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get unique races
    races = season_data['race_name'].unique()
    n_test_races = max(1, int(len(races) * test_race_ratio))
    
    # Randomly select test races
    np.random.seed(42)
    test_races = np.random.choice(races, size=n_test_races, replace=False)
    
    # Split data
    test_data = season_data[season_data['race_name'].isin(test_races)].copy()
    train_data = season_data[~season_data['race_name'].isin(test_races)].copy()
    
    print(f"\n🔀 Train/Test Split:")
    print(f"   Train races: {list(train_data['race_name'].unique())}")
    print(f"   Test races: {list(test_races)}")
    print(f"   Train samples: {len(train_data)} ({train_data['has_sc_event'].sum()} SC events)")
    print(f"   Test samples: {len(test_data)} ({test_data['has_sc_event'].sum()} SC events)")
    
    return train_data, test_data

def analyze_season_data(season_data: pd.DataFrame):
    """
    Analyze the collected season data.
    
    Args:
        season_data: DataFrame with season temporal blocks
    """
    if season_data.empty:
        print("No season data to analyze")
        return
    
    print(f"\n📊 SEASON DATA ANALYSIS")
    print("="*50)
    
    print(f"Total samples: {len(season_data)}")
    print(f"SC events: {season_data['has_sc_event'].sum()}")
    print(f"Normal windows: {(~season_data['has_sc_event']).sum()}")
    print(f"Class balance: {season_data['has_sc_event'].mean():.1%} positive")
    
    # By race analysis
    print(f"\nBy Race:")
    race_stats = season_data.groupby('race_name').agg({
        'has_sc_event': ['count', 'sum'],
        'driver': 'nunique'
    }).round(2)
    race_stats.columns = ['total_samples', 'sc_events', 'drivers']
    print(race_stats.to_string())
    
    # By driver analysis
    print(f"\nTop 10 Drivers by SC Events:")
    driver_stats = season_data.groupby('driver')['has_sc_event'].sum().sort_values(ascending=False).head(10)
    print(driver_stats.to_string())

def prepare_data_for_training(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for model training by removing non-numeric columns.
    
    Args:
        data: Raw temporal blocks data
        
    Returns:
        DataFrame ready for model training
    """
    # Remove metadata columns that shouldn't be used for training
    exclude_cols = ['race_name', 'season_year', 'race_id', 'driver', 'prediction_time', 
                   'window_start', 'window_end']
    
    training_data = data.copy()
    
    # Rename has_sc_event to sc_in_prediction_window for compatibility with existing trainer
    if 'has_sc_event' in training_data.columns and 'sc_in_prediction_window' not in training_data.columns:
        training_data['sc_in_prediction_window'] = training_data['has_sc_event']
    
    # Keep only numeric columns and the target
    feature_cols = [col for col in training_data.columns if col not in exclude_cols + ['has_sc_event']]
    
    # Select only the training columns that exist
    available_cols = [col for col in feature_cols if col in training_data.columns]
    training_data = training_data[available_cols]
    
    # Fill NaN values and handle infinite values
    training_data = training_data.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"   Prepared training data: {training_data.shape}")
    print(f"   Features: {[col for col in training_data.columns if col != 'sc_in_prediction_window']}")
    print(f"   Target column: sc_in_prediction_window")
    
    return training_data

def visualize_season_data(season_data: pd.DataFrame, figsize=(15, 12)):
    """
    Create comprehensive visualizations of the season data.
    
    Args:
        season_data: DataFrame with season temporal blocks
        figsize: Figure size for plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for visualizations")
        return
    
    if season_data.empty:
        print("No data to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Season Safety Car Data Analysis', fontsize=16)
    
    # 1. Class distribution pie chart
    ax1 = axes[0, 0]
    class_counts = season_data['has_sc_event'].value_counts()
    ax1.pie(class_counts.values, 
            labels=['Normal', 'Safety Car'], 
            autopct='%1.1f%%',
            colors=['lightblue', 'red'])
    ax1.set_title('Overall Class Distribution')
    
    # 2. SC events by race
    ax2 = axes[0, 1]
    race_sc_counts = season_data.groupby('race_name')['has_sc_event'].sum().sort_values(ascending=True)
    race_sc_counts.plot(kind='barh', ax=ax2, color='orange')
    ax2.set_title('SC Events by Race')
    ax2.set_xlabel('Number of SC Events')
    
    # 3. Samples by race
    ax3 = axes[0, 2]
    race_total_counts = season_data['race_name'].value_counts().sort_values(ascending=True)
    race_total_counts.plot(kind='barh', ax=ax3, color='green')
    ax3.set_title('Total Samples by Race')
    ax3.set_xlabel('Number of Samples')
    
    # 4. SC events by driver (top 15)
    ax4 = axes[1, 0]
    driver_sc_counts = season_data.groupby('driver')['has_sc_event'].sum().sort_values(ascending=False).head(15)
    driver_sc_counts.plot(kind='bar', ax=ax4, color='purple')
    ax4.set_title('SC Events by Driver (Top 15)')
    ax4.set_xlabel('Driver')
    ax4.set_ylabel('SC Events')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Class balance by race
    ax5 = axes[1, 1]
    race_balance = season_data.groupby('race_name').agg({
        'has_sc_event': ['sum', 'count']
    })
    race_balance.columns = ['sc_events', 'total']
    race_balance['sc_rate'] = race_balance['sc_events'] / race_balance['total']
    race_balance['sc_rate'].plot(kind='bar', ax=ax5, color='red', alpha=0.7)
    ax5.set_title('SC Event Rate by Race')
    ax5.set_ylabel('SC Event Rate')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Distribution of key features (if available)
    ax6 = axes[1, 2]
    numeric_cols = season_data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['has_sc_event', 'season_year'] and 'laptime' in col.lower()]
    
    if len(feature_cols) > 0:
        feature_col = feature_cols[0]  # Take first lap time feature
        season_data.boxplot(column=feature_col, by='has_sc_event', ax=ax6)
        ax6.set_title(f'{feature_col} by SC Event')
        ax6.set_xlabel('Has SC Event')
    else:
        ax6.text(0.5, 0.5, 'No suitable features\nfor distribution plot', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('Feature Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def detailed_evaluation(y_true, y_pred, predictions_list=None):
    """
    Perform detailed evaluation with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        predictions_list: List of prediction objects (optional)
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, classification_report)
    
    print(f"\n📊 DETAILED EVALUATION RESULTS")
    print("="*50)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n🎯 CONFUSION MATRIX")
    print(f"True Positives (Correct SC warnings):   {tp:4d}")
    print(f"False Positives (False alarms):         {fp:4d}")
    print(f"True Negatives (Correct normal):        {tn:4d}")
    print(f"False Negatives (Missed SC events):     {fn:4d}")
    
    # Calculate rates
    if tp + fn > 0:
        miss_rate = fn / (tp + fn)
        print(f"\nMiss Rate (% of SC events missed):      {miss_rate:.1%}")
    
    if fp + tn > 0:
        false_alarm_rate = fp / (fp + tn)
        print(f"False Alarm Rate (% false alarms):     {false_alarm_rate:.1%}")
    
    # Detailed classification report
    print(f"\n📋 CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Safety Car']))
    
    # If we have prediction objects, analyze probabilities
    if predictions_list:
        probabilities = [p.probability for p in predictions_list]
        
        # Separate by actual outcome (using prediction target since we don't have real events)
        sc_indices = [i for i, val in enumerate(y_true) if val == 1]
        normal_indices = [i for i, val in enumerate(y_true) if val == 0]
        
        sc_probs = [probabilities[i] for i in sc_indices] if sc_indices else []
        normal_probs = [probabilities[i] for i in normal_indices] if normal_indices else []
        
        print(f"\n🎲 PROBABILITY ANALYSIS")
        if sc_probs:
            print(f"Average probability for SC events:      {np.mean(sc_probs):.3f}")
        if normal_probs:
            print(f"Average probability for normal events:  {np.mean(normal_probs):.3f}")
        
        # Probability distribution analysis
        print(f"Overall probability range:              {np.min(probabilities):.3f} - {np.max(probabilities):.3f}")
        print(f"Median probability:                     {np.median(probabilities):.3f}")
        
        # Count high confidence predictions
        high_conf_positive = sum(1 for p in probabilities if p > 0.7)
        high_conf_negative = sum(1 for p in probabilities if p < 0.3)
        print(f"High confidence positive (>0.7):        {high_conf_positive} ({high_conf_positive/len(probabilities):.1%})")
        print(f"High confidence negative (<0.3):        {high_conf_negative} ({high_conf_negative/len(probabilities):.1%})")

def demo_simple_approach():
    """
    Demo the simple season collection approach with full validation.
    """
    print("🚀 SIMPLE SEASON COLLECTION DEMO")
    print("="*50)
    
    # Collect data from first 10 races of 2024 season
    season_data = collect_season_temporal_blocks(2024, max_races=10)
    
    if season_data.empty:
        print("No data collected")
        return
    
    # Analyze the data
    analyze_season_data(season_data)
    
    # Visualize the data
    print(f"\n📊 Creating visualizations...")
    try:
        visualize_season_data(season_data)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Create train/test split
    train_data, test_data = simple_train_test_split(season_data)
    
    # Test training
    if len(train_data) > 0 and len(test_data) > 0:
        print(f"\n🧪 TESTING TRAINING...")
        try:
            from poc_3 import TemporalModelTrainer
            
            # Prepare data for training (remove non-numeric columns)
            train_prepared = prepare_data_for_training(train_data)
            test_prepared = prepare_data_for_training(test_data)
            
            print(f"Training features: {train_prepared.shape[1] - 1}")  # -1 for target column
            
            trainer = TemporalModelTrainer()
            model = trainer.train([train_prepared])
            
            print("✅ Model training successful!")
            
            # Test prediction - use original test data (with metadata) for prediction
            # The model will internally select only the feature columns it needs
            events = []
            
            # Add the target column to original test data for compatibility
            test_data_for_prediction = test_data.copy()
            if 'sc_in_prediction_window' not in test_data_for_prediction.columns:
                test_data_for_prediction['sc_in_prediction_window'] = test_data_for_prediction['has_sc_event']
            
            predictions = model.predict(test_data_for_prediction, events)
            
            # Quick evaluation - use the correct target column
            target_col = 'sc_in_prediction_window' if 'sc_in_prediction_window' in test_prepared.columns else 'has_sc_event'
            y_true = test_prepared[target_col].values
            y_pred = [p.prediction for p in predictions]
            
            # Detailed evaluation
            detailed_evaluation(y_true, y_pred, predictions)
            
            return season_data, train_data, test_data, model, predictions
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    return season_data, train_data, test_data

if __name__ == "__main__":
    demo_simple_approach()

# Output:

# 🏆 Season collection complete!
#    Races processed: 10
#    Total temporal blocks: 3078
#    SC events: 230

# 📊 SEASON DATA ANALYSIS
# ==================================================
# Total samples: 3078
# SC events: 230
# Normal windows: 2848
# Class balance: 7.5% positive

# By Race:
#                            total_samples  sc_events  drivers
# race_name
# Australian Grand Prix                185          0       19
# Bahrain Grand Prix                   200          0       20
# Canadian Grand Prix                  597         76       20
# Chinese Grand Prix                   559         76       20
# Emilia Romagna Grand Prix            200          0       20
# Japanese Grand Prix                  182          0       20
# Miami Grand Prix                     439         40       20
# Monaco Grand Prix                    164          0       20
# Saudi Arabian Grand Prix             352         38       20
# Spanish Grand Prix                   200          0       20

# Top 10 Drivers by SC Events:
# driver
# ALB    12
# ALO    12
# VER    12
# TSU    12
# STR    12
# RUS    12
# RIC    12
# PIA    12
# PER    12
# OCO    12

# 📊 Creating visualizations...
# 2025-06-15 11:06:25.690 Python[62781:8955758] +[IMKClient subclass]: chose IMKClient_Modern
# 2025-06-15 11:06:25.690 Python[62781:8955758] +[IMKInputSession subclass]: chose IMKInputSession_Modern

# 🔀 Train/Test Split:
#    Train races: ['Bahrain Grand Prix', 'Australian Grand Prix', 'Japanese Grand Prix', 'Chinese Grand Prix', 'Emilia Romagna Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix']
#    Test races: ['Canadian Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix']
#    Train samples: 1690 (76 SC events)
#    Test samples: 1388 (154 SC events)

# 🧪 TESTING TRAINING...
#    Prepared training data: (1690, 45)
#    Features: ['num_laps_in_window', 'recent_laptime_mean', 'recent_laptime_std', 'recent_laptime_min', 'recent_laptime_max', 'laptime_consistency', 'current_position', 'position_changes', 'avg_position', 'current_tyre_life', 'avg_tyre_life', 'max_tyre_life', 'current_stint', 'stint_changes', 'laptime_trend', 'recent_vs_early_laptime', 'most_recent_laptime_change', 'sudden_laptime_deterioration', 'recent_position_change', 'position_volatility', 'recent_i1speed_mean', 'recent_i1speed_std', 'i1speed_trend', 'recent_i2speed_mean', 'recent_i2speed_std', 'i2speed_trend', 'recent_flspeed_mean', 'recent_flspeed_std', 'flspeed_trend', 'recent_stspeed_mean', 'recent_stspeed_std', 'stspeed_trend', 'recent_s1_mean', 'recent_s1_consistency', 'recent_s2_mean', 'recent_s2_consistency', 'recent_s3_mean', 'recent_s3_consistency', 'recent_s1session_mean', 'recent_s1session_consistency', 'recent_s2session_mean', 'recent_s2session_consistency', 'recent_s3session_mean', 'recent_s3session_consistency']
#    Target column: sc_in_prediction_window
#    Prepared training data: (1388, 45)
#    Features: ['num_laps_in_window', 'recent_laptime_mean', 'recent_laptime_std', 'recent_laptime_min', 'recent_laptime_max', 'laptime_consistency', 'current_position', 'position_changes', 'avg_position', 'current_tyre_life', 'avg_tyre_life', 'max_tyre_life', 'current_stint', 'stint_changes', 'laptime_trend', 'recent_vs_early_laptime', 'most_recent_laptime_change', 'sudden_laptime_deterioration', 'recent_position_change', 'position_volatility', 'recent_i1speed_mean', 'recent_i1speed_std', 'i1speed_trend', 'recent_i2speed_mean', 'recent_i2speed_std', 'i2speed_trend', 'recent_flspeed_mean', 'recent_flspeed_std', 'flspeed_trend', 'recent_stspeed_mean', 'recent_stspeed_std', 'stspeed_trend', 'recent_s1_mean', 'recent_s1_consistency', 'recent_s2_mean', 'recent_s2_consistency', 'recent_s3_mean', 'recent_s3_consistency', 'recent_s1session_mean', 'recent_s1session_consistency', 'recent_s2session_mean', 'recent_s2session_consistency', 'recent_s3session_mean', 'recent_s3session_consistency']
#    Target column: sc_in_prediction_window
# Training features: 44
# ✅ Model training successful!

# 📊 DETAILED EVALUATION RESULTS
# ==================================================
# Accuracy:  0.820
# Precision: 0.038
# Recall:    0.026
# F1 Score:  0.031

# 🎯 CONFUSION MATRIX
# True Positives (Correct SC warnings):      4
# False Positives (False alarms):          100
# True Negatives (Correct normal):        1134
# False Negatives (Missed SC events):      150

# Miss Rate (% of SC events missed):      97.4%
# False Alarm Rate (% false alarms):     8.1%

# 📋 CLASSIFICATION REPORT
#               precision    recall  f1-score   support

#       Normal       0.88      0.92      0.90      1234
#   Safety Car       0.04      0.03      0.03       154

#     accuracy                           0.82      1388
#    macro avg       0.46      0.47      0.47      1388
# weighted avg       0.79      0.82      0.80      1388


# 🎲 PROBABILITY ANALYSIS
# Average probability for SC events:      0.032
# Average probability for normal events:  0.088
# Overall probability range:              0.000 - 1.000
# Median probability:                     0.000
# High confidence positive (>0.7):        94 (6.8%)
# High confidence negative (<0.3):        1271 (91.6%)