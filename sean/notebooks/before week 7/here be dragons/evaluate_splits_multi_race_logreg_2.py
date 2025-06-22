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
                
                # Get balanced windows using new approach
                curr_driver_temporal_blocks = get_balanced_sc_windows(driver_data)
                
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

def get_balanced_sc_windows(driver_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get balanced temporal windows around SC events using the new approach:
    - All windows containing SC events (n windows)
    - n windows right before SC events start
    - n windows right after SC events end
    
    This creates a 1:2 ratio (SC:non-SC) for better balance.
    
    Args:
        driver_data: DataFrame with temporal windows for one driver
        
    Returns:
        DataFrame with balanced windows around SC events
    """
    if driver_data.empty:
        return pd.DataFrame()
    
    # Sort by prediction time
    driver_data = driver_data.sort_values('prediction_time').reset_index(drop=True)
    
    # Find all SC event windows
    sc_windows = driver_data[driver_data['has_sc_event'] == True].copy()
    n_sc_windows = len(sc_windows)
    
    if n_sc_windows == 0:
        # If no SC events for this driver, return some random normal windows
        normal_windows = driver_data[driver_data['has_sc_event'] == False]
        if len(normal_windows) > 10:
            return normal_windows.sample(n=10, random_state=42)
        else:
            return normal_windows
    
    print(f"      Driver has {n_sc_windows} SC windows, collecting {n_sc_windows * 3} total windows")
    
    selected_windows = []
    
    # 1. Add all SC event windows
    selected_windows.append(sc_windows)
    
    # 2. Get n windows right BEFORE SC events start
    before_windows = []
    sc_indices = sc_windows.index.tolist()
    
    for sc_idx in sc_indices:
        # Find the start of this SC event sequence
        sc_start_idx = sc_idx
        
        # Look backwards to find the actual start of the SC sequence
        while (sc_start_idx > 0 and 
               driver_data.iloc[sc_start_idx - 1]['has_sc_event'] == True):
            sc_start_idx -= 1
        
        # Now collect windows before this SC sequence starts
        before_start = max(0, sc_start_idx - n_sc_windows)
        before_end = sc_start_idx
        
        if before_end > before_start:
            before_segment = driver_data.iloc[before_start:before_end]
            before_windows.append(before_segment)
    
    if before_windows:
        before_combined = pd.concat(before_windows, ignore_index=True)
        # Remove duplicates and take up to n windows
        before_combined = before_combined.drop_duplicates(subset=['prediction_time'])
        if len(before_combined) > n_sc_windows:
            before_combined = before_combined.tail(n_sc_windows)  # Take most recent before windows
        selected_windows.append(before_combined)
    
    # 3. Get n windows right AFTER SC events end
    after_windows = []
    
    for sc_idx in sc_indices:
        # Find the end of this SC event sequence
        sc_end_idx = sc_idx
        
        # Look forwards to find the actual end of the SC sequence
        while (sc_end_idx < len(driver_data) - 1 and 
               driver_data.iloc[sc_end_idx + 1]['has_sc_event'] == True):
            sc_end_idx += 1
        
        # Now collect windows after this SC sequence ends
        after_start = sc_end_idx + 1
        after_end = min(len(driver_data), after_start + n_sc_windows)
        
        if after_start < len(driver_data) and after_end > after_start:
            after_segment = driver_data.iloc[after_start:after_end]
            after_windows.append(after_segment)
    
    if after_windows:
        after_combined = pd.concat(after_windows, ignore_index=True)
        # Remove duplicates and take up to n windows
        after_combined = after_combined.drop_duplicates(subset=['prediction_time'])
        if len(after_combined) > n_sc_windows:
            after_combined = after_combined.head(n_sc_windows)  # Take earliest after windows
        selected_windows.append(after_combined)
    
    # Combine all selected windows
    if selected_windows:
        result = pd.concat(selected_windows, ignore_index=True)
        # Remove any duplicates based on prediction time
        result = result.drop_duplicates(subset=['driver', 'prediction_time']).reset_index(drop=True)
        
        # Verify the balance
        sc_count = result['has_sc_event'].sum()
        total_count = len(result)
        print(f"      Selected {total_count} windows: {sc_count} SC ({sc_count/total_count:.1%}), {total_count-sc_count} normal")
        
        return result
    else:
        return pd.DataFrame()

def collect_season_temporal_blocks_balanced(year: int, max_races: int = None, grand_prix_only: bool = True) -> pd.DataFrame:
    """
    Collect balanced temporal blocks from all races in a season using the new approach.
    
    This creates a more balanced dataset by collecting:
    - All SC event windows
    - Equal number of windows before SC events
    - Equal number of windows after SC events
    
    Args:
        year: F1 season year
        max_races: Limit number of races (for testing)
        grand_prix_only: Only include Grand Prix events
        
    Returns:
        DataFrame with balanced temporal blocks from the season
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
            
            # Check if this race has any SC events
            race_sc_count = race_splits_with_sc_markers_df['has_sc_event'].sum()
            if race_sc_count == 0:
                print(f"   ⚠️ No SC events in {race_name}, skipping")
                continue
            
            race_temporal_blocks = pd.DataFrame()
            
            # Loop over all drivers in race
            drivers = race_splits_with_sc_markers_df['driver'].unique()
            print(f"   Processing {len(drivers)} drivers...")
            
            for driver in drivers:
                driver_data = race_splits_with_sc_markers_df[
                    race_splits_with_sc_markers_df['driver'] == driver
                ]
                
                # Get balanced windows using new approach
                curr_driver_temporal_blocks = get_balanced_sc_windows(driver_data)
                
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
            race_sc_total = race_temporal_blocks['has_sc_event'].sum()
            race_total = len(race_temporal_blocks)
            print(f"   ✅ Added {race_total} balanced blocks from {race_name} ({race_sc_total} SC, {race_sc_total/race_total:.1%} positive)")
            
        except Exception as e:
            print(f"   ❌ Error processing {race_name}: {e}")
            continue
    
    if not entire_season_temporal_blocks.empty:
        total_sc = entire_season_temporal_blocks['has_sc_event'].sum()
        total_samples = len(entire_season_temporal_blocks)
        
        print(f"\n🏆 Balanced season collection complete!")
        print(f"   Races processed: {races_processed}")
        print(f"   Total temporal blocks: {total_samples}")
        print(f"   SC events: {total_sc} ({total_sc/total_samples:.1%} positive class)")
        print(f"   Normal events: {total_samples - total_sc} ({(total_samples - total_sc)/total_samples:.1%})")
    
    return entire_season_temporal_blocks

def demo_balanced_approach():
    """
    Demo the new balanced season collection approach.
    """
    print("🚀 BALANCED SEASON COLLECTION DEMO")
    print("="*50)
    
    # Collect balanced data from races with SC events
    season_data = collect_season_temporal_blocks_balanced(2024, max_races=10)
    
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
        print(f"\n🧪 TESTING BALANCED TRAINING...")
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
            
            # Feature importance analysis
            print(f"\n🔍 TOP 10 FEATURE IMPORTANCE")
            print("-"*40)
            feature_importance = model.get_feature_importance()
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<30} | {row['importance']:.3f}")
            
            return season_data, train_data, test_data, model, predictions
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    return season_data, train_data, test_data

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
    # Use the new balanced approach by default
    demo_balanced_approach()