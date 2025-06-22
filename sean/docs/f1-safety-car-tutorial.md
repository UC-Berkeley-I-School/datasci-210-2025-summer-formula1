# Predicting Formula 1 Safety Cars: A Temporal Machine Learning Tutorial

## 🏁 Introduction

Welcome to this hands-on tutorial on predicting safety car deployments in Formula 1 using temporal machine learning! In this notebook, we'll build a system that slides through race data minute-by-minute, asking: **"Will a safety car be deployed in the next 2 minutes?"**

### What You'll Learn:
1. How to work with F1 telemetry data using FastF1
2. The concept of temporal sliding windows for time-series prediction
3. Feature engineering from racing telemetry
4. Training a model for imbalanced classification
5. Evaluating predictions with proper temporal validation

### Prerequisites:
- Basic Python knowledge
- Understanding of pandas DataFrames
- Familiarity with basic ML concepts (classification, train/test split)

Let's start our journey!

```python
# Import all necessary libraries
import pandas as pd
import numpy as np
import fastf1 as f1
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For ML
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Set up nice plotting
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
```

## 📊 Chapter 1: Understanding the Data Source

Before we dive into predictions, let's understand what data we're working with. FastF1 provides access to official F1 timing data.

```python
# Load a session with known safety car deployments
# We'll use Saudi Arabia 2024 as our example
session = f1.get_session(2024, 'Saudi Arabian Grand Prix', 'R')
session.load()

print(f"🏎️ Loaded: {session.event['EventName']} - {session.name}")
print(f"📅 Date: {session.date}")
```

### 1.1 Exploring Lap Data

The core of our analysis is lap-by-lap telemetry for each driver:

```python
# Get the laps dataframe
laps_df = session.laps

print(f"Total laps in dataset: {len(laps_df)}")
print(f"Drivers in race: {laps_df['Driver'].nunique()}")
print(f"\nColumns available:")
print(laps_df.columns.tolist())

# Let's look at a sample of the data
print("\nSample lap data:")
laps_df.head()
```

### 1.2 Understanding Timing in F1 Data

**Critical Concept**: F1 uses "SessionTime" - time elapsed since the session started. This is different from wall clock time!

```python
# Examine different time columns
sample_lap = laps_df.iloc[100]

print("🕐 Time columns explained:")
print(f"LapStartTime: {sample_lap['LapStartTime']} (When the lap began)")
print(f"Time: {sample_lap['Time']} (When the lap ended)")
print(f"LapTime: {sample_lap['LapTime']} (Duration of the lap)")

# These are Timedelta objects - let's convert to minutes for clarity
if pd.notna(sample_lap['LapStartTime']):
    start_minutes = sample_lap['LapStartTime'].total_seconds() / 60
    print(f"\nLap started at: {start_minutes:.1f} minutes into the session")
```

### 1.3 Finding Safety Car Events

Safety cars are tracked in the `track_status` data:

```python
# Get track status data
track_status_df = session.track_status

# Status codes in F1:
# '1' = Track Clear (Green flag)
# '2' = Yellow flag
# '4' = Safety Car
# '5' = Red flag

# Find all safety car deployments
safety_cars = track_status_df[track_status_df['Status'] == '4']

print(f"🚨 Found {len(safety_cars)} safety car deployments:")
for _, sc in safety_cars.iterrows():
    # Convert SessionTime to minutes for readability
    time_minutes = sc['Time'].total_seconds() / 60
    print(f"   Safety car at: {time_minutes:.1f} minutes into race")
```

## 📈 Chapter 2: The Temporal Sliding Window Approach

Now comes the key insight: We want to predict safety cars BEFORE they happen. We'll use a sliding window approach.

### 2.1 Conceptual Overview

```python
# Let's visualize the sliding window concept
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Create a timeline
session_duration = 120  # minutes (example)
timeline = np.arange(0, session_duration, 1)

# Example safety car at 45 minutes
sc_time = 45

# Show multiple prediction windows
window_times = [20, 30, 40, 43]  # When we make predictions
lookback = 5  # How far back we look
horizon = 2   # How far ahead we predict

for i, pred_time in enumerate(window_times):
    # Draw lookback window
    ax.barh(i, lookback, left=pred_time-lookback, height=0.8, 
            alpha=0.3, color='blue', label='Lookback' if i==0 else '')
    
    # Draw prediction window
    ax.barh(i, horizon, left=pred_time, height=0.8, 
            alpha=0.3, color='orange', label='Prediction' if i==0 else '')
    
    # Mark prediction time
    ax.plot(pred_time, i, 'ko', markersize=8)
    
    # Check if this window should predict the SC
    if pred_time <= sc_time <= pred_time + horizon:
        ax.text(pred_time + horizon + 1, i, '✓ Should predict SC!', 
                va='center', color='green', fontweight='bold')
    else:
        ax.text(pred_time + horizon + 1, i, '✗ No SC', 
                va='center', color='gray')

# Mark the safety car
ax.axvline(sc_time, color='red', linestyle='--', linewidth=2, label='Safety Car')

ax.set_xlabel('Session Time (minutes)')
ax.set_ylabel('Prediction Windows')
ax.set_title('Sliding Window Approach for Safety Car Prediction')
ax.legend()
ax.set_xlim(0, 60)
plt.tight_layout()
plt.show()
```

### 2.2 Creating Temporal Windows for One Driver

Let's implement this for a single driver first:

```python
# Pick a driver to analyze
driver = 'VER'  # Max Verstappen
driver_laps = laps_df[laps_df['Driver'] == driver].copy()

print(f"🏎️ Analyzing {driver}")
print(f"Total laps: {len(driver_laps)}")

# Create sliding windows
def create_driver_windows(driver_laps, safety_car_times, 
                         time_step=60, lookback=300, horizon=120):
    """
    Create temporal windows for prediction.
    
    Parameters:
    - time_step: How often to predict (seconds)
    - lookback: Historical data window (seconds)
    - horizon: Prediction window (seconds)
    """
    windows = []
    
    # Get session bounds
    session_start = driver_laps['LapStartTime'].min()
    session_end = driver_laps['Time'].max()
    
    # Start when we have enough history
    current_time = session_start + pd.Timedelta(seconds=lookback)
    
    while current_time + pd.Timedelta(seconds=horizon) <= session_end:
        # Define windows
        window_start = current_time - pd.Timedelta(seconds=lookback)
        window_end = current_time
        pred_start = current_time
        pred_end = current_time + pd.Timedelta(seconds=horizon)
        
        # Get laps in lookback window
        window_laps = driver_laps[
            (driver_laps['LapStartTime'] >= window_start) & 
            (driver_laps['LapStartTime'] <= window_end)
        ]
        
        # Check if SC will occur in prediction window
        sc_in_window = any(
            pred_start <= sc_time <= pred_end 
            for sc_time in safety_car_times
        )
        
        if len(window_laps) > 0:  # Only if we have data
            windows.append({
                'window_start': window_start,
                'window_end': window_end,
                'prediction_time': current_time,
                'prediction_end': pred_end,
                'laps_data': window_laps,
                'num_laps': len(window_laps),
                'should_predict_sc': sc_in_window
            })
        
        # Slide forward
        current_time += pd.Timedelta(seconds=time_step)
    
    return windows

# Extract safety car times
sc_times = [sc['Time'] for _, sc in safety_cars.iterrows()]

# Create windows for our driver
driver_windows = create_driver_windows(driver_laps, sc_times)

print(f"\n📊 Created {len(driver_windows)} prediction windows")
print(f"Windows with SC: {sum(w['should_predict_sc'] for w in driver_windows)}")
print(f"Windows without SC: {sum(not w['should_predict_sc'] for w in driver_windows)}")
```

## 🔧 Chapter 3: Feature Engineering

Now we'll extract meaningful features from each temporal window. The key question: **What observable patterns might precede a safety car?**

### 3.1 Basic Lap Time Features

```python
def extract_laptime_features(window_laps):
    """Extract features related to lap times."""
    features = {}
    
    if 'LapTime' not in window_laps.columns or len(window_laps) == 0:
        return features
    
    # Convert lap times to seconds
    lap_times = window_laps['LapTime'].dropna()
    if len(lap_times) == 0:
        return features
    
    # Handle timedelta conversion
    if pd.api.types.is_timedelta64_dtype(lap_times):
        lap_seconds = lap_times.dt.total_seconds()
    else:
        lap_seconds = pd.to_numeric(lap_times, errors='coerce')
    
    lap_seconds = lap_seconds.dropna()
    
    if len(lap_seconds) > 0:
        features['laptime_mean'] = lap_seconds.mean()
        features['laptime_std'] = lap_seconds.std() if len(lap_seconds) > 1 else 0
        features['laptime_min'] = lap_seconds.min()
        features['laptime_max'] = lap_seconds.max()
        
        # Consistency metric - how variable are the lap times?
        if lap_seconds.mean() > 0:
            features['laptime_consistency'] = lap_seconds.std() / lap_seconds.mean()
        else:
            features['laptime_consistency'] = 0
            
    return features

# Test on a window
test_window = driver_windows[50]
laptime_features = extract_laptime_features(test_window['laps_data'])

print("🏁 Lap time features extracted:")
for feature, value in laptime_features.items():
    print(f"   {feature}: {value:.2f}")
```

### 3.2 Trend Detection Features

Are lap times getting worse? This might indicate a problem:

```python
def extract_trend_features(window_laps):
    """Extract features about trends and changes."""
    features = {}
    
    if len(window_laps) < 2:
        return features
    
    # Sort by time
    laps_sorted = window_laps.sort_values('LapStartTime')
    
    # Lap time trend
    if 'LapTime' in laps_sorted.columns:
        lap_times = laps_sorted['LapTime'].dropna()
        if len(lap_times) >= 2:
            # Convert to seconds
            if pd.api.types.is_timedelta64_dtype(lap_times):
                lap_seconds = lap_times.dt.total_seconds()
            else:
                lap_seconds = pd.to_numeric(lap_times, errors='coerce')
            
            lap_seconds = lap_seconds.dropna().values
            
            if len(lap_seconds) >= 2:
                # Linear trend - are times improving (negative) or worsening (positive)?
                x = np.arange(len(lap_seconds))
                if np.var(x) > 0:
                    correlation = np.corrcoef(x, lap_seconds)[0, 1]
                    features['laptime_trend'] = correlation if not np.isnan(correlation) else 0
                
                # Recent vs early performance
                if len(lap_seconds) >= 4:
                    mid = len(lap_seconds) // 2
                    early_mean = lap_seconds[:mid].mean()
                    recent_mean = lap_seconds[mid:].mean()
                    features['recent_vs_early'] = recent_mean - early_mean
                
                # Most recent change
                features['last_lap_delta'] = lap_seconds[-1] - lap_seconds[-2]
    
    return features

# Test trend features
trend_features = extract_trend_features(test_window['laps_data'])

print("📈 Trend features extracted:")
for feature, value in trend_features.items():
    if abs(value) < 0.01:
        print(f"   {feature}: {value:.4f}")
    else:
        print(f"   {feature}: {value:.2f}")
```

### 3.3 Position and Competition Features

```python
def extract_position_features(window_laps):
    """Extract features about race position and changes."""
    features = {}
    
    if 'Position' not in window_laps.columns or len(window_laps) == 0:
        return features
    
    positions = window_laps['Position'].dropna()
    if len(positions) == 0:
        return features
    
    features['current_position'] = positions.iloc[-1]
    features['avg_position'] = positions.mean()
    
    if len(positions) > 1:
        features['position_changes'] = abs(positions.diff()).sum()
        features['position_gained'] = positions.iloc[0] - positions.iloc[-1]
        features['position_volatility'] = positions.std()
    
    return features

# Test position features
position_features = extract_position_features(test_window['laps_data'])

print("🏆 Position features extracted:")
for feature, value in position_features.items():
    print(f"   {feature}: {value:.2f}")
```

### 3.4 Tire and Strategy Features

```python
def extract_tire_features(window_laps):
    """Extract features about tire wear and strategy."""
    features = {}
    
    # Tire age
    if 'TyreLife' in window_laps.columns:
        tyre_life = window_laps['TyreLife'].dropna()
        if len(tyre_life) > 0:
            features['current_tyre_age'] = tyre_life.iloc[-1]
            features['avg_tyre_age'] = tyre_life.mean()
            features['max_tyre_age'] = tyre_life.max()
    
    # Stint information
    if 'Stint' in window_laps.columns:
        stints = window_laps['Stint'].dropna()
        if len(stints) > 0:
            features['current_stint'] = stints.iloc[-1]
            features['stint_changes'] = stints.nunique() - 1
    
    return features

# Test tire features
tire_features = extract_tire_features(test_window['laps_data'])

print("🛞 Tire/Strategy features extracted:")
for feature, value in tire_features.items():
    print(f"   {feature}: {value:.0f}")
```

### 3.5 Combining All Features

```python
def extract_all_features(window):
    """Extract all features from a temporal window."""
    laps_data = window['laps_data']
    
    # Start with metadata
    features = {
        'prediction_time': window['prediction_time'],
        'num_laps_in_window': window['num_laps'],
        'target_sc': int(window['should_predict_sc'])  # Our target variable!
    }
    
    # Add all feature groups
    features.update(extract_laptime_features(laps_data))
    features.update(extract_trend_features(laps_data))
    features.update(extract_position_features(laps_data))
    features.update(extract_tire_features(laps_data))
    
    return features

# Process all windows for our driver
driver_features = []
for window in driver_windows:
    features = extract_all_features(window)
    driver_features.append(features)

# Convert to DataFrame
driver_features_df = pd.DataFrame(driver_features)

print(f"📊 Feature matrix shape: {driver_features_df.shape}")
print(f"\nFeatures extracted:")
for col in driver_features_df.columns:
    if col not in ['prediction_time', 'target_sc']:
        print(f"   - {col}")
```

## 🎯 Chapter 4: Understanding the Target Variable

Let's examine our prediction target more closely:

```python
# Analyze target distribution
target_counts = driver_features_df['target_sc'].value_counts()

plt.figure(figsize=(8, 6))
target_counts.plot(kind='bar')
plt.title(f'Target Distribution for {driver}')
plt.xlabel('Safety Car in Next 2 Minutes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)

# Add percentage labels
total = len(driver_features_df)
for i, count in enumerate(target_counts):
    plt.text(i, count + 1, f'{count}\n({count/total*100:.1f}%)', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"⚠️ Class imbalance: {target_counts[1]/total*100:.1f}% positive samples")
print("This is typical - safety cars are rare events!")
```

## 🤖 Chapter 5: Building the Prediction Model

Now let's train a model! We'll start simple with logistic regression.

### 5.1 Preparing Data for All Drivers

```python
# Process all drivers
all_features = []

for driver in laps_df['Driver'].unique():
    driver_laps = laps_df[laps_df['Driver'] == driver]
    
    # Create windows
    windows = create_driver_windows(driver_laps, sc_times)
    
    # Extract features
    for window in windows:
        features = extract_all_features(window)
        features['driver'] = driver
        all_features.append(features)

# Create full dataset
full_df = pd.DataFrame(all_features)
print(f"🏁 Full dataset shape: {full_df.shape}")
print(f"Drivers included: {full_df['driver'].nunique()}")
```

### 5.2 Feature Preparation

```python
# Separate features and target
feature_cols = [col for col in full_df.columns 
                if col not in ['prediction_time', 'target_sc', 'driver']]

X = full_df[feature_cols].fillna(0)  # Simple imputation
y = full_df['target_sc']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame for easier analysis
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
```

### 5.3 Training the Model

```python
# Train logistic regression with balanced class weights
model = LogisticRegression(
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    max_iter=1000
)

model.fit(X_scaled, y)

print("✅ Model trained!")
print(f"Model coefficients shape: {model.coef_.shape}")
```

### 5.4 Understanding Feature Importance

```python
# Extract feature importance (absolute coefficients)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(model.coef_[0]),
    'coefficient': model.coef_[0]
}).sort_values('importance', ascending=False)

# Visualize top features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)

colors = ['red' if x > 0 else 'blue' for x in top_features['coefficient']]
bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)

plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()

# Add legend
red_patch = plt.Rectangle((0,0),1,1,fc="red", alpha=0.7)
blue_patch = plt.Rectangle((0,0),1,1,fc="blue", alpha=0.7)
plt.legend([red_patch, blue_patch], 
           ['Positive (increases SC probability)', 'Negative (decreases SC probability)'],
           loc='lower right')

plt.tight_layout()
plt.show()

print("🔍 Top 5 features increasing safety car probability:")
for _, row in feature_importance[feature_importance['coefficient'] > 0].head(5).iterrows():
    print(f"   {row['feature']}: +{row['coefficient']:.3f}")

print("\n🔍 Top 5 features decreasing safety car probability:")
for _, row in feature_importance[feature_importance['coefficient'] < 0].head(5).iterrows():
    print(f"   {row['feature']}: {row['coefficient']:.3f}")
```

## 📊 Chapter 6: Model Evaluation

### 6.1 Making Predictions

```python
# Make predictions
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Add predictions back to our dataframe
full_df['predicted_sc'] = y_pred
full_df['sc_probability'] = y_pred_proba

print(f"Predictions made: {len(y_pred)}")
print(f"Predicted safety cars: {sum(y_pred)}")
print(f"Actual safety cars: {sum(y)}")
```

### 6.2 Confusion Matrix Analysis

```python
# Create confusion matrix
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No SC', 'SC'], 
            yticklabels=['No SC', 'SC'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Add detailed annotations
for i in range(2):
    for j in range(2):
        if i == j:
            label = "Correct" if i == 0 else "Correct Warning"
        else:
            label = "Missed Event" if i == 1 else "False Alarm"
        plt.text(j + 0.5, i + 0.7, label, 
                ha='center', va='center', fontsize=9, color='darkred')

plt.tight_layout()
plt.show()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"📈 Model Performance:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   Precision: {precision:.1%} (When we predict SC, how often are we right?)")
print(f"   Recall: {recall:.1%} (What % of actual SCs did we predict?)")
print(f"   F1 Score: {f1:.3f}")
```

### 6.3 Analyzing Predictions by Driver

```python
# Driver-level performance
driver_performance = []

for driver in full_df['driver'].unique():
    driver_data = full_df[full_df['driver'] == driver]
    
    if len(driver_data) > 0:
        driver_accuracy = (driver_data['predicted_sc'] == driver_data['target_sc']).mean()
        driver_performance.append({
            'driver': driver,
            'accuracy': driver_accuracy,
            'predictions': len(driver_data),
            'sc_events': driver_data['target_sc'].sum()
        })

driver_perf_df = pd.DataFrame(driver_performance).sort_values('accuracy', ascending=False)

# Visualize
plt.figure(figsize=(12, 6))
bars = plt.bar(driver_perf_df['driver'], driver_perf_df['accuracy'])

# Color code by performance
for i, (bar, acc) in enumerate(zip(bars, driver_perf_df['accuracy'])):
    if acc >= 0.8:
        bar.set_color('green')
    elif acc >= 0.6:
        bar.set_color('orange')
    else:
        bar.set_color('red')

plt.axhline(y=accuracy, color='black', linestyle='--', label=f'Overall: {accuracy:.1%}')
plt.xlabel('Driver')
plt.ylabel('Prediction Accuracy')
plt.title('Prediction Accuracy by Driver')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
```

## 🎯 Chapter 7: Temporal Analysis of Predictions

### 7.1 Prediction Lead Times

```python
# Find successful predictions
successful_predictions = full_df[
    (full_df['predicted_sc'] == 1) & 
    (full_df['target_sc'] == 1)
].copy()

# For each successful prediction, calculate how far in advance we predicted
if len(successful_predictions) > 0 and len(sc_times) > 0:
    lead_times = []
    
    for _, pred in successful_predictions.iterrows():
        pred_time = pred['prediction_time']
        
        # Find the closest SC after this prediction
        future_scs = [sc for sc in sc_times if sc > pred_time]
        if future_scs:
            closest_sc = min(future_scs)
            lead_time = (closest_sc - pred_time).total_seconds() / 60
            if lead_time <= 2:  # Within our prediction window
                lead_times.append(lead_time)
    
    if lead_times:
        plt.figure(figsize=(10, 6))
        plt.hist(lead_times, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(lead_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(lead_times):.1f} min')
        plt.xlabel('Lead Time (minutes)')
        plt.ylabel('Count')
        plt.title('How Far in Advance Did We Predict Safety Cars?')
        plt.legend()
        plt.tight_layout()
        plt.show()
```

### 7.2 Probability Calibration

```python
# Analyze probability calibration
prob_bins = np.linspace(0, 1, 11)
calibration_data = []

for i in range(len(prob_bins) - 1):
    bin_mask = (full_df['sc_probability'] >= prob_bins[i]) & \
               (full_df['sc_probability'] < prob_bins[i+1])
    
    if bin_mask.sum() > 0:
        actual_rate = full_df.loc[bin_mask, 'target_sc'].mean()
        predicted_rate = full_df.loc[bin_mask, 'sc_probability'].mean()
        
        calibration_data.append({
            'bin_center': (prob_bins[i] + prob_bins[i+1]) / 2,
            'predicted_probability': predicted_rate,
            'actual_frequency': actual_rate,
            'count': bin_mask.sum()
        })

calib_df = pd.DataFrame(calibration_data)

# Plot calibration
plt.figure(figsize=(8, 8))
plt.scatter(calib_df['predicted_probability'], 
           calib_df['actual_frequency'], 
           s=calib_df['count']*2, alpha=0.6)

# Perfect calibration line
plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Actual Frequency')
plt.title('Probability Calibration Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 🎓 Chapter 8: Key Insights and Takeaways

### 8.1 What Makes This Approach "Temporal"?

```python
print("🕐 TEMPORAL ASPECTS OF OUR APPROACH:\n")

print("1. SLIDING WINDOWS:")
print("   - We move through the race minute-by-minute")
print("   - Each prediction uses only past data (no future leakage)")
print("   - Mimics real-time prediction scenario\n")

print("2. TIME-BASED FEATURES:")
print("   - Trends over time (lap time deterioration)")
print("   - Recent changes (last lap delta)")
print("   - Time-dependent factors (tire age)\n")

print("3. TEMPORAL VALIDATION:")
print("   - We know WHEN each prediction was made")
print("   - Can measure lead times for warnings")
print("   - Proper temporal ordering prevents data leakage")
```

### 8.2 Feature Engineering Insights

```python
# Summarize feature categories and their importance
feature_categories = {
    'Lap Performance': ['laptime_mean', 'laptime_std', 'laptime_consistency'],
    'Trends': ['laptime_trend', 'recent_vs_early', 'last_lap_delta'],
    'Position': ['current_position', 'position_changes', 'position_volatility'],
    'Strategy': ['current_tyre_age', 'current_stint', 'stint_changes']
}

print("📊 FEATURE ENGINEERING INSIGHTS:\n")

for category, features in feature_categories.items():
    cat_features = feature_importance[feature_importance['feature'].isin(features)]
    if len(cat_features) > 0:
        avg_importance = cat_features['importance'].mean()
        print(f"{category}:")
        print(f"   Average importance: {avg_importance:.3f}")
        
        # Most important in category
        top_feature = cat_features.iloc[0]
        print(f"   Most important: {top_feature['feature']} ({top_feature['importance']:.3f})")
        print(f"   Direction: {'Increases' if top_feature['coefficient'] > 0 else 'Decreases'} SC probability\n")

### 8.3 Model Performance Summary

```python
# Create a comprehensive performance summary
print("🏁 MODEL PERFORMANCE SUMMARY:\n")

print(f"Overall Metrics:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   Precision: {precision:.1%}")
print(f"   Recall: {recall:.1%}")
print(f"   F1 Score: {f1:.3f}\n")

print(f"Prediction Statistics:")
print(f"   Total predictions: {len(full_df)}")
print(f"   Positive predictions: {sum(y_pred)} ({sum(y_pred)/len(y_pred)*100:.1f}%)")
print(f"   Actual positives: {sum(y)} ({sum(y)/len(y)*100:.1f}%)\n")

print(f"Error Analysis:")
print(f"   False alarms: {fp} ({fp/len(y)*100:.1f}% of all predictions)")
print(f"   Missed events: {fn} ({fn/sum(y)*100:.1f}% of actual events)")
```

## 💡 Chapter 9: Practical Applications and Improvements

### 9.1 Real-World Applications

```python
print("🏆 POTENTIAL APPLICATIONS:\n")

applications = [
    ("Team Strategy", "Predict optimal pit stop timing to avoid SC disruption"),
    ("Broadcasting", "Alert TV directors to potential incidents before they occur"),
    ("Fan Engagement", "Real-time probability displays for betting/fantasy sports"),
    ("Safety Systems", "Pre-position marshals and equipment based on risk levels"),
    ("Driver Coaching", "Identify high-risk situations for defensive driving")
]

for title, desc in applications:
    print(f"• {title}:")
    print(f"  {desc}\n")
```

### 9.2 Visualizing a Race Timeline with Predictions

```python
# Create a timeline visualization showing predictions vs actual events
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Get timeline data for one driver
timeline_driver = 'VER'
driver_timeline = full_df[full_df['driver'] == timeline_driver].sort_values('prediction_time')

# Convert times to minutes for plotting
times_minutes = [(t - session.t0_date).total_seconds() / 60 
                 for t in driver_timeline['prediction_time']]

# Plot 1: Probability over time
ax1.plot(times_minutes, driver_timeline['sc_probability'], 
         'b-', alpha=0.7, label='SC Probability')
ax1.fill_between(times_minutes, driver_timeline['sc_probability'], 
                 alpha=0.3)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Mark actual SC events
for sc_time in sc_times:
    sc_minutes = (sc_time - session.t0_date).total_seconds() / 60
    ax1.axvline(x=sc_minutes, color='red', linestyle='-', 
                linewidth=2, alpha=0.7, label='Actual SC' if sc_time == sc_times[0] else '')

ax1.set_ylabel('Safety Car Probability')
ax1.set_title(f'Safety Car Prediction Timeline - {timeline_driver}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Lap times with predictions
if 'laptime_mean' in driver_timeline.columns:
    ax2.plot(times_minutes, driver_timeline['laptime_mean'], 
             'g-', alpha=0.7, label='Mean Lap Time (5min window)')
    
    # Highlight high-probability predictions
    high_prob_mask = driver_timeline['sc_probability'] > 0.7
    if high_prob_mask.any():
        high_prob_times = [times_minutes[i] for i in range(len(times_minutes)) if high_prob_mask.iloc[i]]
        high_prob_laptimes = driver_timeline.loc[high_prob_mask, 'laptime_mean']
        ax2.scatter(high_prob_times, high_prob_laptimes, 
                   color='red', s=50, alpha=0.7, label='High SC Risk')

ax2.set_xlabel('Race Time (minutes)')
ax2.set_ylabel('Lap Time (seconds)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9.3 Suggested Improvements

```python
print("🚀 SUGGESTED IMPROVEMENTS:\n")

improvements = {
    "More Data": [
        "• Use multiple races for training (currently single session)",
        "• Include weather data (rain increases SC probability)",
        "• Add circuit characteristics (street circuits = more SC)"
    ],
    
    "Advanced Features": [
        "• Sector time analysis (identify specific problem areas)",
        "• Speed trap data (sudden speed drops)",
        "• G-force telemetry (detect impacts)",
        "• Radio transcripts (driver reports issues)"
    ],
    
    "Model Enhancements": [
        "• Try ensemble methods (Random Forest, XGBoost)",
        "• LSTM for sequential pattern learning",
        "• Separate models for different SC causes",
        "• Online learning to adapt during race"
    ],
    
    "Validation Strategy": [
        "• Cross-validation across multiple races",
        "• Leave-one-race-out validation",
        "• Temporal validation (train on early season, test on late)",
        "• Driver-specific model evaluation"
    ]
}

for category, items in improvements.items():
    print(f"📌 {category}:")
    for item in items:
        print(f"  {item}")
    print()
```

## 🎯 Chapter 10: Hands-On Exercises

Now it's your turn! Here are some exercises to deepen your understanding:

### Exercise 1: Feature Investigation

```python
# TODO: Create a new feature that captures "lap time volatility" 
# in the last 3 laps. High volatility might indicate problems.

def calculate_recent_volatility(window_laps, n_laps=3):
    """
    Calculate the standard deviation of the last n lap times.
    
    Your task: Implement this function
    Hint: Use the last n laps, convert to seconds, calculate std
    """
    # YOUR CODE HERE
    pass

# Test your function
# volatility = calculate_recent_volatility(test_window['laps_data'])
# print(f"Recent volatility: {volatility}")
```

### Exercise 2: Alternative Target Definition

```python
# TODO: Instead of predicting "SC in next 2 minutes", create a target
# for "SC in next 5 laps". How does this change model performance?

def create_lap_based_target(windows, sc_times, n_laps_ahead=5):
    """
    Create a target based on laps instead of time.
    
    Your task: For each window, check if SC occurs within next n laps
    Hint: You'll need to track lap numbers and match with SC timing
    """
    # YOUR CODE HERE
    pass
```

### Exercise 3: Driver-Specific Models

```python
# TODO: Train separate models for each driver
# Question: Do some drivers have more predictable patterns?

def train_driver_specific_models(full_df, feature_cols):
    """
    Train one model per driver.
    
    Your task: 
    1. Group data by driver
    2. Train individual models
    3. Compare performance
    
    Return: Dictionary of {driver: model}
    """
    # YOUR CODE HERE
    pass
```

## 📚 Final Thoughts

### What We've Learned:

1. **Temporal Data Processing**: How to create sliding windows that respect time ordering
2. **Feature Engineering**: Extracting meaningful patterns from telemetry data
3. **Imbalanced Classification**: Handling rare events with appropriate techniques
4. **Model Evaluation**: Beyond accuracy - understanding precision, recall, and lead times
5. **Domain Knowledge**: How F1-specific factors influence our approach

### Key Takeaways:

- 🕐 **Time matters**: Proper temporal handling prevents data leakage
- 📊 **Features are crucial**: Domain knowledge helps create meaningful features
- ⚖️ **Balance is hard**: Rare events require special handling
- 🎯 **Metrics matter**: Accuracy alone doesn't tell the whole story
- 🔄 **Iterate**: Start simple, add complexity gradually

### Next Steps:

1. Try this approach on different races
2. Experiment with different window sizes
3. Add weather and track data
4. Explore more sophisticated models
5. Build a real-time prediction system

## 🏁 Conclusion

Congratulations! You've built a temporal machine learning system for predicting Formula 1 safety cars. This approach combines:

- Real-time data processing
- Domain-specific feature engineering  
- Temporal machine learning concepts
- Practical evaluation metrics

The same principles can be applied to many other temporal prediction problems:
- Equipment failure prediction
- Traffic incident detection
- Market event forecasting
- Medical emergency prediction

Remember: In temporal ML, respecting time ordering is crucial. Always ask yourself: "What information would actually be available at prediction time?"

Happy predicting! 🏎️💨

---

### Appendix: Complete Feature Reference

```python
# Display all features with descriptions
feature_descriptions = {
    'num_laps_in_window': 'Number of completed laps in lookback window',
    'laptime_mean': 'Average lap time in window (seconds)',
    'laptime_std': 'Standard deviation of lap times',
    'laptime_min': 'Fastest lap in window',
    'laptime_max': 'Slowest lap in window',
    'laptime_consistency': 'Coefficient of variation (std/mean)',
    'laptime_trend': 'Linear trend correlation (-1 to 1)',
    'recent_vs_early': 'Recent performance vs early window (seconds)',
    'last_lap_delta': 'Change from previous lap (seconds)',
    'current_position': 'Current race position',
    'avg_position': 'Average position in window',
    'position_changes': 'Total position changes',
    'position_gained': 'Net positions gained in window',
    'position_volatility': 'Standard deviation of positions',
    'current_tyre_age': 'Current tire age (laps)',
    'avg_tyre_age': 'Average tire age in window',
    'max_tyre_age': 'Maximum tire age reached',
    'current_stint': 'Current stint number',
    'stint_changes': 'Number of pit stops in window'
}

print("📖 FEATURE REFERENCE GUIDE:\n")
for feature, description in feature_descriptions.items():
    if feature in feature_cols:
        importance_row = feature_importance[feature_importance['feature'] == feature]
        if not importance_row.empty:
            imp_val = importance_row.iloc[0]['importance']
            print(f"• {feature}:")
            print(f"  Description: {description}")
            print(f"  Importance: {imp_val:.3f}\n")
```