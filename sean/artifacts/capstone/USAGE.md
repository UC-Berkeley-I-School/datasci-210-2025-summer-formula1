# About the F1 ETL Pipeline

This document explains how the F1 ETL pipeline transforms raw Formula 1 telemetry data into machine learning-ready time series datasets, covering the data science principles and preprocessing techniques involved.

## Overview

The pipeline converts high-frequency F1 telemetry data into supervised learning datasets suitable for time series classification tasks. The primary use case is safety car prediction, but the framework supports any classification task using F1 telemetry features.

## Pipeline Architecture

```
Raw FastF1 Data → Aggregation → Time Series Generation → Feature Engineering → ML-Ready Dataset
```

## Stage 1: Raw Data Extraction

### Data Sources
The pipeline extracts multiple data streams from FastF1:

1. **Telemetry Data** (`car_data`): High-frequency sensor readings
   - Speed, RPM, gear, throttle, brake position
   - Sampling rate: ~4-10 Hz depending on circuit
   
2. **Position Data** (`pos_data`): GPS coordinates and derived metrics
   - X, Y coordinates, distance, differential distance
   
3. **Track Status**: Official race control status
   - Green flag, yellow flag, safety car, VSC, red flag
   - Lower frequency (~1 Hz) but critical for prediction tasks

4. **Lap Data**: Timing information and lap-level metrics

### Data Characteristics
- **High frequency**: Car data ~240ms intervals (~4.2 Hz), Position data ~220ms intervals (~4.5 Hz)
- **Asynchronous streams**: Car and position data have different sampling intervals requiring alignment
- **Mixed types**: Numeric (speed), categorical (gear), boolean (brake)
- **Temporal alignment**: Different data streams may have slight timing offsets
- **Missing values**: GPS dropouts, sensor failures, transmission issues

## Stage 2: Data Aggregation & Temporal Alignment

### Merging Data Streams
The pipeline merges telemetry and position data using FastF1's `merge_channels()` method:

```python
merged = car_data.merge_channels(pos_data, frequency='original')
```

**Key considerations:**
- Preserves original sampling frequency to avoid information loss
- Handles slight timing misalignments between data streams
- Adds derived features (distance, differential distance)

### Track Status Alignment
Track status data is temporally aligned with telemetry using `pd.merge_asof()`:

```python
telemetry_with_status = pd.merge_asof(
    telemetry.sort_values('Date'),
    track_status.sort_values('Date'),
    on='Date',
    direction='backward'  # Forward-fill track status
)
```

**Why this matters:**
- Track status changes are sparse but critical for safety car prediction
- Forward-fill ensures every telemetry sample has associated track status
- Backward direction captures the "current" status at each timestamp

## Stage 3: Time Series Sequence Generation

### Sliding Window Approach

The pipeline creates fixed-length sequences using a sliding window **before** feature engineering:

```python
window_size = 100        # 100 samples per sequence
step_size = 50          # 50% overlap between sequences
prediction_horizon = 10  # Predict 10 steps ahead
```

### Data Type Processing During Sequence Generation

The `TimeSeriesGenerator` handles mixed data types during sequence creation:

#### Numeric Features
- **Continuous**: Speed, RPM, throttle percentage (0-100)
- **Discrete**: Gear (1-8), brake binary (0/1)
- **Spatial**: X, Y coordinates, distance

#### Categorical Features
- **Track Status**: Encoded as integers (1=green, 2=yellow, 4=safety_car, etc.)
- **Driver IDs**: Label encoded for consistency
- **Session metadata**: Year, race, session type

#### Non-numeric Handling Strategy
```python
handle_non_numeric='encode'  # Label encode categorical features
handle_non_numeric='drop'    # Remove non-numeric features
```

**Recommendation**: Use 'encode' for safety car prediction as categorical features (gear, track status) contain valuable information.

### Sequence Structure

Each sequence has shape `(window_size, n_features)`:
- **Temporal dimension**: 100 consecutive telemetry samples (~23-25 seconds at 4.2-4.5 Hz)
- **Feature dimension**: Speed, RPM, gear, throttle, brake, X, Y, distance, etc.

### Label Generation

Labels are extracted at `current_time + prediction_horizon`:

```python
# For safety car prediction
label_time = sequence_end_time + prediction_horizon
label = track_status_at(label_time)
```

**Prediction horizons in real-time:**
- 10 samples ≈ 2.3-2.5 seconds (at 4.2-4.5 Hz)
- 20 samples ≈ 4.5-4.8 seconds
- 50 samples ≈ 11-12 seconds

### Overlap Strategy

Default 50% overlap provides:
- **More training data**: Doubles the number of sequences
- **Temporal stability**: Reduces variance in sequence boundaries
- **Better class balance**: More opportunities to capture rare events

## Stage 4: Feature Engineering

Feature engineering is applied **after** sequence generation to the 3D arrays:

### 4.1 Missing Value Treatment

Missing values occur due to:
- GPS signal loss in tunnels/covered sections
- Sensor malfunctions
- Data transmission errors
- Cars in pits (no position data)

#### Imputation Strategies

**Forward Fill** (Default):
```python
strategy='forward_fill'
```
- Carries last valid observation forward
- Appropriate for telemetry as values change gradually
- Preserves temporal dependencies

**Mean Imputation**:
```python
strategy='mean_fill'
```
- Replaces missing values with feature mean across all sequences
- May break temporal patterns but provides stable baseline

**Zero Fill**:
```python
strategy='zero_fill'
```
- Simple but can introduce artifacts
- Use only when missing values represent true zeros

### 4.2 Normalization

Normalization is applied to the 3D sequence arrays and is critical due to different feature scales:
- Speed: 0-350 km/h
- RPM: 8,000-15,000
- Throttle: 0-100%
- Coordinates: Circuit-specific ranges

#### Normalization Methods

**Standard Normalization** (Recommended):
```python
method='standard'  # Z-score normalization
X_norm = (X - μ) / σ
```
- Centers data at zero with unit variance
- Preserves feature relationships
- Works well with most ML algorithms

**Min-Max Scaling**:
```python
method='minmax'  # Scale to [0,1]
X_norm = (X - X_min) / (X_max - X_min)
```
- Bounded output range
- Preserves zero values (important for brake/throttle)

**Per-Sequence Normalization**:
```python
method='per_sequence'  # Normalize each sequence independently
```
- Removes driver-specific biases
- Useful when comparing different cars/setups
- May lose absolute performance information

## Stage 5: Label Encoding & Class Handling

### Track Status Encoding

Raw track status codes are mapped to meaningful labels:

```python
track_status_mapping = {
    '1': 'green',        # Normal racing
    '2': 'yellow',       # Local yellow flag
    '4': 'safety_car',   # Full safety car
    '5': 'red',          # Session stopped
    '6': 'vsc',          # Virtual safety car
    '7': 'vsc_ending'    # VSC ending
}
```

Then label encoded for ML algorithms:
```python
encoded_labels = LabelEncoder().fit_transform(mapped_labels)
# Output: [0, 1, 2, 3, 4, 5] corresponding to alphabetical order
```

### Class Imbalance Considerations

Safety car events are rare (~2-5% of race time):
- **Green flag**: ~85-90% of samples
- **Yellow flags**: ~5-10% of samples  
- **Safety car**: ~2-5% of samples
- **VSC**: ~1-3% of samples

**Handling strategies:**
1. **Class weights**: Use `class_weight='balanced'` in sklearn
2. **Resampling**: SMOTE for time series or custom resampling
3. **Focal loss**: For deep learning approaches
4. **Ensemble methods**: Combine multiple models with different sampling

## Data Science Best Practices

### 1. Temporal Splits
Never use random train/test splits with time series data:

```python
# Wrong: Random split
X_train, X_test = train_test_split(X, y, test_size=0.2)

# Correct: Temporal split
split_time = pd.Timestamp('2024-07-01')  # Mid-season split
train_mask = metadata['start_time'] < split_time
X_train, X_test = X[train_mask], X[~train_mask]
```

### 2. Feature Selection
Important features for safety car prediction:
- **Speed patterns**: Sudden deceleration indicates incidents
- **Distance/position**: Sector-specific incident probabilities
- **Throttle/brake**: Driver behavior changes before incidents
- **Gear**: Lower gears may indicate slow zones

### 3. Sequence Length Selection
Balance between context and computation:
- **Short sequences (50-100 samples)**: Fast training, less context
- **Medium sequences (100-200 samples)**: Good balance for most tasks
- **Long sequences (200+ samples)**: Maximum context, slower training

### 4. Prediction Horizon Tuning
Consider practical requirements:
- **Short horizon (5-10 samples)**: High accuracy, limited reaction time
- **Medium horizon (10-30 samples)**: Balanced accuracy/reaction time
- **Long horizon (30+ samples)**: Early warning, lower accuracy

## Integration with Time Series Libraries

### aeon (Recommended)
The pipeline outputs are compatible with aeon:

```python
from aeon.classification.convolution_based import RocketClassifier

# Reshape for aeon (n_samples, n_channels, n_timepoints)
X_aeon = X.transpose(0, 2, 1)

clf = RocketClassifier()
clf.fit(X_aeon, y)
```

### scikit-learn
Flatten sequences for traditional ML:

```python
# Reshape to 2D (n_samples, n_features)
X_flat = X.reshape(X.shape[0], -1)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_flat, y)
```

### Deep Learning
Direct compatibility with TensorFlow/PyTorch:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=50, validation_split=0.2)
```

## Performance Considerations

### Memory Management
- Each sequence uses ~8KB (100 samples × 10 features × 8 bytes)
- Full season dataset: ~100K sequences = ~800MB
- Use data generators for very large datasets

### Computational Efficiency
- Caching raw FastF1 data reduces extraction time by 90%
- Parallel processing for multiple sessions
- Incremental feature engineering for streaming data

## Validation Strategies

### Cross-Validation for Time Series
Use time-aware CV strategies:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and validate model
```

### Evaluation Metrics
For imbalanced classification:
- **Precision/Recall**: More informative than accuracy
- **F1-score**: Harmonic mean of precision/recall
- **ROC-AUC**: Overall discriminative ability
- **Confusion matrix**: Understanding specific errors

This ETL pipeline provides a robust foundation for F1 time series classification tasks, handling the complexities of real-world motorsport data while maintaining data science best practices.