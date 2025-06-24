
# The `f1_etl` package

This package contains an ETL pipeline for extracting, transforming, and preparing Formula 1 telemetry data for time series classification tasks, specifically designed for safety car prediction and other F1 data science applications.

## Features

- **Automated Data Extraction**: Pull telemetry data from FastF1 for entire seasons
- **Time Series Generation**: Create sliding window sequences from raw telemetry
- **Feature Engineering**: Handle missing values, normalization, and data type conversion
- **Track Status Integration**: Align telemetry with track status for safety car prediction
- **Flexible Configuration**: Support for custom features, window sizes, and prediction horizons
- **Caching Support**: Cache raw data to avoid repeated API calls

## Installation

The project is managed with `uv` but you can just use `pip` if that is preferable.

Install:

- From Source...
  ```bash
  uv pip install -e .
  ```
- From Wheel...
  ```bash
  uv build
  uv pip install dist/f1_etl-0.1.0-py3-none-any.whl
  ```

Verify:

```bash
uv pip list | grep f1-etl
```

## Quick Start

### Basic Usage - Single Race

```python
from f1_etl import SessionConfig, DataConfig, create_safety_car_dataset

# Define a single race session
session = SessionConfig(
    year=2024,
    race="Monaco Grand Prix",
    session_type="R"  # Race
)

# Configure the dataset
config = DataConfig(
    sessions=[session],
    cache_dir="./f1_cache"
)

# Generate the dataset
dataset = create_safety_car_dataset(
    config=config,
    window_size=100,
    prediction_horizon=10
)

print(f"Generated {dataset['config']['n_sequences']} sequences")
print(f"Features: {dataset['config']['feature_names']}")
print(f"Class distribution: {dataset['class_distribution']}")
```

### Full Season Dataset

```python
from f1_etl import create_season_configs

# Generate configs for all 2024 races
race_configs = create_season_configs(2024, session_types=['R'])

# Create dataset configuration
config = DataConfig(
    sessions=race_configs,
    cache_dir="./f1_cache"
)

# Generate the complete dataset
dataset = create_safety_car_dataset(
    config=config,
    window_size=150,
    prediction_horizon=20,
    normalization_method='standard'
)

# Access the data
X = dataset['X']  # Shape: (n_sequences, window_size, n_features)
y = dataset['y']  # Encoded labels
metadata = dataset['metadata']  # Sequence metadata
```

### Multiple Session Types

```python
# Include practice, qualifying, and race sessions
all_configs = create_season_configs(
    2024, 
    session_types=['FP1', 'FP2', 'FP3', 'Q', 'R']
)

config = DataConfig(
    sessions=all_configs,
    drivers=['HAM', 'VER', 'LEC'],  # Specific drivers only
    cache_dir="./f1_cache"
)

dataset = create_safety_car_dataset(config=config)
```

### Custom Target Variable

```python
# Use a different target column (not track status)
dataset = create_safety_car_dataset(
    config=config,
    target_column='Speed',  # Predict speed instead
    window_size=50,
    prediction_horizon=5
)
```

### Machine Learning Integration

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate dataset
dataset = create_safety_car_dataset(config=config)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    dataset['X'], dataset['y'], test_size=0.2, random_state=42
)

# For sklearn models, reshape to 2D
n_samples, n_timesteps, n_features = X_train.shape
X_train_2d = X_train.reshape(n_samples, n_timesteps * n_features)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# Train a model
clf = RandomForestClassifier()
clf.fit(X_train_2d, y_train)
score = clf.score(X_test_2d, y_test)
print(f"Accuracy: {score:.3f}")
```

### Advanced Configuration

```python
# Custom feature engineering
dataset = create_safety_car_dataset(
    config=config,
    window_size=200,
    prediction_horizon=15,
    handle_non_numeric='encode',  # or 'drop'
    normalization_method='minmax',  # or 'standard', 'per_sequence'
    target_column='TrackStatus',
    enable_debug=True  # Detailed logging
)

# Access preprocessing components for reuse
feature_engineer = dataset['feature_engineer']
label_encoder = dataset['label_encoder']

# Use on new data
new_X_normalized = feature_engineer.normalize_sequences(new_X, fit=False)
new_y_encoded = label_encoder.transform(new_y)
```

## Configuration Options

### SessionConfig
- `year`: F1 season year
- `race`: Race name (e.g., "Monaco Grand Prix")
- `session_type`: Session type ('R', 'Q', 'FP1', etc.)

### DataConfig
- `sessions`: List of SessionConfig objects
- `drivers`: Optional list of driver abbreviations
- `cache_dir`: Directory for caching raw data
- `include_weather`: Include weather data (default: True)

### Pipeline Parameters
- `window_size`: Length of each time series sequence
- `prediction_horizon`: Steps ahead to predict
- `handle_non_numeric`: How to handle non-numeric features ('encode' or 'drop')
- `normalization_method`: Normalization strategy ('standard', 'minmax', 'per_sequence')
- `target_column`: Column to predict (default: 'TrackStatus')

## Output Structure

```python
dataset = {
    'X': np.ndarray,              # Normalized feature sequences
    'y': np.ndarray,              # Encoded target labels
    'y_raw': np.ndarray,          # Original target values
    'metadata': List[Dict],       # Sequence metadata
    'label_encoder': LabelEncoder, # For inverse transformation
    'feature_engineer': FeatureEngineer,  # For applying to new data
    'raw_telemetry': pd.DataFrame, # Original telemetry data
    'class_distribution': Dict,    # Label distribution
    'config': Dict                # Pipeline configuration
}
```

## Error Handling

The pipeline includes robust error handling:
- Missing telemetry data for specific drivers
- Insufficient data for sequence generation
- Track status alignment issues
- Feature processing errors

Enable debug logging to troubleshoot issues:

```python
dataset = create_safety_car_dataset(config=config, enable_debug=True)
```

## Performance Tips

1. **Use caching**: Set `cache_dir` to avoid re-downloading data
2. **Filter drivers**: Specify `drivers` list to reduce data volume
3. **Adjust window size**: Smaller windows = more sequences but less context
4. **Choose appropriate step size**: Default is `window_size // 2` for 50% overlap

## License

TBD