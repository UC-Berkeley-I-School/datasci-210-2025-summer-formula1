<p align="center">
  <img src="docs/assets/f1_etl_logo.png" alt="F1 ETL Logo" width="500"/>
</p>

# The `f1_etl` package

This package contains an ETL pipeline for extracting, transforming, and preparing Formula 1 telemetry data for time series classification tasks, specifically designed for safety car prediction and other F1 data science applications.

## Features

- **Automated Data Extraction**: Pull telemetry data from FastF1 for entire seasons
- **Time Series Generation**: Create sliding window sequences from raw telemetry
- **Feature Engineering**: Handle missing values, normalization, and data type conversion
- **Class Imbalance Handling**: ADASYN, SMOTE, and Borderline SMOTE resampling strategies
- **Dimensionality Reduction**: PCA feature transformation for high-dimensional data
- **Track Status Integration**: Align telemetry with track status for safety car prediction
- **Machine Learning Training**: Comprehensive training, evaluation, and metadata tracking
- **Flexible Configuration**: Support for custom features, window sizes, and prediction horizons
- **Caching Support**: Cache raw data to avoid repeated API calls

## Installation

### From PyPI (Recommended)

The package is available on PyPI and can be installed using your preferred package manager:

```bash
# Using pip
pip install f1-etl

# Using uv
uv add f1-etl

# Using poetry
poetry add f1-etl
```

### From Source

For development or latest features:

```bash
# Clone and install in development mode
git clone <repository-url>
cd f1-etl
uv pip install -e .

# Or build and install wheel
uv build
uv pip install dist/f1_etl-0.1.0-py3-none-any.whl
```

### Verify Installation

```bash
# Check installation
pip list | grep f1-etl

# Or with uv
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

### Handling Class Imbalance

The library provides sophisticated resampling techniques to handle class imbalance in F1 telemetry data:

```python
# Apply SMOTE resampling to balance classes
dataset = create_safety_car_dataset(
    config=config,
    resampling_strategy='smote',  # 'adasyn', 'smote', 'borderline_smote'
    resampling_target_class='2',  # Focus on safety car class
)

# Custom resampling configuration
dataset = create_safety_car_dataset(
    config=config,
    resampling_strategy='adasyn',
    resampling_config={'2': 1000000},  # Resample class 2 to 1M samples
)

# Alternative: percentage-based resampling
dataset = create_safety_car_dataset(
    config=config,
    resampling_strategy='borderline_smote',
    resampling_config={'2': 0.5},  # Resample class 2 to 50% of majority
)
```

### Feature Dimensionality Reduction

Use PCA to reduce feature dimensionality while preserving variance:

```python
# Apply PCA transformation
dataset = create_safety_car_dataset(
    config=config,
    feature_transform='pca',
    pca_components=50,  # Use 50 principal components
)

# Auto-determine components based on variance threshold
dataset = create_safety_car_dataset(
    config=config,
    feature_transform='pca',
    pca_variance_threshold=0.95,  # Capture 95% of variance
)

# Access PCA transformation details
pca_info = dataset['config']
print(f"Reduced from {pca_info['original_n_features']} to {pca_info['pca_n_components']} features")
print(f"Explained variance: {pca_info['pca_explained_variance']}")
```

### Advanced Configuration

```python
# Comprehensive preprocessing pipeline
dataset = create_safety_car_dataset(
    config=config,
    window_size=200,
    prediction_horizon=15,
    handle_non_numeric='encode',
    handle_missing=True,
    missing_strategy='forward_fill',
    normalize=True,
    normalization_method='standard',
    target_column='TrackStatus',
    # Class imbalance handling
    resampling_strategy='smote',
    resampling_target_class='2',
    resampling_config={'2': 0.3},
    # Feature engineering
    feature_transform='pca',
    pca_variance_threshold=0.95,
    enable_debug=True
)

# Access preprocessing components for reuse
feature_engineer = dataset['feature_engineer']
label_encoder = dataset['label_encoder']
pca_transformer = dataset['pca_transformer']

# Apply to new data
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
- `handle_missing`: Whether to apply missing value imputation (default: True)
- `missing_strategy`: Strategy for handling missing values ('forward_fill', 'mean_fill', 'zero_fill')
- `normalize`: Whether to apply normalization (default: True)
- `normalization_method`: Normalization strategy ('standard', 'minmax', 'per_sequence')
- `target_column`: Column to predict (default: 'TrackStatus')
- `resampling_strategy`: Class imbalance technique ('adasyn', 'smote', 'borderline_smote', None)
- `resampling_target_class`: Specific class to focus resampling on (e.g., '2' for safety car)
- `resampling_config`: Custom sampling configuration (dict or 'not majority')
- `feature_transform`: Feature transformation strategy ('none', 'pca')
- `pca_components`: Number of PCA components (None for auto-determination)
- `pca_variance_threshold`: Variance threshold for PCA component selection (default: 0.95)

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
    'config': Dict,               # Pipeline configuration
    # PCA objects (if feature_transform='pca')
    'pca_transformer': PCA,       # Fitted PCA transformer
    'pca_scaler': StandardScaler, # Pre-PCA standardization
    'original_features': List     # Feature names before PCA
}
```

## Machine Learning Training (`f1_etl.train`)

The `train` sub-package provides comprehensive training, evaluation, and metadata tracking capabilities for time series classification models on F1 telemetry data.

### Key Features

- **Structured Metadata Tracking**: Track datasets, models, and evaluations with comprehensive metadata
- **Comprehensive Evaluation Suite**: Detailed metrics, confusion matrices, and performance summaries
- **Temporal Data Splitting**: Train/validation/test splitting with temporal awareness
- **External Dataset Validation**: Test models on completely different races/seasons
- **Model Factory**: Pre-configured model pipelines for common use cases

### Basic Training Workflow

```python
from f1_etl.train import (
    create_metadata_from_f1_dataset,
    prepare_data_with_validation,
    ModelEvaluationSuite,
    train_and_validate_model,
    create_basic_models
)

# Create dataset
dataset = create_safety_car_dataset(config=config)

# Prepare metadata
dataset_metadata = create_metadata_from_f1_dataset(
    data_config=config,
    dataset=dataset,
    features_used=dataset['config']['feature_names']
)

# Prepare data splits
splits = prepare_data_with_validation(
    dataset,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

# Create evaluation suite
evaluator = ModelEvaluationSuite(output_dir="training_results")

# Create and train models
models = create_basic_models()
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    model_metadata = create_model_metadata(
        model_type=model_name,
        model_params=model.get_params(),
        training_config={'epochs': 100}
    )
    
    results = train_and_validate_model(
        model=model,
        splits=splits,
        class_names=['normal', 'vsc', 'safety_car'],
        evaluator=evaluator,
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata
    )
```

### External Dataset Evaluation

Test model generalization on different races or seasons:

```python
from f1_etl.train import evaluate_on_external_dataset

# Train on one race
training_config = DataConfig(
    sessions=[SessionConfig(2024, "Monaco Grand Prix", "R")],
    cache_dir="./cache"
)

# Test on a different race
external_config = DataConfig(
    sessions=[SessionConfig(2024, "Silverstone Grand Prix", "R")],
    cache_dir="./cache"
)

# Evaluate trained model on external dataset
external_results = evaluate_on_external_dataset(
    trained_model=trained_model,
    external_config=external_config,
    original_dataset_metadata=dataset_metadata,
    model_metadata=model_metadata,
    class_names=['normal', 'vsc', 'safety_car'],
    evaluator=evaluator
)
```

### Model Factory Options

```python
from f1_etl.train import create_basic_models, create_advanced_models, create_balanced_pipeline

# Basic models for quick testing
basic_models = create_basic_models()
# Returns: Random Forest, SVM, Logistic Regression

# Advanced time series models
advanced_models = create_advanced_models()
# Returns: ROCKET, MiniRocket, Arsenal, etc.

# Create balanced pipeline for imbalanced data
balanced_model = create_balanced_pipeline(
    base_estimator='random_forest',
    sampling_strategy='smote'
)
```

### Evaluation Metrics

The evaluation suite provides comprehensive metrics:

- **Overall Performance**: Accuracy, F1-macro, F1-weighted
- **Class-specific Metrics**: Precision, recall, F1-score per class
- **Target Class Focus**: Specialized metrics for safety car prediction
- **Confusion Matrices**: Visual and numerical confusion matrices
- **Classification Reports**: Detailed sklearn classification reports

### Output Organization

Training results are automatically organized with timestamps and unique identifiers:

```
training_results/
├── run_20240805_143022/
│   ├── RandomForest_evaluation.json
│   ├── RandomForest_confusion_matrices.png
│   ├── SVM_evaluation.json
│   ├── external_Monaco_vs_Silverstone.json
│   └── training_summary.json
```

## Error Handling

The pipeline includes robust error handling:
- Missing telemetry data for specific drivers
- Insufficient data for sequence generation
- Track status alignment issues
- Feature processing errors
- Class imbalance handling failures
- PCA transformation issues

Enable debug logging to troubleshoot issues:

```python
dataset = create_safety_car_dataset(config=config, enable_debug=True)
```

## Performance Tips

1. **Use caching**: Set `cache_dir` to avoid re-downloading data
2. **Filter drivers**: Specify `drivers` list to reduce data volume
3. **Adjust window size**: Smaller windows = more sequences but less context
4. **Choose appropriate step size**: Default is `window_size // 2` for 50% overlap
5. **Apply PCA judiciously**: Use for high-dimensional data but consider information loss
6. **Resampling considerations**: Apply resampling before PCA for better synthetic sample quality
7. **Memory management**: Use smaller datasets or batch processing for large seasons

## License

TBD