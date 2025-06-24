"""Main ETL pipeline for safety car dataset creation"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder

from .config import DataConfig
from .extraction import RawDataExtractor
from .aggregation import DataAggregator
from .time_series import TimeSeriesGenerator
from .feature_engineering import FeatureEngineer
from .encoders import TrackStatusLabelEncoder
from .logging import setup_logger, logger


def create_safety_car_dataset(config: DataConfig, 
                             window_size: int = 100,
                             prediction_horizon: int = 10,
                             handle_non_numeric: str = 'encode',
                             normalization_method: str = 'standard',
                             target_column: str = 'TrackStatus',
                             enable_debug: bool = False) -> Dict[str, Any]:
    """Complete ETL pipeline for safety car prediction dataset"""
    
    # Setup logging
    global logger
    logger = setup_logger(enable_debug=enable_debug)
    
    # Step 1: Extract raw data
    extractor = RawDataExtractor(config.cache_dir)
    sessions_data = [extractor.extract_session(session_config) 
                    for session_config in config.sessions]
    
    # Step 2: Aggregate data with track status alignment
    aggregator = DataAggregator()
    telemetry_data = aggregator.aggregate_telemetry_data(sessions_data, config.drivers)
    
    if telemetry_data.empty:
        raise ValueError("No telemetry data extracted")
    
    # Step 3: Encode track status labels (if using track status)
    label_encoder = None
    if target_column == 'TrackStatus':
        label_encoder = TrackStatusLabelEncoder()
        if 'TrackStatus' in telemetry_data.columns:
            encoded_labels = label_encoder.fit_transform(telemetry_data['TrackStatus'])
            telemetry_data['TrackStatusEncoded'] = encoded_labels
        else:
            raise ValueError("TrackStatus column not found in telemetry data")
    elif target_column not in telemetry_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in telemetry data")
    
    # Step 4: Generate time series sequences with built-in preprocessing
    ts_generator = TimeSeriesGenerator(
        window_size=window_size, 
        step_size=window_size // 2,
        prediction_horizon=prediction_horizon,
        handle_non_numeric=handle_non_numeric,
        target_column=target_column
    )
    
    X, y, metadata = ts_generator.generate_sequences(telemetry_data)
    
    if len(X) == 0:
        raise ValueError("No sequences generated")
    
    logger.info(f"Generated {len(X)} sequences with shape {X.shape}")
    
    # Step 5: Apply feature engineering (missing values + normalization)
    engineer = FeatureEngineer()
    
    # Handle missing values (only if they exist)
    X_clean = engineer.handle_missing_values(X, strategy='forward_fill')
    
    # Normalize sequences
    X_normalized = engineer.normalize_sequences(X_clean, method=normalization_method)
    
    # Encode prediction labels if using track status
    if label_encoder:
        y_encoded = label_encoder.transform(pd.Series(y))
    else:
        # For non-track status targets, create a simple label encoder
        simple_encoder = LabelEncoder()
        y_encoded = simple_encoder.fit_transform(y)
        label_encoder = simple_encoder
    
    # Calculate class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    if hasattr(label_encoder, 'inverse_transform'):
        try:
            class_labels = label_encoder.inverse_transform(unique)
        except:
            class_labels = unique
    else:
        class_labels = unique
    
    class_distribution = dict(zip(class_labels, counts))
    
    return {
        'X': X_normalized,
        'y': y_encoded,
        'y_raw': y,
        'metadata': metadata,
        'label_encoder': label_encoder,
        'feature_engineer': engineer,
        'raw_telemetry': telemetry_data,
        'class_distribution': class_distribution,
        'config': {
            'window_size': window_size,
            'prediction_horizon': prediction_horizon,
            'handle_non_numeric': handle_non_numeric,
            'normalization_method': normalization_method,
            'target_column': target_column,
            'n_sequences': len(X_normalized),
            'n_features': X_normalized.shape[2],
            'feature_names': metadata[0]['features_used'] if metadata else []
        }
    }