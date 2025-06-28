"""Main ETL pipeline for safety car dataset creation"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .aggregation import DataAggregator
from .config import DataConfig
from .encoders import TrackStatusLabelEncoder
from .extraction import RawDataExtractor
from .feature_engineering import FeatureEngineer
from .logging import setup_logger
from .time_series import TimeSeriesGenerator


def create_safety_car_dataset(
    config: DataConfig,
    window_size: int = 100,
    prediction_horizon: int = 10,
    handle_non_numeric: str = "encode",
    # New preprocessing controls
    handle_missing: bool = True,
    missing_strategy: str = "forward_fill",
    normalize: bool = True,
    normalization_method: str = "standard",
    # Existing parameters
    target_column: str = "TrackStatus",
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Complete ETL pipeline for safety car prediction dataset

    Parameters:
    -----------
    config : DataConfig
        Configuration for data extraction and processing
    window_size : int, default=100
        Size of sliding window for time series sequences
    prediction_horizon : int, default=10
        Number of time steps ahead to predict
    handle_non_numeric : str, default='encode'
        How to handle non-numeric features ('encode' or 'drop')
    handle_missing : bool, default=True
        Whether to apply missing value imputation
    missing_strategy : str, default='forward_fill'
        Strategy for handling missing values ('forward_fill', 'mean_fill', 'zero_fill')
    normalize : bool, default=True
        Whether to apply normalization to features
    normalization_method : str, default='standard'
        Normalization method ('standard', 'minmax', 'per_sequence', 'none')
        Note: If normalize=False, this parameter is ignored
    target_column : str, default='TrackStatus'
        Column to use as prediction target
    enable_debug : bool, default=False
        Enable debug logging

    Returns:
    --------
    Dict containing processed dataset and metadata
    """

    # Setup logging
    global logger
    logger = setup_logger(enable_debug=enable_debug)

    # Log preprocessing configuration
    logger.info("Preprocessing configuration:")
    logger.info(
        f"  Missing values: {'enabled' if handle_missing else 'disabled'} ({missing_strategy})"
    )
    logger.info(
        f"  Normalization: {'enabled' if normalize else 'disabled'} ({normalization_method if normalize else 'N/A'})"
    )

    # Step 1: Extract raw data
    extractor = RawDataExtractor(config.cache_dir)
    sessions_data = [
        extractor.extract_session(session_config) for session_config in config.sessions
    ]

    # Step 2: Aggregate data with track status alignment
    aggregator = DataAggregator()
    telemetry_data = aggregator.aggregate_telemetry_data(sessions_data, config.drivers)

    if telemetry_data.empty:
        raise ValueError("No telemetry data extracted")

    # Step 3: Encode track status labels (if using track status)
    label_encoder = None
    if target_column == "TrackStatus":
        label_encoder = TrackStatusLabelEncoder()
        if "TrackStatus" in telemetry_data.columns:
            encoded_labels = label_encoder.fit_transform(telemetry_data["TrackStatus"])
            telemetry_data["TrackStatusEncoded"] = encoded_labels
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
        target_column=target_column,
    )

    X, y, metadata = ts_generator.generate_sequences(telemetry_data)

    if len(X) == 0:
        raise ValueError("No sequences generated")

    logger.info(f"Generated {len(X)} sequences with shape {X.shape}")

    # Step 5: Apply configurable feature engineering
    engineer = FeatureEngineer()
    X_processed = X  # Start with raw sequences

    # Handle missing values (conditionally)
    if handle_missing:
        # Check if missing values actually exist
        if np.isnan(X_processed).any():
            logger.info(
                f"Applying missing value imputation with strategy: {missing_strategy}"
            )
            X_processed = engineer.handle_missing_values(
                X_processed, strategy=missing_strategy
            )
        else:
            logger.info("No missing values detected, skipping imputation")
    else:
        logger.info("Missing value handling disabled")
        if np.isnan(X_processed).any():
            logger.warning(
                "Missing values detected but handling is disabled - may cause issues with some models"
            )

    # Normalize sequences (conditionally)
    if normalize:
        logger.info(f"Applying normalization with method: {normalization_method}")
        X_final = engineer.normalize_sequences(X_processed, method=normalization_method)
    else:
        logger.info("Normalization disabled - using raw feature values")
        X_final = X_processed

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
    if hasattr(label_encoder, "inverse_transform"):
        try:
            class_labels = label_encoder.inverse_transform(unique)
        except (ValueError, AttributeError):
            class_labels = unique
    else:
        class_labels = unique

    class_distribution = dict(zip(class_labels, counts))

    # Enhanced configuration tracking
    processing_config = {
        "window_size": window_size,
        "prediction_horizon": prediction_horizon,
        "handle_non_numeric": handle_non_numeric,
        "handle_missing": handle_missing,
        "missing_strategy": missing_strategy if handle_missing else None,
        "normalize": normalize,
        "normalization_method": normalization_method if normalize else None,
        "target_column": target_column,
        "n_sequences": len(X_final),
        "n_features": X_final.shape[2],
        "feature_names": metadata[0]["features_used"] if metadata else [],
        "has_missing_values": np.isnan(X).any(),
        "missing_values_handled": handle_missing and np.isnan(X).any(),
        "normalization_applied": normalize,
    }

    return {
        "X": X_final,
        "y": y_encoded,
        "y_raw": y,
        "metadata": metadata,
        "label_encoder": label_encoder,
        "feature_engineer": engineer,
        "raw_telemetry": telemetry_data,
        "class_distribution": class_distribution,
        "config": processing_config,
    }


# Convenience functions for model-specific preprocessing
def create_catch22_dataset(
    config: DataConfig,
    window_size: int = 100,
    prediction_horizon: int = 10,
    handle_non_numeric: str = "encode",
    missing_strategy: str = "forward_fill",
    normalization_method: str = "per_sequence",
    target_column: str = "TrackStatus",
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Create dataset optimized for Catch22 classifier
    - Disables missing value handling (Catch22 can handle internally)
    - Uses per-sequence normalization or no normalization to preserve variation
    """
    return create_safety_car_dataset(
        config=config,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        handle_non_numeric=handle_non_numeric,
        handle_missing=False,  # Catch22 handles missing values internally
        missing_strategy=missing_strategy,
        normalize=True,
        normalization_method=normalization_method,
        target_column=target_column,
        enable_debug=enable_debug,
    )


def create_rocket_dataset(
    config: DataConfig,
    window_size: int = 100,
    prediction_horizon: int = 10,
    handle_non_numeric: str = "encode",
    missing_strategy: str = "forward_fill",
    normalization_method: str = "standard",
    target_column: str = "TrackStatus",
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Create dataset optimized for ROCKET classifier
    - Enables missing value handling (ROCKET cannot handle missing values)
    - Uses standard normalization
    """
    return create_safety_car_dataset(
        config=config,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        handle_non_numeric=handle_non_numeric,
        handle_missing=True,  # ROCKET needs missing values handled
        missing_strategy=missing_strategy,
        normalize=True,
        normalization_method=normalization_method,
        target_column=target_column,
        enable_debug=enable_debug,
    )


def create_raw_dataset(
    config: DataConfig,
    window_size: int = 100,
    prediction_horizon: int = 10,
    handle_non_numeric: str = "encode",
    missing_strategy: str = "forward_fill",
    normalization_method: str = "standard",
    target_column: str = "TrackStatus",
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Create dataset with minimal preprocessing for analysis or custom preprocessing
    - Disables both missing value handling and normalization
    """
    return create_safety_car_dataset(
        config=config,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        handle_non_numeric=handle_non_numeric,
        handle_missing=False,  # No missing value handling
        missing_strategy=missing_strategy,
        normalize=False,  # No normalization
        normalization_method=normalization_method,
        target_column=target_column,
        enable_debug=enable_debug,
    )
