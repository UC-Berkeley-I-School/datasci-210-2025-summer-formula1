"""Main ETL pipeline for safety car dataset creation"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .aggregation import DataAggregator
from .config import DataConfig
from .encoders_new import FixedVocabTrackStatusEncoder
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
    use_onehot_labels: bool = False,
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
    use_onehot_labels : bool, default=False
        If True, labels are one-hot encoded vectors. If False, integer labels.
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

    # Step 3: Setup fixed vocabulary encoder for track status
    logger.info("Creating new fixed vocabulary encoder")
    label_encoder = FixedVocabTrackStatusEncoder(use_onehot=use_onehot_labels)

    if target_column == "TrackStatus":
        # Analyze distributions before encoding (optional but useful)
        label_encoder.analyze_data(telemetry_data["TrackStatus"], "training_data")

        if "TrackStatus" not in telemetry_data.columns:
            raise ValueError("TrackStatus column not found in telemetry data")

        # Fit and transform
        encoded_labels = label_encoder.fit_transform(telemetry_data["TrackStatus"])

        telemetry_data["TrackStatusEncoded"] = (
            encoded_labels.tolist() if use_onehot_labels else encoded_labels
        )

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

    # Encode prediction labels using fixed vocabulary encoder
    if target_column == "TrackStatus":
        y_encoded = label_encoder.transform(pd.Series(y))
    else:
        # For non-track status targets, use simple label encoder
        simple_encoder = LabelEncoder()
        y_encoded = simple_encoder.fit_transform(y)
        label_encoder = simple_encoder

    # Calculate class distribution
    if target_column == "TrackStatus":
        class_distribution = label_encoder.get_class_distribution(pd.Series(y))
        all_classes = label_encoder.get_classes()
        n_classes = label_encoder.get_n_classes()
    else:
        unique, counts = np.unique(y_encoded, return_counts=True)
        try:
            class_labels = label_encoder.inverse_transform(unique)
        except (ValueError, AttributeError):
            class_labels = unique
        class_distribution = dict(zip(class_labels, counts))
        all_classes = list(class_distribution.keys())
        n_classes = len(all_classes)

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
        "use_onehot_labels": use_onehot_labels,
        "n_sequences": len(X_final),
        "n_features": X_final.shape[2],
        "n_classes": n_classes,
        "feature_names": metadata[0]["features_used"] if metadata else [],
        "has_missing_values": np.isnan(X).any(),
        "missing_values_handled": handle_missing and np.isnan(X).any(),
        "normalization_applied": normalize,
        "all_possible_classes": list(all_classes),
        "classes_present": [k for k, v in class_distribution.items() if v > 0],
        "label_shape": "one-hot" if use_onehot_labels else "integer",
        "y_shape": y_encoded.shape if hasattr(y_encoded, "shape") else len(y_encoded),
    }

    # Log final results
    logger.info("Final dataset summary:")
    logger.info(f"  Sequences: {len(X_final)}")
    logger.info(f"  Features: {X_final.shape[2]}")
    logger.info(f"  Classes: {n_classes} ({processing_config['label_shape']})")
    logger.info(
        f"  Label shape: {y_encoded.shape if hasattr(y_encoded, 'shape') else len(y_encoded)}"
    )

    for class_name, count in class_distribution.items():
        if count > 0:
            percentage = count / len(y_encoded) * 100
            logger.info(
                f"    {class_name:12s}: {count:5d} samples ({percentage:5.1f}%)"
            )

    return {
        "X": X_final,
        "y": y_encoded,
        "y_raw": y,
        "metadata": metadata,
        "label_encoder": label_encoder,
        "feature_engineer": engineer,
        "raw_telemetry": telemetry_data,
        "class_distribution": class_distribution,
        "all_classes": all_classes,
        "n_classes": n_classes,
        "config": processing_config,
    }
