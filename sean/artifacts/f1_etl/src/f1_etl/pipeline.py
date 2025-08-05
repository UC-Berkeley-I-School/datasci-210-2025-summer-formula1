"""Main ETL pipeline for safety car dataset creation"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .aggregation import DataAggregator
from .config import DataConfig
from .encoders_new import FixedVocabTrackStatusEncoder
from .extraction import RawDataExtractor
from .feature_engineering import FeatureEngineer
from .logging import setup_logger
from .resampling import apply_resampling
from .time_series import TimeSeriesGenerator


def create_safety_car_dataset(
    config: DataConfig,
    window_size: int = 100,
    prediction_horizon: int = 10,
    handle_non_numeric: str = "encode",
    # Preprocessing controls
    handle_missing: bool = True,
    missing_strategy: str = "forward_fill",
    normalize: bool = True,
    normalization_method: str = "standard",
    target_column: str = "TrackStatus",
    use_onehot_labels: bool = False,
    resampling_strategy: Optional[str] = None,
    resampling_target_class: Optional[str] = None,
    resampling_config: Optional[Dict[str, Any]] = None,
    # PCA parameters
    feature_transform: str = "none",  # "none", "pca"
    pca_components: Optional[int] = None,
    pca_variance_threshold: float = 0.95,
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
    resampling_strategy : str, optional
        Resampling strategy to apply before windowing ('adasyn', 'smote', 'borderline_smote', None)
    resampling_target_class : str, optional
        Specific class to focus resampling on (e.g., '2' for safety car)
    resampling_config : dict, optional
        Custom sampling configuration. Examples:
        - {'2': 1000000}: resample class 2 to have 1M samples
        - {'2': 0.5}: resample class 2 to 50% of majority class
        - 'not majority': resample all but the majority class
    feature_transform : str, default='none'
        Feature transformation strategy ('none', 'pca')
    pca_components : int, optional
        Number of PCA components. If None, uses pca_variance_threshold
    pca_variance_threshold : float, default=0.95
        Variance threshold when pca_components is None
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
    logger.info(f"  Feature transform: {feature_transform}")
    logger.info(
        f"  Resampling: {resampling_strategy if resampling_strategy else 'disabled'}"
    )

    # Log driver configuration for debugging
    logger.info("Driver configuration:")
    logger.info(f"  Global drivers: {config.drivers}")
    for session in config.sessions:
        effective_drivers = config.get_effective_drivers(session)
        logger.info(f"  {session.race}: {effective_drivers}")

    # Step 1: Extract raw data
    extractor = RawDataExtractor(config.cache_dir)
    sessions_data = [
        extractor.extract_session(session_config) for session_config in config.sessions
    ]

    # Step 2: Aggregate data with per-session driver filtering
    aggregator = DataAggregator()
    telemetry_data = aggregator.aggregate_telemetry_data(
        sessions_data, config, config.sessions
    )

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

    # Capture original class distribution before resampling
    original_class_distribution = None
    if resampling_strategy:
        if target_column == "TrackStatus":
            original_class_distribution = label_encoder.get_class_distribution(
                telemetry_data["TrackStatus"]
            )
        else:
            # For non-track status targets, calculate distribution manually
            unique_labels, counts = np.unique(
                telemetry_data[target_column].dropna(), return_counts=True
            )
            original_class_distribution = dict(
                zip(unique_labels.astype(str), counts.astype(int))
            )
        logger.info(
            f"Original class distribution before resampling: {original_class_distribution}"
        )

    # Step 3.5: Apply resampling if requested (BEFORE PCA and windowing)
    if resampling_strategy:
        telemetry_data = apply_resampling(
            telemetry_data,
            target_column=target_column,
            strategy=resampling_strategy,
            logger=logger,
            target_class=resampling_target_class,
            sampling_strategy=resampling_config,
        )

    # Step 3.6: Apply PCA transformation AFTER resampling but BEFORE windowing
    pca_transformer = None
    pca_scaler = None
    original_features = None
    original_n_features = None

    # Initialize TimeSeriesGenerator first
    ts_generator = TimeSeriesGenerator(
        window_size=window_size,
        step_size=window_size // 2,
        prediction_horizon=prediction_horizon,
        handle_non_numeric=handle_non_numeric,
        target_column=target_column,
    )

    if feature_transform == "pca":
        logger.info("Applying PCA transformation to telemetry features")
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # Get the features that TimeSeriesGenerator would use
        original_features = ts_generator.features.copy()

        # Filter to only include features that exist in the data
        available_features = [
            f for f in original_features if f in telemetry_data.columns
        ]

        # Further filter to only numeric features
        numeric_features = []
        for feature in available_features:
            col_dtype = telemetry_data[feature].dtype
            # Check if column is numeric (int, float) and not timedelta
            if pd.api.types.is_numeric_dtype(
                telemetry_data[feature]
            ) and not pd.api.types.is_timedelta64_dtype(telemetry_data[feature]):
                numeric_features.append(feature)
            else:
                logger.debug(
                    f"Skipping non-numeric feature '{feature}' (dtype: {col_dtype})"
                )

        if not numeric_features:
            raise ValueError(
                f"No numeric features found for PCA. Available features: {available_features}"
            )

        original_n_features = len(numeric_features)
        logger.info(
            f"Applying PCA to {len(numeric_features)} numeric features: {numeric_features}"
        )

        # Extract feature data - now guaranteed to be numeric
        X_features = telemetry_data[numeric_features].values

        # Convert to float to ensure compatibility
        try:
            X_features = X_features.astype(np.float64)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert features to float: {e}")
            logger.error(
                f"Feature dtypes: {[(f, telemetry_data[f].dtype) for f in numeric_features]}"
            )
            raise

        # Handle missing values before PCA
        if np.isnan(X_features).any():
            logger.info("Imputing missing values before PCA")
            imputer = SimpleImputer(strategy="mean")
            X_features = imputer.fit_transform(X_features)

        # Standardize for PCA
        pca_scaler = StandardScaler()
        X_scaled = pca_scaler.fit_transform(X_features)

        # Determine number of components
        if pca_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= pca_variance_threshold) + 1
            logger.info(
                f"Using {n_components} components to capture {pca_variance_threshold * 100}% variance"
            )
        else:
            n_components = min(pca_components, len(numeric_features))
            logger.info(f"Using {n_components} components (user specified)")

        # Apply PCA
        pca_transformer = PCA(n_components=n_components)
        X_pca = pca_transformer.fit_transform(X_scaled)

        logger.info(
            f"PCA reduced features from {len(numeric_features)} to {n_components}"
        )
        logger.info(
            f"Explained variance ratio: {pca_transformer.explained_variance_ratio_}"
        )
        logger.info(
            f"Cumulative variance: {np.cumsum(pca_transformer.explained_variance_ratio_)}"
        )

        # Replace original features with PCA components
        telemetry_data = telemetry_data.drop(columns=numeric_features)

        # Add PCA components as new columns
        pca_feature_names = [f"PC{i + 1}" for i in range(n_components)]
        for i, col_name in enumerate(pca_feature_names):
            telemetry_data[col_name] = X_pca[:, i]

        # Update TimeSeriesGenerator to use PCA features
        ts_generator.features = pca_feature_names

        # Store the numeric features that were actually used
        original_features = numeric_features

        logger.info(f"Updated features for windowing: {pca_feature_names}")

    # Step 4: Generate time series sequences (with PCA features if enabled)
    X, y, metadata = ts_generator.generate_sequences(telemetry_data)

    if len(X) == 0:
        raise ValueError("No sequences generated")

    logger.info(f"Generated {len(X)} sequences with shape {X.shape}")

    # Step 5: Apply feature engineering (missing values and normalization)
    engineer = FeatureEngineer()
    X_processed = X  # Start with raw sequences

    # Handle missing values
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
            logger.warning("Missing values detected but handling is disabled")

    # Normalize sequences (conditionally)
    if normalize:
        # Skip normalization for PCA features (already standardized)
        if feature_transform == "pca":
            logger.info(
                "Skipping normalization for PCA features (already standardized)"
            )
            X_final = X_processed
        else:
            logger.info(f"Applying normalization with method: {normalization_method}")
            X_final = engineer.normalize_sequences(
                X_processed, method=normalization_method
            )
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

    # Update metadata if PCA was applied
    if feature_transform == "pca" and metadata:
        for meta in metadata:
            meta["feature_transform"] = "pca"
            meta["original_features"] = original_features
            meta["pca_components"] = ts_generator.features

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
        "resampling_strategy": resampling_strategy,
        "resampling_config": resampling_config,
        "original_class_distribution": original_class_distribution,
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
        # PCA-specific info
        "feature_transform": feature_transform,
        "original_n_features": original_n_features
        if feature_transform == "pca"
        else X_final.shape[2],
        "pca_n_components": X_final.shape[2] if feature_transform == "pca" else None,
        "pca_explained_variance": pca_transformer.explained_variance_ratio_.tolist()
        if pca_transformer
        else None,
        "pca_cumulative_variance": np.cumsum(
            pca_transformer.explained_variance_ratio_
        ).tolist()
        if pca_transformer
        else None,
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
        # PCA objects for potential inverse transform
        "pca_transformer": pca_transformer,
        "pca_scaler": pca_scaler,
        "original_features": original_features,
    }
