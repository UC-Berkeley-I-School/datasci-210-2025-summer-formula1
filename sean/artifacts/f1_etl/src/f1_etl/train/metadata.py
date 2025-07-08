"""Metadata classes and utilities for tracking dataset and model configurations"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DatasetMetadata:
    """Captures dataset configuration and characteristics"""

    scope: str
    sessions_config: List[Dict[str, Any]]
    drivers: List[str]
    include_weather: bool
    window_size: int
    prediction_horizon: int
    handle_non_numeric: str
    handle_missing: bool
    missing_strategy: str
    normalize: bool
    normalization_method: str
    target_column: str
    total_samples: int
    n_features: int
    n_timesteps: int
    feature_names: Optional[List[str]] = None
    class_distribution: Optional[Dict[str, int]] = None  # Final class distribution after any resampling
    features_used: str = "all"
    is_multivariate: bool = True
    preprocessing_steps: Optional[List[str]] = None
    resampling_strategy: Optional[str] = None
    resampling_config: Optional[Dict[str, Any]] = None
    original_class_distribution: Optional[Dict[str, int]] = None  # Class distribution before resampling


@dataclass
class ModelMetadata:
    """Captures model configuration and hyperparameters"""

    model_type: str
    base_estimator: str
    wrapper: str = "Catch22Classifier"
    hyperparameters: Optional[Dict[str, Any]] = None
    class_weights: Optional[Dict[int, float]] = None
    custom_weights_applied: bool = False
    random_state: Optional[int] = 42
    cv_strategy: Optional[str] = None


@dataclass
class EvaluationMetadata:
    """Captures evaluation context and settings"""

    evaluation_id: str
    timestamp: str
    test_size: float
    stratified_split: bool = True
    target_class_focus: str = "safety_car"
    evaluation_metrics: Optional[List[str]] = None


def create_dataset_metadata_from_f1_config(
    dataset_config, dataset, processing_config=None, features_used="all"
):
    """
    Create DatasetMetadata from F1 ETL configuration and dataset object

    Parameters:
    -----------
    dataset_config : DataConfig
        The F1 ETL DataConfig object
    dataset : dict
        The dataset dictionary returned by create_safety_car_dataset
    processing_config : dict, optional
        The processing config from dataset['config'] if available
    features_used : str
        Description of which features were used
    """

    X = dataset["X"]
    y = dataset["y"]

    # Use processing config from dataset if available
    if processing_config is None and "config" in dataset:
        processing_config = dataset["config"]

    # Determine scope description
    sessions = dataset_config.sessions if hasattr(dataset_config, "sessions") else []
    if len(sessions) == 1:
        session = sessions[0]
        year = getattr(session, "year", "unknown")
        race = getattr(session, "race", "unknown")
        session_type = getattr(session, "session_type", "unknown")
        scope = f"single_session_{year}_{race}_{session_type}".replace(" ", "_")
    elif len(sessions) > 1:
        years = list(set(getattr(s, "year", None) for s in sessions))
        years = [y for y in years if y is not None]
        session_types = list(set(getattr(s, "session_type", None) for s in sessions))
        session_types = [st for st in session_types if st is not None]

        if years and session_types:
            year_str = "-".join(map(str, sorted(years)))
            type_str = "_".join(sorted(session_types))
            scope = f"multi_session_{year_str}_{type_str}_{len(sessions)}sessions"
        else:
            scope = f"multi_session_{len(sessions)}sessions"
    else:
        scope = "unknown_scope"

    # Get class distribution
    unique, counts = np.unique(y, return_counts=True)
    label_encoder = dataset.get("label_encoder")
    class_dist = {}
    if label_encoder and hasattr(label_encoder, "class_to_idx"):
        idx_to_class = {v: k for k, v in label_encoder.class_to_idx.items()}
        class_dist = {
            str(idx_to_class.get(idx, f"class_{idx}")): int(count)
            for idx, count in zip(unique, counts)
        }
    elif label_encoder and hasattr(label_encoder, "classes_"):
        # Standard sklearn LabelEncoder
        class_names = label_encoder.classes_
        class_dist = {
            str(class_names[idx]): int(count)
            for idx, count in zip(unique, counts)
            if idx < len(class_names)
        }

    # Extract feature names if available
    feature_names = None
    if processing_config and "feature_names" in processing_config:
        feature_names = processing_config["feature_names"]
    elif "metadata" in dataset and dataset["metadata"]:
        # Try to get from first metadata entry
        meta_entry = (
            dataset["metadata"][0]
            if isinstance(dataset["metadata"], list)
            else dataset["metadata"]
        )
        if isinstance(meta_entry, dict) and "features_used" in meta_entry:
            feature_names = meta_entry["features_used"]

    # Get preprocessing steps
    preprocessing_steps = []
    if processing_config:
        if processing_config.get("missing_values_handled", False):
            preprocessing_steps.append(
                f"missing_values_handled_{processing_config.get('missing_strategy', 'unknown')}"
            )
        if processing_config.get("normalization_applied", False):
            preprocessing_steps.append(
                f"normalized_{processing_config.get('normalization_method', 'unknown')}"
            )
        if processing_config.get("resampling_strategy"):
            preprocessing_steps.append(
                f"resampled_{processing_config.get('resampling_strategy')}"
            )
    
    # Get resampling information
    resampling_strategy = None
    resampling_config = None
    original_class_dist = None
    
    if processing_config:
        resampling_strategy = processing_config.get("resampling_strategy")
        resampling_config = processing_config.get("resampling_config")
        
        # If resampling was applied, try to get original class distribution
        if resampling_strategy and "original_class_distribution" in processing_config:
            original_class_dist = processing_config["original_class_distribution"]

    return DatasetMetadata(
        scope=scope,
        sessions_config=[
            {
                "year": getattr(s, "year", None),
                "race": getattr(s, "race", None),
                "session_type": getattr(s, "session_type", None),
            }
            for s in sessions
        ],
        drivers=getattr(dataset_config, "drivers", []),
        include_weather=getattr(dataset_config, "include_weather", False),
        window_size=processing_config.get("window_size", 100)
        if processing_config
        else 100,
        prediction_horizon=processing_config.get("prediction_horizon", 10)
        if processing_config
        else 10,
        handle_non_numeric=processing_config.get("handle_non_numeric", "encode")
        if processing_config
        else "encode",
        handle_missing=processing_config.get("handle_missing", True)
        if processing_config
        else True,
        missing_strategy=processing_config.get("missing_strategy", "forward_fill")
        if processing_config
        else "forward_fill",
        normalize=processing_config.get("normalize", True)
        if processing_config
        else True,
        normalization_method=processing_config.get(
            "normalization_method", "per_sequence"
        )
        if processing_config
        else "per_sequence",
        target_column=processing_config.get("target_column", "TrackStatus")
        if processing_config
        else "TrackStatus",
        total_samples=X.shape[0],
        n_features=X.shape[1] if len(X.shape) > 1 else 1,
        n_timesteps=X.shape[2] if len(X.shape) > 2 else X.shape[1],
        feature_names=feature_names,
        class_distribution=class_dist,
        features_used=features_used,
        is_multivariate=len(X.shape) > 2 and X.shape[1] > 1,
        preprocessing_steps=preprocessing_steps,
        resampling_strategy=resampling_strategy,
        resampling_config=resampling_config,
        original_class_distribution=original_class_dist,
    )


def create_metadata_from_f1_dataset(
    data_config, dataset, features_used="multivariate_all_9_features"
):
    """
    Convenience function to create metadata from F1 dataset
    """
    return create_dataset_metadata_from_f1_config(
        dataset_config=data_config,
        dataset=dataset,
        processing_config=dataset.get("config"),  # Use the config from the dataset
        features_used=features_used,
    )


def create_model_metadata(model_name, model, class_weights=None):
    """Create ModelMetadata from model configuration"""

    # Extract hyperparameters
    hyperparams = {}
    if hasattr(model, "estimator") and hasattr(model.estimator, "get_params"):
        hyperparams = model.estimator.get_params()
    elif hasattr(model, "get_params"):
        hyperparams = model.get_params()

    # Determine base estimator name
    base_estimator = "Unknown"
    if hasattr(model, "estimator"):
        base_estimator = model.estimator.__class__.__name__
    else:
        base_estimator = model.__class__.__name__

    return ModelMetadata(
        model_type=model_name,
        base_estimator=base_estimator,
        wrapper="Catch22Classifier" if hasattr(model, "estimator") else "Direct",
        hyperparameters=hyperparams,
        class_weights=class_weights,
        custom_weights_applied=class_weights is not None,
        random_state=hyperparams.get("random_state", None),
    )
