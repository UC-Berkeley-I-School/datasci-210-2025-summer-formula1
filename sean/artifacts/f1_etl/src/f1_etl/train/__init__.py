"""
F1 Safety Car Prediction Training Module

Provides comprehensive training, evaluation, and metadata tracking capabilities
for time series classification models on F1 telemetry data.

Features:
- Structured metadata tracking for datasets, models, and evaluations
- Comprehensive evaluation suite with detailed metrics and reporting
- Train/validation/test splitting with temporal awareness
- Support for both basic and advanced time series models
- External dataset validation capabilities
- Organized output with confusion matrices and performance summaries

Example usage:
    from f1_etl.train import (
        create_metadata_from_f1_dataset,
        prepare_data_with_validation,
        ModelEvaluationSuite,
        train_and_validate_model
    )

    # Create evaluation suite
    evaluator = ModelEvaluationSuite(output_dir="results")

    # Prepare data
    splits = prepare_data_with_validation(dataset)

    # Train and evaluate
    results = train_and_validate_model(
        model, splits, class_names, evaluator,
        dataset_metadata, model_metadata
    )
"""

# Metadata classes
# Data preparation
from .data_preparation import analyze_class_distribution, prepare_data_with_validation

# Evaluation
from .evaluation import ModelEvaluationSuite
from .metadata import (
    DatasetMetadata,
    EvaluationMetadata,
    ModelMetadata,
    create_dataset_metadata_from_f1_config,
    create_metadata_from_f1_dataset,
    create_model_metadata,
)

# Model creation
from .model_factory import (
    create_advanced_models,
    create_balanced_pipeline,
    create_basic_models,
)

# Training orchestration
from .training import (
    compare_performance_across_datasets,
    evaluate_on_external_dataset,
    train_and_validate_model,
)

__all__ = [
    # Metadata
    "DatasetMetadata",
    "ModelMetadata",
    "EvaluationMetadata",
    "create_dataset_metadata_from_f1_config",
    "create_metadata_from_f1_dataset",
    "create_model_metadata",
    # Evaluation
    "ModelEvaluationSuite",
    # Data preparation
    "prepare_data_with_validation",
    "analyze_class_distribution",
    # Model creation
    "create_basic_models",
    "create_advanced_models",
    "create_balanced_pipeline",
    # Training
    "train_and_validate_model",
    "evaluate_on_external_dataset",
    "compare_performance_across_datasets",
]
