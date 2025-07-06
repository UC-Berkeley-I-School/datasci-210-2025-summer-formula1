"""Training orchestration and external evaluation utilities"""

from dataclasses import asdict
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score

from ..pipeline import create_safety_car_dataset
from .metadata import EvaluationMetadata, create_metadata_from_f1_dataset


def train_and_validate_model(
    model,
    splits,
    class_names,
    evaluator,
    dataset_metadata,
    model_metadata,
    validate_during_training=True,
):
    """
    Train model with validation monitoring

    Args:
        model: Model to train
        splits: Dictionary from prepare_data_with_validation
        class_names: List of class names
        evaluator: ModelEvaluationSuite instance
        dataset_metadata: DatasetMetadata instance
        model_metadata: ModelMetadata instance
        validate_during_training: Whether to evaluate on validation set

    Returns:
        Dictionary with training results and validation performance
    """
    print(f"\n{'=' * 80}")
    print(f"TRAINING WITH VALIDATION: {model_metadata.model_type}")
    print(f"{'=' * 80}")

    # Train the model
    print("Training on train set...")
    model.fit(splits["X_train"], splits["y_train"])

    results = {}

    # Evaluate on validation set if requested
    if validate_during_training:
        print("\nEvaluating on validation set...")
        val_pred = model.predict(splits["X_val"])

        # Quick validation metrics
        val_accuracy = accuracy_score(splits["y_val"], val_pred)
        val_f1_macro = f1_score(
            splits["y_val"], val_pred, average="macro", zero_division=0
        )

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Macro: {val_f1_macro:.4f}")

        # Store validation results
        results["validation"] = {
            "accuracy": val_accuracy,
            "f1_macro": val_f1_macro,
            "predictions": val_pred.tolist(),
            "y_true": splits["y_val"].tolist(),
        }

    # Full evaluation on test set
    print("\nRunning full evaluation on test set...")
    test_results = evaluator.evaluate_model(
        model=model,
        model_name=model_metadata.model_type,
        X_train=splits["X_train"],  # Pass train data for metadata
        X_test=splits["X_test"],
        y_train=splits["y_train"],
        y_test=splits["y_test"],
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        class_names=list(class_names),
        target_class="safety_car",
        save_results=True,
        evaluation_suffix="",  # No suffix for primary evaluation
    )

    results["test"] = test_results
    results["model"] = model  # Store trained model

    return results


def evaluate_on_external_dataset(
    trained_model,
    external_config,
    original_dataset_metadata,
    model_metadata,
    class_names,
    evaluator,
    resampling_strategy=None,
):
    """
    Evaluate a trained model on a completely different dataset (e.g., different race)

    Args:
        trained_model: Already trained model
        external_config: DataConfig for the external dataset
        original_dataset_metadata: Metadata from training dataset
        model_metadata: ModelMetadata instance
        class_names: List of class names
        evaluator: ModelEvaluationSuite instance
        resampling_strategy: Optional resampling strategy for external dataset
                           ('adasyn', 'smote', 'borderline_smote', None)
                           Note: Usually you want None to evaluate on natural distribution

    Returns:
        Evaluation results on external dataset
    """
    print(f"\n{'=' * 80}")
    print("EXTERNAL DATASET EVALUATION")
    print(f"{'=' * 80}")

    # Load external dataset with same preprocessing as training
    print("Loading external dataset...")
    if resampling_strategy:
        print(f"Note: Applying {resampling_strategy} resampling to external dataset")

    external_dataset = create_safety_car_dataset(
        config=external_config,
        window_size=original_dataset_metadata.window_size,
        prediction_horizon=original_dataset_metadata.prediction_horizon,
        handle_non_numeric="encode",
        handle_missing=True,
        missing_strategy="forward_fill",
        normalize=True,
        normalization_method="per_sequence",
        target_column="TrackStatus",
        resampling_strategy=resampling_strategy,  # Add resampling support
        enable_debug=False,
    )

    # Convert to Aeon format
    X_external = external_dataset["X"].transpose(0, 2, 1)
    y_external = external_dataset["y"]

    print(f"External dataset size: {len(y_external):,} samples")

    # Create metadata for external dataset
    external_metadata = create_metadata_from_f1_dataset(
        data_config=external_config,
        dataset=external_dataset,
        features_used=original_dataset_metadata.features_used,
    )
    external_metadata.scope = f"external_{external_metadata.scope}"

    # Predict on external dataset
    print("Generating predictions...")
    y_pred = trained_model.predict(X_external)

    # Calculate metrics
    metrics = evaluator._calculate_comprehensive_metrics(
        y_external, y_pred, None, list(class_names), "safety_car"
    )

    # Create evaluation metadata with unified ID
    eval_metadata = EvaluationMetadata(
        evaluation_id=f"{model_metadata.model_type}_{evaluator.run_id}_external",
        timestamp=datetime.now().isoformat(),
        test_size=1.0,  # All external data is test data
        target_class_focus="safety_car",
        evaluation_metrics=[
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision",
            "recall",
        ],
    )

    # Create results structure
    results = {
        "evaluation_metadata": asdict(eval_metadata),
        "dataset_metadata": asdict(external_metadata),
        "model_metadata": asdict(model_metadata),
        "metrics": metrics,
        "predictions": {
            "y_true": y_external.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": None,
        },
        "class_info": {
            "class_names": list(class_names),
            "target_class": "safety_car",
            "target_class_index": list(class_names).index("safety_car"),
        },
        "note": f"External dataset evaluation - model trained on different data{' (with resampling)' if resampling_strategy else ''}",
    }

    # Print and save results
    evaluator._print_detailed_analysis(results)
    evaluator._save_results(results, eval_metadata.evaluation_id)

    return results


def compare_performance_across_datasets(training_results, external_results):
    """
    Print performance comparison across different datasets

    Args:
        training_results: Results from train_and_validate_model
        external_results: Results from evaluate_on_external_dataset
    """
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"{'Dataset':<20} {'Accuracy':<10} {'F1-Macro':<10} {'Target F1':<10}")
    print("-" * 60)

    # Validation performance
    if "validation" in training_results:
        print(
            f"{'Validation':<20} {training_results['validation']['accuracy']:<10.4f} "
            f"{training_results['validation']['f1_macro']:<10.4f} {'N/A':<10}"
        )

    # Test performance (same race holdout)
    test_metrics = training_results["test"]["metrics"]
    print(
        f"{'Test (same race)':<20} {test_metrics['overall']['accuracy']:<10.4f} "
        f"{test_metrics['overall']['f1_macro']:<10.4f} "
        f"{test_metrics['target_class_metrics']['f1'] if test_metrics['target_class_metrics'] else 0:<10.4f}"
    )

    # External test performance (different race)
    ext_metrics = external_results["metrics"]
    print(
        f"{'Test (diff race)':<20} {ext_metrics['overall']['accuracy']:<10.4f} "
        f"{ext_metrics['overall']['f1_macro']:<10.4f} "
        f"{ext_metrics['target_class_metrics']['f1'] if ext_metrics['target_class_metrics'] else 0:<10.4f}"
    )
