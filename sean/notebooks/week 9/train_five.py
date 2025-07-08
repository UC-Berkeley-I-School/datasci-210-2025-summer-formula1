"""Example: Training basic models on F1 safety car prediction"""

import argparse
from datetime import datetime
from aeon.classification.convolution_based import RocketClassifier


from f1_etl import (
    DataConfig,
    SessionConfig,
    create_safety_car_dataset,
)
from f1_etl.train import (
    ModelEvaluationSuite,
    create_metadata_from_f1_dataset,
    prepare_data_with_validation,
    create_model_metadata,
    train_and_validate_model,
    evaluate_on_external_dataset,
    compare_performance_across_datasets,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train F1 safety car prediction models')
    parser.add_argument(
        '--driver', 
        type=str,
        action='append',
        dest='drivers',
        help='Driver number(s) to include in training. Can be specified multiple times.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert driver numbers to strings and use empty list if none specified
    drivers = [str(d) for d in args.drivers] if args.drivers else []
    
    # 1. Load dataset
    data_config = DataConfig(
        sessions=[
            SessionConfig(2024, "Qatar Grand Prix", "R"),
            SessionConfig(2024, "Chinese Grand Prix", "R"),
            SessionConfig(2024, "Mexico City Grand Prix", "R"),
            SessionConfig(2024, "São Paulo Grand Prix", "R"),
            SessionConfig(2024, "Miami Grand Prix", "R"),
            SessionConfig(2024, "United States Grand Prix", "R"),
            SessionConfig(2024, "Monaco Grand Prix", "R"),
        ],
        drivers=drivers,
        include_weather=False,
    )

    dataset = create_safety_car_dataset(
        config=data_config,
        window_size=50,
        prediction_horizon=100,
        normalize=True,
        target_column="TrackStatus",
        resampling_strategy="smote",
    )

    # 2. Create metadata
    dataset_metadata = create_metadata_from_f1_dataset(
        data_config=data_config,
        dataset=dataset,
        features_used="multivariate_all_9_features",
    )

    # 3. Prepare data
    splits = prepare_data_with_validation(dataset, val_size=0.15, test_size=0.15)
    class_names = list(dataset["label_encoder"].class_to_idx.keys())

    # 5. Train model
    model_name = "rocket_sc_smote_train_sc_eval_driver1"
    model = RocketClassifier(n_kernels=1000)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"

    evaluator = ModelEvaluationSuite(
        output_dir="evaluation_results",
        run_id=run_id,
    )

    model_metadata = create_model_metadata(
        model_name=model_name,
        model=model,
    )

    training_results = train_and_validate_model(
        model=model,
        splits=splits,
        class_names=class_names,
        evaluator=evaluator,
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
    )

    # 6. Evaluate on external dataset
    external_config = DataConfig(
        sessions=[
            SessionConfig(2024, "Canadian Grand Prix", "R"),
            SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
        ],
        drivers=drivers,  # Use same drivers as training
        include_weather=False,
    )

    external_results = evaluate_on_external_dataset(
        trained_model=training_results["model"],
        external_config=external_config,
        original_dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        class_names=class_names,
        evaluator=evaluator,
    )

    # 7. Compare results
    compare_performance_across_datasets(training_results, external_results)


if __name__ == "__main__":
    main()