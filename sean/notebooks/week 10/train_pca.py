"""Example: Training basic models on F1 safety car prediction"""

import argparse
from datetime import datetime
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.deep_learning import ResNetClassifier

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
    parser.add_argument(
        '--window-size', 
        type=int,
        action='append',
        dest='window_size',
        help='Window size for time series generation'
    )
    parser.add_argument(
        '--prediction-horizon', 
        type=int,
        action='append',
        dest='prediction_horizon',
        help='Prediction horizon for time series generation and classifier'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert driver numbers to strings and use empty list if none specified
    drivers = [str(d) for d in args.drivers] if args.drivers else []

    # print("window_size=", args.window_size, "type=", type(args.window_size))
    # print("prediction_horizon=", args.prediction_horizon, "type=", type(args.prediction_horizon))

    window_size = args.window_size[0]
    prediction_horizon = args.prediction_horizon[0]
    
    print("window_size=", window_size, "type=", type(window_size))
    print("prediction_horizon=", prediction_horizon, "type=", type(prediction_horizon))

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
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        normalize=True,
        target_column="TrackStatus",
        # resampling_strategy="adasyn",
        # feature_transform="pca",  # Enable PCA            <---- NEW FEATURE
        # pca_variance_threshold=0.95,  # Keep 95% variance <---- NEW FEATURE
        # pca_components=6                                  <---- NEW FEATURE
    )

    # 2. Create metadata
    dataset_metadata = create_metadata_from_f1_dataset(
        data_config=data_config,
        dataset=dataset,
        features_used="multivariate_all_9_features",
    )

    # 3. Prepare data
    splits = prepare_data_with_validation(dataset, val_size=0.00, test_size=0.30)
    class_names = list(dataset["label_encoder"].class_to_idx.keys())

    # 5. Train model

    # model_name = f"rocket_pca_driver{"_".join(drivers)}"
    # model = RocketClassifier(n_kernels=1000)

    model_name = f"rocket_rf_driver{"_".join(drivers)}"
    model = RocketClassifier(
        n_kernels=1000,
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            # class_weight=cls_weight,
            max_depth=10,
        ),
    )

    # model_name = f"catch22_rf_driver{"_".join(drivers)}"
    # model = Catch22Classifier(
    #     estimator=RandomForestClassifier(
    #         n_estimators=100,
    #         random_state=42,
    #         # class_weight=cls_weight,
    #         max_depth=10,
    #     ),
    #     outlier_norm=True,
    #     random_state=42,
    # )

    # model_name = f"catch22_logistic_driver{"_".join(drivers)}"
    # model = Catch22Classifier(
    #     estimator=LogisticRegression(
    #         random_state=42,
    #         max_iter=3000,
    #         # solver='liblinear',
    #         solver="saga",
    #         penalty="l1",
    #         C=0.1,
    #         # class_weight=cls_weight,
    #     ),
    #     outlier_norm=True,
    #     random_state=42,
    # )

    # TODO fix for binary classification
    # model_name = f"catch22_linear_driver{"_".join(drivers)}"
    # model = Catch22Classifier(
    #     estimator=LinearRegression(),
    #     outlier_norm=True,
    #     random_state=42,
    # )

    # model_name = f"inceptiontime_driver{'_'.join(drivers)}"
    # model = InceptionTimeClassifier(
    #     n_classifiers=5,  # ensemble of 5 models
    #     depth=6,  # network depth
    #     n_filters=32,  # number of filters
    #     n_epochs=100,
    #     batch_size=16,
    #     random_state=42,
    #     verbose=False,
    # )

    # model_name = f"resnet_driver{'_'.join(drivers)}"
    # model = ResNetClassifier(
    #     n_residual_blocks=3,
    #     n_filters=[128, 256, 128],
    #     n_epochs=100,
    #     batch_size=16,
    #     random_state=42,
    #     verbose=False
    # )

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_windows_{window_size}_horizon_{prediction_horizon}_{model_name}"

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
        resampling_strategy=dataset_metadata.resampling_strategy,
        resampling_config=dataset_metadata.resampling_config,
        original_dataset_config=dataset['config']
    )

    # 7. Compare results
    compare_performance_across_datasets(training_results, external_results)


if __name__ == "__main__":
    main()
