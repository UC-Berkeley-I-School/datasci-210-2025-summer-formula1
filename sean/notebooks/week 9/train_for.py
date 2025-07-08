"""Example: Training basic models on F1 safety car prediction"""

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


def main():
    # 1. Load dataset
    data_config = DataConfig(
        # sessions=create_multi_session_configs(year=2023, session_types=['R'], include_testing=False),
        sessions=[
            SessionConfig(2024, "Qatar Grand Prix", "R"),
            SessionConfig(2024, "Chinese Grand Prix", "R"),
            # SessionConfig(2024, "Canadian Grand Prix", "R"),
            SessionConfig(2024, "Mexico City Grand Prix", "R"),
            SessionConfig(2024, "São Paulo Grand Prix", "R"),
            SessionConfig(2024, "Miami Grand Prix", "R"),
            # SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
            SessionConfig(2024, "United States Grand Prix", "R"),
            SessionConfig(2024, "Monaco Grand Prix", "R"),
        ],
        drivers=["1"],  # <-- optionally filter down drivers
        include_weather=False,
    )
    # data_config = DataConfig(
    #     # sessions=create_multi_session_configs(year=2023, session_types=['R'], include_testing=False),
    #     sessions=[
    #         SessionConfig(2024, "Qatar Grand Prix", "R", drivers=["27", "31", "43", "23", "77", "11"]),
    #         SessionConfig(2024, "Chinese Grand Prix", "R", drivers=["77", "18", "3", "20", "22"]),
    #         # SessionConfig(2024, "Canadian Grand Prix", "R", drivers=["2", "55", "23"]),
    #         SessionConfig(2024, "Mexico City Grand Prix", "R", drivers=["22", "23"]),
    #         SessionConfig(2024, "São Paulo Grand Prix", "R", drivers=["27", "43", "55"]),
    #         SessionConfig(2024, "Miami Grand Prix", "R", drivers=["1", "20", "2"]),
    #         # SessionConfig(2024, "Saudi Arabian Grand Prix", "R", drivers=["18"]),
    #         SessionConfig(2024, "United States Grand Prix", "R", drivers=["44"]),
    #         SessionConfig(2024, "Monaco Grand Prix", "R", drivers=["11", "20", "27"]),
    #     ],
    #     # drivers=[],  # <-- optionally filter down drivers
    #     include_weather=False,
    # )

    dataset = create_safety_car_dataset(
        config=data_config,
        window_size=50,
        prediction_horizon=100,
        normalize=True,
        target_column="TrackStatus",
        resampling_strategy="smote",
        # resampling_config={"2": 0.5},
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

    # 4. Define class weights
    # y_enc = dataset["label_encoder"]
    # class_weight = {
    #     y_enc.class_to_idx["green"]: 1.0,
    #     y_enc.class_to_idx["red"]: 1.0,
    #     y_enc.class_to_idx["safety_car"]: 5.0,
    #     y_enc.class_to_idx["unknown"]: 1.0,
    #     y_enc.class_to_idx["vsc"]: 1.0,
    #     y_enc.class_to_idx["vsc_ending"]: 1.0,
    #     y_enc.class_to_idx["yellow"]: 1.0,
    # }

    # 5. Train model
    model_name = "rocket_sc_smote_train_sc_eval_driver1"
    # model_name = "rocket_with_imb_pipeline"
    # model_name = "random_forest"

    # models = create_basic_models(class_weight=class_weight)
    # models = create_basic_models()
    # model = models[model_name]

    model = RocketClassifier(n_kernels=1000)
    # model = RocketClassifier(n_kernels=1000, class_weight=class_weight)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"

    evaluator = ModelEvaluationSuite(
        output_dir="evaluation_results",
        run_id=run_id,
    )

    model_metadata = create_model_metadata(
        # model_name=model_name, model=model, class_weights=class_weight
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
            # SessionConfig(2024, "Qatar Grand Prix", "R"),
            # SessionConfig(2024, "Chinese Grand Prix", "R"),
            SessionConfig(2024, "Canadian Grand Prix", "R"),
            # SessionConfig(2024, "Mexico City Grand Prix", "R"),
            # SessionConfig(2024, "São Paulo Grand Prix", "R"),
            # SessionConfig(2024, "Miami Grand Prix", "R"),
            SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
            # SessionConfig(2024, "United States Grand Prix", "R"),
            # SessionConfig(2024, "Monaco Grand Prix", "R"),
        ],
        drivers=["1"],  # <-- optionally filter down drivers
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
