"""Example: Evaluating pretrained F1 safety car prediction models"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime

from f1_etl import (
    DataConfig,
    SessionConfig,
    create_safety_car_dataset,
)
from f1_etl.train import (
    ModelEvaluationSuite,
    create_metadata_from_f1_dataset,
    create_model_metadata,
    evaluate_on_external_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pretrained F1 safety car prediction models')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the saved model pickle file'
    )
    parser.add_argument(
        '--driver', 
        type=str,
        action='append',
        dest='drivers',
        help='Driver number(s) to include in evaluation. Can be specified multiple times.'
    )
    parser.add_argument(
        '--window-size', 
        type=int,
        required=True,
        help='Window size for time series generation (must match training)'
    )
    parser.add_argument(
        '--prediction-horizon', 
        type=int,
        required=True,
        help='Prediction horizon for time series generation (must match training)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the model for metadata'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert driver numbers to strings
    drivers = [str(d) for d in args.drivers] if args.drivers else []

    # Load the pretrained model
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # Create evaluation dataset config
    eval_config = DataConfig(
        sessions=[
            SessionConfig(2024, "Canadian Grand Prix", "R"),
            SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
        ],
        drivers=drivers,
        include_weather=False,
    )

    # Create dataset for getting metadata (using same config as training)
    # This is just to get the label encoder and other metadata
    dummy_config = DataConfig(
        sessions=[SessionConfig(2024, "Qatar Grand Prix", "R")],
        drivers=drivers,
        include_weather=False,
    )
    
    dummy_dataset = create_safety_car_dataset(
        config=dummy_config,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        normalize=True,
        target_column="TrackStatus",
    )

    # Create metadata objects
    dataset_metadata = create_metadata_from_f1_dataset(
        data_config=eval_config,
        dataset=dummy_dataset,
        features_used="multivariate_all_9_features",
    )

    model_metadata = create_model_metadata(
        model_name=args.model_name,
        model=model,
    )

    # Set up evaluator
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval_{args.model_name}"
    evaluator = ModelEvaluationSuite(
        output_dir="evaluation_results",
        run_id=run_id,
    )

    # Get class names from the dummy dataset
    class_names = list(dummy_dataset["label_encoder"].class_to_idx.keys())

    # Evaluate on external dataset
    external_results = evaluate_on_external_dataset(
        trained_model=model,
        external_config=eval_config,
        original_dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        class_names=class_names,
        evaluator=evaluator,
        resampling_strategy=None,  # No resampling for evaluation
        resampling_config=None,
        original_dataset_config=dummy_dataset['config']
    )

    print(f"\nEvaluation complete. Results saved to: {evaluator.run_dir}")
    print(f"Test F1 Score: {external_results['metrics']['f1_score']:.3f}")


if __name__ == "__main__":
    main()