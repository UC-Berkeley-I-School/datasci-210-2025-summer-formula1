# %%
# F1 Safety Car Prediction - Step-by-Step Jupyter Notebook Implementation

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.dummy import DummyClassifier


from f1_etl import SessionConfig, DataConfig, create_safety_car_dataset


# %%
@dataclass
class DatasetMetadata:
    """Captures dataset configuration and characteristics"""

    scope: str  # e.g., "2024_season_races", "single_session", etc.
    sessions_config: Dict[str, Any]  # Original sessions configuration
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

    # Dataset characteristics
    total_samples: int
    n_features: int
    n_timesteps: int
    feature_names: Optional[List[str]] = None
    class_distribution: Optional[Dict[str, int]] = None

    # Processing details
    features_used: str = "all"  # "all", "speed_only", "custom_subset", etc.
    is_multivariate: bool = True
    preprocessing_steps: List[str] = None


@dataclass
class ModelMetadata:
    """Captures model configuration and hyperparameters"""

    model_type: str  # e.g., "logistic_regression", "random_forest"
    base_estimator: str  # e.g., "LogisticRegression", "RandomForestClassifier"
    wrapper: str = "Catch22Classifier"  # Aeon wrapper used

    # Hyperparameters
    hyperparameters: Dict[str, Any] = None
    class_weights: Optional[Dict[int, float]] = None
    custom_weights_applied: bool = False

    # Training details
    random_state: Optional[int] = 42
    cv_strategy: Optional[str] = None  # If cross-validation used


@dataclass
class EvaluationMetadata:
    """Captures evaluation context and settings"""

    evaluation_id: str
    timestamp: str
    test_size: float
    stratified_split: bool = True
    target_class_focus: str = "safety_car"
    evaluation_metrics: List[str] = None


class ModelEvaluationSuite:
    """Comprehensive model evaluation with metadata tracking and file output"""

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def evaluate_model(
        self,
        model,
        model_name: str,
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_metadata: DatasetMetadata,
        model_metadata: ModelMetadata,
        class_names: List[str],
        target_class: str = "safety_car",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with metadata capture
        """

        # Generate evaluation metadata
        eval_metadata = EvaluationMetadata(
            evaluation_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            test_size=len(X_test) / (len(X_train) + len(X_test)),
            target_class_focus=target_class,
            evaluation_metrics=[
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "precision",
                "recall",
            ],
        )

        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"Evaluation ID: {eval_metadata.evaluation_id}")
        print(f"{'=' * 80}")

        try:
            # Train model
            print("Training model...")
            model.fit(X_train, y_train)

            # Generate predictions
            print("Generating predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba, class_names, target_class
            )

            # Create results structure
            results = {
                "evaluation_metadata": asdict(eval_metadata),
                "dataset_metadata": asdict(dataset_metadata),
                "model_metadata": asdict(model_metadata),
                "metrics": metrics,
                "predictions": {
                    "y_true": y_test.tolist()
                    if hasattr(y_test, "tolist")
                    else list(y_test),
                    "y_pred": y_pred.tolist()
                    if hasattr(y_pred, "tolist")
                    else list(y_pred),
                    "y_pred_proba": y_pred_proba.tolist()
                    if y_pred_proba is not None
                    else None,
                },
                "class_info": {
                    "class_names": class_names,
                    "target_class": target_class,
                    "target_class_index": class_names.index(target_class)
                    if target_class in class_names
                    else None,
                },
            }

            # Print detailed analysis
            self._print_detailed_analysis(results)

            # Save results if requested
            if save_results:
                self._save_results(results, eval_metadata.evaluation_id)

            return results

        except Exception as e:
            error_results = {
                "evaluation_metadata": asdict(eval_metadata),
                "dataset_metadata": asdict(dataset_metadata),
                "model_metadata": asdict(model_metadata),
                "error": str(e),
                "model_name": model_name,
            }

            if save_results:
                self._save_results(error_results, eval_metadata.evaluation_id)

            print(f"ERROR: {str(e)}")
            return error_results

    def _calculate_comprehensive_metrics(
        self, y_true, y_pred, y_pred_proba, class_names, target_class
    ):
        """Calculate comprehensive evaluation metrics"""

        # Convert class_names to list consistently at the start
        class_names_list = (
            class_names.tolist()
            if hasattr(class_names, "tolist")
            else list(class_names)
        )

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Target class specific metrics
        target_metrics = {}
        target_idx = None
        if target_class in class_names_list:
            target_idx = class_names_list.index(target_class)
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

            if target_idx in unique_classes:
                target_in_cm = unique_classes.index(target_idx)
                if target_in_cm < cm.shape[0] and target_in_cm < cm.shape[1]:
                    tp = cm[target_in_cm, target_in_cm]
                    fn = cm[target_in_cm, :].sum() - tp
                    fp = cm[:, target_in_cm].sum() - tp
                    tn = cm.sum() - tp - fn - fp

                    target_metrics = {
                        "true_positives": int(tp),
                        "false_negatives": int(fn),
                        "false_positives": int(fp),
                        "true_negatives": int(tn),
                        "precision": float(
                            precision[target_in_cm]
                            if target_in_cm < len(precision)
                            else 0
                        ),
                        "recall": float(
                            recall[target_in_cm] if target_in_cm < len(recall) else 0
                        ),
                        "f1": float(f1[target_in_cm] if target_in_cm < len(f1) else 0),
                        "support": int(
                            support[target_in_cm] if target_in_cm < len(support) else 0
                        ),
                    }

        # Per-class metrics dictionary
        per_class_metrics = {}
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        # Convert class_names to list if it's a numpy array
        class_names_list = (
            class_names.tolist()
            if hasattr(class_names, "tolist")
            else list(class_names)
        )
        for i, class_idx in enumerate(unique_classes):
            if class_idx < len(class_names_list) and i < len(precision):
                per_class_metrics[class_names_list[class_idx]] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }

        return {
            "overall": {
                "accuracy": float(accuracy),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
            },
            "per_class": per_class_metrics,
            "target_class_metrics": target_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=[
                    class_names_list[i]
                    for i in unique_classes
                    if i < len(class_names_list)
                ],
                zero_division=0,
                output_dict=True,
            ),
        }

    def _print_detailed_analysis(self, results):
        """Print comprehensive analysis to console"""

        metrics = results["metrics"]
        target_class = results["class_info"]["target_class"]

        print("\n📊 OVERALL PERFORMANCE")
        print(f"{'=' * 50}")
        print(f"Accuracy:    {metrics['overall']['accuracy']:.4f}")
        print(f"F1-Macro:    {metrics['overall']['f1_macro']:.4f}")
        print(f"F1-Weighted: {metrics['overall']['f1_weighted']:.4f}")

        if metrics["target_class_metrics"]:
            print(f"\n🎯 TARGET CLASS ANALYSIS: {target_class.upper()}")
            print(f"{'=' * 50}")
            tm = metrics["target_class_metrics"]
            print(f"Precision:       {tm['precision']:.4f}")
            print(f"Recall:          {tm['recall']:.4f}")
            print(f"F1-Score:        {tm['f1']:.4f}")
            print(f"True Positives:  {tm['true_positives']:4d}")
            print(
                f"False Negatives: {tm['false_negatives']:4d} (missed {target_class} events)"
            )
            print(
                f"False Positives: {tm['false_positives']:4d} (false {target_class} alarms)"
            )
            print(f"True Negatives:  {tm['true_negatives']:4d}")

        print("\n📈 PER-CLASS PERFORMANCE")
        print(f"{'=' * 50}")
        for class_name, class_metrics in metrics["per_class"].items():
            print(
                f"{class_name:12s}: P={class_metrics['precision']:.3f}, "
                f"R={class_metrics['recall']:.3f}, "
                f"F1={class_metrics['f1']:.3f}, "
                f"N={class_metrics['support']}"
            )

        print("\n🔍 CONFUSION MATRIX")
        print(f"{'=' * 50}")
        cm = np.array(metrics["confusion_matrix"])
        class_names_list = results["class_info"]["class_names"]
        # Convert to list if it's a numpy array
        if hasattr(class_names_list, "tolist"):
            class_names_list = class_names_list.tolist()
        unique_classes = sorted(
            np.unique(
                np.concatenate(
                    [results["predictions"]["y_true"], results["predictions"]["y_pred"]]
                )
            )
        )
        present_class_names = [
            class_names_list[i] for i in unique_classes if i < len(class_names_list)
        ]

        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in present_class_names],
            columns=[f"Pred_{name}" for name in present_class_names],
        )
        print(cm_df.to_string())

    def _save_results(self, results, evaluation_id):
        """Save results to JSON and summary text files"""

        # Save complete results as JSON
        json_path = self.output_dir / f"{evaluation_id}_complete.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save human-readable summary
        summary_path = self.output_dir / f"{evaluation_id}_summary.txt"
        with open(summary_path, "w") as f:
            self._write_summary_report(results, f)

        print("\n💾 Results saved:")
        print(f"  Complete: {json_path}")
        print(f"  Summary:  {summary_path}")

    def _write_summary_report(self, results, file_handle):
        """Write human-readable summary report"""

        f = file_handle
        eval_meta = results["evaluation_metadata"]
        dataset_meta = results["dataset_metadata"]
        model_meta = results["model_metadata"]

        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Evaluation Overview
        f.write("EVALUATION OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Evaluation ID: {eval_meta['evaluation_id']}\n")
        f.write(f"Timestamp: {eval_meta['timestamp']}\n")
        f.write(f"Target Class: {eval_meta['target_class_focus']}\n")
        f.write(f"Test Size: {eval_meta['test_size']:.1%}\n\n")

        # Dataset Information
        f.write("DATASET CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Scope: {dataset_meta['scope']}\n")
        f.write(f"Drivers: {', '.join(dataset_meta['drivers'])}\n")
        f.write(f"Window Size: {dataset_meta['window_size']}\n")
        f.write(f"Prediction Horizon: {dataset_meta['prediction_horizon']}\n")
        f.write(f"Features Used: {dataset_meta['features_used']}\n")
        f.write(f"Multivariate: {dataset_meta['is_multivariate']}\n")
        f.write(f"Total Samples: {dataset_meta['total_samples']:,}\n")
        f.write(
            f"Shape: ({dataset_meta['total_samples']}, {dataset_meta['n_features']}, {dataset_meta['n_timesteps']})\n"
        )
        if dataset_meta["class_distribution"]:
            f.write("Class Distribution:\n")
            for class_name, count in dataset_meta["class_distribution"].items():
                f.write(f"  {class_name}: {count:,}\n")
        f.write("\n")

        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: {model_meta['model_type']}\n")
        f.write(f"Base Estimator: {model_meta['base_estimator']}\n")
        f.write(f"Wrapper: {model_meta['wrapper']}\n")
        f.write(f"Custom Weights: {model_meta['custom_weights_applied']}\n")
        if model_meta["hyperparameters"]:
            f.write("Hyperparameters:\n")
            for param, value in model_meta["hyperparameters"].items():
                f.write(f"  {param}: {value}\n")
        if model_meta["class_weights"]:
            f.write("Class Weights:\n")
            for class_idx, weight in model_meta["class_weights"].items():
                f.write(f"  Class {class_idx}: {weight}\n")
        f.write("\n")

        # Performance Results
        if "metrics" in results:
            metrics = results["metrics"]
            target_class = results["class_info"]["target_class"]

            f.write("PERFORMANCE RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}\n")
            f.write(f"F1-Macro: {metrics['overall']['f1_macro']:.4f}\n")
            f.write(f"F1-Weighted: {metrics['overall']['f1_weighted']:.4f}\n\n")

            if metrics["target_class_metrics"]:
                f.write(f"TARGET CLASS ANALYSIS: {target_class.upper()}\n")
                f.write("-" * 40 + "\n")
                tm = metrics["target_class_metrics"]
                f.write(f"Precision: {tm['precision']:.4f}\n")
                f.write(f"Recall: {tm['recall']:.4f}\n")
                f.write(f"F1-Score: {tm['f1']:.4f}\n")
                f.write(f"True Positives: {tm['true_positives']:,}\n")
                f.write(f"False Negatives: {tm['false_negatives']:,} (missed events)\n")
                f.write(f"False Positives: {tm['false_positives']:,} (false alarms)\n")
                f.write(f"True Negatives: {tm['true_negatives']:,}\n\n")

            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n"
            )
            f.write("-" * 60 + "\n")
            for class_name, class_metrics in metrics["per_class"].items():
                f.write(
                    f"{class_name:<12} {class_metrics['precision']:<10.3f} "
                    f"{class_metrics['recall']:<10.3f} {class_metrics['f1']:<10.3f} "
                    f"{class_metrics['support']:<10}\n"
                )

        else:
            f.write("ERROR OCCURRED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Error: {results.get('error', 'Unknown error')}\n")


# Helper functions for easy metadata creation
def create_dataset_metadata_from_config(dataset_config, dataset, features_used="all"):
    """Create DatasetMetadata from dataset configuration and dataset object"""

    X = dataset["X"]
    y = dataset["y"]

    # Determine scope description
    sessions = dataset_config.sessions if hasattr(dataset_config, "sessions") else []
    if len(sessions) == 1:
        # SessionConfig is a dataclass, access attributes directly
        session = sessions[0]
        year = getattr(session, "year", "unknown")
        scope = f"single_session_{year}"
    elif len(sessions) > 1:
        # Extract years from SessionConfig objects
        years = list(set(getattr(s, "year", None) for s in sessions))
        years = [y for y in years if y is not None]  # Filter out None values
        if years:
            scope = f"multi_session_{'-'.join(map(str, sorted(years)))}"
        else:
            scope = "multi_session_unknown_years"
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

    return DatasetMetadata(
        scope=scope,
        sessions_config=[
            {
                "year": getattr(s, "year", None),
                "race": getattr(s, "race", None),
                "session_type": getattr(s, "session_type", None),
            }
            for s in sessions
        ],  # Convert SessionConfig objects to dicts
        drivers=getattr(dataset_config, "drivers", []),
        include_weather=getattr(dataset_config, "include_weather", False),
        window_size=getattr(dataset_config, "window_size", 100),
        prediction_horizon=getattr(dataset_config, "prediction_horizon", 10),
        handle_non_numeric="encode",  # Default values
        handle_missing=True,
        missing_strategy="forward_fill",
        normalize=True,
        normalization_method="per_sequence",
        target_column="TrackStatus",
        total_samples=X.shape[0],
        n_features=X.shape[1] if len(X.shape) > 1 else 1,
        n_timesteps=X.shape[2] if len(X.shape) > 2 else X.shape[1],
        class_distribution=class_dist,
        features_used=features_used,
        is_multivariate=len(X.shape) > 2 and X.shape[1] > 1,
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


# %%
# 1. Create the evaluation suite
evaluator = ModelEvaluationSuite(output_dir="evaluation_results")

# %%
# sessions_2024_season = create_multi_session_configs(
#     year=2024,
#     session_types=['R'],
#     include_testing=False
# )
# data_config = DataConfig(
#     sessions=sessions_2024_season,
#     drivers=['2'],
#     include_weather=False
# )
data_config = DataConfig(
    sessions=[SessionConfig(2024, "Saudia Arabian Grand Prix", "R")],
    drivers=["1"],
    include_weather=False,
)

# %%
dataset = create_safety_car_dataset(
    config=data_config,
    window_size=100,
    prediction_horizon=10,
    handle_non_numeric="encode",
    handle_missing=True,
    missing_strategy="forward_fill",
    normalize=True,
    normalization_method="per_sequence",
    target_column="TrackStatus",
    enable_debug=False,
)

# %%
# Create dataset metadata from configuration
dataset_metadata = create_metadata_from_f1_dataset(
    data_config=data_config,
    dataset=dataset,
    features_used="multivariate_all_9_features",
)

# %%
# 3. Create class names list from your label encoder
class_names = list(dataset["label_encoder"].class_to_idx.keys())


# %%
def prepare_data(dataset, test_size=0.2):
    """Prepare train/test splits and convert to Aeon format"""

    X = dataset["X"]  # Shape: (n_samples, n_timesteps, n_features)
    y = dataset["y"]  # Encoded labels

    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)

    # Use only Speed feature (index 0) for simplicity
    # X_speed = X_aeon[:, 0:1, :]  # Keep 3D: (n_samples, 1, n_timesteps)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        # X_speed, y, test_size=test_size, random_state=42, stratify=y
        X_aeon,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def analyze_class_distribution(dataset, y_train):
    """Analyze and display class distribution"""

    # Get class names
    label_encoder = dataset["label_encoder"]
    class_names = label_encoder.get_classes()

    # Count classes
    unique, counts = np.unique(y_train, return_counts=True)

    print("\n=== CLASS DISTRIBUTION ===")
    for class_id, count in zip(unique, counts):
        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"Class_{class_id}"
        )
        percentage = count / len(y_train) * 100
        print(f"{class_name:12s}: {count:5d} samples ({percentage:5.1f}%)")

    imbalance_ratio = max(counts) / min(counts)
    print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")

    return class_names, dict(zip(unique, counts))


# Example usage:
X_train, X_test, y_train, y_test = prepare_data(dataset)
class_names, class_dist = analyze_class_distribution(dataset, y_train)

# %%
X_train.shape

# %%
y_enc = dataset["label_encoder"]
y_enc.class_to_idx

# %%
# class_weight = {
#     y_enc.class_to_idx['green']: 1.0,        # 0
#     y_enc.class_to_idx['red']: 10.0,         # 1
#     y_enc.class_to_idx['safety_car']: 20.0,  # 2 (your target class)
#     y_enc.class_to_idx['unknown']: 5.0,      # 3
#     y_enc.class_to_idx['vsc']: 30.0,         # 4
#     y_enc.class_to_idx['vsc_ending']: 100.0, # 5
#     y_enc.class_to_idx['yellow']: 8.0        # 6
# }

class_weight = {
    y_enc.class_to_idx["green"]: 1.0,
    y_enc.class_to_idx["red"]: 25.0,
    y_enc.class_to_idx["safety_car"]: 100.0,  # Much higher
    y_enc.class_to_idx["unknown"]: 25.0,
    y_enc.class_to_idx["vsc"]: 50.0,
    y_enc.class_to_idx["vsc_ending"]: 50.0,
    y_enc.class_to_idx["yellow"]: 25.0,
}


# %%
def create_models(class_weight: Optional[Dict] = None):
    """Create dictionary of models to test"""

    cls_weight = "balanced" if class_weight is None else class_weight

    models = {
        "dummy_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified"),
        "logistic_regression": Catch22Classifier(
            estimator=LogisticRegression(
                random_state=42,
                max_iter=3000,
                # solver='liblinear',
                solver="saga",
                penalty="l1",
                C=0.1,
                class_weight=cls_weight,
            ),
            outlier_norm=True,
            random_state=42,
        ),
        "random_forest": Catch22Classifier(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight=cls_weight, max_depth=10
            ),
            outlier_norm=True,
            random_state=42,
        ),
    }

    return models


def train_single_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""

    print(f"\nTraining {model_name}...")

    try:
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        results = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "predictions": y_pred,
            "true_labels": y_test,
        }

        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Macro: {results['f1_macro']:.4f}")
        print(f"  F1-Weighted: {results['f1_weighted']:.4f}")

        return results

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {"model_name": model_name, "error": str(e)}


# Example usage:
# models = create_models()
# results = {}
# for model_name, model in models.items():
#     results[model_name] = train_single_model(model, model_name, X_train, X_test, y_train, y_test)

# %%
models = create_models(class_weight=class_weight)
results = {}

model_name = "logistic_regression"
model = models[model_name]

# results[model_name] = train_single_model(model, model_name, X_train, X_test, y_train, y_test)

# %%
# Create model metadata
model_metadata = create_model_metadata(
    model_name=model_name, model=model, class_weights=class_weight
)

# %%
# Run comprehensive evaluation
results = evaluator.evaluate_model(
    model=model,
    model_name=model_name,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    dataset_metadata=dataset_metadata,
    model_metadata=model_metadata,
    class_names=list(class_names),  # Ensure it's a list
    target_class="safety_car",
    save_results=True,
)


# %%
# 5. For evaluating against a different test set
def evaluate_on_different_test_set(
    model, test_data_path, evaluator, model_metadata, dataset_metadata
):
    """Evaluate trained model on a different test set"""

    # Load different test set (adjust based on your data loading)
    test_dataset = load_test_dataset(test_data_path)
    X_test_new = test_dataset["X"].transpose(0, 2, 1)  # Convert to Aeon format
    y_test_new = test_dataset["y"]

    # Create new dataset metadata for this test set
    test_dataset_metadata = create_dataset_metadata_from_config(
        dataset_config=test_data_config,  # Different config for test set
        dataset=test_dataset,
        features_used="multivariate_all_9_features",
    )
    test_dataset_metadata.scope = f"external_test_set_{test_dataset_metadata.scope}"

    # Evaluate without retraining (model already fitted)
    try:
        y_pred_new = model.predict(X_test_new)

        # Create mock training data for the evaluator (since model is pre-trained)
        # We pass empty arrays since the model won't be retrained
        dummy_train = (np.array([]), np.array([]))

        # Create a custom evaluation that skips training
        eval_metadata = EvaluationMetadata(
            evaluation_id=f"{model_metadata.model_type}_external_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            test_size=1.0,  # 100% test data
            target_class_focus="safety_car",
        )

        # Calculate metrics directly
        metrics = evaluator._calculate_comprehensive_metrics(
            y_test_new, y_pred_new, None, class_names, "safety_car"
        )

        # Create results structure
        results = {
            "evaluation_metadata": asdict(eval_metadata),
            "dataset_metadata": asdict(test_dataset_metadata),
            "model_metadata": asdict(model_metadata),
            "metrics": metrics,
            "predictions": {
                "y_true": y_test_new.tolist(),
                "y_pred": y_pred_new.tolist(),
                "y_pred_proba": None,
            },
            "class_info": {
                "class_names": class_names,
                "target_class": "safety_car",
                "target_class_index": class_names.index("safety_car"),
            },
            "note": "External test set evaluation - model was pre-trained",
        }

        # Print and save results
        evaluator._print_detailed_analysis(results)
        evaluator._save_results(results, eval_metadata.evaluation_id)

        return results

    except Exception as e:
        print(f"Error evaluating on external test set: {e}")
        return None


# 6. Batch evaluation of multiple models
def run_comprehensive_model_comparison():
    """Run evaluation on all models and save results"""

    models = create_models()
    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 100}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'=' * 100}")

        # Create model-specific metadata
        model_metadata = create_model_metadata(
            model_name=model_name,
            model=model,
            class_weights=class_weight
            if "logistic" in model_name or "forest" in model_name
            else None,
        )

        # Run evaluation
        results = evaluator.evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_metadata=dataset_metadata,
            model_metadata=model_metadata,
            class_names=class_names,
            target_class="safety_car",
            save_results=True,
        )

        all_results[model_name] = results

    # Save comparison summary
    comparison_id = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    comparison_path = evaluator.output_dir / f"{comparison_id}_comparison.json"

    with open(comparison_path, "w") as f:
        json.dump(
            {
                "comparison_id": comparison_id,
                "timestamp": datetime.now().isoformat(),
                "models_compared": list(all_results.keys()),
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n📊 Model comparison saved: {comparison_path}")
    return all_results


# 7. Example with different feature configurations
def evaluate_feature_configurations():
    """Evaluate same model with different feature configurations"""

    configurations = [
        {
            "name": "speed_only",
            "X_data": X_aeon[:, 0:1, :],  # Speed only
            "description": "univariate_speed_only",
        },
        {
            "name": "speed_and_throttle",
            "X_data": X_aeon[:, 0:2, :],  # Speed and throttle
            "description": "bivariate_speed_throttle",
        },
        {
            "name": "all_features",
            "X_data": X_aeon,  # All features
            "description": "multivariate_all_9_features",
        },
    ]

    base_model_name = "logistic_regression"
    base_model = models[base_model_name]

    for config in configurations:
        print(f"\n{'=' * 80}")
        print(f"FEATURE CONFIGURATION: {config['name'].upper()}")
        print(f"{'=' * 80}")

        # Split data for this configuration
        X_train_config, X_test_config, y_train_config, y_test_config = train_test_split(
            config["X_data"], y, test_size=0.2, random_state=42, stratify=y
        )

        # Update dataset metadata
        config_dataset_metadata = dataset_metadata
        config_dataset_metadata.features_used = config["description"]
        config_dataset_metadata.n_features = config["X_data"].shape[1]
        config_dataset_metadata.is_multivariate = config["X_data"].shape[1] > 1

        # Create fresh model instance
        from sklearn.base import clone

        model_config = clone(base_model)

        # Update model metadata
        config_model_metadata = create_model_metadata(
            model_name=f"{base_model_name}_{config['name']}",
            model=model_config,
            class_weights=class_weight,
        )

        # Run evaluation
        results = evaluator.evaluate_model(
            model=model_config,
            model_name=f"{base_model_name}_{config['name']}",
            X_train=X_train_config,
            X_test=X_test_config,
            y_train=y_train_config,
            y_test=y_test_config,
            dataset_metadata=config_dataset_metadata,
            model_metadata=config_model_metadata,
            class_names=class_names,
            target_class="safety_car",
            save_results=True,
        )


# 8. Quick evaluation function for iterative testing
def quick_eval(model_name, custom_weights=None, feature_subset=None):
    """Quick evaluation for rapid iteration"""

    # Prepare data
    X_data = X_aeon if feature_subset is None else X_aeon[:, feature_subset, :]
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        X_data, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    if model_name == "logistic_regression":
        from aeon.classification.feature_based import Catch22Classifier
        from sklearn.linear_model import LogisticRegression

        model = Catch22Classifier(
            estimator=LogisticRegression(
                random_state=42,
                max_iter=3000,
                solver="saga",
                penalty="l1",
                C=0.1,
                class_weight=custom_weights or class_weight,
            ),
            outlier_norm=True,
            random_state=42,
        )
    else:
        raise ValueError(f"Quick eval not implemented for {model_name}")

    # Create metadata
    feature_desc = (
        f"custom_subset_{len(feature_subset) if feature_subset else 'all'}_features"
    )

    quick_dataset_metadata = dataset_metadata
    quick_dataset_metadata.features_used = feature_desc
    quick_dataset_metadata.n_features = X_data.shape[1]

    quick_model_metadata = create_model_metadata(
        model_name=f"{model_name}_quick",
        model=model,
        class_weights=custom_weights or class_weight,
    )

    # Run evaluation
    return evaluator.evaluate_model(
        model=model,
        model_name=f"{model_name}_quick",
        X_train=X_train_q,
        X_test=X_test_q,
        y_train=y_train_q,
        y_test=y_test_q,
        dataset_metadata=quick_dataset_metadata,
        model_metadata=quick_model_metadata,
        class_names=class_names,
        target_class="safety_car",
        save_results=True,
    )


# 9. Load and analyze previous results
def analyze_previous_results(results_dir="evaluation_results"):
    """Load and analyze previous evaluation results"""

    results_path = Path(results_dir)
    json_files = list(results_path.glob("*_complete.json"))

    if not json_files:
        print("No previous results found")
        return

    print(f"Found {len(json_files)} previous evaluations:")

    summaries = []
    for json_file in sorted(json_files):
        with open(json_file, "r") as f:
            result = json.load(f)

        if "metrics" in result:
            summary = {
                "file": json_file.name,
                "timestamp": result["evaluation_metadata"]["timestamp"],
                "model": result["model_metadata"]["model_type"],
                "features": result["dataset_metadata"]["features_used"],
                "accuracy": result["metrics"]["overall"]["accuracy"],
                "target_f1": result["metrics"]["target_class_metrics"].get("f1", 0)
                if result["metrics"]["target_class_metrics"]
                else 0,
                "target_recall": result["metrics"]["target_class_metrics"].get(
                    "recall", 0
                )
                if result["metrics"]["target_class_metrics"]
                else 0,
                "target_precision": result["metrics"]["target_class_metrics"].get(
                    "precision", 0
                )
                if result["metrics"]["target_class_metrics"]
                else 0,
            }
        else:
            summary = {
                "file": json_file.name,
                "timestamp": result["evaluation_metadata"]["timestamp"],
                "model": result["model_metadata"]["model_type"],
                "error": result.get("error", "Unknown error"),
            }

        summaries.append(summary)

    # Create comparison DataFrame
    df = pd.DataFrame(summaries)

    if "accuracy" in df.columns:
        df_success = df[df["accuracy"].notna()].copy()
        df_success = df_success.sort_values("target_f1", ascending=False)

        print("\n📊 PERFORMANCE COMPARISON (sorted by target class F1):")
        print("=" * 100)
        print(
            f"{'Model':<20} {'Features':<25} {'Accuracy':<10} {'Target F1':<10} {'Recall':<10} {'Precision':<10}"
        )
        print("-" * 100)

        for _, row in df_success.iterrows():
            print(
                f"{row['model']:<20} {row['features']:<25} {row['accuracy']:<10.4f} "
                f"{row['target_f1']:<10.4f} {row['target_recall']:<10.4f} {row['target_precision']:<10.4f}"
            )

    if "error" in df.columns and df["error"].notna().any():
        df_errors = df[df["error"].notna()]
        print(f"\n❌ FAILED EVALUATIONS ({len(df_errors)}):")
        for _, row in df_errors.iterrows():
            print(f"  {row['model']}: {row['error']}")

    return df


# 10. Example usage patterns

# Basic single model evaluation
print("=== BASIC SINGLE MODEL EVALUATION ===")
results = evaluator.evaluate_model(
    model=models["logistic_regression"],
    model_name="logistic_regression",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    dataset_metadata=dataset_metadata,
    model_metadata=create_model_metadata(
        "logistic_regression", models["logistic_regression"], class_weight
    ),
    class_names=class_names,
    target_class="safety_car",
)

# Quick iteration testing
print("\n=== QUICK ITERATION TESTING ===")
# Test with just speed feature
speed_results = quick_eval("logistic_regression", feature_subset=[0])

# Test with different class weights
high_safety_weights = {
    dataset["label_encoder"].class_to_idx["green"]: 1.0,
    dataset["label_encoder"].class_to_idx["red"]: 50.0,
    dataset["label_encoder"].class_to_idx["safety_car"]: 200.0,  # Even higher
    dataset["label_encoder"].class_to_idx["unknown"]: 50.0,
    dataset["label_encoder"].class_to_idx["vsc"]: 100.0,
    dataset["label_encoder"].class_to_idx["vsc_ending"]: 100.0,
    dataset["label_encoder"].class_to_idx["yellow"]: 50.0,
}
high_weight_results = quick_eval(
    "logistic_regression", custom_weights=high_safety_weights
)

# Comprehensive model comparison
print("\n=== COMPREHENSIVE MODEL COMPARISON ===")
# all_model_results = run_comprehensive_model_comparison()

# Analyze previous results
print("\n=== PREVIOUS RESULTS ANALYSIS ===")
# previous_results_df = analyze_previous_results()
