"""Model evaluation suite with comprehensive metrics and reporting"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .metadata import DatasetMetadata, EvaluationMetadata, ModelMetadata


class ModelEvaluationSuite:
    """Comprehensive model evaluation with metadata tracking and file output"""

    def __init__(
        self, output_dir: str = "evaluation_results", run_id: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate run ID once for the entire script execution
        if run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_id = run_id

        # Create subdirectory for this run
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

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
        evaluation_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with metadata capture

        Args:
            evaluation_suffix: Optional suffix for the evaluation (e.g., "external" for external dataset)
        """

        # Generate evaluation metadata with unified ID
        eval_id_parts = [model_name, self.run_id]
        if evaluation_suffix:
            eval_id_parts.append(evaluation_suffix)

        eval_metadata = EvaluationMetadata(
            evaluation_id="_".join(eval_id_parts),
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
        print(
            f"EVALUATING: {model_name.upper()}{' - ' + evaluation_suffix.upper() if evaluation_suffix else ''}"
        )
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
        """Save results to JSON and summary text files in run subdirectory"""

        # Save complete results as JSON
        json_path = self.run_dir / f"{evaluation_id}_complete.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save human-readable summary
        summary_path = self.run_dir / f"{evaluation_id}_summary.txt"
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
        f.write(
            f"Drivers: {', '.join(dataset_meta['drivers']) if dataset_meta['drivers'] else 'all_drivers'}\n"
        )
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

            # Add confusion matrix to the report
            f.write("\nCONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            cm = np.array(metrics["confusion_matrix"])
            class_names_list = results["class_info"]["class_names"]
            # Convert to list if it's a numpy array
            if hasattr(class_names_list, "tolist"):
                class_names_list = class_names_list.tolist()
            unique_classes = sorted(
                np.unique(
                    np.concatenate(
                        [
                            results["predictions"]["y_true"],
                            results["predictions"]["y_pred"],
                        ]
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
            f.write(cm_df.to_string())
            f.write("\n")

        else:
            f.write("ERROR OCCURRED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Error: {results.get('error', 'Unknown error')}\n")
