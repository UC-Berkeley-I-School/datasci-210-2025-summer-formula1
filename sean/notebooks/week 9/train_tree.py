# F1 Safety Car Prediction - Step-by-Step Jupyter Notebook Implementation

import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from aeon.classification.deep_learning import InceptionTimeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from aeon.transformations.collection import Tabularizer
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.dummy import DummyClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

import fastf1

from f1_etl import SessionConfig, DataConfig, create_safety_car_dataset, DriverLabelEncoder, FixedVocabTrackStatusEncoder
from f1_etl.config import create_multi_session_configs


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
    hyperparameters: Optional[Dict[str, Any]] = None
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
        
    def evaluate_model(self, 
                      model, 
                      model_name: str,
                      X_train, X_test, y_train, y_test,
                      dataset_metadata: DatasetMetadata,
                      model_metadata: ModelMetadata,
                      class_names: List[str],
                      target_class: str = "safety_car",
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with metadata capture
        """
        
        # Generate evaluation metadata
        eval_metadata = EvaluationMetadata(
            evaluation_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            test_size=len(X_test) / (len(X_train) + len(X_test)),
            target_class_focus=target_class,
            evaluation_metrics=["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]
        )
        
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"Evaluation ID: {eval_metadata.evaluation_id}")
        print(f"{'='*80}")
        
        try:
            # Train model
            print("Training model...")
            model.fit(X_train, y_train)
            
            # Generate predictions
            print("Generating predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
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
                    "y_true": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    "y_pred": y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    "y_pred_proba": y_pred_proba.tolist() if y_pred_proba is not None else None
                },
                "class_info": {
                    "class_names": class_names,
                    "target_class": target_class,
                    "target_class_index": class_names.index(target_class) if target_class in class_names else None
                }
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
                "model_name": model_name
            }
            
            if save_results:
                self._save_results(error_results, eval_metadata.evaluation_id)
            
            print(f"ERROR: {str(e)}")
            return error_results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, class_names, target_class):
        """Calculate comprehensive evaluation metrics"""
        
        # Convert class_names to list consistently at the start
        class_names_list = class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
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
                        "precision": float(precision[target_in_cm] if target_in_cm < len(precision) else 0),
                        "recall": float(recall[target_in_cm] if target_in_cm < len(recall) else 0),
                        "f1": float(f1[target_in_cm] if target_in_cm < len(f1) else 0),
                        "support": int(support[target_in_cm] if target_in_cm < len(support) else 0)
                    }
        
        # Per-class metrics dictionary
        per_class_metrics = {}
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        # Convert class_names to list if it's a numpy array
        class_names_list = class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names)
        for i, class_idx in enumerate(unique_classes):
            if class_idx < len(class_names_list) and i < len(precision):
                per_class_metrics[class_names_list[class_idx]] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }
        
        return {
            "overall": {
                "accuracy": float(accuracy),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted)
            },
            "per_class": per_class_metrics,
            "target_class_metrics": target_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_true, y_pred, 
                target_names=[class_names_list[i] for i in unique_classes if i < len(class_names_list)],
                zero_division=0, 
                output_dict=True
            )
        }
    
    def _print_detailed_analysis(self, results):
        """Print comprehensive analysis to console"""
        
        metrics = results["metrics"]
        target_class = results["class_info"]["target_class"]
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Accuracy:    {metrics['overall']['accuracy']:.4f}")
        print(f"F1-Macro:    {metrics['overall']['f1_macro']:.4f}")
        print(f"F1-Weighted: {metrics['overall']['f1_weighted']:.4f}")
        
        if metrics['target_class_metrics']:
            print(f"\n🎯 TARGET CLASS ANALYSIS: {target_class.upper()}")
            print(f"{'='*50}")
            tm = metrics['target_class_metrics']
            print(f"Precision:       {tm['precision']:.4f}")
            print(f"Recall:          {tm['recall']:.4f}")
            print(f"F1-Score:        {tm['f1']:.4f}")
            print(f"True Positives:  {tm['true_positives']:4d}")
            print(f"False Negatives: {tm['false_negatives']:4d} (missed {target_class} events)")
            print(f"False Positives: {tm['false_positives']:4d} (false {target_class} alarms)")
            print(f"True Negatives:  {tm['true_negatives']:4d}")
        
        print(f"\n📈 PER-CLASS PERFORMANCE")
        print(f"{'='*50}")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:12s}: P={class_metrics['precision']:.3f}, "
                  f"R={class_metrics['recall']:.3f}, "
                  f"F1={class_metrics['f1']:.3f}, "
                  f"N={class_metrics['support']}")
        
        print(f"\n🔍 CONFUSION MATRIX")
        print(f"{'='*50}")
        cm = np.array(metrics['confusion_matrix'])
        class_names_list = results['class_info']['class_names']
        # Convert to list if it's a numpy array
        if hasattr(class_names_list, 'tolist'):
            class_names_list = class_names_list.tolist()
        unique_classes = sorted(np.unique(np.concatenate([
            results['predictions']['y_true'], 
            results['predictions']['y_pred']
        ])))
        present_class_names = [class_names_list[i] for i in unique_classes if i < len(class_names_list)]
        
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{name}" for name in present_class_names],
            columns=[f"Pred_{name}" for name in present_class_names]
        )
        print(cm_df.to_string())
    
    def _save_results(self, results, evaluation_id):
        """Save results to JSON and summary text files"""
        
        # Save complete results as JSON
        json_path = self.output_dir / f"{evaluation_id}_complete.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = self.output_dir / f"{evaluation_id}_summary.txt"
        with open(summary_path, 'w') as f:
            self._write_summary_report(results, f)
        
        print(f"\n💾 Results saved:")
        print(f"  Complete: {json_path}")
        print(f"  Summary:  {summary_path}")
    
    def _write_summary_report(self, results, file_handle):
        """Write human-readable summary report"""
        
        f = file_handle
        eval_meta = results["evaluation_metadata"]
        dataset_meta = results["dataset_metadata"]
        model_meta = results["model_metadata"]
        
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
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
        f.write(f"Shape: ({dataset_meta['total_samples']}, {dataset_meta['n_features']}, {dataset_meta['n_timesteps']})\n")
        if dataset_meta['class_distribution']:
            f.write("Class Distribution:\n")
            for class_name, count in dataset_meta['class_distribution'].items():
                f.write(f"  {class_name}: {count:,}\n")
        f.write("\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: {model_meta['model_type']}\n")
        f.write(f"Base Estimator: {model_meta['base_estimator']}\n")
        f.write(f"Wrapper: {model_meta['wrapper']}\n")
        f.write(f"Custom Weights: {model_meta['custom_weights_applied']}\n")
        if model_meta['hyperparameters']:
            f.write("Hyperparameters:\n")
            for param, value in model_meta['hyperparameters'].items():
                f.write(f"  {param}: {value}\n")
        if model_meta['class_weights']:
            f.write("Class Weights:\n")
            for class_idx, weight in model_meta['class_weights'].items():
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
            
            if metrics['target_class_metrics']:
                f.write(f"TARGET CLASS ANALYSIS: {target_class.upper()}\n")
                f.write("-" * 40 + "\n")
                tm = metrics['target_class_metrics']
                f.write(f"Precision: {tm['precision']:.4f}\n")
                f.write(f"Recall: {tm['recall']:.4f}\n")
                f.write(f"F1-Score: {tm['f1']:.4f}\n")
                f.write(f"True Positives: {tm['true_positives']:,}\n")
                f.write(f"False Negatives: {tm['false_negatives']:,} (missed events)\n")
                f.write(f"False Positives: {tm['false_positives']:,} (false alarms)\n")
                f.write(f"True Negatives: {tm['true_negatives']:,}\n\n")
            
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"{class_name:<12} {class_metrics['precision']:<10.3f} "
                       f"{class_metrics['recall']:<10.3f} {class_metrics['f1']:<10.3f} "
                       f"{class_metrics['support']:<10}\n")
        
        else:
            f.write("ERROR OCCURRED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Error: {results.get('error', 'Unknown error')}\n")

# Helper functions for easy metadata creation
def create_dataset_metadata_from_f1_config(dataset_config, dataset, processing_config=None, features_used="all"):
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
    
    X = dataset['X']
    y = dataset['y']
    
    # Use processing config from dataset if available
    if processing_config is None and 'config' in dataset:
        processing_config = dataset['config']
    
    # Determine scope description
    sessions = dataset_config.sessions if hasattr(dataset_config, 'sessions') else []
    if len(sessions) == 1:
        session = sessions[0]
        year = getattr(session, 'year', 'unknown')
        race = getattr(session, 'race', 'unknown')
        session_type = getattr(session, 'session_type', 'unknown')
        scope = f"single_session_{year}_{race}_{session_type}".replace(' ', '_')
    elif len(sessions) > 1:
        years = list(set(getattr(s, 'year', None) for s in sessions))
        years = [y for y in years if y is not None]
        session_types = list(set(getattr(s, 'session_type', None) for s in sessions))
        session_types = [st for st in session_types if st is not None]
        
        if years and session_types:
            year_str = '-'.join(map(str, sorted(years)))
            type_str = '_'.join(sorted(session_types))
            scope = f"multi_session_{year_str}_{type_str}_{len(sessions)}sessions"
        else:
            scope = f"multi_session_{len(sessions)}sessions"
    else:
        scope = "unknown_scope"
    
    # Get class distribution
    unique, counts = np.unique(y, return_counts=True)
    label_encoder = dataset.get('label_encoder')
    class_dist = {}
    if label_encoder and hasattr(label_encoder, 'class_to_idx'):
        idx_to_class = {v: k for k, v in label_encoder.class_to_idx.items()}
        class_dist = {str(idx_to_class.get(idx, f"class_{idx}")): int(count) 
                     for idx, count in zip(unique, counts)}
    elif label_encoder and hasattr(label_encoder, 'classes_'):
        # Standard sklearn LabelEncoder
        class_names = label_encoder.classes_
        class_dist = {str(class_names[idx]): int(count) 
                     for idx, count in zip(unique, counts) if idx < len(class_names)}
    
    # Extract feature names if available
    feature_names = None
    if processing_config and 'feature_names' in processing_config:
        feature_names = processing_config['feature_names']
    elif 'metadata' in dataset and dataset['metadata']:
        # Try to get from first metadata entry
        meta_entry = dataset['metadata'][0] if isinstance(dataset['metadata'], list) else dataset['metadata']
        if isinstance(meta_entry, dict) and 'features_used' in meta_entry:
            feature_names = meta_entry['features_used']
    
    # Get preprocessing steps
    preprocessing_steps = []
    if processing_config:
        if processing_config.get('missing_values_handled', False):
            preprocessing_steps.append(f"missing_values_handled_{processing_config.get('missing_strategy', 'unknown')}")
        if processing_config.get('normalization_applied', False):
            preprocessing_steps.append(f"normalized_{processing_config.get('normalization_method', 'unknown')}")
    
    return DatasetMetadata(
        scope=scope,
        sessions_config=[{
            'year': getattr(s, 'year', None),
            'race': getattr(s, 'race', None), 
            'session_type': getattr(s, 'session_type', None)
        } for s in sessions],
        drivers=getattr(dataset_config, 'drivers', []),
        include_weather=getattr(dataset_config, 'include_weather', False),
        window_size=processing_config.get('window_size', 100) if processing_config else 100,
        prediction_horizon=processing_config.get('prediction_horizon', 10) if processing_config else 10,
        handle_non_numeric=processing_config.get('handle_non_numeric', 'encode') if processing_config else 'encode',
        handle_missing=processing_config.get('handle_missing', True) if processing_config else True,
        missing_strategy=processing_config.get('missing_strategy', 'forward_fill') if processing_config else 'forward_fill',
        normalize=processing_config.get('normalize', True) if processing_config else True,
        normalization_method=processing_config.get('normalization_method', 'per_sequence') if processing_config else 'per_sequence',
        target_column=processing_config.get('target_column', 'TrackStatus') if processing_config else 'TrackStatus',
        total_samples=X.shape[0],
        n_features=X.shape[1] if len(X.shape) > 1 else 1,
        n_timesteps=X.shape[2] if len(X.shape) > 2 else X.shape[1],
        feature_names=feature_names,
        class_distribution=class_dist,
        features_used=features_used,
        is_multivariate=len(X.shape) > 2 and X.shape[1] > 1,
        preprocessing_steps=preprocessing_steps
    )

def create_metadata_from_f1_dataset(data_config, dataset, features_used="multivariate_all_9_features"):
    """
    Convenience function to create metadata from F1 dataset
    """
    return create_dataset_metadata_from_f1_config(
        dataset_config=data_config,
        dataset=dataset,
        processing_config=dataset.get('config'),  # Use the config from the dataset
        features_used=features_used
    )

def create_model_metadata(model_name, model, class_weights=None):
    """Create ModelMetadata from model configuration"""
    
    # Extract hyperparameters
    hyperparams = {}
    if hasattr(model, 'estimator') and hasattr(model.estimator, 'get_params'):
        hyperparams = model.estimator.get_params()
    elif hasattr(model, 'get_params'):
        hyperparams = model.get_params()
    
    # Determine base estimator name
    base_estimator = "Unknown"
    if hasattr(model, 'estimator'):
        base_estimator = model.estimator.__class__.__name__
    else:
        base_estimator = model.__class__.__name__
    
    return ModelMetadata(
        model_type=model_name,
        base_estimator=base_estimator,
        wrapper="Catch22Classifier" if hasattr(model, 'estimator') else "Direct",
        hyperparameters=hyperparams,
        class_weights=class_weights,
        custom_weights_applied=class_weights is not None,
        random_state=hyperparams.get('random_state', None)
    )

def prepare_data_with_validation(dataset, val_size=0.15, test_size=0.15, lookback=100, random_state=42):
    """
    Prepare train/val/test splits for time series data with proper temporal ordering
    
    Args:
        dataset: Dataset from create_safety_car_dataset
        val_size: Proportion of data for validation (default 0.15)
        test_size: Proportion of data for testing (default 0.15)
        lookback: Number of timesteps to remove from val/test to prevent data leakage (default 100)
        random_state: Random seed for reproducibility (only used for train shuffle)
    
    Returns:
        Dictionary with train/val/test splits
    """
    X = dataset['X']  # Shape: (n_samples, n_timesteps, n_features)
    y = dataset['y']  # Encoded labels
    
    # Convert to Aeon format: (n_samples, n_features, n_timesteps)
    X_aeon = X.transpose(0, 2, 1)
    
    n_samples = len(y)
    
    # Calculate split indices (no shuffling to preserve temporal order)
    train_end = int(n_samples * (1 - val_size - test_size))
    val_end = int(n_samples * (1 - test_size))
    
    # Split data temporally
    X_train = X_aeon[:train_end]
    y_train = y[:train_end]
    
    X_val = X_aeon[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X_aeon[val_end:]
    y_test = y[val_end:]
    
    # Remove first `lookback` samples from val and test to prevent data leakage
    if len(X_val) > lookback:
        X_val = X_val[lookback:]
        y_val = y_val[lookback:]
    
    if len(X_test) > lookback:
        X_test = X_test[lookback:]
        y_test = y_test[lookback:]
    
    # Shuffle only the training data
    np.random.seed(random_state)
    train_indices = np.random.permutation(len(y_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    # Print split information
    print(f"\n=== DATA SPLIT SUMMARY ===")
    print(f"Total samples: {n_samples:,}")
    print(f"Train: {len(y_train):,} ({len(y_train)/n_samples:.1%})")
    print(f"Val:   {len(y_val):,} ({len(y_val)/n_samples:.1%}) - removed {lookback} samples")
    print(f"Test:  {len(y_test):,} ({len(y_test)/n_samples:.1%}) - removed {lookback} samples")
    
    # Analyze class distribution in each split
    splits_info = {}
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        splits_info[split_name] = dict(zip(unique, counts))
        print(f"\n{split_name.capitalize()} class distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx}: {count:,} ({count/len(y_split):.1%})")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'splits_info': splits_info
    }

def train_and_validate_model(model, splits, class_names, evaluator, 
                           dataset_metadata, model_metadata, 
                           validate_during_training=True):
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
    print(f"\n{'='*80}")
    print(f"TRAINING WITH VALIDATION: {model_metadata.model_type}")
    print(f"{'='*80}")
    
    # Train the model
    print("Training on train set...")
    model.fit(splits['X_train'], splits['y_train'])
    
    results = {}
    
    # Evaluate on validation set if requested
    if validate_during_training:
        print("\nEvaluating on validation set...")
        val_pred = model.predict(splits['X_val'])
        
        # Quick validation metrics
        val_accuracy = accuracy_score(splits['y_val'], val_pred)
        val_f1_macro = f1_score(splits['y_val'], val_pred, average='macro', zero_division=0)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Macro: {val_f1_macro:.4f}")
        
        # Store validation results
        results['validation'] = {
            'accuracy': val_accuracy,
            'f1_macro': val_f1_macro,
            'predictions': val_pred.tolist(),
            'y_true': splits['y_val'].tolist()
        }
    
    # Full evaluation on test set
    print("\nRunning full evaluation on test set...")
    test_results = evaluator.evaluate_model(
        model=model,
        model_name=model_metadata.model_type,
        X_train=splits['X_train'],  # Pass train data for metadata
        X_test=splits['X_test'],
        y_train=splits['y_train'],
        y_test=splits['y_test'],
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        class_names=list(class_names),
        target_class="safety_car",
        save_results=True
    )
    
    results['test'] = test_results
    results['model'] = model  # Store trained model
    
    return results

def evaluate_on_external_dataset(trained_model, external_config, 
                               original_dataset_metadata, model_metadata,
                               class_names, evaluator):
    """
    Evaluate a trained model on a completely different dataset (e.g., different race)
    
    Args:
        trained_model: Already trained model
        external_config: DataConfig for the external dataset
        original_dataset_metadata: Metadata from training dataset
        model_metadata: ModelMetadata instance
        class_names: List of class names
        evaluator: ModelEvaluationSuite instance
    
    Returns:
        Evaluation results on external dataset
    """
    print(f"\n{'='*80}")
    print(f"EXTERNAL DATASET EVALUATION")
    print(f"{'='*80}")
    
    # Load external dataset with same preprocessing as training
    print("Loading external dataset...")
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
        enable_debug=False
    )
    
    # Convert to Aeon format
    X_external = external_dataset['X'].transpose(0, 2, 1)
    y_external = external_dataset['y']
    
    print(f"External dataset size: {len(y_external):,} samples")
    
    # Create metadata for external dataset
    external_metadata = create_metadata_from_f1_dataset(
        data_config=external_config,
        dataset=external_dataset,
        features_used=original_dataset_metadata.features_used
    )
    external_metadata.scope = f"external_{external_metadata.scope}"
    
    # Predict on external dataset
    print("Generating predictions...")
    y_pred = trained_model.predict(X_external)
    
    # Calculate metrics
    metrics = evaluator._calculate_comprehensive_metrics(
        y_external, y_pred, None, list(class_names), "safety_car"
    )
    
    # Create evaluation metadata
    eval_metadata = EvaluationMetadata(
        evaluation_id=f"{model_metadata.model_type}_external_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        test_size=1.0,  # All external data is test data
        target_class_focus="safety_car",
        evaluation_metrics=["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]
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
            "y_pred_proba": None
        },
        "class_info": {
            "class_names": list(class_names),
            "target_class": "safety_car",
            "target_class_index": list(class_names).index("safety_car")
        },
        "note": "External dataset evaluation - model trained on different data"
    }
    
    # Print and save results
    evaluator._print_detailed_analysis(results)
    evaluator._save_results(results, eval_metadata.evaluation_id)
    
    return results

def create_models(class_weight: Optional[Dict] = None):
    """Create dictionary of models to test"""

    cls_weight = 'balanced' if class_weight is None else class_weight
    
    models = {
        "dummy_frequent": DummyClassifier(strategy='most_frequent'),
        
        "dummy_stratified": DummyClassifier(strategy='stratified'),
        
        "logistic_regression": Catch22Classifier(
            estimator=LogisticRegression(
                random_state=42, 
                max_iter=3000, 
                # solver='liblinear',
                solver='saga',
                penalty='l1',
                C=0.1,
                # class_weight=cls_weight,
            ),
            outlier_norm=True,
            random_state=42,
            class_weight=cls_weight
        ),
        
        "random_forest": Catch22Classifier(
            estimator=RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10
                # class_weight=cls_weight,
            ),
            outlier_norm=True,
            random_state=42,
            class_weight=cls_weight
        )
    }
    
    return models

def create_advanced_models(class_weight=None):
    models = {
        # 1. Deep learning - good at finding complex patterns
        "inception_time": InceptionTimeClassifier(
            n_epochs=50,
            batch_size=64,
            use_custom_filters=True
        ),
        
        # 2. HIVE-COTE - ensemble of different approaches
        "hivecote": HIVECOTEV2(
            time_limit_in_minutes=10,  # Limit computation time
            n_jobs=-1
        ),
        
        # 3. Time Series Forest - handles imbalanced data well
        "ts_forest": TimeSeriesForestClassifier(
            n_estimators=200,
            min_interval_length=10,  # Look at minimum 10 timestep intervals
            n_jobs=-1
        ),
        
        # 4. Shapelets - finds discriminative subsequences
        "shapelet": ShapeletTransformClassifier(
            n_shapelet_samples=100,
            max_shapelets=20,
            batch_size=100
        ),
        
        # 5. KNN with DTW - simple but effective baseline
        "knn_dtw": KNeighborsTimeSeriesClassifier(
            n_neighbors=5,
            distance="dtw",
            distance_params={"window": 0.1}  # 10% warping window
        )
    }
    
    # Add class weighting where possible
    if class_weight:
        # Some models don't support class_weight directly
        # You might need to use sample weighting or SMOTE
        pass
        
    return models

def create_balanced_pipeline(classifier, sampling_strategy='auto'):
    """
    Create a pipeline that handles imbalanced data
    """
    # Tabularizer flattens time series for SMOTE
    return ImbPipeline([
        ('tabularize', Tabularizer()),
        ('smote', SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
        ('classify', classifier)
    ])

# Example usage workflow:

data_config = DataConfig(
    sessions=[SessionConfig(2024, 'Saudia Arabian Grand Prix', 'R')],
    # sessions=create_multi_session_configs(year=2023, session_types=['R'], include_testing=False),
    drivers=['1'],
    include_weather=False
)

dataset = create_safety_car_dataset(
    config=data_config,
    window_size=1000,
    prediction_horizon=500,
    handle_non_numeric="encode",
    handle_missing=True,
    missing_strategy="forward_fill",
    normalize=True,
    normalization_method="per_sequence",
    target_column="TrackStatus",
    enable_debug=False
)

# Create dataset metadata from configuration
dataset_metadata = create_metadata_from_f1_dataset(
    data_config=data_config,
    dataset=dataset,
    features_used="multivariate_all_9_features"
)

# 1. Prepare data with train/val/test split
splits = prepare_data_with_validation(dataset, val_size=0.15, test_size=0.15)

# 3. Create class names list from your label encoder
class_names = list(dataset['label_encoder'].class_to_idx.keys())

# 2. Analyze class distribution in each split
print("\n=== CLASS DISTRIBUTION BY SPLIT ===")
for split_name, class_counts in splits['splits_info'].items():
    print(f"\n{split_name.upper()}:")
    for class_idx, count in class_counts.items():
        class_name = class_names[class_idx]
        print(f"  {class_name}: {count}")

# 3. Train and validate model

y_enc = dataset['label_encoder']
y_enc.class_to_idx

class_weight = {
    y_enc.class_to_idx['green']: 1.0,
    y_enc.class_to_idx['red']: 10.0,         
    y_enc.class_to_idx['safety_car']: 50.0,  # Much higher
    y_enc.class_to_idx['unknown']: 1.0,      
    y_enc.class_to_idx['vsc']: 20.0,
    y_enc.class_to_idx['vsc_ending']: 20.0,
    y_enc.class_to_idx['yellow']: 10.0
}

# models = create_models(class_weight=class_weight)
models = create_advanced_models(class_weight=class_weight)
# model_name = 'random_forest'
model_name = "ts_forest"
model = models[model_name]

evaluator = ModelEvaluationSuite(output_dir="evaluation_results_3")

model_metadata = create_model_metadata(
    model_name=model_name,
    model=model,
    class_weights=class_weight
)
training_results = train_and_validate_model(
    model=models['logistic_regression'],
    splits=splits,
    class_names=class_names,
    evaluator=evaluator,
    dataset_metadata=dataset_metadata,
    model_metadata=model_metadata,
    validate_during_training=True
)

# 4. Get the trained model
trained_model = training_results['model']

# 5. Evaluate on a completely different race
external_config = DataConfig(
    sessions=[SessionConfig(2025, 'Saudi Arabian Grand Prix', 'R')],  # Different race
    # sessions=create_multi_session_configs(year=2024, session_types=['R'], include_testing=False),
    drivers=['1'],
    include_weather=False
)

external_results = evaluate_on_external_dataset(
    trained_model=trained_model,
    external_config=external_config,
    original_dataset_metadata=dataset_metadata,
    model_metadata=model_metadata,
    class_names=class_names,
    evaluator=evaluator
)

# 6. Compare performance across datasets
print("\n=== PERFORMANCE COMPARISON ===")
print(f"{'Dataset':<20} {'Accuracy':<10} {'F1-Macro':<10} {'Target F1':<10}")
print("-" * 60)

# Validation performance
if 'validation' in training_results:
    print(f"{'Validation':<20} {training_results['validation']['accuracy']:<10.4f} "
          f"{training_results['validation']['f1_macro']:<10.4f} {'N/A':<10}")

# Test performance (same race holdout)
test_metrics = training_results['test']['metrics']
print(f"{'Test (same race)':<20} {test_metrics['overall']['accuracy']:<10.4f} "
      f"{test_metrics['overall']['f1_macro']:<10.4f} "
      f"{test_metrics['target_class_metrics']['f1'] if test_metrics['target_class_metrics'] else 0:<10.4f}")

# External test performance (different race)
ext_metrics = external_results['metrics']
print(f"{'Test (diff race)':<20} {ext_metrics['overall']['accuracy']:<10.4f} "
      f"{ext_metrics['overall']['f1_macro']:<10.4f} "
      f"{ext_metrics['target_class_metrics']['f1'] if ext_metrics['target_class_metrics'] else 0:<10.4f}")