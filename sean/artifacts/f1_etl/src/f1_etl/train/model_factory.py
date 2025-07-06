"""Model creation utilities for F1 safety car prediction"""

from typing import Dict, Optional

from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.dummy import DummyClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.transformations.collection import Tabularizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_basic_models(class_weight: Optional[Dict] = None):
    """
    Create dictionary of basic models for quick experimentation

    Args:
        class_weight: Optional dictionary mapping class indices to weights

    Returns:
        Dictionary of model_name -> model instance
    """
    cls_weight = "balanced" if class_weight is None else class_weight

    models = {
        "dummy_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified"),
        "logistic_regression": Catch22Classifier(
            estimator=LogisticRegression(
                random_state=42,
                max_iter=3000,
                solver="saga",
                penalty="l1",
                C=0.1,
            ),
            outlier_norm=True,
            random_state=42,
            class_weight=cls_weight,
        ),
        "random_forest": Catch22Classifier(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            ),
            outlier_norm=True,
            random_state=42,
            class_weight=cls_weight,
        ),
    }

    return models


def create_advanced_models(class_weight=None):
    models = {
        # 1. Deep learning - good at finding complex patterns
        "inception_time": InceptionTimeClassifier(
            n_epochs=50, batch_size=64, use_custom_filters=True
        ),
        # 2. HIVE-COTE - ensemble of different approaches
        "hivecote": HIVECOTEV2(
            time_limit_in_minutes=10,  # Limit computation time
            n_jobs=-1,
        ),
        # 3. Time Series Forest - handles imbalanced data well
        "ts_forest": TimeSeriesForestClassifier(
            n_estimators=200,
            min_interval_length=10,  # Look at minimum 10 timestep intervals
            n_jobs=-1,
        ),
        # 4. Shapelets - finds discriminative subsequences
        "shapelet": ShapeletTransformClassifier(
            n_shapelet_samples=100, max_shapelets=20, batch_size=100
        ),
        # 5. KNN with DTW - simple but effective baseline
        "knn_dtw": KNeighborsTimeSeriesClassifier(
            n_neighbors=5,
            distance="dtw",
            distance_params={"window": 0.1},  # 10% warping window
        ),
    }

    # Add class weighting where possible
    if class_weight:
        # Some models don't support class_weight directly
        # You might need to use sample weighting or SMOTE
        pass

    return models


def create_balanced_pipeline(classifier, sampling_strategy="auto"):
    """
    Create a pipeline that handles imbalanced data using SMOTE

    WARNING: This applies SMOTE AFTER windowing, which can cause data leakage
    between overlapping windows. For time series data, it's recommended to use
    the resampling_strategy parameter in create_safety_car_dataset() instead,
    which applies resampling BEFORE windowing.

    This function is kept for backwards compatibility and special cases where
    post-windowing resampling is specifically desired.

    Args:
        classifier: Base classifier to use
        sampling_strategy: SMOTE sampling strategy

    Returns:
        ImbPipeline instance
    """
    return ImbPipeline(
        [
            ("tabularize", Tabularizer()),
            ("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
            ("classify", classifier),
        ]
    )
