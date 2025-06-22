"""
F1 Dataset Processing Package

This package provides tools for processing Formula 1 telemetry data into 
machine learning datasets suitable for temporal classification tasks.
"""

from .preprocessor import (
    F1DatasetPreprocessor,
    PreprocessorConfig,
    BaseFeatures
)

from .aggregator import (
    F1SeasonAggregator,
    AggregatorConfig,
    SessionProcessor,
    create_aggregator
)

__version__ = "0.1.0"

__all__ = [
    "F1DatasetPreprocessor",
    "PreprocessorConfig", 
    "BaseFeatures",
    "F1SeasonAggregator",
    "AggregatorConfig",
    "SessionProcessor",
    "create_aggregator"
]