"""
F1 Safety Car Prediction ETL Pipeline

A comprehensive ETL pipeline for extracting, transforming, and preparing Formula 1
telemetry data for time series classification tasks. Built on FastF1 and designed
for safety car prediction and other F1 data science applications.

Features:
- Automated data extraction from FastF1 for entire seasons
- Time series generation with sliding windows
- Feature engineering including missing value handling and normalization
- Track status integration for safety car prediction
- Flexible configuration for different dataset compositions
- Caching support to avoid repeated API calls

Example usage:
    from f1_etl import SessionConfig, DataConfig, create_safety_car_dataset

    # Single race
    session = SessionConfig(2024, "Monaco Grand Prix", "R")
    config = DataConfig(sessions=[session])
    dataset = create_safety_car_dataset(config)

    # Full season
    from f1_etl import create_season_configs
    sessions = create_season_configs(2024)
    config = DataConfig(sessions=sessions)
    dataset = create_safety_car_dataset(config)
"""

__version__ = "0.1.0"

# Import main classes and functions
from .aggregation import DataAggregator
from .config import DataConfig, SessionConfig
from .encoders import DriverLabelEncoder, TrackStatusLabelEncoder
from .encoders_new import FixedVocabTrackStatusEncoder, compare_race_distributions
from .extraction import RawDataExtractor
from .feature_engineering import FeatureEngineer
from .logging import setup_logger
from .pipeline import create_safety_car_dataset
from .time_series import TimeSeriesGenerator

__all__ = [
    # Configuration
    "SessionConfig",
    "DataConfig",
    # Core pipeline
    "RawDataExtractor",
    "DataAggregator",
    "TimeSeriesGenerator",
    "FeatureEngineer",
    # Encoders
    "TrackStatusLabelEncoder",
    "DriverLabelEncoder",
    "FixedVocabTrackStatusEncoder",
    "compare_race_distributions",
    # Main functions
    "create_safety_car_dataset",
    # Utilities
    "setup_logger",
]
