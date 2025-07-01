"""Label encoders for F1 data"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TrackStatusLabelEncoder:
    """
    @deprecated - use ``FixedVocabTrackStatusEncoder`` instead

    Encodes track status labels for safety car prediction
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.track_status_mapping = {
            "1": "green",
            "2": "yellow",
            "4": "safety_car",
            "5": "red",
            "6": "vsc",
            "7": "vsc_ending",
        }

    def fit(self, track_status_data: pd.Series) -> "TrackStatusLabelEncoder":
        mapped_labels = track_status_data.map(self.track_status_mapping).fillna(
            "unknown"
        )
        self.label_encoder.fit(mapped_labels)
        self.is_fitted = True
        return self

    def transform(self, track_status_data: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        mapped_labels = track_status_data.map(self.track_status_mapping).fillna(
            "unknown"
        )
        return self.label_encoder.transform(mapped_labels)

    def fit_transform(self, track_status_data: pd.Series) -> np.ndarray:
        return self.fit(track_status_data).transform(track_status_data)

    def inverse_transform(self, encoded_labels: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(encoded_labels)

    def get_classes(self) -> np.ndarray:
        return self.label_encoder.classes_


class DriverLabelEncoder:
    """Encodes driver identifiers for consistency"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.driver_to_number = {}  # Maps abbreviations to driver numbers
        self.number_to_driver = {}  # Maps driver numbers to abbreviations

    def fit_session(self, session) -> "DriverLabelEncoder":
        """Fit the encoder using session driver data"""
        driver_numbers = session.drivers

        for driver_number in driver_numbers:
            driver_info = session.get_driver(driver_number)
            abbreviation = driver_info["Abbreviation"]

            self.driver_to_number[abbreviation] = driver_number
            self.number_to_driver[driver_number] = abbreviation

        # Fit encoder on abbreviations for consistent encoding
        abbreviations = list(self.driver_to_number.keys())
        self.label_encoder.fit(abbreviations)
        self.is_fitted = True
        return self

    def transform_driver_to_number(self, drivers):
        """Transform driver abbreviations to driver numbers"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted")
        return [self.driver_to_number[driver] for driver in drivers]

    def transform_number_to_driver(self, numbers):
        """Transform driver numbers to abbreviations"""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted")
        return [self.number_to_driver[number] for number in numbers]
