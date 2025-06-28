"""Feature engineering for time series data"""

import numpy as np

from .logging import logger


class FeatureEngineer:
    """Applies feature engineering to time series data"""

    def __init__(self):
        self.normalization_params = {}
        self.is_fitted = False

    def handle_missing_values(
        self, X: np.ndarray, strategy: str = "forward_fill"
    ) -> np.ndarray:
        """Handle missing values in numeric time series data"""
        if not np.isnan(X).any():
            logger.info("No missing values detected, skipping imputation")
            return X

        logger.info(f"Handling missing values with strategy: {strategy}")
        X_filled = X.copy()

        if strategy == "forward_fill":
            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    series = X_filled[i, :, j]
                    mask = np.isnan(series)
                    if mask.any():
                        last_valid = None
                        for k in range(len(series)):
                            if not np.isnan(series[k]):
                                last_valid = series[k]
                            elif last_valid is not None:
                                series[k] = last_valid

                        if np.isnan(series[0]):
                            valid_indices = np.where(~np.isnan(series))[0]
                            if len(valid_indices) > 0:
                                fill_value = series[valid_indices[0]]
                                for k in range(valid_indices[0]):
                                    series[k] = fill_value
                        X_filled[i, :, j] = series
        elif strategy == "mean_fill":
            for j in range(X.shape[2]):
                feature_data = X[:, :, j]
                feature_mean = np.nanmean(feature_data)
                X_filled[:, :, j] = np.where(
                    np.isnan(feature_data), feature_mean, feature_data
                )
        elif strategy == "zero_fill":
            X_filled = np.where(np.isnan(X_filled), 0, X_filled)

        return X_filled

    def normalize_sequences(
        self, X: np.ndarray, method: str = "standard", fit: bool = True
    ) -> np.ndarray:
        """Normalize time series sequences"""
        if method == "standard":
            if fit or not self.is_fitted:
                means = np.mean(X, axis=(0, 1), keepdims=True)
                stds = np.std(X, axis=(0, 1), keepdims=True)
                stds = np.where(stds == 0, 1, stds)
                self.normalization_params = {"means": means, "stds": stds}
                self.is_fitted = True

            params = self.normalization_params
            return (X - params["means"]) / params["stds"]

        elif method == "minmax":
            if fit or not self.is_fitted:
                mins = np.min(X, axis=(0, 1), keepdims=True)
                maxs = np.max(X, axis=(0, 1), keepdims=True)
                ranges = maxs - mins
                ranges = np.where(ranges == 0, 1, ranges)
                self.normalization_params = {"mins": mins, "ranges": ranges}
                self.is_fitted = True

            params = self.normalization_params
            return (X - params["mins"]) / params["ranges"]

        elif method == "per_sequence":
            normalized = np.zeros_like(X)
            for i in range(X.shape[0]):
                seq = X[i]
                seq_mean = np.mean(seq, axis=0, keepdims=True)
                seq_std = np.std(seq, axis=0, keepdims=True)
                seq_std = np.where(seq_std == 0, 1, seq_std)
                normalized[i] = (seq - seq_mean) / seq_std
            return normalized

        else:
            raise ValueError(f"Unknown normalization method: {method}")
