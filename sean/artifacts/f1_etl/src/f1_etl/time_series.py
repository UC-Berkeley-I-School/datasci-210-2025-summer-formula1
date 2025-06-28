"""Time series generation from telemetry data"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .logging import logger


class TimeSeriesGenerator:
    """Generates sliding window time series sequences from telemetry data"""

    def __init__(
        self,
        window_size: int,
        step_size: int = 1,
        features: Optional[List[str]] = None,
        prediction_horizon: int = 1,
        handle_non_numeric: str = "encode",  # 'encode' or 'drop'
        target_column: str = "TrackStatus",
    ):  # Configurable target column
        self.window_size = window_size
        self.step_size = step_size
        self.prediction_horizon = prediction_horizon
        self.handle_non_numeric = handle_non_numeric
        self.target_column = target_column
        self.features = features or [
            "Speed",
            "RPM",
            "nGear",
            "Throttle",
            "Brake",
            "X",
            "Y",
            "Distance",
            "DifferentialDistance",
        ]

    def _process_features(
        self, group_data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process features to handle non-numeric data types
        Returns numpy array with proper dtype and list of processed feature names
        """
        available_features = [f for f in self.features if f in group_data.columns]
        if not available_features:
            raise ValueError(
                f"No requested features found in data. Available: {list(group_data.columns)}"
            )

        feature_data = group_data[available_features].copy()
        processed_features = []

        for feature in available_features:
            col = feature_data[feature]

            if pd.api.types.is_numeric_dtype(col):
                # Already numeric, keep as-is
                processed_features.append(feature)
            elif pd.api.types.is_bool_dtype(col) or col.dtype == "bool":
                # Boolean - encode as 0/1
                if self.handle_non_numeric == "encode":
                    feature_data[feature] = col.astype(int)
                    processed_features.append(feature)
                    logger.debug(f"Encoded boolean feature '{feature}' as 0/1")
                elif self.handle_non_numeric == "drop":
                    logger.debug(f"Dropping boolean feature '{feature}'")
                    feature_data = feature_data.drop(columns=[feature])
            elif col.dtype == "object":
                # Check if it's actually numeric stored as object
                try:
                    converted = pd.to_numeric(col, errors="coerce")
                    if not converted.isna().all():
                        feature_data[feature] = converted
                        processed_features.append(feature)
                        logger.debug(f"Converted object feature '{feature}' to numeric")
                    else:
                        # Non-numeric object
                        if self.handle_non_numeric == "encode":
                            # Simple label encoding for categorical
                            unique_vals = col.unique()
                            mapping = {val: i for i, val in enumerate(unique_vals)}
                            feature_data[feature] = col.map(mapping)
                            processed_features.append(feature)
                            logger.debug(
                                f"Label encoded categorical feature '{feature}': {mapping}"
                            )
                        elif self.handle_non_numeric == "drop":
                            logger.debug(f"Dropping non-numeric feature '{feature}'")
                            feature_data = feature_data.drop(columns=[feature])
                except Exception as e:
                    logger.warning(f"Could not process feature '{feature}': {e}")
                    if self.handle_non_numeric == "drop":
                        feature_data = feature_data.drop(columns=[feature])
            else:
                # Other data types
                if self.handle_non_numeric == "drop":
                    logger.debug(
                        f"Dropping unsupported feature '{feature}' (dtype: {col.dtype})"
                    )
                    feature_data = feature_data.drop(columns=[feature])
                else:
                    logger.warning(
                        f"Attempting to convert '{feature}' (dtype: {col.dtype}) to numeric"
                    )
                    try:
                        feature_data[feature] = pd.to_numeric(col, errors="coerce")
                        processed_features.append(feature)
                    except (ValueError, TypeError):
                        feature_data = feature_data.drop(columns=[feature])

        if feature_data.empty:
            raise ValueError("No valid features remaining after processing")

        # Convert to numeric numpy array
        try:
            feature_array = feature_data[processed_features].astype(np.float64).values
        except Exception as e:
            logger.error(f"Error converting to float64: {e}")
            logger.error(f"Data types: {feature_data[processed_features].dtypes}")
            raise

        return feature_array, processed_features

    def generate_sequences(
        self, telemetry_data: pd.DataFrame, group_by: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Generate sliding window sequences with built-in preprocessing"""
        if group_by is None:
            group_by = ["SessionId", "Driver"]

        sequences = []
        labels = []
        metadata = []

        logger.info(f"Processing {len(telemetry_data)} total telemetry rows")
        logger.info(f"Grouping by: {group_by}")
        logger.debug(f"Available columns: {list(telemetry_data.columns)}")

        group_count = 0
        for group_keys, group_data in telemetry_data.groupby(group_by):
            group_count += 1
            logger.debug(f"Processing group {group_count}: {group_keys}")
            logger.debug(f"  Group size: {len(group_data)} rows")
            logger.debug(
                f"  Required minimum rows: {self.window_size + self.prediction_horizon}"
            )

            try:
                group_sequences, group_labels, group_metadata = (
                    self._generate_group_sequences(group_data, group_keys, group_by)
                )
                logger.debug(f"  Generated {len(group_sequences)} sequences")
                sequences.extend(group_sequences)
                labels.extend(group_labels)
                metadata.extend(group_metadata)

            except Exception as e:
                logger.warning(f"Error processing group {group_keys}: {e}")
                logger.debug("Full traceback:", exc_info=True)
                continue

        logger.info(f"Total sequences generated: {len(sequences)}")
        if not sequences:
            logger.error("No sequences generated - debugging info:")
            logger.error(f"  Total groups processed: {group_count}")
            logger.error(f"  Window size: {self.window_size}")
            logger.error(f"  Prediction horizon: {self.prediction_horizon}")
            logger.error(f"  Required features: {self.features}")
            logger.error(f"  Target column: {self.target_column}")

            # Check if target column exists
            if self.target_column not in telemetry_data.columns:
                logger.error(
                    f"  ERROR: {self.target_column} column missing! Available: {list(telemetry_data.columns)}"
                )
            else:
                logger.error(
                    f"  {self.target_column} values: {telemetry_data[self.target_column].unique()}"
                )

            raise ValueError("No sequences generated - see debug output above")

        return np.array(sequences), np.array(labels), metadata

    def _generate_group_sequences(
        self, group_data: pd.DataFrame, group_keys: Tuple, group_by: List[str]
    ) -> Tuple[List, List, List]:
        """Generate sequences for a single group"""
        # Sort by time
        if "Date" not in group_data.columns:
            raise ValueError(
                f"Date column missing from group data. Available: {list(group_data.columns)}"
            )

        group_data_sorted = group_data.sort_values("Date").reset_index(drop=True)
        logger.debug(f"    Sorted group data: {len(group_data_sorted)} rows")

        # Process features to handle non-numeric data
        try:
            feature_array, processed_features = self._process_features(
                group_data_sorted
            )
            logger.debug(f"    Processed features: {processed_features}")
            logger.debug(f"    Feature array shape: {feature_array.shape}")
        except Exception as e:
            logger.debug(f"    Feature processing failed: {e}")
            raise

        sequences = []
        labels = []
        metadata = []

        max_start_idx = (
            len(feature_array) - self.window_size - self.prediction_horizon + 1
        )
        logger.debug(
            f"    Max start index: {max_start_idx} (need >= 0 to generate sequences)"
        )

        if max_start_idx <= 0:
            logger.debug(
                f"    Insufficient data: need {self.window_size + self.prediction_horizon} rows, have {len(feature_array)}"
            )
            return sequences, labels, metadata

        # Check for target column
        if self.target_column not in group_data_sorted.columns:
            logger.debug(
                f"    {self.target_column} column missing! Available: {list(group_data_sorted.columns)}"
            )
            return sequences, labels, metadata

        sequences_generated = 0
        for i in range(0, max_start_idx, self.step_size):
            sequence = feature_array[i : i + self.window_size]

            label_idx = i + self.window_size + self.prediction_horizon - 1
            if label_idx < len(group_data_sorted):
                label = group_data_sorted.iloc[label_idx][self.target_column]
            else:
                logger.debug(
                    f"    Label index {label_idx} out of bounds (max: {len(group_data_sorted) - 1})"
                )
                continue

            seq_metadata = {
                "start_time": group_data_sorted.iloc[i]["Date"],
                "end_time": group_data_sorted.iloc[i + self.window_size - 1]["Date"],
                "prediction_time": group_data_sorted.iloc[label_idx]["Date"],
                "sequence_length": self.window_size,
                "prediction_horizon": self.prediction_horizon,
                "features_used": processed_features,
                "target_column": self.target_column,
            }

            for j, key in enumerate(group_by):
                seq_metadata[key] = (
                    group_keys[j] if isinstance(group_keys, tuple) else group_keys
                )

            sequences.append(sequence)
            labels.append(label)
            metadata.append(seq_metadata)
            sequences_generated += 1

        logger.debug(f"    Successfully generated {sequences_generated} sequences")
        return sequences, labels, metadata
