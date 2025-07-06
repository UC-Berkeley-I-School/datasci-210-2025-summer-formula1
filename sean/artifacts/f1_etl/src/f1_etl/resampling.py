"""Resampling strategies for handling class imbalance in F1 telemetry data"""

from typing import Optional

import numpy as np
import pandas as pd


def apply_resampling(
    telemetry_data: pd.DataFrame,
    target_column: str,
    strategy: str,
    logger,
    target_class=None,
    sampling_strategy=None,
) -> pd.DataFrame:
    """
    Apply resampling to telemetry data before windowing.

    This resamples at the session/driver level to handle class imbalance
    before time series windowing.

    Parameters:
    -----------
    telemetry_data : pd.DataFrame
        Raw telemetry data with all features
    target_column : str
        Column name containing target labels
    strategy : str
        Resampling strategy ('adasyn', 'smote', 'borderline_smote')
    logger : Logger
        Logger instance for output
    target_class : str or int, optional
        Specific class to focus resampling on (e.g., '2' for safety car)
    sampling_strategy : dict or str, optional
        Custom sampling strategy. If None, uses 'minority'.
        Examples:
        - 'minority': resample all minority classes to match majority
        - 'not majority': resample all but majority class
        - {'2': 1000000}: resample class 2 to have 1M samples
        - {'2': 0.5}: resample class 2 to 50% of majority class

    Returns:
    --------
    pd.DataFrame : Resampled telemetry data
    """
    try:
        from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn package required for resampling. "
            "Install with: pip install imbalanced-learn"
        )

    # Determine sampling strategy
    if sampling_strategy is None:
        if target_class is not None:
            # Get class distribution
            class_counts = telemetry_data[target_column].value_counts()
            majority_count = class_counts.max()

            # Create custom strategy focusing on target class
            # Bring target class up to 20% of majority class by default
            target_ratio = 0.2
            sampling_strategy = {str(target_class): int(majority_count * target_ratio)}
            logger.info(
                f"Focusing resampling on class {target_class}, targeting {sampling_strategy[str(target_class)]:,} samples"
            )
        else:
            sampling_strategy = "minority"

    logger.info(f"Applying {strategy} resampling at session/driver level")
    logger.info(f"Sampling strategy: {sampling_strategy}")

    # Group by session and driver to resample within each group
    resampled_groups = []

    for group_keys, group_data in telemetry_data.groupby(["SessionId", "Driver"]):
        logger.debug(f"Resampling group: {group_keys}")

        # Get features for resampling (numeric columns only, excluding time-based)
        numeric_cols = group_data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target, metadata, and time-related columns
        exclude_cols = [target_column, "TrackStatusEncoded"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Also exclude any columns that might contain time deltas or timestamps
        time_related = [
            "Time",
            "SessionTime",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
        ]
        feature_cols = [col for col in feature_cols if col not in time_related]

        if not feature_cols:
            logger.warning(
                f"No numeric features found for resampling in group {group_keys}"
            )
            resampled_groups.append(group_data)
            continue

        # Check for any Timedelta columns that might have slipped through
        valid_feature_cols = []
        for col in feature_cols:
            if pd.api.types.is_timedelta64_dtype(group_data[col]):
                logger.debug(f"Skipping timedelta column: {col}")
                continue
            valid_feature_cols.append(col)

        feature_cols = valid_feature_cols

        if not feature_cols:
            logger.warning(
                f"No valid features after filtering timedelta columns in group {group_keys}"
            )
            resampled_groups.append(group_data)
            continue

        # Extract features and handle NaN values
        X_group = group_data[feature_cols].copy()

        # Check for NaN values
        if X_group.isna().any().any():
            logger.debug(
                f"Found NaN values in group {group_keys}, applying forward fill"
            )
            # Forward fill then backward fill to handle NaN at the beginning
            X_group = X_group.ffill().bfill()

            # If still NaN (entire column is NaN), fill with 0
            if X_group.isna().any().any():
                logger.debug(f"Still NaN after forward/backward fill, using zero fill")
                X_group = X_group.fillna(0)

        X_group = X_group.values
        y_group = group_data[target_column].values

        # Check if resampling is needed (multiple classes present)
        unique_classes = np.unique(y_group)
        if len(unique_classes) < 2:
            logger.debug(f"Only one class in group {group_keys}, skipping resampling")
            resampled_groups.append(group_data)
            continue

        # Calculate safe k_neighbors value
        min_class_samples = min([np.sum(y_group == cls) for cls in unique_classes])
        k_neighbors = min(5, min_class_samples - 1, len(X_group) - 1)

        if k_neighbors < 1:
            logger.warning(f"Not enough samples for resampling in group {group_keys}")
            resampled_groups.append(group_data)
            continue

        # Initialize resampler with custom sampling strategy
        if strategy == "adasyn":
            resampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors,
                random_state=42,
            )
        elif strategy == "smote":
            resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42,
            )
        elif strategy == "borderline_smote":
            m_neighbors = min(10, len(X_group) - 1)
            resampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                m_neighbors=m_neighbors,
                kind="borderline-1",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown resampling strategy: {strategy}")

        try:
            X_resampled, y_resampled = resampler.fit_resample(X_group, y_group)

            # Create synthetic rows for the new samples
            n_synthetic = len(X_resampled) - len(X_group)
            logger.debug(
                f"Generated {n_synthetic} synthetic samples for group {group_keys}"
            )

            if n_synthetic > 0:
                # Create DataFrame for synthetic samples
                synthetic_data = pd.DataFrame(
                    X_resampled[len(X_group) :], columns=feature_cols
                )

                # Add metadata columns from original group
                for col in group_data.columns:
                    if col not in feature_cols and col != target_column:
                        if col in [
                            "SessionId",
                            "Driver",
                            "SessionYear",
                            "SessionRace",
                            "SessionType",
                        ]:
                            # Keep session/driver info
                            synthetic_data[col] = group_data[col].iloc[0]
                        elif col == "Date":
                            # Interpolate dates for synthetic samples
                            date_min = group_data["Date"].min()
                            date_max = group_data["Date"].max()
                            date_range = pd.date_range(
                                start=date_min, end=date_max, periods=n_synthetic
                            )
                            synthetic_data[col] = date_range
                        elif pd.api.types.is_timedelta64_dtype(group_data[col]):
                            # For timedelta columns, use interpolation based on existing values
                            existing_values = group_data[col].dropna()
                            if len(existing_values) > 0:
                                # Random sample from existing timedelta values
                                synthetic_data[col] = np.random.choice(
                                    existing_values, size=n_synthetic
                                )
                            else:
                                synthetic_data[col] = pd.NaT
                        else:
                            # Use mode or first value for other columns
                            mode_values = group_data[col].mode()
                            if len(mode_values) > 0:
                                synthetic_data[col] = mode_values[0]
                            else:
                                synthetic_data[col] = group_data[col].iloc[0]

                # Add synthetic target values
                synthetic_data[target_column] = y_resampled[len(X_group) :]

                # Combine original and synthetic data
                combined_group = pd.concat(
                    [group_data, synthetic_data], ignore_index=True
                )

                # Sort by date to maintain temporal order
                if "Date" in combined_group.columns:
                    combined_group = combined_group.sort_values("Date")

                resampled_groups.append(combined_group)
            else:
                resampled_groups.append(group_data)

        except Exception as e:
            logger.warning(f"Resampling failed for group {group_keys}: {e}")
            logger.debug("Exception details:", exc_info=True)
            resampled_groups.append(group_data)

    # Combine all resampled groups
    resampled_data = pd.concat(resampled_groups, ignore_index=True)

    # Log summary statistics
    original_counts = telemetry_data[target_column].value_counts()
    resampled_counts = resampled_data[target_column].value_counts()

    logger.info(
        f"Resampling complete: {len(telemetry_data)} -> {len(resampled_data)} samples"
    )
    logger.info("Class distribution before resampling:")
    for cls, count in original_counts.items():
        logger.info(f"  {cls}: {count}")
    logger.info("Class distribution after resampling:")
    for cls, count in resampled_counts.items():
        logger.info(f"  {cls}: {count}")

    return resampled_data
