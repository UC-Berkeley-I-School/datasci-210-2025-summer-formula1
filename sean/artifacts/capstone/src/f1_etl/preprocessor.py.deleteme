import fastf1.core
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Tuple, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

class BaseFeatures(Enum):
    DRIVER = 'Driver'
    SAFETY_CAR = 'SafetyCar'
    LOCATION = 'Location'
    COMPOUND = 'Compound'


@dataclass
class PreprocessorConfig:
    """Configuration for the F1 dataset preprocessor"""

    interval_seconds: float = 1.0
    balance_features: list["BaseFeatures"] = None
    balance_method: str = "remove_insufficient"
    min_samples: int = 1000
    target_samples: Optional[int] = None
    include_track_status: bool = True
    include_event_info: bool = True

    def __post_init__(self):
        if self.balance_features is None:
            self.balance_features = [BaseFeatures.DRIVER, BaseFeatures.SAFETY_CAR]


class F1DatasetPreprocessor:
    """
    Efficient F1 dataset preprocessor that minimizes redundant operations
    and memory usage for time series classification tasks.
    """

    def __init__(self, config: PreprocessorConfig = None):
        self.config = config or PreprocessorConfig()
        self._session_cache = {}

    def process_dataset(self, session: fastf1.core.Session) -> DataFrame:
        """
        Main processing pipeline that efficiently applies all transformations.
        """
        if not session:
            raise ValueError("session must be defined")

        if self.config.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        # Single-pass processing with all transformations
        return self._process_single_pass(session)

    def _process_single_pass(self, session: fastf1.core.Session) -> DataFrame:
        """
        Single-pass processing that combines all transformation steps.
        """
        # Pre-fetch and cache all session data once
        session_data = self._extract_session_data(session)

        # Create time grid
        time_grid = self._create_time_grid(session_data["laps"])

        # Process all drivers in single loop
        all_records = []
        for driver in session_data["laps"]["Driver"].unique():
            driver_records = self._process_driver_telemetry(
                driver, session_data, time_grid
            )
            all_records.extend(driver_records)

        # Convert to DataFrame once
        df = pd.DataFrame(all_records)

        if df.empty:
            return df

        # Apply balancing if requested
        if self.config.balance_features:
            df = self._balance_features(df)

        return df

    def _extract_session_data(self, session: fastf1.core.Session) -> Dict[str, Any]:
        """
        Extract and cache all required session data in a single operation.
        """
        session_id = id(session)
        if session_id in self._session_cache:
            return self._session_cache[session_id]

        data = {"laps": session.laps, "track_status": None, "event_info": None}

        # Extract track status if needed
        if self.config.include_track_status:
            track_status = session.track_status.copy()
            track_status["StatusTimeSeconds"] = track_status["Time"].dt.total_seconds()
            track_status = track_status.sort_values("StatusTimeSeconds").reset_index(
                drop=True
            )
            data["track_status"] = track_status

        # Extract event info if needed
        if self.config.include_event_info:
            session_info = session.session_info
            data["event_info"] = {
                "session_name": session_info.get("Meeting", {}).get("Name", "Unknown"),
                "country": session_info.get("Meeting", {})
                .get("Country", {})
                .get("Name", "Unknown"),
                "session_type": session_info.get("Type", "Unknown"),
                "start_date": session_info.get("StartDate"),
                "location": session_info.get("Meeting", {}).get("Location", "Unknown"),
            }

        self._session_cache[session_id] = data
        return data

    def _create_time_grid(self, laps_df: DataFrame) -> np.ndarray:
        """Create time grid for temporal alignment."""
        session_start = laps_df["LapStartTime"].min()
        session_end = laps_df["Time"].max()

        start_seconds = session_start.total_seconds()
        end_seconds = session_end.total_seconds()

        return np.arange(start_seconds, end_seconds, self.config.interval_seconds)

    def _process_driver_telemetry(
        self, driver: str, session_data: Dict, time_grid: np.ndarray
    ) -> list:
        """
        Process telemetry for a single driver, creating records aligned to time grid.
        """
        driver_laps = session_data["laps"][session_data["laps"]["Driver"] == driver]
        records = []

        for _, lap in driver_laps.iterrows():
            try:
                # Get telemetry data for this lap
                car_data = lap.get_car_data()
                pos_data = lap.get_pos_data()
                weather_data = lap.get_weather_data()

                if car_data.empty:
                    continue

                # Convert to seconds for alignment
                car_data = car_data.copy()
                car_data["SessionSeconds"] = car_data["SessionTime"].dt.total_seconds()

                if not pos_data.empty:
                    pos_data = pos_data.copy()
                    pos_data["SessionSeconds"] = pos_data[
                        "SessionTime"
                    ].dt.total_seconds()

                # Create base lap info
                lap_info = {
                    "Driver": driver,
                    "LapNumber": lap["LapNumber"],
                    "Compound": lap["Compound"],
                    "TyreLife": lap["TyreLife"],
                    "AirTemp": (
                        weather_data.get("AirTemp")
                        if pd.notna(weather_data.get("AirTemp"))
                        else None
                    ),
                    "TrackTemp": (
                        weather_data.get("TrackTemp")
                        if pd.notna(weather_data.get("TrackTemp"))
                        else None
                    ),
                    "Humidity": (
                        weather_data.get("Humidity")
                        if pd.notna(weather_data.get("Humidity"))
                        else None
                    ),
                }

                # Add event info if enabled
                if self.config.include_event_info and session_data["event_info"]:
                    lap_info.update(session_data["event_info"])

                # Align telemetry to time grid
                lap_start = lap["LapStartTime"].total_seconds()
                lap_end = lap["Time"].total_seconds()

                # Find relevant time grid points for this lap
                lap_grid_mask = (time_grid >= lap_start) & (time_grid <= lap_end)
                lap_time_points = time_grid[lap_grid_mask]

                # Create record for each time point in this lap
                for time_point in lap_time_points:
                    record = lap_info.copy()
                    record["SessionTimeSeconds"] = time_point

                    # Interpolate telemetry data to this time point
                    self._add_telemetry_at_time(record, car_data, pos_data, time_point)

                    # Add track status if enabled
                    if self.config.include_track_status:
                        self._add_track_status_at_time(
                            record, session_data["track_status"], time_point
                        )

                    records.append(record)

            except Exception as e:
                print(f"Skipping lap {lap['LapNumber']} for {driver}: {e}")
                continue

        return records

    def _add_telemetry_at_time(
        self, record: dict, car_data: DataFrame, pos_data: DataFrame, time_point: float
    ):
        """Add interpolated telemetry data to record at specific time point."""
        # Find closest telemetry point
        if not car_data.empty:
            time_diffs = np.abs(car_data["SessionSeconds"] - time_point)
            closest_idx = time_diffs.idxmin()
            closest_car = car_data.loc[closest_idx]

            # Add car telemetry
            record.update(
                {
                    "Speed": closest_car.get("Speed"),
                    "RPM": closest_car.get("RPM"),
                    "Gear": closest_car.get("nGear"),
                    "Throttle": closest_car.get("Throttle"),
                    "Brake": closest_car.get("Brake"),
                    "DRS": closest_car.get("DRS"),
                }
            )

        # Add position data if available
        if not pos_data.empty:
            pos_time_diffs = np.abs(pos_data["SessionSeconds"] - time_point)
            closest_pos_idx = pos_time_diffs.idxmin()
            closest_pos = pos_data.loc[closest_pos_idx]

            record.update(
                {
                    "X": closest_pos.get("X"),
                    "Y": closest_pos.get("Y"),
                    "Z": closest_pos.get("Z"),
                }
            )

    def _add_track_status_at_time(
        self, record: dict, track_status: DataFrame, time_point: float
    ):
        """Add track status information at specific time point."""
        if track_status is None or track_status.empty:
            record.update(
                {"TrackStatus": "1", "TrackStatusMessage": "AllClear", "SafetyCar": 0}
            )
            return

        # Find most recent track status change
        valid_statuses = track_status[track_status["StatusTimeSeconds"] <= time_point]

        if not valid_statuses.empty:
            latest_status = valid_statuses.iloc[-1]
            record.update(
                {
                    "TrackStatus": latest_status["Status"],
                    "TrackStatusMessage": latest_status["Message"],
                    "SafetyCar": 1 if latest_status["Status"] == "4" else 0,
                }
            )
        else:
            record.update(
                {"TrackStatus": "1", "TrackStatusMessage": "AllClear", "SafetyCar": 0}
            )

    def _balance_features(self, df: DataFrame) -> DataFrame:
        """Apply feature balancing using the existing logic."""
        feature_cols = [feature.value for feature in self.config.balance_features]

        # Create combination column for multi-feature balancing
        if len(feature_cols) == 1:
            combo_col = feature_cols[0]
        else:
            combo_col = "FeatureCombination"
            df[combo_col] = df[feature_cols].apply(
                lambda row: "_".join(str(val) for val in row), axis=1
            )

        combo_counts = df[combo_col].value_counts()

        if self.config.balance_method == "remove_insufficient":
            sufficient_combos = combo_counts[
                combo_counts >= self.config.min_samples
            ].index
            df = df[df[combo_col].isin(sufficient_combos)].copy()

        elif self.config.balance_method == "undersample_to_min":
            viable_combos = combo_counts[combo_counts >= self.config.min_samples]
            if not viable_combos.empty:
                min_count = viable_combos.min()
                balanced_dfs = []
                for combo in viable_combos.index:
                    combo_data = df[df[combo_col] == combo]
                    sampled = combo_data.sample(n=min_count, random_state=42)
                    balanced_dfs.append(sampled)
                df = pd.concat(balanced_dfs, ignore_index=True)

        elif self.config.balance_method == "undersample_to_target":
            target = self.config.target_samples or 3000
            viable_combos = combo_counts[combo_counts >= self.config.min_samples]
            balanced_dfs = []
            for combo in viable_combos.index:
                combo_data = df[df[combo_col] == combo]
                sample_size = min(len(combo_data), target)
                sampled = combo_data.sample(n=sample_size, random_state=42)
                balanced_dfs.append(sampled)
            df = pd.concat(balanced_dfs, ignore_index=True)

        # Clean up temporary column
        if len(feature_cols) > 1 and combo_col in df.columns:
            df = df.drop(columns=[combo_col])

        return df

    def clear_cache(self):
        """Clear session data cache."""
        self._session_cache.clear()


def test():
    # Basic usage with defaults
    session = fastf1.get_session(2024, "São Paulo Grand Prix", "R")
    session.load()

    ########################################################
    # Test 1 - Default config
    ########################################################

    # Default config
    config = PreprocessorConfig()
    preprocessor = F1DatasetPreprocessor(config)
    result = preprocessor.process_dataset(session)

    ########################################################
    # Test 2 - Custom config
    ########################################################

    print("Results:")
    print(f"Shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

    # Customized preprocessing
    config = PreprocessorConfig(
        interval_seconds=0.5,
        balance_features=[BaseFeatures.DRIVER, BaseFeatures.COMPOUND],
        balance_method="undersample_to_target",
        target_samples=2000,
        include_track_status=True,
    )
    preprocessor = F1DatasetPreprocessor(config)
    result = preprocessor.process_dataset(session)

    print("Results:")
    print(f"Shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")


if __name__ == "__main__":
    test()
    # Output:
    # ```
    # Processing Summary:
    # Success rate: 100.0%
    # Total data points: 2434317
    # INFO:__main__:Combined dataset: 24 races, 2434317 data points, 21 countries
    # ```