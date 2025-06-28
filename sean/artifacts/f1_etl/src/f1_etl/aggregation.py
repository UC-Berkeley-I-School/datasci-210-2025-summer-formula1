"""Data aggregation across multiple sessions"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from .logging import logger


class DataAggregator:
    """Aggregates raw data across multiple sessions"""

    def __init__(self):
        self.aggregated_data = defaultdict(list)

    def aggregate_telemetry_data(
        self, sessions_data: List[Dict[str, Any]], drivers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aggregate telemetry data across sessions with track status alignment"""
        all_telemetry = []

        for session_data in sessions_data:
            session_telemetry = self._merge_session_telemetry(session_data, drivers)

            session_telemetry["SessionYear"] = session_data["session_info"]["year"]
            session_telemetry["SessionRace"] = session_data["session_info"]["race"]
            session_telemetry["SessionType"] = session_data["session_info"][
                "session_type"
            ]
            session_telemetry["SessionId"] = (
                f"{session_data['session_info']['year']}_{session_data['session_info']['race']}_{session_data['session_info']['session_type']}"
            )

            track_status = session_data.get("track_status", pd.DataFrame())
            t0_date = session_data["session_info"]["t0_date"]
            session_telemetry = self._align_track_status(
                session_telemetry, track_status, t0_date
            )

            all_telemetry.append(session_telemetry)

        return pd.concat(all_telemetry, ignore_index=True)

    def _align_track_status(
        self, telemetry: pd.DataFrame, track_status: pd.DataFrame, t0_date
    ) -> pd.DataFrame:
        """Align track status with telemetry timestamps using forward fill"""
        if track_status is None or track_status.empty or telemetry.empty:
            if not telemetry.empty:
                telemetry["TrackStatus"] = "1"
                telemetry["TrackStatusMessage"] = "AllClear"
            return telemetry

        if "Date" not in telemetry.columns:
            logger.warning(
                "No Date column in telemetry data, skipping track status alignment"
            )
            telemetry["TrackStatus"] = "1"
            telemetry["TrackStatusMessage"] = "AllClear"
            return telemetry

        if "Time" not in track_status.columns or "Status" not in track_status.columns:
            logger.warning("Track status data missing required columns, using default")
            telemetry["TrackStatus"] = "1"
            telemetry["TrackStatusMessage"] = "AllClear"
            return telemetry

        try:
            track_status_with_date = track_status.copy()
            track_status_with_date["Date"] = t0_date + track_status_with_date["Time"]

            status_cols = ["Date", "Status"]
            if "Message" in track_status_with_date.columns:
                status_cols.append("Message")

            telemetry_with_status = pd.merge_asof(
                telemetry.sort_values("Date"),
                track_status_with_date[status_cols].sort_values("Date"),
                on="Date",
                direction="backward",
            ).fillna({"Status": "1", "Status_y": "1"})  # Handle both possible names

            # Check which Status column exists and rename appropriately
            status_col = (
                "Status_y" if "Status_y" in telemetry_with_status.columns else "Status"
            )
            message_col = (
                "Message_y"
                if "Message_y" in telemetry_with_status.columns
                else "Message"
            )

            if message_col not in telemetry_with_status.columns:
                telemetry_with_status[message_col] = "AllClear"
            else:
                telemetry_with_status[message_col] = telemetry_with_status[
                    message_col
                ].fillna("AllClear")

            # Rename using the correct column names
            rename_dict = {status_col: "TrackStatus"}
            if message_col in telemetry_with_status.columns:
                rename_dict[message_col] = "TrackStatusMessage"

            telemetry_with_status = telemetry_with_status.rename(columns=rename_dict)

            return telemetry_with_status

        except Exception as e:
            logger.warning(f"Failed to align track status: {e}")
            telemetry["TrackStatus"] = "1"
            telemetry["TrackStatusMessage"] = "AllClear"
            return telemetry

    def _merge_session_telemetry(
        self, session_data: Dict[str, Any], drivers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Merge car and position data for a single session"""
        session_drivers = drivers if drivers else session_data["drivers"]
        session_telemetry = []

        for driver_number in session_drivers:
            if driver_number not in session_data["car_data"]:
                continue

            try:
                car_data = session_data["car_data"][driver_number]
                pos_data = session_data["pos_data"][driver_number]

                merged = car_data.merge_channels(pos_data, frequency="original")
                merged = merged.add_distance().add_differential_distance()
                merged["Driver"] = driver_number

                session_telemetry.append(merged)

            except Exception as e:
                logger.warning(
                    f"Could not merge telemetry for driver {driver_number}: {e}"
                )

        if session_telemetry:
            result = pd.concat(session_telemetry, ignore_index=True)
            if "Date" not in result.columns and "Time" in result.columns:
                result = result.rename(columns={"Time": "Date"})
            elif (
                "Date" not in result.columns
                and hasattr(result, "index")
                and hasattr(result.index, "name")
            ):
                result = result.reset_index()
                if "index" in result.columns:
                    result = result.rename(columns={"index": "Date"})
            return result
        else:
            return pd.DataFrame()
