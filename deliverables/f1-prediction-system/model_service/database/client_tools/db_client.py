"""
data_access.py - Data Access Object for F1 TimescaleDB

Provides methods to retrieve data for the three REST API endpoints:
1. Get available sessions
2. Get features for predictions
3. Get telemetry coordinates
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SessionInfo:
    """Data class for session information."""
    session_id: str
    year: int
    race_name: str
    session_type: str
    session_date: datetime
    driver_count: int
    window_count: int

@dataclass
class DriverCoordinates:
    """Data class for driver coordinate data."""
    driver_number: int
    driver_abbreviation: str
    coordinates: List[Dict[str, float]]

class F1TimescaleDAO:
    """Data Access Object for F1 TimescaleDB."""

    def __init__(self, db_config: Dict[str, any]):
        """
        Initialize DAO with database configuration.

        Args:
            db_config: Dictionary with keys: host, port, database, user, password
        """
        self.db_config = db_config

    def _get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(**self.db_config)

    def get_available_sessions(self) -> List[SessionInfo]:
        """
        Get all available sessions with statistics.

        Returns:
            List of SessionInfo objects
        """
        query = """
            SELECT
                s.session_id,
                s.year,
                s.race_name,
                s.session_type,
                s.session_date,
                COALESCE(stats.driver_count, 0) as driver_count,
                COALESCE(stats.window_count, 0) as window_count
            FROM sessions s
            LEFT JOIN (
                SELECT
                    session_id,
                    COUNT(DISTINCT driver_number) as driver_count,
                    COUNT(*) as window_count
                FROM time_series_windows
                GROUP BY session_id
            ) stats ON s.session_id = stats.session_id
            ORDER BY s.session_date DESC, s.session_type
        """

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(query)
            results = cursor.fetchall()

            sessions = []
            for row in results:
                sessions.append(SessionInfo(
                    session_id=row['session_id'],
                    year=row['year'],
                    race_name=row['race_name'],
                    session_type=row['session_type'],
                    session_date=row['session_date'],
                    driver_count=row['driver_count'],
                    window_count=row['window_count']
                ))

            return sessions

        finally:
            cursor.close()
            conn.close()

    def get_session_features_for_prediction(self, session_id: str) -> Optional[Dict[str, any]]:
        """
        Get all features for a session, ready for model prediction.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary with:
                - features_by_driver: Dict mapping driver_number to (X, y_true, metadata)
                - session_info: Basic session information
                - feature_names: List of feature names
        """
        # Query to get all features grouped by driver
        query = """
            SELECT
                d.driver_number,
                d.driver_abbreviation,
                array_agg(w.window_index ORDER BY w.window_index) as window_indices,
                array_agg(f.feature_values ORDER BY w.window_index) as all_features,
                array_agg(w.y_true ORDER BY w.window_index) as y_true_values,
                array_agg(w.start_time ORDER BY w.window_index) as start_times,
                array_agg(w.prediction_time ORDER BY w.window_index) as prediction_times,
                MAX(w.sequence_length) as n_timesteps,
                MAX(array_length(w.features_used,1)) as n_features,
                MAX(w.features_used) as feature_names
            FROM time_series_windows w
            JOIN window_features f ON
                w.session_id = f.session_id AND
                w.driver_number = f.driver_number AND
                w.window_index = f.window_index
            JOIN drivers d ON w.driver_number = d.driver_number
            WHERE w.session_id = %s
            GROUP BY d.driver_number, d.driver_abbreviation
            ORDER BY d.driver_number
        """

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(query, (session_id,))
            results = cursor.fetchall()

            if not results:
                return None

            features_by_driver = {}
            feature_names = None

            for row in results:
                driver_number = row['driver_number']
                n_windows = len(row['window_indices'])
                n_timesteps = row['n_timesteps']
                n_features = row['n_features']

                if feature_names is None:
                    feature_names = row['feature_names']

                # Reconstruct X array for this driver
                X_list = []
                for i in range(n_windows):
                    # Get features for this window (stored as 1D)
                    features_1d = np.array(row['all_features'][i])

                    # Reshape to (n_timesteps, n_features)
                    features_2d = features_1d.reshape(n_timesteps, n_features)

                    # Transpose to (n_features, n_timesteps) for aeon
                    features_aeon = features_2d.T

                    X_list.append(features_aeon)

                # Stack into final array
                X = np.array(X_list)  # Shape: (n_windows, n_features, n_timesteps)
                y_true = np.array(row['y_true_values'])

                # Create metadata
                metadata = {
                    'driver_abbreviation': row['driver_abbreviation'],
                    'window_indices': row['window_indices'],
                    'start_times': row['start_times'],
                    'prediction_times': row['prediction_times']
                }

                features_by_driver[driver_number] = {
                    'X': X,
                    'y_true': y_true,
                    'metadata': metadata
                }

            # Get session info
            cursor.execute("""
                SELECT year, race_name, session_type, session_date
                FROM sessions
                WHERE session_id = %s
            """, (session_id,))

            session_info = cursor.fetchone()

            return {
                'features_by_driver': features_by_driver,
                'session_info': session_info,
                'feature_names': feature_names
            }

        finally:
            cursor.close()
            conn.close()

    def get_telemetry_coordinates(self, session_id: str) -> List[DriverCoordinates]:
        """
        Get X, Y, Z coordinates for all drivers in a session.

        Args:
            session_id: The session identifier

        Returns:
            List of DriverCoordinates objects
        """
        # First, try to get coordinates from telemetry_coordinates table
        query_telemetry = """
            SELECT
                t.driver_number,
                d.driver_abbreviation,
                array_agg(
                    json_build_object('X', t.x, 'Y', t.y, 'Z', t.z)
                    ORDER BY t.window_index
                ) as coordinates
            FROM telemetry_coordinates t
            JOIN drivers d ON t.driver_number = d.driver_number
            WHERE t.session_id = %s
            GROUP BY t.driver_number, d.driver_abbreviation
            ORDER BY t.driver_number
        """

        # Fallback: extract from feature values if telemetry_coordinates is empty
        query_features = """
            SELECT
                w.driver_number,
                d.driver_abbreviation,
                array_agg(
                    json_build_object(
                        'X', f.feature_values[6],  -- X is 6th feature (0-indexed)
                        'Y', f.feature_values[7],  -- Y is 7th feature
                        'Z', 0.0                   -- Z not in features
                    )
                    ORDER BY w.window_index
                ) as coordinates
            FROM time_series_windows w
            JOIN window_features f ON
                w.session_id = f.session_id AND
                w.driver_number = f.driver_number AND
                w.window_index = f.window_index
            JOIN drivers d ON w.driver_number = d.driver_number
            WHERE w.session_id = %s
                AND 'X' = ANY(w.features_used)
                AND 'Y' = ANY(w.features_used)
            GROUP BY w.driver_number, d.driver_abbreviation
            ORDER BY w.driver_number
        """

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Try telemetry_coordinates first
            cursor.execute(query_telemetry, (session_id,))
            results = cursor.fetchall()

            # If no results, try extracting from features
            if not results:
                cursor.execute(query_features, (session_id,))
                results = cursor.fetchall()

            coordinates_list = []
            for row in results:
                coordinates_list.append(DriverCoordinates(
                    driver_number=row['driver_number'],
                    driver_abbreviation=row['driver_abbreviation'],
                    coordinates=row['coordinates']
                ))

            return coordinates_list

        finally:
            cursor.close()
            conn.close()

    def store_predictions(self, session_id: str, driver_number: int,
                        predictions: List[Tuple[int, int, List[float], str]],
                        create_columns_if_missing: bool = True) -> int:
        """
        Store model predictions back to the database.

        Args:
            session_id: The session identifier
            driver_number: Driver number
            predictions: List of tuples (window_index, y_pred, y_proba, model_version)
            create_columns_if_missing: Whether to create prediction columns if they don't exist

        Returns:
            Number of predictions stored (0 if columns don't exist and create_columns_if_missing=False)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Check if prediction columns exist
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_name = 'time_series_windows'
                AND column_name = 'prediction_timestamp'
            """)

            columns_exist = cursor.fetchone()[0] > 0

            if not columns_exist:
                if not create_columns_if_missing:
                    return 0

                # Add columns
                cursor.execute("""
                    ALTER TABLE time_series_windows
                        ADD COLUMN IF NOT EXISTS y_pred INTEGER,
                        ADD COLUMN IF NOT EXISTS y_proba FLOAT[],
                        ADD COLUMN IF NOT EXISTS model_version TEXT,
                        ADD COLUMN IF NOT EXISTS prediction_timestamp TIMESTAMPTZ
                """)
                conn.commit()

            # Store predictions
            query = """
                UPDATE time_series_windows
                SET
                    y_pred = %s,
                    y_proba = %s,
                    model_version = %s,
                    prediction_timestamp = NOW()
                WHERE
                    session_id = %s AND
                    driver_number = %s AND
                    window_index = %s
            """

            count = 0
            for window_index, y_pred, y_proba, model_version in predictions:
                cursor.execute(query, (
                    y_pred,
                    y_proba,
                    model_version,
                    session_id,
                    driver_number,
                    window_index
                ))
                count += cursor.rowcount

            conn.commit()
            return count

        except Exception as e:
            conn.rollback()
            raise e

        finally:
            cursor.close()
            conn.close()