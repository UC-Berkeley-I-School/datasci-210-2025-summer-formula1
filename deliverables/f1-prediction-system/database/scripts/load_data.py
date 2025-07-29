#!/usr/bin/env python3
"""
ETL script to load F1 telemetry data into TimescaleDB.
This script transforms the data from f1_etl format into the database schema.

The original implementation hard‑coded the host and port for the PostgreSQL
connection and ignored the standard PGHOST/PGPORT environment variables.
When the ETL runs inside the docker entrypoint it should connect over
the Unix socket provided by the base image, but the TCP settings caused
intermittent connection failures. To address this, get_db_connection
now prefers PGHOST/PGPORT if present and falls back to the traditional
POSTGRES_* variables or the default DB_CONFIG. Additional explanatory
comments have been added throughout to aid future maintenance.
"""

import os
import psycopg2
from psycopg2.extras import Json
from datetime import datetime  # noqa: F401 (kept for future use)
import numpy as np
from tqdm import tqdm
import logging
import time
import gc

# Import user configuration
import sys
sys.path.append('/scripts')
from data_config import CONFIG, WINDOW_SIZE, PREDICTION_HORIZON, NORMALIZE, DB_CONFIG  # noqa: E402

# Import f1_etl
from f1_etl import create_safety_car_dataset, DriverLabelEncoder  # noqa: E402,F401

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """Create a database connection using environment variables or fallback config.

    At runtime the PostgreSQL host and port may be provided via either the
    traditional POSTGRES_* variables (e.g. POSTGRES_HOST, POSTGRES_PORT) or
    via PGHOST/PGPORT when connecting over a Unix socket. To support both
    approaches, this helper inspects PGHOST/PGPORT first and only falls back
    to POSTGRES_HOST/POSTGRES_PORT or DB_CONFIG when necessary. It also
    preserves other connection options for reliability.
    """
    # Base connection parameters with defaults
    conn_params = {
        "database": os.environ.get("POSTGRES_DB", DB_CONFIG["database"]),
        "user": os.environ.get("POSTGRES_USER", DB_CONFIG["user"]),
        "password": os.environ.get("POSTGRES_PASSWORD", DB_CONFIG["password"]),
        "connect_timeout": 60,
        # Disable statement timeout and idle transaction timeout to avoid
        # unexpected interruptions when processing large batches.
        "options": "-c statement_timeout=0 -c idle_in_transaction_session_timeout=0",
        # Keepalive settings to mitigate network interruptions
        "keepalives": 1,
        "keepalives_idle": 10,
        "keepalives_interval": 5,
        "keepalives_count": 3,
    }

    # Prefer PGHOST/PGPORT if provided (e.g. to use Unix domain sockets)
    pg_host = os.environ.get("PGHOST")
    pg_port = os.environ.get("PGPORT")
    if pg_host:
        conn_params["host"] = pg_host
        if pg_port:
            conn_params["port"] = int(pg_port)
    else:
        conn_params["host"] = os.environ.get("POSTGRES_HOST", DB_CONFIG["host"])
        conn_params["port"] = int(os.environ.get("POSTGRES_PORT", DB_CONFIG["port"]))

    return psycopg2.connect(**conn_params)


def process_dataset_in_chunks():
    """Process dataset in smaller chunks to manage memory."""
    logger.info("Starting F1 telemetry data ETL process...")

    # First, just get the metadata without loading all the data
    logger.info("Loading dataset metadata...")

    # Create dataset
    dataset = create_safety_car_dataset(
        config=CONFIG,
        window_size=WINDOW_SIZE,
        prediction_horizon=PREDICTION_HORIZON,
        normalize=NORMALIZE,
        target_column="TrackStatus",
    )

    # Save essential data
    X = dataset['X']
    metadata = dataset['metadata']
    y = dataset['y']

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Extract driver mappings
    number_to_abbreviation = {}
    known_mappings = {
        '1': 'VER', '11': 'PER', '16': 'LEC', '55': 'SAI',
        '63': 'RUS', '44': 'HAM', '14': 'ALO', '18': 'STR',
        '4': 'NOR', '81': 'PIA', '23': 'ALB', '2': 'SAR',
        '77': 'BOT', '24': 'ZHO', '27': 'HUL', '20': 'MAG',
        '22': 'TSU', '31': 'OCO', '10': 'GAS', '3': 'RIC',
        '30': 'LAW', '43': 'COL'
    }

    unique_drivers = set(meta['Driver'] for meta in metadata)
    for driver in unique_drivers:
        driver_str = str(driver)
        driver_num = int(driver)
        abbr = known_mappings.get(driver_str, f'D{driver_num:02d}')
        number_to_abbreviation[driver_str] = abbr
        number_to_abbreviation[driver_num] = abbr

    # Insert basic metadata
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        insert_drivers(cursor, metadata, number_to_abbreviation)
        insert_sessions(cursor, metadata)
        conn.commit()
        logger.info("Metadata inserted successfully")
    finally:
        cursor.close()
        conn.close()

    # Process windows by session to reduce memory usage
    session_drivers = {}
    for idx, meta in enumerate(metadata):
        key = (meta['SessionId'], int(meta['Driver']))
        if key not in session_drivers:
            session_drivers[key] = []
        session_drivers[key].append(idx)

    logger.info(f"Processing {len(session_drivers)} session-driver combinations")

    # Process each session-driver combination separately
    for (session_id, driver_number), indices in tqdm(session_drivers.items(), desc="Processing sessions"):
        process_session_driver_with_features(session_id, driver_number, indices, metadata, X, y)
        # Force garbage collection after each session
        gc.collect()
        # Small delay to reduce CPU pressure
        time.sleep(0.1)


def process_session_driver_with_features(session_id, driver_number, indices, metadata, X, y):
    """Process a single session-driver combination with features."""
    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Process in very small batches
        batch_size = 10
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            for local_idx, global_idx in enumerate(batch_indices):
                meta = metadata[global_idx]
                window_index = i + local_idx  # Correct window index

                # Insert window metadata
                cursor.execute(
                    """
                    INSERT INTO time_series_windows 
                    (session_id, driver_number, window_index, start_time, 
                     end_time, prediction_time, y_true, sequence_length, 
                     prediction_horizon, features_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        session_id, driver_number, window_index,
                        meta['start_time'], meta['end_time'],
                        meta['prediction_time'], int(y[global_idx]),
                        meta['sequence_length'], meta['prediction_horizon'],
                        meta['features_used'],
                    ),
                )

                # Insert features - flatten the 2D array to 1D for storage
                feature_values = X[global_idx].flatten().tolist()  # Flatten to 1D

                cursor.execute(
                    """
                    INSERT INTO window_features 
                    (session_id, driver_number, window_index, feature_values)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        session_id, driver_number, window_index,
                        feature_values,
                    ),
                )

            # Commit frequently to avoid holding long transactions
            conn.commit()
            # Brief pause every batch
            time.sleep(0.01)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_drivers(cursor, metadata_list, number_to_abbreviation):
    """Insert unique drivers from metadata."""
    drivers = set()
    for meta in metadata_list:
        driver_num = int(meta['Driver'])
        if driver_num in number_to_abbreviation:
            drivers.add((driver_num, number_to_abbreviation[driver_num]))

    for driver_num, driver_abbr in drivers:
        cursor.execute(
            """
            INSERT INTO drivers (driver_number, driver_abbreviation)
            VALUES (%s, %s)
            ON CONFLICT (driver_number) DO NOTHING
            """,
            (driver_num, driver_abbr),
        )

    logger.info(f"Inserted {len(drivers)} drivers")


def insert_sessions(cursor, metadata_list):
    """Insert unique sessions from metadata."""
    sessions = {}
    for meta in metadata_list:
        session_id = meta['SessionId']
        if session_id not in sessions:
            parts = session_id.split('_')
            year = int(parts[0])
            session_type = parts[-1]
            race_name = '_'.join(parts[1:-1])

            sessions[session_id] = {
                'year': year,
                'race_name': race_name,
                'session_type': session_type,
                'session_date': meta['start_time'],
            }

    for session_id, session_data in sessions.items():
        cursor.execute(
            """
            INSERT INTO sessions (session_id, year, race_name, session_type, session_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (
                session_id,
                session_data['year'],
                session_data['race_name'],
                session_data['session_type'],
                session_data['session_date'],
            ),
        )

    logger.info(f"Inserted {len(sessions)} sessions")


def verify_data():
    """Verify loaded data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM sessions")
        session_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT driver_number) FROM drivers")
        driver_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM time_series_windows")
        window_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM window_features")
        feature_count = cursor.fetchone()[0]

        logger.info(
            f"Database contains: {session_count} sessions, {driver_count} drivers, {window_count} windows, {feature_count} feature matrices"
        )
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    try:
        process_dataset_in_chunks()
        verify_data()
        logger.info("Data loading completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise