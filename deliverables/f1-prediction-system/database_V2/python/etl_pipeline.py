#!/usr/bin/env python3
"""
ETL script to load F1 telemetry data into TimescaleDB.
This script transforms the data from f1_etl format into the database schema.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
import fastf1

# Import user configuration
import sys
sys.path.append('/scripts')
from data_config import CONFIG, WINDOW_SIZE, PREDICTION_HORIZON, NORMALIZE, DB_CONFIG

# Import f1_etl
from f1_etl import create_safety_car_dataset, DriverLabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_driver_mappings(config):
    """Build driver mappings by loading each session and extracting driver info."""
    driver_to_number = {}
    number_to_abbreviation = {}
    
    logger.info("Building driver mappings from sessions...")
    
    for session_config in config.sessions:
        try:
            # Load the session
            session = fastf1.get_session(
                session_config.year,
                session_config.race_name,
                session_config.session_type
            )
            session.load()
            
            # Create and fit encoder for this session
            encoder = DriverLabelEncoder()
            encoder.fit_session(session)
            
            # Merge mappings - keep both string and int versions
            for abbr, num_str in encoder.driver_to_number.items():
                driver_num = int(num_str)
                driver_to_number[abbr] = num_str  # Keep as string
                number_to_abbreviation[num_str] = abbr  # String key
                number_to_abbreviation[driver_num] = abbr  # Int key for compatibility
                
        except Exception as e:
            logger.warning(f"Failed to load session {session_config}: {e}")
            continue
    
    logger.info(f"Found {len(driver_to_number)} unique drivers across all sessions")
    return driver_to_number, number_to_abbreviation

def get_db_connection():
    """Create database connection using environment variables or config."""
    # During initialization, PostgreSQL only accepts Unix socket connections
    # Check if we're in the initialization phase by looking for POSTGRES_USER env var
    if os.environ.get("POSTGRES_USER") and not os.environ.get("POSTGRES_HOST"):
        # Use Unix socket connection during initialization
        conn_params = {
            "database": os.environ.get("POSTGRES_DB", DB_CONFIG["database"]),
            "user": os.environ.get("POSTGRES_USER", DB_CONFIG["user"]),
            # No host parameter = use Unix socket
        }
    else:
        # Normal TCP/IP connection
        conn_params = {
            "host": os.environ.get("POSTGRES_HOST", DB_CONFIG["host"]),
            "port": os.environ.get("POSTGRES_PORT", DB_CONFIG["port"]),
            "database": os.environ.get("POSTGRES_DB", DB_CONFIG["database"]),
            "user": os.environ.get("POSTGRES_USER", DB_CONFIG["user"]),
            "password": os.environ.get("POSTGRES_PASSWORD", DB_CONFIG["password"])
        }
    return psycopg2.connect(**conn_params)

def insert_drivers(cursor, metadata_list, number_to_abbreviation):
    """Insert unique drivers from metadata."""
    drivers = set()
    for meta in metadata_list:
        # Driver number in metadata is stored as string
        driver_num_str = meta['Driver']
        driver_num = int(driver_num_str)
        
        # Check both string and int versions in the mapping
        if driver_num in number_to_abbreviation:
            drivers.add((driver_num, number_to_abbreviation[driver_num]))
        elif driver_num_str in number_to_abbreviation:
            drivers.add((driver_num, number_to_abbreviation[driver_num_str]))
    
    for driver_num, driver_abbr in drivers:
        cursor.execute("""
            INSERT INTO drivers (driver_number, driver_abbreviation)
            VALUES (%s, %s)
            ON CONFLICT (driver_number) DO NOTHING
        """, (driver_num, driver_abbr))
    
    logger.info(f"Inserted {len(drivers)} drivers")

def insert_sessions(cursor, metadata_list):
    """Insert unique sessions from metadata."""
    sessions = {}
    for meta in metadata_list:
        session_id = meta['SessionId']
        if session_id not in sessions:
            # Parse session_id format: "YYYY_Race Name_SessionType"
            parts = session_id.split('_')
            year = int(parts[0])
            session_type = parts[-1]
            race_name = '_'.join(parts[1:-1])
            
            sessions[session_id] = {
                'year': year,
                'race_name': race_name,
                'session_type': session_type,
                'session_date': meta['start_time']
            }
    
    for session_id, session_data in sessions.items():
        cursor.execute("""
            INSERT INTO sessions (session_id, year, race_name, session_type, session_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
        """, (
            session_id,
            session_data['year'],
            session_data['race_name'],
            session_data['session_type'],
            session_data['session_date']
        ))
    
    logger.info(f"Inserted {len(sessions)} sessions")

def prepare_window_data(X, y, metadata, window_idx):
    """Prepare data for a single time window."""
    meta = metadata[window_idx]
    
    # X shape is (n_samples, n_features, n_timesteps)
    # We need to transpose to (n_timesteps, n_features) for storage
    feature_matrix = X[window_idx].T.tolist()
    
    return {
        'start_time': meta['start_time'].isoformat(),
        'end_time': meta['end_time'].isoformat(),
        'prediction_time': meta['prediction_time'].isoformat(),
        'feature_matrix': feature_matrix,
        'y_true': int(y[window_idx]),
        'sequence_length': meta['sequence_length'],
        'prediction_horizon': meta['prediction_horizon'],
        'features_used': meta['features_used']
    }

def extract_coordinates_from_telemetry(raw_telemetry, metadata, number_to_abbreviation):
    """Extract X, Y, Z coordinates from raw telemetry data."""
    coordinates_by_session_driver = {}
    
    # raw_telemetry contains data from all sessions - group by SessionId and Driver
    for (session_id, driver), group_df in raw_telemetry.groupby(['SessionId', 'Driver']):
        driver_num = int(driver)
        if driver_num not in number_to_abbreviation:
            continue
            
        key = (session_id, driver_num)
        
        # Sample coordinates at regular intervals to match windows
        coordinates = []
        
        # Get metadata entries for this specific session/driver combination
        driver_metadata = [m for m in metadata 
                          if m['SessionId'] == session_id and int(m['Driver']) == driver_num]
        
        for meta in driver_metadata:
            # Find telemetry point closest to window start time
            # Use Date column for datetime comparison
            mask = (group_df['Date'] >= meta['start_time']) & \
                   (group_df['Date'] <= meta['end_time'])
            window_data = group_df.loc[mask]
            
            if not window_data.empty and 'X' in window_data.columns:
                coord = {
                    'X': float(window_data.iloc[0]['X']) if not np.isnan(window_data.iloc[0]['X']) else 0.0,
                    'Y': float(window_data.iloc[0]['Y']) if not np.isnan(window_data.iloc[0]['Y']) else 0.0,
                    'Z': float(window_data.iloc[0]['Z']) if 'Z' in window_data.columns and not np.isnan(window_data.iloc[0]['Z']) else 0.0
                }
                coordinates.append(coord)
        
        if coordinates:
            coordinates_by_session_driver[key] = coordinates
    
    return coordinates_by_session_driver

def load_data():
    """Main ETL function."""
    logger.info("Starting F1 telemetry data ETL process...")
    
    # Create dataset using f1_etl
    logger.info("Creating safety car dataset...")
    dataset = create_safety_car_dataset(
        config=CONFIG,
        window_size=WINDOW_SIZE,
        prediction_horizon=PREDICTION_HORIZON,
        normalize=NORMALIZE,
        target_column="TrackStatus",
    )
    
    X = dataset['X']
    y = dataset['y']
    metadata = dataset['metadata']
    raw_telemetry = dataset.get('raw_telemetry', [])
    
    logger.info(f"Dataset created: {X.shape[0]} windows, {X.shape[1]} features, {X.shape[2]} timesteps")
    
    # Extract driver mappings from metadata
    driver_to_number = {}
    number_to_abbreviation = {}
    
    # Get unique drivers from metadata
    for meta in metadata:
        driver_num = int(meta['Driver'])
        # We need to get abbreviations from somewhere - check if it's in metadata
        # If not, we'll need to extract from raw_telemetry or use a placeholder
        if 'driver_abbreviation' in meta:
            abbr = meta['driver_abbreviation']
        else:
            # Try to find abbreviation from raw telemetry
            driver_mask = raw_telemetry['Driver'] == driver_num
            if driver_mask.any():
                # Use driver number as abbreviation if not available
                abbr = f"D{driver_num:02d}"
            else:
                continue
        
        driver_to_number[abbr] = driver_num
        number_to_abbreviation[driver_num] = abbr
    
    logger.info(f"Found {len(number_to_abbreviation)} unique drivers in dataset")
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert drivers and sessions
        insert_drivers(cursor, metadata, number_to_abbreviation)
        insert_sessions(cursor, metadata)
        conn.commit()
        
        # Extract coordinates from raw telemetry
        logger.info("Extracting telemetry coordinates...")
        coordinates_data = extract_coordinates_from_telemetry(raw_telemetry, metadata, number_to_abbreviation)
        
        # Group windows by session and driver
        windows_by_session_driver = {}
        for idx, meta in enumerate(metadata):
            session_id = meta['SessionId']
            driver_num = int(meta['Driver'])
            key = (session_id, driver_num)
            
            if key not in windows_by_session_driver:
                windows_by_session_driver[key] = []
            
            window_data = prepare_window_data(X, y, metadata, idx)
            windows_by_session_driver[key].append(window_data)
        
        # Insert time series windows in batches
        logger.info("Inserting time series windows...")
        total_windows = 0
        
        for (session_id, driver_num), windows in tqdm(windows_by_session_driver.items()):
            if driver_num not in number_to_abbreviation:
                continue
            
            # Insert windows
            cursor.execute(
                "SELECT insert_time_series_batch(%s, %s, %s)",
                (session_id, driver_num, Json(windows))
            )
            
            # Insert coordinates if available
            key = (session_id, driver_num)
            if key in coordinates_data:
                cursor.execute(
                    "SELECT insert_telemetry_coordinates_batch(%s, %s, %s)",
                    (session_id, driver_num, Json(coordinates_data[key]))
                )
            
            total_windows += len(windows)
        
        conn.commit()
        logger.info(f"Successfully loaded {total_windows} time windows")
        
        # Verify data
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT driver_number) FROM drivers")
        driver_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM time_series_windows")
        window_count = cursor.fetchone()[0]
        
        logger.info(f"Database now contains: {session_count} sessions, {driver_count} drivers, {window_count} windows")
        
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_data()