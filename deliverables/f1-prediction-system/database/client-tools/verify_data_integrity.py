#!/usr/bin/env python3
"""
Diagnose any data discrepancy between original dataset and stored values.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from f1_etl import create_safety_car_dataset
from ..config.data_config import CONFIG, WINDOW_SIZE, PREDICTION_HORIZON, NORMALIZE

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'f1_telemetry',
    'user': 'f1_user',
    'password': 'f1_password'
}

def diagnose_data_mismatch():
    """Compare original dataset with stored values to find discrepancies."""
    
    print("=== RECREATING ORIGINAL DATASET ===")
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
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Total samples: {len(metadata)}")
    
    # Group metadata by session/driver to match database structure
    metadata_lookup = {}
    for idx, meta in enumerate(metadata):
        key = (meta['SessionId'], int(meta['Driver']))
        if key not in metadata_lookup:
            metadata_lookup[key] = []
        metadata_lookup[key].append((idx, meta))
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Check a specific session/driver combination
        session_id = '2024_Chinese Grand Prix_R'
        driver_number = 1
        
        print(f"\n=== CHECKING {session_id} - Driver {driver_number} ===")
        
        # Get stored values from database
        cursor.execute("""
            SELECT 
                w.window_index,
                f.feature_values,
                w.y_true
            FROM time_series_windows w
            JOIN window_features f ON 
                w.session_id = f.session_id AND 
                w.driver_number = f.driver_number AND 
                w.window_index = f.window_index
            WHERE w.session_id = %s AND w.driver_number = %s
            ORDER BY w.window_index
            LIMIT 5
        """, (session_id, driver_number))
        
        db_windows = cursor.fetchall()
        
        # Get original data for same session/driver
        key = (session_id, driver_number)
        if key in metadata_lookup:
            original_data = metadata_lookup[key]
            
            print(f"\nOriginal data: {len(original_data)} windows")
            print(f"Database data: {len(db_windows)} windows")
            
            # Compare first few windows
            for db_window in db_windows[:3]:
                window_idx = db_window['window_index']
                
                # Get corresponding original data
                if window_idx < len(original_data):
                    original_idx, original_meta = original_data[window_idx]
                    original_features = X[original_idx]
                    original_y = y[original_idx]
                    
                    # Reshape database features
                    db_features = np.array(db_window['feature_values']).reshape(100, 9)
                    
                    print(f"\n--- Window {window_idx} ---")
                    print(f"Original index in dataset: {original_idx}")
                    print(f"Original metadata: {original_meta['start_time']} to {original_meta['end_time']}")
                    
                    # Compare first timestep
                    print(f"\nFirst timestep comparison:")
                    print(f"Original: {original_features[0, :5]}...")
                    print(f"Database: {db_features[0, :5]}...")
                    print(f"Match: {np.allclose(original_features[0], db_features[0])}")
                    
                    # Compare last timestep
                    print(f"\nLast timestep comparison:")
                    print(f"Original: {original_features[-1, :5]}...")
                    print(f"Database: {db_features[-1, :5]}...")
                    
                    # Check variance
                    print(f"\nVariance check:")
                    print(f"Original unique timesteps: {len(np.unique(original_features, axis=0))}")
                    print(f"Database unique timesteps: {len(np.unique(db_features, axis=0))}")
                    
                    # Check if y_true matches
                    print(f"\ny_true: Original={original_y}, Database={db_window['y_true']}")
        
        # Check if windows are ordered correctly
        print("\n=== CHECKING WINDOW ORDERING ===")
        cursor.execute("""
            SELECT 
                w.window_index,
                w.start_time,
                w.end_time,
                w.prediction_time
            FROM time_series_windows w
            WHERE w.session_id = %s AND w.driver_number = %s
            ORDER BY w.window_index
            LIMIT 10
        """, (session_id, driver_number))
        
        time_windows = cursor.fetchall()
        
        print("\nDatabase window timings:")
        for tw in time_windows[:5]:
            print(f"Window {tw['window_index']}: {tw['start_time']} -> {tw['end_time']}")
        
        # Check metadata ordering
        if key in metadata_lookup:
            print("\nOriginal metadata timings:")
            for i, (idx, meta) in enumerate(metadata_lookup[key][:5]):
                print(f"Window {i}: {meta['start_time']} -> {meta['end_time']} (dataset index: {idx})")
        
    finally:
        cursor.close()
        conn.close()

def check_data_uniqueness():
    """Check how many unique values exist across all data."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        print("\n=== DATA UNIQUENESS CHECK ===")
        
        # Sample multiple windows
        cursor.execute("""
            SELECT 
                session_id,
                driver_number,
                window_index,
                feature_values[1:9] as first_timestep
            FROM window_features
            ORDER BY session_id, driver_number, window_index
            LIMIT 100
        """)
        
        results = cursor.fetchall()
        
        # Collect all first timesteps
        first_timesteps = []
        for row in results:
            first_timesteps.append(tuple(row[3]))  # Convert to tuple for hashing
        
        unique_timesteps = set(first_timesteps)
        print(f"Unique first timesteps across 100 windows: {len(unique_timesteps)}")
        
        if len(unique_timesteps) < 10:
            print("\nUnique values found:")
            for i, ts in enumerate(unique_timesteps):
                print(f"{i}: {ts[:3]}...")  # Show first 3 features
                
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    diagnose_data_mismatch()
    check_data_uniqueness()