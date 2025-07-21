#!/usr/bin/env python3
"""
Final validation that data is ready for API consumption.
"""

import psycopg2
import numpy as np
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'f1_telemetry',
    'user': 'f1_user',
    'password': 'f1_password'
}

def validate_api_readiness():
    """Validate that data can be retrieved and reshaped for model predictions."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Test the get_session_features function
        print("=== TESTING API DATA RETRIEVAL ===")
        
        session_id = '2024_Monaco Grand Prix_R'
        
        # Get features for one driver
        cursor.execute("""
            SELECT 
                w.driver_number,
                d.driver_abbreviation,
                COUNT(*) as n_windows,
                array_agg(w.window_index ORDER BY w.window_index) as window_indices,
                array_agg(f.feature_values ORDER BY w.window_index) as all_features,
                array_agg(w.y_true ORDER BY w.window_index) as y_true_values
            FROM time_series_windows w
            JOIN window_features f ON 
                w.session_id = f.session_id AND 
                w.driver_number = f.driver_number AND 
                w.window_index = f.window_index
            JOIN drivers d ON w.driver_number = d.driver_number
            WHERE w.session_id = %s AND w.driver_number = 1
            GROUP BY w.driver_number, d.driver_abbreviation
        """, (session_id,))
        
        result = cursor.fetchone()
        
        if result:
            print(f"Driver: {result['driver_abbreviation']} (#{result['driver_number']})")
            print(f"Windows: {result['n_windows']}")
            
            # Reconstruct data for model
            X_list = []
            for i in range(result['n_windows']):
                # Get features for this window
                features_1d = np.array(result['all_features'][i])
                
                # Reshape to (100, 9)
                features_2d = features_1d.reshape(100, 9)
                
                # Transpose to (9, 100) for aeon
                features_aeon = features_2d.T
                
                X_list.append(features_aeon)
            
            # Stack into batch
            X = np.array(X_list)
            y_true = np.array(result['y_true_values'])
            
            print(f"\nFinal shapes for model:")
            print(f"X shape: {X.shape} (n_windows, n_features, n_timesteps)")
            print(f"y_true shape: {y_true.shape}")
            
            # Validate shape
            expected_shape = (result['n_windows'], 9, 100)
            assert X.shape == expected_shape, f"Expected {expected_shape}, got {X.shape}"
            print("✓ Shape validation passed!")
            
            # Check data range (should be normalized)
            print(f"\nData range check:")
            print(f"X min: {X.min():.3f}, max: {X.max():.3f}")
            print(f"X mean: {X.mean():.3f}, std: {X.std():.3f}")
            
            # Check label distribution
            unique_labels, counts = np.unique(y_true, return_counts=True)
            print(f"\nLabel distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  Label {label}: {count} windows ({100*count/len(y_true):.1f}%)")
            
            print("\n✓ Data is ready for API and model predictions!")
            
        else:
            print(f"No data found for session {session_id}")
            
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    validate_api_readiness()