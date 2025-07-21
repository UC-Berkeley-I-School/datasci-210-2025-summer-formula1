--- Check that features were loaded
-- docker exec -it f1-timescaledb psql -U f1_user -d f1_telemetry -c "
SELECT 
    COUNT(*) as total_features,
    COUNT(DISTINCT session_id) as sessions_with_features,
    COUNT(DISTINCT driver_number) as drivers_with_features
FROM window_features
WHERE feature_values IS NOT NULL;

--- Check feature dimensions (1D array)
-- docker exec -it f1-timescaledb psql -U f1_user -d f1_telemetry -c "
SELECT 
    session_id,
    driver_number,
    window_index,
    array_length(feature_values, 1) as total_values,
    array_length(feature_values, 1) / 9 as n_timesteps,
    9 as n_features
FROM window_features
WHERE feature_values IS NOT NULL
LIMIT 5;

--- Get sample values
-- docker exec -it f1-timescaledb psql -U f1_user -d f1_telemetry -c "
SELECT 
    session_id,
    driver_number,
    window_index,
    feature_values[1:9] as first_timestep_features
FROM window_features
WHERE feature_values IS NOT NULL
LIMIT 1;