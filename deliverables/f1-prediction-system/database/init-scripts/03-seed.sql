-- Sample data

-- Sample drivers
INSERT INTO drivers (driver_id, name, team, car_number) VALUES
('VER', 'Max Verstappen', 'Red Bull Racing', 1),
('HAM', 'Lewis Hamilton', 'Mercedes', 44),
('LEC', 'Charles Leclerc', 'Ferrari', 16) 
ON CONFLICT (driver_id) DO NOTHING;

-- Sample session
SELECT create_session(
    '2023_Spanish_Grand_Prix_Q',
    '2023 Spanish Grand Prix',
    'Q',
    'Circuit de Barcelona-Catalunya',
    '2023-06-04 14:00:00+00'::TIMESTAMPTZ,
    '{"weather": "sunny", "temperature": 25}'::JSONB
);

-- Sample telemetry data
INSERT INTO telemetry (time, session_id, driver_id, speed, rpm, ngear, throttle, brake, source)
VALUES 
('2023-06-04 14:05:00+00', '2023_Spanish_Grand_Prix_Q', 'VER', 250.5, 11000, 6, 95.0, false, 'sample'),
('2023-06-04 14:05:01+00', '2023_Spanish_Grand_Prix_Q', 'VER', 245.2, 10800, 6, 90.0, false, 'sample'),
('2023-06-04 14:05:02+00', '2023_Spanish_Grand_Prix_Q', 'HAM', 248.1, 10950, 6, 93.0, false, 'sample');

-- Sample predictions
INSERT INTO predictions (session_id, driver_id, model_version, predicted_track_status, confidence)
VALUES 
('2023_Spanish_Grand_Prix_Q', 'VER', 'v1.0', 'green', 0.95),
('2023_Spanish_Grand_Prix_Q', 'HAM', 'v1.0', 'green', 0.92);

CALL refresh_continuous_aggregate('telemetry_1min', NULL, NULL);

SELECT 'Sample data loaded successfully' as status;
