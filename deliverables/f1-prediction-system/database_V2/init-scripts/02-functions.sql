-- Database Functions for F1 Telemetry System

-- Function to batch insert telemetry data
CREATE OR REPLACE FUNCTION insert_telemetry_batch(
    p_session_id VARCHAR(100),
    p_driver_id VARCHAR(20),
    p_telemetry_data JSONB
) RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    telemetry_point JSONB;
BEGIN
    FOR telemetry_point IN SELECT * FROM jsonb_array_elements(p_telemetry_data)
    LOOP
        INSERT INTO telemetry (
            time, session_id, driver_id, speed, rpm, ngear, throttle, brake, drs,
            x, y, z, status, session_time, source, raw_data
        ) VALUES (
            COALESCE((telemetry_point->>'Date')::TIMESTAMPTZ, NOW()),
            p_session_id,
            p_driver_id,
            (telemetry_point->>'Speed')::DECIMAL,
            (telemetry_point->>'RPM')::DECIMAL,
            (telemetry_point->>'nGear')::INTEGER,
            (telemetry_point->>'Throttle')::DECIMAL,
            (telemetry_point->>'Brake')::BOOLEAN,
            (telemetry_point->>'DRS')::INTEGER,
            (telemetry_point->>'X')::DECIMAL,
            (telemetry_point->>'Y')::DECIMAL,
            (telemetry_point->>'Z')::DECIMAL,
            (telemetry_point->>'Status')::VARCHAR,
            (telemetry_point->>'SessionTime')::INTERVAL,
            COALESCE((telemetry_point->>'Source')::VARCHAR, 'api'),
            telemetry_point
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to create sessions
CREATE OR REPLACE FUNCTION create_session(
    p_session_id VARCHAR(100),
    p_race_name VARCHAR(200),
    p_session_type VARCHAR(20),
    p_track_name VARCHAR(200) DEFAULT NULL,
    p_start_time TIMESTAMPTZ DEFAULT NOW(),
    p_metadata JSONB DEFAULT '{}'
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO sessions (session_id, race_name, session_type, track_name, start_time, metadata)
    VALUES (p_session_id, p_race_name, p_session_type, p_track_name, p_start_time, p_metadata)
    ON CONFLICT (session_id) DO UPDATE SET
        race_name = EXCLUDED.race_name,
        session_type = EXCLUDED.session_type,
        track_name = EXCLUDED.track_name,
        start_time = EXCLUDED.start_time,
        metadata = EXCLUDED.metadata;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to store predictions
CREATE OR REPLACE FUNCTION store_predictions(
    p_session_id VARCHAR(100),
    p_predictions JSONB,
    p_model_version VARCHAR(50) DEFAULT 'v1.0'
) RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    prediction JSONB;
BEGIN
    FOR prediction IN SELECT * FROM jsonb_array_elements(p_predictions)
    LOOP
        INSERT INTO predictions (
            session_id, driver_id, model_version, predicted_track_status,
            confidence, model_metadata
        ) VALUES (
            p_session_id,
            prediction->>'driver_id',
            p_model_version,
            prediction->>'predicted_track_status',
            (prediction->>'confidence')::DECIMAL,
            prediction->'metadata'
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;
