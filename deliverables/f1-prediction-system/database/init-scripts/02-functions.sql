-- Function to efficiently insert time series windows in batches
CREATE OR REPLACE FUNCTION insert_time_series_batch(
    p_session_id VARCHAR(100),
    p_driver_number INTEGER,
    p_windows JSONB
) RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    window_data JSONB;
    window_idx INTEGER := 0;
BEGIN
    FOR window_data IN SELECT * FROM jsonb_array_elements(p_windows)
    LOOP
        INSERT INTO time_series_windows (
            session_id,
            driver_number,
            window_index,
            start_time,
            end_time,
            prediction_time,
            feature_matrix,
            y_true,
            sequence_length,
            prediction_horizon,
            features_used
        ) VALUES (
            p_session_id,
            p_driver_number,
            window_idx,
            (window_data->>'start_time')::TIMESTAMPTZ,
            (window_data->>'end_time')::TIMESTAMPTZ,
            (window_data->>'prediction_time')::TIMESTAMPTZ,
            window_data->'feature_matrix',
            (window_data->>'y_true')::INTEGER,
            (window_data->>'sequence_length')::INTEGER,
            (window_data->>'prediction_horizon')::INTEGER,
            ARRAY(SELECT jsonb_array_elements_text(window_data->'features_used'))
        )
        ON CONFLICT (session_id, driver_number, window_index) DO NOTHING;
        
        inserted_count := inserted_count + 1;
        window_idx := window_idx + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to insert telemetry coordinates
CREATE OR REPLACE FUNCTION insert_telemetry_coordinates_batch(
    p_session_id VARCHAR(100),
    p_driver_number INTEGER,
    p_coordinates JSONB
) RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    coord_data JSONB;
    coord_idx INTEGER := 0;
BEGIN
    FOR coord_data IN SELECT * FROM jsonb_array_elements(p_coordinates)
    LOOP
        INSERT INTO telemetry_coordinates (
            session_id,
            driver_number,
            window_index,
            x,
            y,
            z
        ) VALUES (
            p_session_id,
            p_driver_number,
            coord_idx,
            (coord_data->>'X')::DECIMAL,
            (coord_data->>'Y')::DECIMAL,
            (coord_data->>'Z')::DECIMAL
        )
        ON CONFLICT (session_id, driver_number, window_index) DO NOTHING;
        
        inserted_count := inserted_count + 1;
        coord_idx := coord_idx + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get session summary
CREATE OR REPLACE FUNCTION get_session_summary(p_session_id VARCHAR(100))
RETURNS TABLE (
    session_id VARCHAR(100),
    year INTEGER,
    race_name VARCHAR(200),
    session_type VARCHAR(20),
    session_date TIMESTAMPTZ,
    driver_count BIGINT,
    window_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.session_id,
        s.year,
        s.race_name,
        s.session_type,
        s.session_date,
        COUNT(DISTINCT tsw.driver_number) as driver_count,
        COUNT(*) as window_count
    FROM sessions s
    LEFT JOIN time_series_windows tsw ON s.session_id = tsw.session_id
    WHERE s.session_id = p_session_id
    GROUP BY s.session_id, s.year, s.race_name, s.session_type, s.session_date;
END;
$$ LANGUAGE plpgsql;

-- Function to get predictions data for API
CREATE OR REPLACE FUNCTION get_predictions_data(p_session_id VARCHAR(100))
RETURNS TABLE (
    driver_number INTEGER,
    driver_abbreviation VARCHAR(3),
    window_data JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.driver_number,
        d.driver_abbreviation,
        jsonb_agg(
            jsonb_build_object(
                'window_index', tsw.window_index,
                'feature_matrix', tsw.feature_matrix,
                'y_true', tsw.y_true,
                'start_time', tsw.start_time,
                'end_time', tsw.end_time,
                'prediction_time', tsw.prediction_time
            ) ORDER BY tsw.window_index
        ) as window_data
    FROM time_series_windows tsw
    JOIN drivers d ON tsw.driver_number = d.driver_number
    WHERE tsw.session_id = p_session_id
    GROUP BY d.driver_number, d.driver_abbreviation;
END;
$$ LANGUAGE plpgsql;