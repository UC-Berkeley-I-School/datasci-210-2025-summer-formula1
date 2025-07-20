-- Function to get features for prediction
CREATE OR REPLACE FUNCTION get_session_features(p_session_id TEXT)
RETURNS TABLE (
    driver_number INTEGER,
    driver_abbreviation VARCHAR(3),
    window_index INTEGER,
    feature_matrix FLOAT[][],
    y_true INTEGER,
    start_time TIMESTAMPTZ,
    prediction_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        w.driver_number,
        d.driver_abbreviation,
        w.window_index,
        f.feature_matrix,
        w.y_true,
        w.start_time,
        w.prediction_time
    FROM time_series_windows w
    JOIN window_features f ON 
        w.session_id = f.session_id AND 
        w.driver_number = f.driver_number AND 
        w.window_index = f.window_index
    JOIN drivers d ON w.driver_number = d.driver_number
    WHERE w.session_id = p_session_id
    ORDER BY w.driver_number, w.window_index;
END;
$$ LANGUAGE plpgsql;

-- Function to store predictions
CREATE OR REPLACE FUNCTION store_predictions(
    p_session_id TEXT,
    p_driver_number INTEGER,
    p_window_index INTEGER,
    p_y_pred INTEGER,
    p_y_proba FLOAT[],
    p_model_version TEXT
) RETURNS VOID AS $$
BEGIN
    UPDATE time_series_windows
    SET 
        y_pred = p_y_pred,
        y_proba = p_y_proba,
        model_version = p_model_version,
        prediction_timestamp = NOW()
    WHERE 
        session_id = p_session_id AND
        driver_number = p_driver_number AND
        window_index = p_window_index;
END;
$$ LANGUAGE plpgsql;

-- Function to get predictions data for API (updated to handle features correctly)
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
                'y_true', tsw.y_true,
                'y_pred', tsw.y_pred,
                'y_proba', tsw.y_proba,
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