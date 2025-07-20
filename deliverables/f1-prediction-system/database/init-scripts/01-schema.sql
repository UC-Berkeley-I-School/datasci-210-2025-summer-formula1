-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Drivers table
CREATE TABLE drivers (
    driver_number INTEGER PRIMARY KEY,
    driver_abbreviation VARCHAR(3) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    year INTEGER NOT NULL,
    race_name TEXT NOT NULL,
    session_type VARCHAR(20) NOT NULL,
    session_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Simplified time_series_windows table
CREATE TABLE time_series_windows (
    session_id TEXT NOT NULL,
    driver_number INTEGER NOT NULL,
    window_index INTEGER NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    
    -- Just store the ground truth
    y_true INTEGER NOT NULL,
    
    -- Store window parameters for validation
    sequence_length INTEGER NOT NULL,
    prediction_horizon INTEGER NOT NULL,
    features_used TEXT[] NOT NULL,
    
    -- Optional: Store pre-computed predictions
    y_pred INTEGER,
    y_proba FLOAT[],
    model_version TEXT,
    
    PRIMARY KEY (session_id, driver_number, window_index)
);

-- Since coordinates are needed per window, store them with the windows:
ALTER TABLE time_series_windows ADD COLUMN x FLOAT, ADD COLUMN y FLOAT, ADD COLUMN z FLOAT;


-- Store raw features separately if needed for on-the-fly predictions
CREATE TABLE window_features (
    session_id TEXT NOT NULL,
    driver_number INTEGER NOT NULL,
    window_index INTEGER NOT NULL,
    feature_values FLOAT[], -- Store as array instead of JSONB
    
    PRIMARY KEY (session_id, driver_number, window_index)
);

-- Telemetry coordinates table for track visualization
CREATE TABLE telemetry_coordinates (
    session_id TEXT NOT NULL,
    driver_number INTEGER NOT NULL,
    window_index INTEGER NOT NULL,
    x DECIMAL(12,3),
    y DECIMAL(12,3),
    z DECIMAL(12,3),
    
    -- Constraints
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (driver_number) REFERENCES drivers(driver_number),
    PRIMARY KEY (session_id, driver_number, window_index)
);

-- Indexes for API query patterns
CREATE INDEX idx_time_series_session ON time_series_windows (session_id);
CREATE INDEX idx_time_series_driver ON time_series_windows (driver_number);
CREATE INDEX idx_time_series_start_time ON time_series_windows (start_time);
CREATE INDEX idx_telemetry_coords_session ON telemetry_coordinates (session_id);

-- Simple view for session statistics (not continuous aggregate)
CREATE VIEW session_stats AS
SELECT 
    s.session_id,
    s.year,
    s.race_name,
    s.session_type,
    s.session_date,
    COUNT(DISTINCT tsw.driver_number) as driver_count,
    COUNT(DISTINCT tsw.window_index) as window_count
FROM sessions s
LEFT JOIN time_series_windows tsw ON s.session_id = tsw.session_id
GROUP BY s.session_id, s.year, s.race_name, s.session_type, s.session_date;