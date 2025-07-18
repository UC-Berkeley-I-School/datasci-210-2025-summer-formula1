-- TimescaleDB Schema
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Sessions table for race sessions
CREATE TABLE sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    race_name VARCHAR(200) NOT NULL,
    session_type VARCHAR(20) NOT NULL,
    track_name VARCHAR(200),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Drivers table
CREATE TABLE drivers (
    driver_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    team VARCHAR(100),
    car_number INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Telemetry data table
CREATE TABLE telemetry (
    time TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    driver_id VARCHAR(20) NOT NULL,
    
    -- FastF1 Car Data fields
    speed DECIMAL(8,3),       
    rpm DECIMAL(8,1),         
    ngear INTEGER,            
    throttle DECIMAL(6,3),    
    brake BOOLEAN,             
    drs INTEGER,             
    
    -- FastF1 Position Data fields  
    x DECIMAL(12,3),       
    y DECIMAL(12,3),      
    z DECIMAL(12,3),          
    status VARCHAR(20),       
    
    -- FastF1 Time Data
    session_time INTERVAL,    
    source VARCHAR(20),       
    
    -- Speed Traps
    speed_i1 DECIMAL(6,2),
    speed_i2 DECIMAL(6,2),
    speed_fl DECIMAL(6,2),
    speed_st DECIMAL(6,2),
    
    -- Additional computed fields
    distance DECIMAL(10,3),
    differential_distance DECIMAL(8,3),
    relative_distance DECIMAL(6,5),
    driver_ahead VARCHAR(20),
    distance_to_driver_ahead DECIMAL(8,2),
    
    -- Extended telemetry
    tire_temp_fl DECIMAL(5,2),
    tire_temp_fr DECIMAL(5,2),
    tire_temp_rl DECIMAL(5,2),
    tire_temp_rr DECIMAL(5,2),
    fuel_remaining DECIMAL(6,2),
    lap_number INTEGER,
    sector INTEGER,
    track_status VARCHAR(10),
    
    -- Raw telemetry for flexibility
    raw_data JSONB,
    
    -- Constraints
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

-- Convert to hypertable
SELECT create_hypertable('telemetry', 'time', chunk_time_interval => INTERVAL '5 minutes');

-- Predictions table
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    driver_id VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    predicted_track_status VARCHAR(20),
    confidence DECIMAL(4,3),
    prediction_start_time TIMESTAMPTZ,
    prediction_end_time TIMESTAMPTZ,
    feature_importance JSONB,
    model_metadata JSONB,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

SELECT create_hypertable('predictions', 'prediction_time', chunk_time_interval => INTERVAL '1 day');

-- Weather data table
CREATE TABLE weather_data (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    time_recorded TIMESTAMPTZ NOT NULL,
    session_time INTERVAL,
    air_temp DECIMAL(4,1),
    humidity DECIMAL(4,1),
    pressure DECIMAL(6,1),
    rainfall BOOLEAN,
    track_temp DECIMAL(4,1),
    wind_direction INTEGER,
    wind_speed DECIMAL(4,1),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

SELECT create_hypertable('weather_data', 'time_recorded', chunk_time_interval => INTERVAL '1 day');

-- Indexes
CREATE INDEX idx_telemetry_session_driver ON telemetry (session_id, driver_id, time DESC);
CREATE INDEX idx_telemetry_time ON telemetry (time DESC);
CREATE INDEX idx_predictions_session_driver ON predictions (session_id, driver_id, prediction_time DESC);

-- Continuous aggregates
CREATE MATERIALIZED VIEW telemetry_1min AS
SELECT 
    time_bucket('1 minute', time) AS bucket,
    session_id,
    driver_id,
    AVG(speed) as avg_speed,
    MAX(speed) as max_speed,
    AVG(rpm) as avg_rpm,
    AVG(throttle) as avg_throttle,
    COUNT(*) as data_points
FROM telemetry
GROUP BY bucket, session_id, driver_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('telemetry_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute', 
    schedule_interval => INTERVAL '1 minute');

-- Data retention and compression
SELECT add_retention_policy('telemetry', INTERVAL '6 months');
SELECT add_compression_policy('telemetry', INTERVAL '7 days');
