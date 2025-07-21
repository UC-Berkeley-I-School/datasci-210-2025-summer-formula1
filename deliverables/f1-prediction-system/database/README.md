# F1 TimescaleDB Setup

This repository contains everything needed to bootstrap a TimescaleDB instance with Formula 1 telemetry data for machine learning predictions.

## Quick Start

1. **Configure data loading** by editing `config/data_config.py`:
   ```python
   SESSIONS = [
       SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
       SessionConfig(2024, "Monaco Grand Prix", "R"),
       # Add more sessions as needed
   ]
   ```

2. **Build the Docker image**:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

3. **Run the container**:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

The database will automatically be populated with the configured F1 data on first startup.

## Directory Structure

```
database/
├── Dockerfile               # Main Docker configuration
├── README.md               # This file
├── build.sh                # Build script
├── run.sh                  # Run script
├── pyproject.toml          # Python dependencies
├── uv.lock                 # Locked dependencies
├── init-scripts/           # Database initialization
│   ├── 01-schema.sql       # Database schema
│   ├── 02-functions.sql    # Stored procedures
│   └── 99-load-data.sh     # Data loading trigger
├── scripts/                # ETL scripts
│   └── load_data.py        # Main ETL script
├── config/                 # Configuration
│   ├── data_config.py      # User data configuration
│   └── postgresql.conf     # PostgreSQL settings
└── client-tools/           # Database interaction tools
    ├── db_client.py                # DAO implementation
    ├── example_queries.sql         # Sample SQL queries
    ├── validate_api_readiness.py   # API validation script
    └── verify_data_integrity.py    # Data validation script
```

## Database Schema

### Tables (01-schema.sql)

The database contains four main tables:

1. **sessions**: F1 session metadata
   - `session_id` (TEXT): Unique identifier in format "YYYY_Race Name_SessionType"
   - `year` (INTEGER): Race year
   - `race_name` (TEXT): Name of the Grand Prix
   - `session_type` (VARCHAR): Session type (R=Race, Q=Qualifying, FP1-3=Practice)
   - `session_date` (TIMESTAMPTZ): When the session occurred

2. **drivers**: Driver information
   - `driver_number` (INTEGER): Official driver number
   - `driver_abbreviation` (VARCHAR): 3-letter driver code (e.g., VER, HAM)

3. **time_series_windows**: Core telemetry data
   - `session_id`, `driver_number`, `window_index`: Composite primary key
   - `start_time`, `end_time`, `prediction_time`: Window timestamps
   - `y_true` (INTEGER): Ground truth label (track status)
   - `sequence_length` (INTEGER): Number of timesteps (typically 100)
   - `features_used` (TEXT[]): List of feature names
   - Optional prediction columns: `y_pred`, `y_proba`, `model_version`

4. **window_features**: Raw feature data
   - `feature_values` (FLOAT[]): Flattened array of shape (n_timesteps × n_features)
   - Stores the actual telemetry values for model predictions

5. **telemetry_coordinates**: Track position data
   - `x`, `y`, `z` (DECIMAL): 3D coordinates for visualization

### Functions (02-functions.sql)

1. **get_session_features(session_id)**: Retrieves all features for a session
   - Returns driver info, window data, and feature arrays
   - Includes metadata for reshaping (n_timesteps, n_features)

2. **store_predictions(...)**: Stores model predictions
   - Updates y_pred, y_proba, and model_version
   - Timestamps predictions automatically

3. **get_predictions_data(session_id)**: Returns aggregated prediction data
   - Groups windows by driver
   - Returns JSONB format for easy API consumption

## API Integration

The database is designed to support three API endpoints:

1. **GET /api/v1/sessions** - Returns available sessions
2. **POST /api/v1/predict** - Returns predictions for a session
3. **GET /api/v1/telemetry** - Returns track coordinates

## Client Tools

The `client-tools/` directory contains utilities for interacting with the database:

- **db_client.py**: Complete DAO (Data Access Object) implementation for REST API integration
- **example_queries.sql**: Collection of useful SQL queries for data exploration
- **validate_api_readiness.py**: Script to verify data is correctly formatted for API consumption
- **verify_data_integrity.py**: Script to validate data storage and check for issues

## Configuration Options

Edit `config/data_config.py` to control:

- Which F1 sessions to load
- Window size and prediction horizon
- Feature normalization
- Database connection parameters

## Monitoring

View container logs:
```bash
docker logs -f f1-timescaledb
```

Connect to the database:
```bash
docker exec -it f1-timescaledb psql -U f1_user -d f1_telemetry
```

## Smoke Test SQL Queries

### Basic Health Checks

```sql
-- Check loaded sessions
SELECT session_id, year, race_name, session_type 
FROM sessions 
ORDER BY session_date DESC;

-- View session statistics
SELECT * FROM session_stats;

-- Check driver list
SELECT driver_number, driver_abbreviation 
FROM drivers 
ORDER BY driver_number;
```

### Data Validation

```sql
-- Check window counts per session
SELECT 
    session_id, 
    COUNT(DISTINCT driver_number) as drivers,
    COUNT(*) as total_windows
FROM time_series_windows 
GROUP BY session_id;

-- Verify features are loaded
SELECT 
    COUNT(*) as windows_with_features,
    MIN(array_length(feature_values, 1)) as min_values,
    MAX(array_length(feature_values, 1)) as max_values
FROM window_features;

-- Sample feature data for one window
SELECT 
    w.session_id,
    w.driver_number,
    w.window_index,
    w.features_used,
    f.feature_values[1:9] as first_timestep
FROM time_series_windows w
JOIN window_features f USING (session_id, driver_number, window_index)
LIMIT 1;
```

### API-Ready Queries

```sql
-- Get available sessions (for GET /api/v1/sessions)
SELECT 
    s.session_id,
    s.year,
    s.race_name,
    COUNT(DISTINCT w.driver_number) as driver_count,
    COUNT(w.*) as window_count
FROM sessions s
LEFT JOIN time_series_windows w ON s.session_id = w.session_id
GROUP BY s.session_id, s.year, s.race_name, s.session_date
ORDER BY s.session_date DESC;

-- Get features for prediction (sample)
SELECT * FROM get_session_features('2024_Monaco Grand Prix_R') 
LIMIT 5;

-- Check label distribution
SELECT 
    y_true as track_status,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
FROM time_series_windows
GROUP BY y_true
ORDER BY y_true;
```

### Performance Checks

```sql
-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

Driver mappings are automatically built by loading each configured session. The ETL process extracts all unique drivers from the telemetry data, ensuring complete coverage across all races.
