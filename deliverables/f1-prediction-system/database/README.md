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
.
├── Dockerfile                 # Main Docker configuration
├── build.sh                  # Build script
├── run.sh                    # Run script
├── init-scripts/             # Database initialization
│   ├── 01-schema.sql        # Database schema
│   ├── 02-functions.sql     # Stored procedures
│   └── 99-load-data.sh      # Data loading trigger
├── scripts/                  # ETL scripts
│   ├── load_data.py         # Main ETL script
│   └── requirements.txt     # Python dependencies
└── config/                   # Configuration
    ├── data_config.py       # User data configuration
    └── postgresql.conf      # PostgreSQL settings
```

## Database Schema

The database contains four main tables:

- **sessions**: F1 session metadata (race, year, type)
- **drivers**: Driver information (number, abbreviation) - automatically extracted from sessions
- **time_series_windows**: Feature matrices and labels for each time window
- **telemetry_coordinates**: X,Y,Z coordinates for track visualization

Driver mappings are automatically built by loading each configured session and using the `DriverLabelEncoder` from `f1_etl`. This ensures all drivers across all races are properly captured.

## API Integration

The database is designed to support three API endpoints:

1. **GET /api/v1/sessions** - Returns available sessions
2. **POST /api/v1/predict** - Returns predictions for a session
3. **GET /api/v1/telemetry** - Returns track coordinates

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

Check data statistics:
```sql
SELECT * FROM session_stats;
```

## Troubleshooting

- **Data loading fails**: Check that `f1-etl` is properly installed
- **Connection refused**: Ensure port 5432 is not already in use
- **Out of memory**: Increase Docker memory limits or load fewer sessions

## Data Retention

By default:
- Data is retained for 12 months
- Compression is applied after 7 days
- Continuous aggregates update every 5 minutes