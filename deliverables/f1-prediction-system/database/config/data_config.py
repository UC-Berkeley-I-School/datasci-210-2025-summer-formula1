"""
User configuration for F1 data loading.
Modify this file to control which sessions and drivers are loaded into the database.
"""

from f1_etl import DataConfig, SessionConfig

# Define which sessions to load
# Format: SessionConfig(year, race_name, session_type)
# Session types: FP1, FP2, FP3, Q (Qualifying), R (Race)
SESSIONS = [
    SessionConfig(2024, "Saudi Arabian Grand Prix", "R"),
    SessionConfig(2024, "Qatar Grand Prix", "R"),
    SessionConfig(2024, "Chinese Grand Prix", "R"),
    SessionConfig(2024, "Mexico City Grand Prix", "R"),
    SessionConfig(2024, "São Paulo Grand Prix", "R"),
    SessionConfig(2024, "Miami Grand Prix", "R"),
    SessionConfig(2024, "United States Grand Prix", "R"),
    SessionConfig(2024, "Monaco Grand Prix", "R"),
]

# Optionally filter by specific drivers (None = all drivers)
# Example: DRIVERS = ["VER", "HAM", "LEC"]
DRIVERS = None

# ETL configuration
CONFIG = DataConfig(
    sessions=SESSIONS,
    drivers=DRIVERS,
    include_weather=False,  # Weather data not used by models
)

# Model configuration
WINDOW_SIZE = 100  # Number of time steps in each window
PREDICTION_HORIZON = 10  # How far ahead to predict (in time steps)
NORMALIZE = True  # Whether to normalize features

# Database connection (uses environment variables by default)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "racing_telemetry",
    "user": "f1_user",
    "password": "f1_password"
}