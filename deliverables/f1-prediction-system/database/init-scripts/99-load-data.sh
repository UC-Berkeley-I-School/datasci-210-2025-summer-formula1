#!/bin/sh
# This script runs after the database is initialized to load F1 data

echo "Starting F1 data loading process..."

# During initialization, use Unix socket connection
export PGHOST=/var/run/postgresql

# Wait for PostgreSQL to be fully ready
sleep 5

# Check if data already exists
DATA_EXISTS=$(psql -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM sessions;")

if [ "$DATA_EXISTS" -gt 0 ]; then
    echo "Data already exists in database. Skipping data load."
    exit 0
fi

# Run the Python ETL script
echo "Loading F1 telemetry data..."
cd /app/database && .venv/bin/python load_data.py

if [ $? -eq 0 ]; then
    echo "Data loading completed successfully!"
else
    echo "Error: Data loading failed!"
    exit 1
fi