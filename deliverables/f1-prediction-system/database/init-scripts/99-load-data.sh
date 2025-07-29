#!/bin/sh
# This script is executed by the official TimescaleDB entrypoint once the
# database initialization is complete. Its purpose is to load Formula 1
# telemetry data into the freshly created database. The original version
# waited a fixed five seconds before attempting to connect to the database
# over TCP. Unfortunately PostgreSQL may take longer than five seconds to
# start accepting connections and, more importantly, the Python ETL
# process always attempted to connect via TCP (host=localhost, port=5432)
# even when a Unix socket is available. This caused intermittent
# “connection refused” errors and the container would exit. To make the
# loading process robust, this script now actively waits for PostgreSQL to
# become ready using pg_isready, falls back to a Unix domain socket if
# available, and exports the required environment so both psql and
# load_data.py can connect reliably.

set -e

echo "Starting F1 data loading process..."

# When run inside the container the TimescaleDB server listens on a Unix
# socket under /var/run/postgresql. Use PGHOST to direct psql and
# pg_isready to this socket instead of TCP. If PGHOST is already set
# externally it will be respected.
export PGHOST=${PGHOST:-/var/run/postgresql}

# Export PGPASSWORD so pg_isready and psql can authenticate without
# prompting. Use the same credentials that were provided via
# POSTGRES_USER/POSTGRES_PASSWORD during container creation.
export PGPASSWORD="${POSTGRES_PASSWORD}"

# Wait until the server reports that it is ready to accept connections.
# Retry up to 30 times with a short delay. If the server does not
# become ready within this timeframe, exit with an error so Docker will
# report a failure.
max_attempts=30
attempt=1
until pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -q; do
    if [ $attempt -ge $max_attempts ]; then
        echo "Database did not become ready after ${max_attempts} attempts. Exiting."
        exit 1
    fi
    echo "Waiting for PostgreSQL to become ready... (attempt ${attempt}/${max_attempts})"
    attempt=$((attempt + 1))
    sleep 2
done

# Check if the core tables already contain data. This allows the
# container to be restarted without duplicating data. COUNT(*) on
# sessions will return zero if the ETL has not been run before.
DATA_EXISTS=$(psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -t -c "SELECT COUNT(*) FROM sessions;" 2>/dev/null || echo 0)

if [ "$DATA_EXISTS" -gt 0 ]; then
    echo "Data already exists in database. Skipping data load."
    exit 0
fi

echo "Loading F1 telemetry data..."

# Change into the database directory and run the Python ETL script from
# within its virtual environment. The .venv is created during the
# Dockerfile build by uv sync. Using the absolute path ensures the
# correct Python interpreter is invoked.
cd /app/database
if [ -x .venv/bin/python ]; then
    .venv/bin/python load_data.py
else
    # Fallback in case the virtual environment is not present (e.g. during local testing)
    python3 load_data.py
fi

# Capture the exit status of the ETL process and report success or failure.
status=$?
if [ $status -eq 0 ]; then
    echo "Data loading completed successfully!"
else
    echo "Error: Data loading failed (exit code $status)"
    exit $status
fi