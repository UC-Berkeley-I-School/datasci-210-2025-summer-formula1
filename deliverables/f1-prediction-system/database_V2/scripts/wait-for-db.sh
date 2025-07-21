#!/bin/bash
# Wait for database to be ready

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-racing_user}
DB_NAME=${DB_NAME:-racing_telemetry}

echo "Waiting for database to be ready..."

for i in {1..30}; do
    if PGPASSWORD=racing_password psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;" >/dev/null 2>&1; then
        echo "[OK] Database is ready!"
        exit 0
    fi
    echo "Attempt $i/30 failed, waiting 5 seconds..."
    sleep 5
done

echo "[ERROR] Database failed to become ready"
exit 1
