#!/bin/bash
set -e

echo "[F1] Setting up F1 TimescaleDB System..."

# Check requirements
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3."
    exit 1
fi

# Virtual environment
if [ ! -d "venv" ]; then
    echo "[PACKAGE] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Dependencies
echo "[PACKAGE] Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Start Docker services
echo "[DOCKER] Starting Docker services..."
cd docker
docker-compose up -d

# Wait for database
echo "[WAIT] Waiting for database to be ready..."
cd ..
./scripts/wait-for-db.sh

echo "[OK] Setup complete!"
echo ""
echo "[INFO] Access Points:"
echo "   Database: postgresql://racing_user:racing_password@localhost:5432/racing_telemetry"
echo "   pgAdmin: http://localhost:8080 (admin@racing.com / admin123)"
echo "   Grafana: http://localhost:3000 (admin / admin123)"
echo ""
echo "[START] Next steps:"
echo "   python python/etl_integration.py  # Test ETL integration"
echo "   ./scripts/utils.sh status         # Check system status"
