#!/bin/bash
set -e

echo "===================================================="
echo "           F1 Production Data Downloader"
echo "===================================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker"
        exit 1
    fi
    
    if ! docker ps &> /dev/null; then
        log_error "Docker is not running. Please start Docker"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

setup_python_env() {
    log_info "Setting up Python environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "Created virtual environment"
    else
        log_info "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    
    log_info "Installing Python packages..."
    pip install fastf1 asyncpg pandas numpy aiofiles python-dotenv
    
    log_info "Python environment setup complete"
}

setup_database() {
    log_info "Setting up TimescaleDB..."
    
    if docker ps | grep -q "racing_timescaledb"; then
        log_info "TimescaleDB container already running"
    else
        log_info "Starting TimescaleDB container..."
        
        if [ -f "docker-compose.yml" ]; then
            docker compose up -d timescaledb
        elif [ -f "docker/docker-compose.yml" ]; then
            cd docker && docker compose up -d timescaledb && cd ..
        else
            log_error "docker-compose.yml not found. Please ensure TimescaleDB is running"
            exit 1
        fi
    fi
    
    log_info "Waiting for database to be ready..."
    sleep 10
    
    if docker exec racing_timescaledb psql -U racing_user -d racing_telemetry -c "SELECT 1;" &> /dev/null; then
        log_info "Database connection successful"
    else
        log_error "Database connection failed"
        exit 1
    fi
}

create_config() {
    log_info "Creating configuration files..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=racing_telemetry
DB_USER=racing_user
DB_PASSWORD=racing_password
F1_CACHE_DIR=./cache
LOG_LEVEL=INFO
BATCH_SIZE=1000
MAX_RETRIES=3
RETRY_DELAY=5.0
MAX_CONCURRENT_DRIVERS=3
PROGRESS_FILE=f1_download_progress.json
EOF
        log_info "Created .env configuration file"
    else
        log_info "Configuration file already exists"
    fi
    
    mkdir -p cache
    mkdir -p logs
}

test_setup() {
    log_info "Testing setup..."
    
    source venv/bin/activate
    
    log_info "Testing database connection..."
    python3 -c "
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect('postgresql://racing_user:racing_password@localhost:5432/racing_telemetry')
        version = await conn.fetchval('SELECT version()')
        print(f'[OK] Database connection successful: {version[:50]}...')
        await conn.close()
        return True
    except Exception as e:
        print(f'[ERROR] Database connection failed: {e}')
        return False

success = asyncio.run(test_db())
exit(0 if success else 1)
"
    
    if [ $? -ne 0 ]; then
        log_error "Database connection test failed"
        exit 1
    fi
    
    log_info "Testing FastF1 import..."
    python3 -c "
import fastf1
import pandas as pd
print('[OK] FastF1 import successful')
print(f'FastF1 version: {fastf1.__version__}')
"
    
    if [ $? -ne 0 ]; then
        log_error "FastF1 test failed"
        exit 1
    fi
    
    log_info "All tests passed!"
}

show_usage() {
    log_info "Setup complete! Here's how to use the F1 downloader:"
    echo ""
    echo "1. Start the download (this will take several hours/days):"
    echo "   source venv/bin/activate"
    echo "   python production_f1_downloader.py"
    echo ""
    echo "2. Monitor progress:"
    echo "   python f1_monitor.py progress"
    echo "   python f1_monitor.py stats"
    echo ""
    echo "3. Run in background:"
    echo "   nohup python production_f1_downloader.py > logs/download.log 2>&1 &"
    echo ""
    log_info "The download will fetch ALL F1 data from 2022-2024 (~500+ sessions)"
    log_warn "This will download several GB of data and take many hours to complete"
}

main() {
    case "${1:-setup}" in
        "setup")
            check_prerequisites
            setup_python_env
            setup_database
            create_config
            test_setup
            show_usage
            ;;
        "test")
            source venv/bin/activate
            test_setup
            ;;
        "clean")
            log_info "Cleaning up..."
            rm -rf venv cache f1_download_progress.json f1_downloader.log .env
            log_info "Cleanup complete"
            ;;
        *)
            echo "Usage: $0 [setup|test|clean]"
            ;;
    esac
}

main "$@"