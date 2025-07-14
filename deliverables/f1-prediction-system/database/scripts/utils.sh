#!/bin/bash
# F1 TimescaleDB System Utilities

case "$1" in
    "status")
        echo "[F1] F1 TimescaleDB System Status"
        echo "================================"
        cd docker && docker-compose ps
        ;;
    "start")
        echo "[START] Starting system..."
        cd docker && docker-compose up -d
        ;;
    "stop")
        echo "[STOP] Stopping system..."
        cd docker && docker-compose down
        ;;
    "logs")
        cd docker && docker-compose logs -f ${2:-}
        ;;
    "test")
        echo "[TEST] Running tests..."
        source venv/bin/activate
        python python/etl_integration.py
        ;;
    *)
        echo "F1 TimescaleDB System Utilities"
        echo "Usage: $0 {status|start|stop|logs|test}"
        ;;
esac
