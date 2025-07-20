#!/bin/bash
# Build script for F1 TimescaleDB Docker image

set -e

echo "=== F1 TimescaleDB Docker Build Script ==="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p init-scripts scripts config

# Check if required files exist
echo "Checking required files..."
required_files=(
    "Dockerfile"
    "init-scripts/01-schema.sql"
    "init-scripts/02-functions.sql"
    "init-scripts/99-load-data.sh"
    "scripts/load_data.py"
    "config/data_config.py"
    "config/postgresql.conf"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file '$file' not found!"
        echo "Please ensure all files are in place before building."
        exit 1
    fi
done

# Make shell scripts executable
chmod +x init-scripts/99-load-data.sh

# Build Docker image
echo "Building Docker image..."
docker build -t f1-timescaledb:latest .

if [ $? -eq 0 ]; then
    echo
    echo "=== Build completed successfully! ==="
    echo
    echo "To run the container, use:"
    echo "  ./run.sh"
    echo
    echo "To customize data loading, edit:"
    echo "  config/data_config.py"
else
    echo
    echo "=== Build failed! ==="
    exit 1
fi