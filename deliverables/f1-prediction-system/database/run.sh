#!/bin/bash
# Run script for F1 TimescaleDB Docker container

set -e

echo "=== F1 TimescaleDB Docker Run Script ==="
echo

# Configuration
CONTAINER_NAME="f1-timescaledb"
IMAGE_NAME="f1-timescaledb:latest"
HOST_PORT=5432
DATA_VOLUME="f1-timescale-data"

# Check if image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo "Error: Docker image '$IMAGE_NAME' not found!"
    echo "Please run ./build.sh first."
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    read -p "Do you want to remove it and start fresh? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping and removing existing container..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME
        
        read -p "Do you also want to remove the data volume? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm $DATA_VOLUME 2>/dev/null || true
        fi
    else
        echo "Starting existing container..."
        docker start $CONTAINER_NAME
        echo "Container started. Use 'docker logs -f $CONTAINER_NAME' to view logs."
        exit 0
    fi
fi

# Run container
echo "Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $HOST_PORT:5432 \
    -v $DATA_VOLUME:/var/lib/postgresql/data \
    -e POSTGRES_DB=f1_telemetry \
    -e POSTGRES_USER=f1_user \
    -e POSTGRES_PASSWORD=f1_password \
    $IMAGE_NAME

echo
echo "=== Container started successfully! ==="
echo
echo "Container name: $CONTAINER_NAME"
echo "Database: f1_telemetry"
echo "User: f1_user"
echo "Password: f1_password"
echo "Port: $HOST_PORT"
echo
echo "Connection string:"
echo "  postgresql://f1_user:f1_password@localhost:$HOST_PORT/f1_telemetry"
echo
echo "To view logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo
echo "To connect with psql:"
echo "  docker exec -it $CONTAINER_NAME psql -U f1_user -d f1_telemetry"
echo
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo
echo "Note: Initial data loading may take several minutes depending on the"
echo "      number of sessions configured in config/data_config.py"