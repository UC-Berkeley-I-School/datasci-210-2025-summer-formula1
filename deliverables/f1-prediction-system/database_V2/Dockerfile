#FROM timescale/timescaledb:latest-pg16

# Copy initialization scripts
#COPY init-scripts/01-schema.sql /docker-entrypoint-initdb.d/
#COPY init-scripts/02-functions.sql /docker-entrypoint-initdb.d/
#COPY init-scripts/03-seed.sql /docker-entrypoint-initdb.d/
#COPY init-scripts/postgresql.conf /etc/postgresql/postgresql.conf
FROM timescale/timescaledb:latest-pg16

# Copy initialization scripts
COPY init-scripts/01-schema.sql /docker-entrypoint-initdb.d/
COPY init-scripts/02-functions.sql /docker-entrypoint-initdb.d/
COPY init-scripts/03-seed.sql /docker-entrypoint-initdb.d/
COPY init-scripts/postgresql.conf /etc/postgresql/postgresql.conf

#EXPOSE 5432

#HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1
#
#VOLUME ["/var/lib/postgresql/data"]

# Set environment variables
#ENV POSTGRES_DB=f1_telemetry
#ENV POSTGRES_USER=f1_user
#ENV POSTGRES_PASSWORD=f1_password
#ENV TIMESCALEDB_TELEMETRY=off



# Dockerfile for ETL Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ETL pipeline code
COPY sean/artifacts/f1_etl/ ./etl/
COPY deliverables/f1-prediction-system/database/ ./database/

# Copy our scripts
COPY etl_cache_pipeline.py .
COPY etl_scheduler.py .

# Create data directory
RUN mkdir -p /app/database/data

# Set environment variables
ENV PYTHONPATH=/app/etl:/app/database/python
ENV DB_PATH=/app/database/data
ENV ETL_SCHEDULE=daily

# Create volumes for persistence
VOLUME ["/app/database/data", "/app/logs"]

# Default command
CMD ["python", "etl_scheduler.py"]