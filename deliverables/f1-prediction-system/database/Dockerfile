FROM timescale/timescaledb:latest-pg16

# Install Python for initialization scripts
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
COPY ./init-scripts /docker-entrypoint-initdb.d/

# Set environment variables
ENV POSTGRES_DB=f1_telemetry
ENV POSTGRES_USER=f1_user
ENV POSTGRES_PASSWORD=f1_password