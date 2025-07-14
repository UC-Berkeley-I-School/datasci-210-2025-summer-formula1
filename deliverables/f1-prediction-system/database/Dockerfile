FROM timescale/timescaledb:latest-pg16

# Install Python for initialization scripts
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
COPY database/01-schema.sql /docker-entrypoint-initdb.d/
COPY database/02-functions.sql /docker-entrypoint-initdb.d/
COPY database/03-seed.sql /docker-entrypoint-initdb.d/
COPY database/postgresql.conf /etc/postgresql/postgresql.conf

EXPOSE 5432

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

VOLUME ["/var/lib/postgresql/data"]

# Set environment variables
ENV POSTGRES_DB=f1_telemetry
ENV POSTGRES_USER=f1_user
ENV POSTGRES_PASSWORD=f1_password
ENV TIMESCALEDB_TELEMETRY=off
