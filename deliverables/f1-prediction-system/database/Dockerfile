FROM timescale/timescaledb:latest-pg16

# Copy initialization scripts
COPY init-scripts/01-schema.sql /docker-entrypoint-initdb.d/
COPY init-scripts/02-functions.sql /docker-entrypoint-initdb.d/
COPY init-scripts/03-seed.sql /docker-entrypoint-initdb.d/
COPY init-scripts/postgresql.conf /etc/postgresql/postgresql.conf

EXPOSE 5432

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

VOLUME ["/var/lib/postgresql/data"]

# Set environment variables
ENV POSTGRES_DB=f1_telemetry
ENV POSTGRES_USER=f1_user
ENV POSTGRES_PASSWORD=f1_password
ENV TIMESCALEDB_TELEMETRY=off
