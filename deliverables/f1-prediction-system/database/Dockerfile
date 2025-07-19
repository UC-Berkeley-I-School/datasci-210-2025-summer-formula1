FROM timescale/timescaledb:latest-pg16

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install build dependencies and Python
RUN apk add --no-cache \
    python3 \
    python3-dev \
    py3-pip \
    gcc \
    g++ \
    make \
    musl-dev \
    postgresql-dev \
    freetype-dev \
    libpng-dev \
    openblas-dev \
    jpeg-dev \
    zlib-dev \
    llvm15-dev \
    llvm15-static \
    # PostgreSQL client for psql commands in scripts
    postgresql-client

# Set LLVM_CONFIG to point to the correct version
ENV LLVM_CONFIG=/usr/lib/llvm15/bin/llvm-config

# Set working directory
WORKDIR /app/database

# Copy project files
COPY pyproject.toml uv.lock* .python-version README.md ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project

# Set up environment to use uv's virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy Python scripts
COPY scripts/load_data.py ./

# Copy initialization scripts
COPY init-scripts/01-schema.sql /docker-entrypoint-initdb.d/
COPY init-scripts/02-functions.sql /docker-entrypoint-initdb.d/
COPY init-scripts/99-load-data.sh /docker-entrypoint-initdb.d/

# Ensure shell script has Unix line endings and is executable
RUN dos2unix /docker-entrypoint-initdb.d/99-load-data.sh 2>/dev/null || true && \
chmod +x /docker-entrypoint-initdb.d/99-load-data.sh

# Copy configuration files
COPY config/data_config.py ./
COPY config/postgresql.conf /etc/postgresql/postgresql.conf

EXPOSE 5432

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

VOLUME ["/var/lib/postgresql/data"]

# Set environment variables
ENV POSTGRES_DB=f1_telemetry
ENV POSTGRES_USER=f1_user
ENV POSTGRES_PASSWORD=f1_password
ENV TIMESCALEDB_TELEMETRY=off
ENV PYTHONUNBUFFERED=1