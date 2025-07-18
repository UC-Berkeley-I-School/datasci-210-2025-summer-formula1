# Stage 1: Build wheels
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y \
    curl \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME='/usr/local' \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry --version

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies directly with poetry
RUN poetry install --only main --no-interaction --no-ansi

# Stage 2: Final application image
FROM python:3.11-slim

WORKDIR /app

# Copy the entire python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application files
COPY src ./src
COPY model_pipeline.pkl .

ENV REDIS_URL="redis://localhost:6379"
ENV LOG_LEVEL="INFO"

# Set the entrypoint
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]