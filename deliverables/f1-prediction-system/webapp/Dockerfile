# Build arguments (can be overridden during build)
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/webapp /app

# Set default environment variables for runtime
# These can be overridden by docker-compose or docker run -e
ENV FLASK_APP=f1dev.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV PYTHONPATH=/app

# Default API configuration - should be overridden at runtime
ENV API_BASE_URL=http://f1_model_service:8000
ENV MONTE_CARLO_BASE_URL=http://f1_monte_carlo_sim:8001

# Server configuration
ENV HOST=0.0.0.0
ENV PORT=5000

# Logging configuration
# Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
ENV LOG_LEVEL=INFO
ENV GUNICORN_LOG_LEVEL=INFO

# Cache configuration
ENV CACHE_TYPE=simple
ENV CACHE_DEFAULT_TIMEOUT=300

# Expose the port the app runs on
EXPOSE 5000

# Run the application with gunicorn for production
# Using shell form to allow environment variable expansion
# Removed --preload to avoid timeout issues during initialization
CMD gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 --log-level ${GUNICORN_LOG_LEVEL} f1dev:app