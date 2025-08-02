FROM python:3.12-slim

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

# Create a cache directory for fastf1
RUN mkdir -p /app/cache

# Set environment variables
ENV FLASK_APP=f1dev.py
ENV PYTHONPATH=/app
ENV FASTF1_CACHE=/app/cache

# Expose the port the app runs on
EXPOSE 5000

# Run the application with gunicorn for production
# Increase timeout to handle slow startup and set preload to share memory between workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--preload", "f1dev:app"]