FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app

# Environment variables
ENV DATABASE_TYPE=remote
ENV EXTERNAL_API_BASE=https://f1capstone.com
ENV DB_HOST=localhost
ENV DB_PORT=5432
ENV DB_NAME=racing_telemetry
ENV DB_USER=racing_user
ENV DB_PASSWORD=racing_password

COPY integrated_dashboard.py /app/integrated_dashboard.py

CMD uv run integrated_dashboard.py