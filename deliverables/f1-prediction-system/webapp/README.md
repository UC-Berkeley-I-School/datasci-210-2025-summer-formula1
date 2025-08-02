# F1 Prediction System - Web Application

A Flask-based web application that visualizes Formula 1 race telemetry and safety car predictions in real-time.

## Screenshots

<div align="center">
  <img src="docs/screenshot-main.png" alt="Main Dashboard" width="800">
  <p><em>Main dashboard showing real-time race visualization</em></p>
</div>

<div align="center">
  <img src="docs/screenshot-telemetry.png" alt="Telemetry View" width="800">
  <p><em>Detailed telemetry view with track positions and predictions</em></p>
</div>

## Features

- **Real-time Race Visualization**: Interactive D3.js-based visualization of F1 race data
- **Safety Car Predictions**: Display probabilistic predictions for safety car deployments
- **Session Selection**: Browse and select from available F1 race sessions
- **Responsive Design**: Works on desktop and mobile devices
- **API Integration**: Connects to the F1 Model Service REST API for telemetry data

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- [just](https://github.com/casey/just) - Command runner
- Docker and Docker Compose (for containerized deployment)

## Installation

### Installing Just

For official instructions, see [Just Installation](https://github.com/casey/just?tab=readme-ov-file#installation).

### Installing uv

For official instructions, see [uv Installation](https://docs.astral.sh/uv/getting-started/installation/).

## Project Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:UC-Berkeley-I-School/datasci-210-2025-summer-formula1.git
   cd datasci-210-2025-summer-formula1/deliverables/f1-prediction-system/webapp
   ```

2. **Initialize the development environment:**
   ```bash
   just init
   ```
   
   This command will:
   - Install all Python dependencies (including dev dependencies)
   - Create a `.env` file from the template
   - Set up necessary directories

3. **Configure environment variables:**
   
   Edit the `.env` file to match your setup:
   ```bash
   # For local development with model service running on host
   API_BASE_URL=http://localhost:8000
   
   # For Docker Compose deployment
   API_BASE_URL=http://f1_model_service:8000
   
   # For remote API
   API_BASE_URL=http://your-api-host:8000
   ```

## Development

### Running the Development Server

**Basic development server:**
```bash
just dev
```

**Development server with auto-reload (watches for file changes):**
```bash
just dev-watch
```

The application will be available at `http://localhost:5001`

### Available Commands

View all available commands:
```bash
just
```

Common commands:
- `just dev` - Run Flask development server
- `just dev-watch` - Run with auto-reload on file changes
- `just test` - Run tests
- `just lint` - Run linting
- `just format` - Format code with black
- `just examine-npy` - Examine the structure of .npy data files

## Docker Deployment

### Building the Docker Image

From the deliverables directory:
```bash
cd ../..  # Navigate to {project-root}/deliverables
docker compose build webapp
```

### Running with Docker Compose

```bash
docker compose up webapp -d
```

This will:
- Build the webapp container
- Start the webapp along with its dependencies (model_service, database)
- Expose the webapp on port 5000

### Using Just commands for Docker:

```bash
# Build Docker image
just docker-build

# Run Docker container
just docker-run

# Run with docker-compose
just compose-up

# Stop docker-compose services
just compose-down
```

## API Integration

The webapp connects to the F1 Model Service REST API to fetch:
- Available race sessions from `/api/v1/sessions`
- Telemetry data from `/api/v1/telemetry?session_id=<id>`

The API base URL is configured via the `API_BASE_URL` environment variable.

## Troubleshooting

### Connection Issues

If you see "Could not load races: HTTP error! status: 404":
1. Check that the model service is running
2. Verify the `API_BASE_URL` in your `.env` file
3. Check the logs for the exact URL being called

### Missing Dependencies

If commands fail with "command not found":
1. Ensure you ran `just init` or `just install-dev`
2. Make sure you're using `uv run` prefix for Python commands
3. Verify that `uv` and `just` are installed

### Port Conflicts

If port 5001 is already in use:
1. Change the port in the Justfile `dev` command
2. Or stop the conflicting service

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `just check`
5. Submit a pull request