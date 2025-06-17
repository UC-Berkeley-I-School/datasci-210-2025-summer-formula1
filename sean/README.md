# Formula 1 Time Series Models

<div align="center">
  <img src="https://img.shields.io/badge/status-under%20development-orange?style=for-the-badge" alt="Under Development">
  <!-- <img src="https://img.shields.io/badge/python-3.13+-blue?style=for-the-badge&logo=python" alt="Python 3.13+"> -->
  <!-- <img src="https://img.shields.io/badge/machine%20learning-temporal-green?style=for-the-badge" alt="Temporal ML"> -->
</div>

<br>

A machine learning project focused on Formula 1 classification tasks using temporal sliding windows and telemetry data.

## Quick Start

### Prerequisites

- Python 3.13 (older versions might work but haven't been tested)
- Formula 1 data (automatically downloaded via FastF1)

### Installation

```bash
# Install dependencies using uv (recommended) or pip
uv sync
source .venv/bin/activate
```

### Update dependencies

```bash
uv add pandas
```

### Execute scripts

```bash
uv run python src/foo.py
# or
uvx python src/bar.py
```

## Data Source
Formula 1 official timing data via the [FastF1](https://docs.fastf1.dev) library

## License

This project is for educational purposes as part of a Master's capstone project.