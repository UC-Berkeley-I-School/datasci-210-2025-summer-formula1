import os
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List

# Initialise the API router with a versioned prefix. All endpoints in this
# module will be available under ``/api/v1``.
router = APIRouter(prefix="/api/v1")

# --- Schemas ----------------------------------------------------------------

class SessionMetadata(BaseModel):
    session_id: str
    year: int
    race_name: str
    session_type: str
    session_date: str  # ISO format
    driver_count: int
    window_count: int


class SessionsResponse(BaseModel):
    sessions: List[SessionMetadata]


class PredictRequest(BaseModel):
    session_id: str


class DriverPrediction(BaseModel):
    driver_number: int
    driver_abbreviation: str
    y_true: List[int]
    y_pred: List[int]
    y_proba: List[List[float]]


class PredictResponse(BaseModel):
    session_id: str
    predictions: List[DriverPrediction]


class Coordinate(BaseModel):
    X: float
    Y: float
    Z: float


class DriverTelemetry(BaseModel):
    driver_number: int
    driver_abbreviation: str
    coordinates: List[Coordinate]


class TelemetryResponse(BaseModel):
    coordinates: List[DriverTelemetry]

# Import the DAO and model manager. The DAO wraps low‑level database queries
# and exposes convenient Python methods for the API. The model manager
# handles caching and loading driver specific models from disk.
from database.client_tools.db_client import F1TimescaleDAO, SessionInfo, DriverCoordinates
from src.model_manager import model_manager

# Load database configuration from the environment. Prefer the more
# conventional POSTGRES_* variables first so that the same docker
# environment variables can be reused across services. Fall back to
# service-specific variables (DB_*) and sensible defaults if nothing is
# provided. Ports are cast to integers because environment variables are
# strings by default.
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST') or os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('POSTGRES_PORT') or os.environ.get('DB_PORT', '5432')),
    'database': os.environ.get('POSTGRES_DB') or os.environ.get('DB_NAME', 'f1_telemetry'),
    'user': os.environ.get('POSTGRES_USER') or os.environ.get('DB_USER', 'f1_user'),
    'password': os.environ.get('POSTGRES_PASSWORD') or os.environ.get('DB_PASSWORD', 'f1_password'),
}

# Instantiate a single DAO instance. Reusing the same DAO avoids
# re-establishing a new connection for every request, which can be
# expensive. Psycopg2 will pool connections internally when used
# appropriately.
dao = F1TimescaleDAO(DB_CONFIG)

# --- Data mapping helpers ---------------------------------------------------

def sessioninfo_to_schema(session: SessionInfo) -> SessionMetadata:
    return SessionMetadata(
        session_id=session.session_id,
        year=session.year,
        race_name=session.race_name,
        session_type=session.session_type,
        session_date=session.session_date.isoformat() if hasattr(session.session_date, 'isoformat') else str(session.session_date),
        driver_count=session.driver_count,
        window_count=session.window_count,
    )


def drivercoordinates_to_schema(dc: DriverCoordinates) -> DriverTelemetry:
    coords = [Coordinate(**c) for c in dc.coordinates]
    return DriverTelemetry(
        driver_number=dc.driver_number,
        driver_abbreviation=dc.driver_abbreviation,
        coordinates=coords,
    )

# --- Endpoint implementations ----------------------------------------------

@router.get("/sessions", response_model=SessionsResponse)
def get_sessions() -> SessionsResponse:
    dao_sessions = dao.get_available_sessions()
    sessions = [sessioninfo_to_schema(s) for s in dao_sessions]
    return SessionsResponse(sessions=sessions)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run predictions for all drivers in a session.

    The request must supply a valid ``session_id``; otherwise a 404
    response is returned. For each driver the corresponding model is
    loaded via the ``model_manager`` and used to produce class labels and
    probability distributions. The results are aggregated into a single
    response.
    """
    try:
        features = dao.get_session_features_for_prediction(request.session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found or error: {e}")
    if not features:
        raise HTTPException(status_code=404, detail="Session not found")
    features_by_driver = features.get('features_by_driver', {})
    predictions: List[DriverPrediction] = []
    import json
    for driver_number, driver_data in features_by_driver.items():
        # Unpack driver_data (dict-based from db_client)
        X = driver_data.get('X')
        y_true = driver_data.get('y_true')
        metadata = driver_data.get('metadata')

        # Defensive: parse y_true if it's a string
        if isinstance(y_true, str):
            try:
                y_true = json.loads(y_true)
            except Exception:
                y_true = []

        # Defensive: parse metadata if it's a string
        if not isinstance(metadata, dict):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        # Determine which model directory corresponds to this driver
        models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
        model_dir = None
        model_id = None
        for d in os.listdir(models_root):
            if d.endswith(f'_driver{driver_number}'):
                model_dir = os.path.join(models_root, d)
                model_id = d  # Use the full directory name as model_id
                break

        y_pred: List[int] = []
        y_proba: List[List[float]] = []
        if model_dir:
            try:
                # Use subdirectory name as model_id for ModelManager
                model_id = os.path.basename(model_dir)
                model = model_manager.get_model(model_id)
                # Run prediction
                y_pred = model.predict(X).tolist()
                # Run predict_proba if available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X).tolist()
            except Exception as e:
                logging.error(f"Model loading/prediction error for driver {driver_number}: {e}")
                y_pred = []
                y_proba = []

        # Build prediction schema
        predictions.append(
            DriverPrediction(
                driver_number=driver_number,
                driver_abbreviation=metadata.get('driver_abbreviation', ''),
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
            )
        )

    if not predictions:
        raise HTTPException(status_code=404, detail="No driver data found for session")

    return PredictResponse(session_id=request.session_id, predictions=predictions)


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry(session_id: str = Query(...)) -> TelemetryResponse:
    try:
        dao_coords = dao.get_telemetry_coordinates(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Telemetry error: {e}")
    coordinates = [drivercoordinates_to_schema(dc) for dc in dao_coords]
    if not coordinates:
        raise HTTPException(status_code=404, detail="Telemetry not found")
    return TelemetryResponse(coordinates=coordinates)