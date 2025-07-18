# src/housing_predict.py
from datetime import datetime
import logging
import os
from fastapi import FastAPI, Query, HTTPException, Request, Response
from redis import asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from contextlib import asynccontextmanager
from redis.asyncio import Redis
from fastapi_cache.decorator import cache
from joblib import load

from src.schemas import (
    PredictionOutput,
    HousingInput,
    HelloResponse,
    BatchHousingInput,
    BatchPredictionOutput,
)
from src.model_manager import model_manager

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Constants
LOCAL_REDIS_URL = "redis://localhost:6379"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Load model
model = load("model_pipeline.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    retries = 3
    while retries > 0:
        try:
            HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
            logger.info(f"Attempting to connect to Redis at {HOST_URL}")
            redis = Redis.from_url(
                HOST_URL, encoding="utf8", decode_responses=True, socket_timeout=5
            )
            await redis.ping()
            FastAPICache.init(RedisBackend(redis), prefix="w255-cache-prediction")
            logger.info("Successfully connected to Redis")
            break
        except Exception as e:
            retries -= 1
            logger.error(
                f"Failed to connect to Redis (attempts left: {retries}): {str(e)}"
            )
            if retries > 0:
                await asyncio.sleep(5)
            else:
                logger.critical(
                    "Could not establish Redis connection after all attempts"
                )
                raise RuntimeError("Failed to initialize Redis cache")

    yield


sub_application_housing_predict = FastAPI(lifespan=lifespan)


@sub_application_housing_predict.post("/predict", response_model=PredictionOutput)
@cache()
async def predict(
    input_data: HousingInput,
    model_id: str = Query("model_pipeline", description="Model ID to use for prediction")
) -> PredictionOutput:
    """Note: request and response params must come first for cache to work properly"""
    logger.info(f"Processing prediction request with model_id={model_id}")
    try:
        model = model_manager.get_model(model_id)
    except FileNotFoundError:
        logger.error(f"Model file not found for model_id={model_id}")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading model: {str(e)}")

    data = [
        [
            input_data.MedInc,
            input_data.HouseAge,
            input_data.AveRooms,
            input_data.AveBedrms,
            input_data.Population,
            input_data.AveOccup,
            input_data.Latitude,
            input_data.Longitude,
        ]
    ]
    prediction = model.predict(data)[0]
    logger.info(f"Prediction successful: {prediction}")
    return PredictionOutput(prediction=prediction)


@sub_application_housing_predict.post(
    "/bulk-predict", response_model=BatchPredictionOutput
)
@cache()
async def multi_predict(
    input_data: BatchHousingInput,
    model_id: str = Query("model_pipeline", description="Model ID to use for prediction")
) -> BatchPredictionOutput:
    """Note: request and response params must come first for cache to work properly"""
    logger.info(f"Processing bulk prediction request with model_id={model_id}")
    try:
        model = model_manager.get_model(model_id)
    except FileNotFoundError:
        logger.error(f"Model file not found for model_id={model_id}")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading model: {str(e)}")

    data = input_data.vectorize()
    logger.info(f"Processing {len(data)} records")
    predictions = model.predict(data).tolist()
    logger.info("Bulk prediction successful")
    return BatchPredictionOutput(predictions=predictions)


@cache()
@sub_application_housing_predict.get("/hello")
async def hello(
    request: Request, response: Response, name: str = Query(..., description="<name>")
) -> HelloResponse:
    """Note: request and response params must come first for cache to work properly"""
    return HelloResponse(message=f"Hello {name}")


@sub_application_housing_predict.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"time": datetime.utcnow().isoformat()}
