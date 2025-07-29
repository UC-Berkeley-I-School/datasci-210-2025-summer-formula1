import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api_v1 import router as api_v1_router
from src.model_manager import model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload all models
    logger.info("Starting model preloading...")
    models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    
    if os.path.exists(models_root):
        model_dirs = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d)) and d.startswith('20250722')]
        logger.info(f"Found {len(model_dirs)} model directories to preload")
        
        for model_dir in model_dirs:
            try:
                # Extract driver number from directory name
                if '_driver' in model_dir:
                    driver_num = model_dir.split('_driver')[-1]
                    logger.info(f"Preloading model for driver {driver_num}: {model_dir}")
                    model_manager.get_model(model_dir)
            except Exception as e:
                logger.error(f"Failed to preload model {model_dir}: {e}")
        
        logger.info(f"Preloaded {len(model_manager.list_loaded_models())} models")
    else:
        logger.warning(f"Models directory not found: {models_root}")
    
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.include_router(api_v1_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
