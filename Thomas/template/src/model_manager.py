import threading
from collections import OrderedDict
from joblib import load
from typing import Any, Dict, Optional

class ModelManager:
    """
    Manages loading and caching of up to N models in memory.
    Uses an LRU (Least Recently Used) eviction policy.
    """
    def __init__(self, max_models: int = 20, model_dir: str = "models"):
        self.max_models = max_models
        self.model_dir = model_dir
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get_model(self, model_id: str) -> Any:
        """Retrieve a model by model_id, loading it if not already in memory."""
        with self._lock:
            if model_id in self._models:
                # Move to end to mark as recently used
                self._models.move_to_end(model_id)
                return self._models[model_id]
            # Otherwise, load model
            model_path = f"{self.model_dir}/{model_id}.pkl"
            model = load(model_path)
            # Evict least recently used if over capacity
            if len(self._models) >= self.max_models:
                self._models.popitem(last=False)
            self._models[model_id] = model
            return model

    def list_loaded_models(self) -> Dict[str, Any]:
        """Return a list of currently loaded model IDs."""
        with self._lock:
            return list(self._models.keys())

    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

# Singleton instance
model_manager = ModelManager()
