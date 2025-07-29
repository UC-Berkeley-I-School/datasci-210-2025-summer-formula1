import threading
import os
from collections import OrderedDict
from joblib import load
from typing import Any, Dict, Optional


class ModelManager:
    """
    Manages loading and caching of up to ``max_models`` models in memory.
    Uses a simple LRU (Least Recently Used) eviction policy: when the
    number of loaded models exceeds ``max_models`` the least recently
    accessed model will be evicted. The default model directory can be
    overridden by passing ``model_dir``, setting the ``MODEL_DIR``
    environment variable or falls back to a path relative to this file.
    """

    def __init__(self, max_models: int = 20, model_dir: Optional[str] = None):
        self.max_models = max_models
        # Determine model directory. Preference order:
        #  1. Explicit ``model_dir`` parameter
        #  2. ``MODEL_DIR`` environment variable
        #  3. "../../models" relative to this file
        if model_dir:
            self.model_dir = model_dir
        else:
            env_dir = os.environ.get("MODEL_DIR")
            if env_dir:
                self.model_dir = env_dir
            else:
                # Construct an absolute path two levels up from this file
                self.model_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../models")
                )
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get_model(self, model_id: str) -> Any:
        """Retrieve a model by ``model_id``. If it is not already loaded
        into memory it will be loaded from disk. Models are stored in
        subdirectories under ``self.model_dir`` with the naming pattern
        ``<model_id>/my_model.pkl``. When loading a new model causes the
        cache size to exceed ``max_models``, the oldest entry will be
        evicted. Exceptions raised during model loading will propagate to
        the caller.
        """
        with self._lock:
            if model_id in self._models:
                # Move to the end to mark as recently used
                self._models.move_to_end(model_id)
                return self._models[model_id]
            model_path = f"{self.model_dir}/{model_id}/my_model.pkl"
            model = load(model_path)
            if len(self._models) >= self.max_models:
                # Evict least recently used item
                self._models.popitem(last=False)
            self._models[model_id] = model
            return model

    def list_loaded_models(self) -> Dict[str, Any]:
        """Return a list of currently loaded model IDs."""
        with self._lock:
            return list(self._models.keys())

    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory. Returns True if the model was
        removed or False if it was not present."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False


# Singleton instance used by API handlers
model_manager = ModelManager()