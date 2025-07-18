import pytest
from unittest import mock
from fastapi.testclient import TestClient
import numpy as np

mock.patch("fastapi_cache.decorator.cache", lambda *args, **kwargs: lambda f: f).start()


class MockModel:
    def predict(self, X):
        # Return predictable test values
        if isinstance(X, list):
            return np.array([4.0] * len(X))
        return np.array([4.0] * len(X))


@pytest.fixture(autouse=True)
def mock_dependencies():
    # Mock both the model and joblib.load
    with mock.patch("src.housing_predict.model", MockModel()):
        with mock.patch("joblib.load", return_value=MockModel()):
            yield


@pytest.fixture
def test_client():
    from src.main import app

    with TestClient(app) as client:
        yield client
