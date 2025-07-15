from fastapi.testclient import TestClient
import pytest
from datetime import datetime
import time
import logging

LOGGER = logging.getLogger(__name__)

from src.main import app
client = TestClient(app)

# Centralized URL paths for all known endpoints
ENDPOINTS = {
    "health": "/lab/health",
    "hello": "/lab/hello",
    "predict": "/lab/predict",
    "bulk_predict": "/lab/bulk-predict",
    "docs": "/docs",
    "openapi_json": "/openapi.json",
}


# Helper function to parse ISO8601 datetime
def is_iso8601(datetime_str):
    try:
        datetime.fromisoformat(datetime_str)
        return True
    except ValueError:
        return False


# Mock the predict method globally
# @pytest.fixture(autouse=True)
# def mock_model_predict():
#     with mock.patch("src.housing_predict.model") as mock_model:
#         mock_model.predict.return_value = [123.45]  # Mocked prediction output
#         yield


def test_health_endpoint():
    response = client.get(ENDPOINTS["health"])
    assert response.status_code == 200
    response_json = response.json()
    assert "time" in response_json
    assert is_iso8601(response_json["time"])


@pytest.mark.parametrize("name", ["John", "Alice", "Bob"])
def test_hello_endpoint_with_name(name):
    response = client.get(f"{ENDPOINTS['hello']}?name={name}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {name}"}


def test_hello_endpoint_without_name():
    response = client.get(ENDPOINTS["hello"])
    assert response.status_code == 422  # Unprocessable Entity


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 404
    assert "Not Found" in response.text


def test_docs_endpoint():
    response = client.get(ENDPOINTS["docs"])
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_json_endpoint():
    response = client.get(ENDPOINTS["openapi_json"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    openapi_schema = response.json()
    assert "openapi" in openapi_schema
    assert openapi_schema["openapi"].startswith("3.")  # Ensure OpenAPI version 3+


@pytest.mark.parametrize(
    "endpoint",
    [
        ENDPOINTS["health"],
        ENDPOINTS["hello"],
        ENDPOINTS["docs"],
        ENDPOINTS["openapi_json"],
    ],
)
def test_endpoints_with_incorrect_method(endpoint):
    response = client.post(endpoint)
    assert response.status_code == 405  # Method Not Allowed


@pytest.mark.parametrize("name", ["", "   ", "123"])
def test_hello_endpoint_with_various_names(name):
    response = client.get(f"{ENDPOINTS['hello']}?name={name}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {name}"}


def test_non_existent_endpoint():
    response = client.get("/non_existent")
    assert response.status_code == 404


def test_health_endpoint_performance():
    start_time = time.time()
    response = client.get(ENDPOINTS["health"])
    end_time = time.time()
    assert response.status_code == 200
    assert end_time - start_time < 0.1  # Ensure response time is less than 100ms


# Test for the /predict endpoint with valid and invalid inputs

valid_input = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def test_predict_endpoint_with_valid_input():
    response = client.post(ENDPOINTS["predict"], json=valid_input)
    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)


@pytest.mark.parametrize(
    "key,value",
    [
        ("Latitude", 95),  # Invalid latitude
        ("Latitude", -95),  # Invalid latitude
        ("Longitude", 200),  # Invalid longitude
        ("Longitude", -200),  # Invalid longitude
    ],
)
def test_predict_endpoint_with_invalid_lat_lon(key, value):
    invalid_input = valid_input.copy()
    invalid_input[key] = value
    response = client.post(ENDPOINTS["predict"], json=invalid_input)
    assert response.status_code == 422
    response_json = response.json()
    expected_msg = f"Value error, Invalid value for {key}"
    assert response_json["detail"][0]["msg"] == expected_msg
    assert response_json["detail"][0]["type"] == "value_error"
    assert response_json["detail"][0]["loc"][-1] == key


def test_predict_endpoint_with_missing_field():
    incomplete_input = valid_input.copy()
    del incomplete_input["MedInc"]
    response = client.post(ENDPOINTS["predict"], json=incomplete_input)
    assert response.status_code == 422
    response_json = response.json()
    assert response_json["detail"][0]["msg"] == "Field required"
    assert response_json["detail"][0]["loc"][-1] == "MedInc"


def test_predict_endpoint_with_extra_field():
    extra_input = valid_input.copy()
    extra_input["ExtraField"] = 123
    response = client.post(ENDPOINTS["predict"], json=extra_input)
    assert response.status_code == 422
    response_json = response.json()
    assert response_json["detail"][0]["type"] == "extra_forbidden"
    assert "ExtraField" in response_json["detail"][0]["loc"]


def test_predict_endpoint_with_invalid_types():
    invalid_type_input = valid_input.copy()
    invalid_type_input["MedInc"] = "invalid"
    response = client.post(ENDPOINTS["predict"], json=invalid_type_input)
    assert response.status_code == 422
    response_json = response.json()
    assert response_json["detail"][0]["type"] == "float_parsing"
    assert response_json["detail"][0]["loc"][-1] == "MedInc"


def test_predict_endpoint_performance():
    start_time = time.time()
    response = client.post(ENDPOINTS["predict"], json=valid_input)
    LOGGER.debug(response.json())
    end_time = time.time()
    assert response.status_code == 200
    assert end_time - start_time < 0.5  # Ensure prediction time is reasonable


# New test for bulk prediction
bulk_valid_input = {
    "houses": [
        valid_input,
        {
            "MedInc": 7.2574,
            "HouseAge": 28.0,
            "AveRooms": 5.703427,
            "AveBedrms": 1.07381,
            "Population": 334.0,
            "AveOccup": 3.333556,
            "Latitude": 36.77,
            "Longitude": -121.87,
        },
    ]
}


def test_bulk_predict_endpoint_with_valid_input():
    response = client.post(ENDPOINTS["bulk_predict"], json=bulk_valid_input)
    assert response.status_code == 200
    response_json = response.json()
    assert "predictions" in response_json
    LOGGER.debug(response_json)
    assert isinstance(response_json["predictions"], list)
    assert len(response_json["predictions"]) == 2
    for prediction in response_json["predictions"]:
        assert isinstance(prediction, float)
