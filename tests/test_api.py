"""
Integration tests for FastAPI REST Inference Service (src/api/main.py).
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient fixture that enters app lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns HTTP 200 with payload {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_request(client):
    """Test POST /predict with a valid 10-element data_point array."""
    valid_payload = {
        "data_point": [10.0, 15.2, 14.1, 13.8, 16.0, 15.5, 14.9, 13.2, 15.0, 14.4]
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert set(data.keys()) == {"input_data", "anomaly_score", "is_anomaly", "threshold"}
    assert data["input_data"] == valid_payload["data_point"]
    assert isinstance(data["anomaly_score"], float)
    assert data["is_anomaly"] in [0, 1]
    assert isinstance(data["threshold"], float)


def test_predict_too_few_elements(client):
    """Test POST /predict with fewer elements than window_size (e.g. 5) returns HTTP 400."""
    invalid_payload = {"data_point": [10.0, 15.2, 14.1, 13.8, 16.0]}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Invalid data_point length" in response.json()["detail"]


def test_predict_too_many_elements(client):
    """Test POST /predict with more elements than window_size (e.g. 15) returns HTTP 400."""
    invalid_payload = {"data_point": [10.0] * 15}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Invalid data_point length" in response.json()["detail"]


def test_predict_missing_data_point(client):
    """Test POST /predict with missing data_point field returns HTTP 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_string_elements(client):
    """Test POST /predict with string values returns HTTP 422."""
    invalid_payload = {"data_point": ["abc", "xyz"]}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_integer_elements(client):
    """Test POST /predict accepts coerced integer numerical values."""
    valid_payload = {"data_point": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["is_anomaly"] in [0, 1]


def test_predict_response_exact_contract(client):
    """Test POST /predict response payload contains EXACTLY the required four keys."""
    valid_payload = {"data_point": [12.0] * 10}
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"input_data", "anomaly_score", "is_anomaly", "threshold"}
    assert set(data.keys()) == expected_keys
