"""
Integration tests for FastAPI REST Inference Service (src/api/main.py).
"""

import math
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient fixture that enters app lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns HTTP 200 with payload EXACTLY {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_request(client):
    """Test POST /predict with a valid 10-element data_point array."""
    window_size = app.state.window_size
    valid_payload = {
        "data_point": [10.0 + i * 0.5 for i in range(window_size)]
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert set(data.keys()) == {"input_data", "anomaly_score", "is_anomaly", "threshold"}
    assert data["input_data"] == valid_payload["data_point"]
    assert isinstance(data["anomaly_score"], float)
    assert math.isfinite(data["anomaly_score"])
    assert isinstance(data["threshold"], float)
    assert math.isfinite(data["threshold"])
    assert data["is_anomaly"] in [0, 1]


def test_predict_too_few_elements(client):
    """Test POST /predict with fewer elements than window_size returns HTTP 400."""
    window_size = app.state.window_size
    invalid_payload = {"data_point": [10.0] * (window_size - 1)}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Invalid data_point length" in response.json()["detail"]


def test_predict_too_many_elements(client):
    """Test POST /predict with more elements than window_size returns HTTP 400."""
    window_size = app.state.window_size
    invalid_payload = {"data_point": [10.0] * (window_size + 5)}
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
    window_size = app.state.window_size
    valid_payload = {"data_point": [10 * (i + 1) for i in range(window_size)]}
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["is_anomaly"] in [0, 1]


def test_predict_response_exact_contract(client):
    """Test POST /predict response payload contains EXACTLY the required four keys."""
    window_size = app.state.window_size
    valid_payload = {"data_point": [12.0] * window_size}
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"input_data", "anomaly_score", "is_anomaly", "threshold"}
    assert set(data.keys()) == expected_keys


def test_predict_zero_file_io_during_inference(client):
    """Test that POST /predict performs ZERO file I/O operations (torch.load, joblib.load, np.load)."""
    window_size = app.state.window_size
    valid_payload = {"data_point": [15.0] * window_size}

    with patch("torch.load") as mock_torch_load, \
         patch("joblib.load") as mock_joblib_load, \
         patch("numpy.load") as mock_np_load:
        
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        
        # Verify no artifact file loading functions were invoked during request handling
        mock_torch_load.assert_not_called()
        mock_joblib_load.assert_not_called()
        mock_np_load.assert_not_called()
