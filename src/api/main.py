"""
FastAPI REST Inference Service for Time-Series Anomaly Detection.

Exposes REST API endpoints:
  - GET /health: Health check endpoint returning {"status": "ok"}.
  - POST /predict: Evaluates a time-series window sequence against the trained LSTM Autoencoder.

Startup / Lifespan Context Manager:
  Loads model.pth, scaler.joblib, and anomaly_threshold.npy ONCE into memory.
  No file I/O operations inside route handlers.
"""

from contextlib import asynccontextmanager
import os
from typing import Dict, Any, List
import joblib
import numpy as np
import yaml
import torch
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.models.anomaly_model import LSTMAutoencoder


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup artifact initialization.
    Loads trained PyTorch model, StandardScaler, and anomaly threshold into application state.
    """
    config = load_config("config.yaml")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    artifact_cfg = config.get("artifacts", {})

    model_path = artifact_cfg.get("model_path", "model.pth")
    scaler_path = artifact_cfg.get("scaler_path", "scaler.joblib")
    threshold_path = artifact_cfg.get("threshold_path", "anomaly_threshold.npy")

    input_dim = int(model_cfg.get("input_dim", 1))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 1))
    window_size = int(train_cfg.get("window_size", 10))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler artifact not found at: {scaler_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold artifact not found at: {threshold_path}")

    # 1. Load PyTorch model state dict
    device = torch.device("cpu")
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load fitted StandardScaler
    scaler = joblib.load(scaler_path)

    # 3. Load anomaly threshold float
    threshold_arr = np.load(threshold_path)
    threshold_val = float(threshold_arr)

    # Store in application state for zero file I/O during inference
    app.state.model = model
    app.state.scaler = scaler
    app.state.threshold = threshold_val
    app.state.window_size = window_size
    app.state.input_dim = input_dim
    app.state.device = device

    print("FastAPI Application Startup: All model artifacts loaded successfully.")
    yield


app = FastAPI(
    title="Time-Series Anomaly Detection API",
    description="REST API for real-time streaming time-series anomaly detection using LSTM Autoencoder.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    data_point: List[float] = Field(
        ...,
        description="Streaming time-series sequence window of numerical values.",
        json_schema_extra={"example": [12.4, 15.2, 14.1, 13.8, 16.0, 15.5, 14.9, 13.2, 15.0, 14.4]},
    )


class PredictResponse(BaseModel):
    input_data: List[float]
    anomaly_score: float
    is_anomaly: int
    threshold: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Lightweight health check endpoint returning HTTP 200 with {'status': 'ok'}."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(payload: PredictRequest, request: Request):
    """
    Evaluates an incoming time-series window sequence for anomalies.

    Pipeline:
      1. Validate incoming sequence length against expected window_size (HTTP 400 on mismatch).
      2. Scale input values using the in-memory StandardScaler.
      3. Construct 3D tensor (1, window_size, 1).
      4. Forward pass through in-memory LSTMAutoencoder in eval mode with torch.no_grad().
      5. Calculate MSE reconstruction error (anomaly_score).
      6. Compare anomaly_score with pre-loaded threshold (is_anomaly: 1 if > threshold else 0).
      7. Return PredictResponse with original input_data, anomaly_score, is_anomaly, and threshold.
    """
    expected_window_size = request.app.state.window_size

    # Length validation: return HTTP 400 Bad Request if length is invalid
    if len(payload.data_point) != expected_window_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data_point length. Expected exactly {expected_window_size} float values, got {len(payload.data_point)}.",
        )

    raw_array = np.array(payload.data_point, dtype=np.float32).reshape(-1, 1)

    # Transform using pre-loaded scaler (NO fit or fit_transform at inference)
    scaled_array = request.app.state.scaler.transform(raw_array)

    # Reshape to PyTorch 3D tensor: (1, window_size, 1)
    input_tensor = (
        torch.tensor(scaled_array, dtype=torch.float32)
        .unsqueeze(0)
        .to(request.app.state.device)
    )

    # In-memory inference
    with torch.no_grad():
        reconstruction = request.app.state.model(input_tensor)

    # Calculate MSE reconstruction error (anomaly score)
    anomaly_score = float(torch.mean((input_tensor - reconstruction) ** 2).item())

    # Threshold comparison
    threshold = request.app.state.threshold
    is_anomaly = 1 if anomaly_score > threshold else 0

    return PredictResponse(
        input_data=payload.data_point,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        threshold=threshold,
    )
