"""
Unit tests for Anomaly Threshold calculation and Artifact Persistence (Phase 9).
"""

import os
import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.models.anomaly_model import LSTMAutoencoder
from train import (
    load_config,
    calculate_reconstruction_errors,
    calculate_anomaly_threshold,
)


def test_calculate_anomaly_threshold_percentile():
    """Test anomaly threshold calculation with exact percentile."""
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    threshold_90 = calculate_anomaly_threshold(errors, 90.0)
    expected_90 = float(np.percentile(errors, 90.0))
    assert pytest.approx(threshold_90) == expected_90


def test_calculate_anomaly_threshold_invalid_percentile():
    """Test that invalid percentile values raise ValueError."""
    errors = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError):
        calculate_anomaly_threshold(errors, -1.0)
    with pytest.raises(ValueError):
        calculate_anomaly_threshold(errors, 101.0)


def test_calculate_reconstruction_errors_per_sequence():
    """Test per-sequence MSE error calculation shape (num_samples,)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=16, num_layers=1)
    model.eval()
    
    # 4 sequences of length 5
    sample_data = torch.randn(4, 5, 1)
    dataset = TensorDataset(sample_data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    device = torch.device("cpu")

    errors = calculate_reconstruction_errors(model, loader, device)
    
    assert errors.ndim == 1
    assert errors.shape == (4,)
    assert np.isfinite(errors).all()


def test_saved_artifacts_reload_and_integrity():
    """Test that model.pth, scaler.joblib, and anomaly_threshold.npy exist and reload cleanly."""
    config = load_config("config.yaml")
    artifact_cfg = config["artifacts"]

    model_path = artifact_cfg["model_path"]
    scaler_path = artifact_cfg["scaler_path"]
    threshold_path = artifact_cfg["threshold_path"]

    # 1. Model artifact reload
    assert os.path.exists(model_path), f"Missing model artifact: {model_path}"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    x = torch.randn(2, config["training"]["window_size"], config["model"]["input_dim"])
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape

    # 2. Scaler artifact reload
    assert os.path.exists(scaler_path), f"Missing scaler artifact: {scaler_path}"
    scaler = joblib.load(scaler_path)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")
    transformed = scaler.transform(np.array([[1.0]]))
    assert transformed.shape == (1, 1)

    # 3. Anomaly threshold artifact reload
    assert os.path.exists(threshold_path), f"Missing threshold artifact: {threshold_path}"
    threshold_arr = np.load(threshold_path)
    threshold = float(threshold_arr)
    assert np.isfinite(threshold)
    assert threshold >= 0.0
