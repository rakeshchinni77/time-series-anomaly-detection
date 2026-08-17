"""
Training and Artifact Persistence Pipeline for Time-Series Anomaly Detection.

Orchestrates offline model training, anomaly threshold calculation, and artifact persistence:
  - Loads configuration parameters from config.yaml.
  - Reuses Phase 4 data preprocessing pipeline (no data leakage).
  - Prepares PyTorch DataLoaders for training and validation.
  - Trains PyTorch LSTMAutoencoder with MSE reconstruction loss.
  - Validates after each epoch using torch.no_grad().
  - Persists state_dict of the best model (lowest validation loss) to model.pth.
  - Reloads the best model and calculates per-sequence reconstruction errors on validation data.
  - Calculates percentile-based anomaly threshold and persists anomaly_threshold.npy.
  - Persists fitted StandardScaler object to scaler.joblib.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data.preprocess import prepare_data
from src.models.anomaly_model import LSTMAutoencoder


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def calculate_reconstruction_errors(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Calculate MSE reconstruction error PER SEQUENCE across the validation dataset.

    Args:
        model (nn.Module): Trained PyTorch LSTMAutoencoder model in eval mode.
        val_loader (DataLoader): DataLoader for validation sequences.
        device (torch.device): Device to execute inference on.

    Returns:
        np.ndarray: 1D array of shape (num_val_sequences,) containing per-sequence MSE errors.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            reconstruction = model(batch_x)

            # MSE per sequence (average across window_size and feature dimensions)
            per_seq_errors = torch.mean((batch_x - reconstruction) ** 2, dim=(1, 2))
            all_errors.extend(per_seq_errors.cpu().numpy())

    errors_arr = np.array(all_errors, dtype=np.float64)

    if errors_arr.ndim != 1 or errors_arr.size == 0:
        raise ValueError(f"Expected 1D non-empty array of errors, got shape {errors_arr.shape}")
    if not np.isfinite(errors_arr).all():
        raise ValueError("Reconstruction errors contain NaN or infinite values.")

    return errors_arr


def calculate_anomaly_threshold(reconstruction_errors: np.ndarray, percentile: float) -> float:
    """
    Calculate scalar anomaly threshold from validation errors using configured percentile.

    Args:
        reconstruction_errors (np.ndarray): 1D array of validation reconstruction errors.
        percentile (float): Percentile value (0 <= percentile <= 100).

    Returns:
        float: Scalar numerical anomaly threshold.
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"threshold_percentile must be between 0 and 100, got {percentile}")

    threshold = float(np.percentile(reconstruction_errors, percentile))

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"Invalid calculated threshold: {threshold}")

    return threshold


def save_artifacts(scaler: Any, scaler_path: str, threshold: float, threshold_path: str) -> None:
    """
    Persist scaler object and anomaly threshold array to disk.

    Args:
        scaler (Any): Fitted StandardScaler instance.
        scaler_path (str): File path for scaler artifact (e.g. scaler.joblib).
        threshold (float): Numerical threshold value.
        threshold_path (str): File path for threshold artifact (e.g. anomaly_threshold.npy).
    """
    scaler_dir = os.path.dirname(scaler_path)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    threshold_dir = os.path.dirname(threshold_path)
    if threshold_dir:
        os.makedirs(threshold_dir, exist_ok=True)
    np.save(threshold_path, np.array(threshold, dtype=np.float64))


def train_model(
    config_path: str = "config.yaml",
) -> Tuple[float, int, float, str, str, str]:
    """
    Main training pipeline execution function.

    Steps:
      1. Load config.yaml hyperparameters and artifact paths.
      2. Preprocess data (chronological split, fit scaler on train ONLY, create sequences).
      3. Create PyTorch DataLoaders.
      4. Train LSTMAutoencoder and track best validation reconstruction loss.
      5. Save best model state dict to model.pth.
      6. Reload best model state dict into eval mode.
      7. Calculate per-sequence validation reconstruction errors.
      8. Calculate percentile threshold and save anomaly_threshold.npy.
      9. Save fitted scaler to scaler.joblib.

    Returns:
        Tuple[float, int, float, str, str, str]:
            (best_val_loss, best_epoch, threshold, model_path, scaler_path, threshold_path)
    """
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    anomaly_cfg = config["anomaly"]
    artifact_cfg = config["artifacts"]

    csv_path = data_cfg["path"]
    value_column = data_cfg["value_column"]
    train_ratio = float(data_cfg["train_ratio"])

    window_size = int(train_cfg["window_size"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    learning_rate = float(train_cfg["learning_rate"])

    input_dim = int(model_cfg["input_dim"])
    hidden_dim = int(model_cfg["hidden_dim"])
    num_layers = int(model_cfg["num_layers"])

    threshold_percentile = float(anomaly_cfg["threshold_percentile"])

    model_path = artifact_cfg["model_path"]
    scaler_path = artifact_cfg["scaler_path"]
    threshold_path = artifact_cfg["threshold_path"]

    # 1. Preprocess data (reusing Phase 4 pipeline; scaler fit on train ONLY)
    train_seq, val_seq, scaler = prepare_data(
        csv_path=csv_path,
        value_column=value_column,
        train_ratio=train_ratio,
        window_size=window_size,
    )

    # 2. Convert to PyTorch Tensors
    train_tensor = torch.tensor(train_seq, dtype=torch.float32)
    val_tensor = torch.tensor(val_seq, dtype=torch.float32)

    # 3. Create PyTorch DataLoaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Model, Loss, Optimizer initialization
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0

    parent_dir = os.path.dirname(model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    print(f"Starting training on device: {device}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, window_size={window_size}")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)

            loss.backward()
            optimizer.step()

            current_batch_size = batch_x.size(0)
            total_train_loss += loss.item() * current_batch_size
            total_train_samples += current_batch_size

        epoch_train_loss = total_train_loss / total_train_samples

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                reconstruction = model(batch_x)
                loss = criterion(reconstruction, batch_x)

                current_batch_size = batch_x.size(0)
                total_val_loss += loss.item() * current_batch_size
                total_val_samples += current_batch_size

        epoch_val_loss = total_val_loss / total_val_samples

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}"
        )

        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Best model state dict saved to: {model_path}")

    # 6. Reload BEST model for post-training threshold calculation
    best_model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    best_model.load_state_dict(state_dict)
    best_model.eval()

    # 7. Calculate per-sequence validation reconstruction errors
    val_errors = calculate_reconstruction_errors(best_model, val_loader, device)

    # 8. Calculate anomaly threshold
    threshold = calculate_anomaly_threshold(val_errors, threshold_percentile)

    # 9. Persist scaler and threshold artifacts
    save_artifacts(scaler, scaler_path, threshold, threshold_path)

    print("\nValidation Reconstruction Errors Analysis:")
    print(f"  Count: {len(val_errors)}")
    print(f"  Min Error:    {np.min(val_errors):.6f}")
    print(f"  Max Error:    {np.max(val_errors):.6f}")
    print(f"  Mean Error:   {np.mean(val_errors):.6f}")
    print(f"  Median Error: {np.median(val_errors):.6f}")

    print("\nArtifact Persistence Summary:")
    print(f"  Threshold Percentile: {threshold_percentile}%")
    print(f"  Calculated Anomaly Threshold: {threshold:.6f}")
    print(f"  Model Artifact: {model_path}")
    print(f"  Scaler Artifact: {scaler_path}")
    print(f"  Threshold Artifact: {threshold_path}")

    return best_val_loss, best_epoch, threshold, model_path, scaler_path, threshold_path


if __name__ == "__main__":
    train_model()
