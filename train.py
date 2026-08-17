"""
Training Pipeline for Time-Series Anomaly Detection.

Orchestrates offline model training driven by config.yaml:
  - Loads configuration parameters.
  - Reuses Phase 4 data preprocessing pipeline (no data leakage).
  - Prepares PyTorch DataLoaders.
  - Trains PyTorch LSTMAutoencoder with MSE reconstruction loss.
  - Validates after each epoch using torch.no_grad().
  - Persists state_dict of the best model (lowest validation reconstruction loss) to model.pth.
"""

import os
from typing import Dict, Any, Tuple
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


def train_model(config_path: str = "config.yaml") -> Tuple[float, int, str]:
    """
    Main training execution function.

    Returns:
        Tuple[float, int, str]: (best_val_loss, best_epoch, model_path)
    """
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
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

    model_path = artifact_cfg["model_path"]

    # 1. Preprocess data (reusing Phase 4 pipeline; scaler fit on train ONLY)
    train_seq, val_seq, _ = prepare_data(
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

    # Ensure target directory exists if model_path includes parent directories
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

        # Check for best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"Best model state dict saved to: {model_path}")

    return best_val_loss, best_epoch, model_path


if __name__ == "__main__":
    train_model()
