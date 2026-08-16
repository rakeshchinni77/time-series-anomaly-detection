"""
Data preprocessing pipeline for Time-Series Anomaly Detection.

Implements data loading, chronological splitting, stateful StandardScaler normalization,
and sliding window sequence creation for univariate time series.

Strict Data Leakage Rule:
StandardScaler MUST be fitted ONLY on the training split of the dataset.
The validation dataset is transformed using the training-fitted scaler instance.
"""

import os
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(csv_path: str, value_column: str = "value") -> pd.DataFrame:
    """
    Load CSV data and validate required columns and numerical integrity.

    Args:
        csv_path (str): Path to the CSV file.
        value_column (str): Name of the target value column.

    Returns:
        pd.DataFrame: Loaded and validated DataFrame with parsed timestamp.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column 'timestamp' in dataset.")
    if value_column not in df.columns:
        raise ValueError(f"Missing required value column '{value_column}' in dataset.")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract numerical values for validation
    values = df[value_column].to_numpy()

    if np.isnan(values).any():
        raise ValueError("Dataset contains NaN values.")
    if np.isinf(values).any():
        raise ValueError("Dataset contains Infinite values.")

    return df


def split_time_series(
    data: np.ndarray, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a strict chronological split on univariate time-series data.

    Args:
        data (np.ndarray): 1D or 2D input array.
        train_ratio (float): Ratio of data to use for training (0 < ratio < 1).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_data, validation_data)
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    n_samples = len(data)
    if n_samples < 2:
        raise ValueError("Insufficient data points to perform split.")

    train_end = int(n_samples * train_ratio)
    if train_end == 0 or train_end >= n_samples:
        raise ValueError("train_ratio results in empty train or validation set.")

    train_data = data[:train_end]
    val_data = data[train_end:]

    return train_data, val_data


def fit_scaler(train_data: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler strictly on the training partition.

    Args:
        train_data (np.ndarray): Training observations.

    Returns:
        StandardScaler: Scaler fitted ONLY on training data.
    """
    scaler = StandardScaler()
    data_2d = train_data.reshape(-1, 1) if train_data.ndim == 1 else train_data
    if data_2d.ndim != 2 or data_2d.shape[1] != 1:
        raise ValueError(f"Expected single-feature data of shape (N, 1), got {train_data.shape}")
    scaler.fit(data_2d)
    return scaler


def transform_data(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Transform observations using a pre-fitted StandardScaler instance.

    Args:
        data (np.ndarray): Input observations.
        scaler (StandardScaler): Fitted scaler.

    Returns:
        np.ndarray: Scaled array of shape (N, 1).
    """
    data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
    if data_2d.ndim != 2 or data_2d.shape[1] != 1:
        raise ValueError(f"Expected single-feature data of shape (N, 1), got {data.shape}")
    return scaler.transform(data_2d)


def create_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create overlapping sliding window sequences from single-feature data.

    Args:
        data (np.ndarray): 1D or 2D array of observations.
        window_size (int): Positive integer length of each temporal window.

    Returns:
        np.ndarray: Array of sequences with shape (num_samples, window_size, 1).
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if data.ndim == 1:
        data_2d = data.reshape(-1, 1)
    elif data.ndim == 2:
        if data.shape[1] != 1:
            raise ValueError(f"Multivariate inputs not supported; expected 1 feature, got {data.shape[1]}.")
        data_2d = data
    else:
        raise ValueError(f"Input data must be 1D or 2D array, got {data.ndim}D array.")

    n_samples = len(data_2d)
    if window_size > n_samples:
        raise ValueError(
            f"window_size ({window_size}) cannot be larger than number of observations ({n_samples})."
        )

    num_sequences = n_samples - window_size + 1
    sequences = np.empty((num_sequences, window_size, 1), dtype=data_2d.dtype)

    for i in range(num_sequences):
        sequences[i] = data_2d[i : i + window_size]

    return sequences


def prepare_data(
    csv_path: str,
    value_column: str = "value",
    train_ratio: float = 0.8,
    window_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    High-level preprocessing pipeline for train.py execution.

    Steps:
      1. Load raw dataset and validate integrity.
      2. Chronological train/validation split.
      3. Fit StandardScaler STRICTLY on train_raw.
      4. Transform train_raw and val_raw using fitted scaler.
      5. Create sliding window sequences.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: (train_sequences, val_sequences, fitted_scaler)
    """
    df = load_data(csv_path, value_column=value_column)
    raw_values = df[value_column].to_numpy()

    # Step 1: Split BEFORE scaling (No Data Leakage)
    train_raw, val_raw = split_time_series(raw_values, train_ratio=train_ratio)

    # Step 2: Fit scaler ONLY on training data
    scaler = fit_scaler(train_raw)

    # Step 3: Transform train and val data using the training-fitted scaler
    train_scaled = transform_data(train_raw, scaler)
    val_scaled = transform_data(val_raw, scaler)

    # Step 4: Generate overlapping sliding windows
    train_sequences = create_sequences(train_scaled, window_size=window_size)
    val_sequences = create_sequences(val_scaled, window_size=window_size)

    return train_sequences, val_sequences, scaler
