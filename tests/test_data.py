"""
Unit tests for data preprocessing pipeline (src/data/preprocess.py).
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import (
    load_data,
    split_time_series,
    fit_scaler,
    transform_data,
    create_sequences,
    prepare_data,
)


def test_create_sequences_shape():
    """Test create_sequences produces shape (samples, window_size, 1)."""
    data = np.arange(10)
    window_size = 4
    sequences = create_sequences(data, window_size)
    assert sequences.shape == (7, 4, 1)


def test_create_sequences_values():
    """Test create_sequences constructs exact sequence values."""
    data = np.array([1, 2, 3, 4, 5])
    window_size = 3
    sequences = create_sequences(data, window_size)
    expected = np.array([
        [[1], [2], [3]],
        [[2], [3], [4]],
        [[3], [4], [5]]
    ])
    np.testing.assert_array_equal(sequences, expected)


def test_create_sequences_overlapping_windows():
    """Explicitly test overlapping sliding windows construction."""
    data = np.array([1, 2, 3, 4, 5])
    window_size = 3
    sequences = create_sequences(data, window_size)
    np.testing.assert_array_equal(sequences[0], np.array([[1], [2], [3]]))
    np.testing.assert_array_equal(sequences[1], np.array([[2], [3], [4]]))
    np.testing.assert_array_equal(sequences[2], np.array([[3], [4], [5]]))


def test_create_sequences_1d_input():
    """Test create_sequences accepts 1D input array."""
    data = np.array([1, 2, 3, 4, 5])
    window_size = 3
    sequences = create_sequences(data, window_size)
    assert sequences.shape == (3, 3, 1)


def test_create_sequences_2d_single_feature_input():
    """Test create_sequences accepts 2D single-feature input array (N, 1)."""
    data = np.array([[1], [2], [3], [4], [5]])
    window_size = 3
    sequences = create_sequences(data, window_size)
    assert sequences.shape == (3, 3, 1)
    expected = np.array([
        [[1], [2], [3]],
        [[2], [3], [4]],
        [[3], [4], [5]]
    ])
    np.testing.assert_array_equal(sequences, expected)


def test_create_sequences_rejects_zero_window():
    """Test create_sequences raises ValueError when window_size is 0."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        create_sequences(data, 0)


def test_create_sequences_rejects_negative_window():
    """Test create_sequences raises ValueError when window_size is negative."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        create_sequences(data, -1)


def test_create_sequences_rejects_oversized_window():
    """Test create_sequences raises ValueError when window_size exceeds observation count."""
    data = np.array([1, 2, 3])
    window_size = 4
    with pytest.raises(ValueError):
        create_sequences(data, window_size)


def test_create_sequences_rejects_multivariate_input():
    """Test create_sequences raises ValueError when input has > 1 feature."""
    data = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40]
    ])
    with pytest.raises(ValueError):
        create_sequences(data, 3)


def test_scaler_fits_training_data_only():
    """Test StandardScaler fit behavior on training values."""
    train_values = np.array([10.0, 20.0, 30.0, 40.0])
    scaler = fit_scaler(train_values)
    assert pytest.approx(scaler.mean_[0]) == 25.0
    
    train_scaled = transform_data(train_values, scaler)
    assert pytest.approx(train_scaled.mean()) == 0.0


def test_validation_uses_training_fitted_scaler():
    """Test validation values are transformed using training-fitted scaler."""
    train_values = np.array([10.0, 20.0, 30.0, 40.0])
    val_values = np.array([100.0, 110.0])
    scaler = fit_scaler(train_values)
    
    train_scaled = transform_data(train_values, scaler)
    val_scaled = transform_data(val_values, scaler)
    
    # Expected transform: (x - mean) / std = (x - 25) / std
    expected_val_0 = (100.0 - 25.0) / scaler.scale_[0]
    assert pytest.approx(val_scaled[0, 0]) == expected_val_0


def test_data_leakage_regression():
    """Regression test ensuring scaler mean reflects ONLY training distribution."""
    train_values = np.array([1.0, 2.0, 3.0, 4.0])
    val_values = np.array([100.0, 101.0])
    
    scaler = fit_scaler(train_values)
    assert pytest.approx(scaler.mean_[0]) == 2.5
    
    combined_mean = (1.0 + 2.0 + 3.0 + 4.0 + 100.0 + 101.0) / 6.0
    assert scaler.mean_[0] != combined_mean


def test_expected_sequence_count():
    """Test sequence count formula N - W + 1."""
    N = 20
    window_size = 5
    data = np.arange(N)
    sequences = create_sequences(data, window_size)
    assert sequences.shape[0] == (N - window_size + 1)
    assert sequences.shape[1:] == (window_size, 1)


def test_temporal_order():
    """Test temporal sequence order is strictly preserved."""
    data = np.arange(6)
    window_size = 3
    sequences = create_sequences(data, window_size)
    np.testing.assert_array_equal(sequences[0], np.array([[0], [1], [2]]))
    np.testing.assert_array_equal(sequences[-1], np.array([[3], [4], [5]]))


def test_high_level_prepare_data_with_temp_csv(tmp_path):
    """Integration unit test for prepare_data using a temporary CSV file."""
    csv_file = tmp_path / "test_data.csv"
    data_content = (
        "timestamp,value\n"
        "2024-01-01 00:00:00,10.0\n"
        "2024-01-01 00:30:00,20.0\n"
        "2024-01-01 01:00:00,30.0\n"
        "2024-01-01 01:30:00,40.0\n"
        "2024-01-01 02:00:00,50.0\n"
        "2024-01-01 02:30:00,60.0\n"
        "2024-01-01 03:00:00,70.0\n"
        "2024-01-01 03:30:00,80.0\n"
        "2024-01-01 04:00:00,90.0\n"
        "2024-01-01 04:30:00,100.0\n"
    )
    csv_file.write_text(data_content, encoding="utf-8")
    
    # 10 rows, train_ratio=0.8 -> 8 train obs, 2 val obs. window_size=3 -> 8-3+1=6 train seqs, 2-3+1 -> ValueError because 3 > 2!
    # Let's adjust window_size=2 -> 8-2+1=7 train seqs, 2-2+1=1 val seq.
    train_seq, val_seq, scaler = prepare_data(
        str(csv_file), value_column="value", train_ratio=0.8, window_size=2
    )
    assert train_seq.shape == (7, 2, 1)
    assert val_seq.shape == (1, 2, 1)
    assert isinstance(scaler, StandardScaler)
    assert pytest.approx(scaler.mean_[0]) == 45.0  # mean of 10,20,30,40,50,60,70,80 is 45.0


def test_load_data_real_dataset():
    """Test lightweight loading against real raw NYC Taxi CSV."""
    real_csv = "data/raw/nyc_taxi.csv"
    if os.path.exists(real_csv):
        df = load_data(real_csv, "value")
        assert df.shape == (10320, 2)
        assert list(df.columns) == ["timestamp", "value"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
