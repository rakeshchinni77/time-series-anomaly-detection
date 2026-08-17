"""
Unit tests for PyTorch LSTM Autoencoder architecture (src/models/anomaly_model.py).
"""

import pytest
import torch
import torch.nn as nn

from src.models.anomaly_model import LSTMAutoencoder


def test_class_inheritance():
    """Test that LSTMAutoencoder inherits from torch.nn.Module."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    assert isinstance(model, nn.Module)


def test_encoder_exists_and_type():
    """Test that encoder module exists and is nn.LSTM."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    assert hasattr(model, "encoder")
    assert isinstance(model.encoder, nn.LSTM)


def test_decoder_exists_and_type():
    """Test that decoder module exists and is nn.LSTM."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    assert hasattr(model, "decoder")
    assert isinstance(model.decoder, nn.LSTM)


def test_encoder_batch_first():
    """Test that encoder uses batch_first=True."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    assert model.encoder.batch_first is True


def test_decoder_batch_first():
    """Test that decoder uses batch_first=True."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    assert model.decoder.batch_first is True


def test_forward_accepts_3d_input():
    """Test that forward pass accepts 3D tensor (batch, seq_len, features)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert output.ndim == 3


def test_output_shape_equals_input_shape():
    """Test that output shape exactly matches input shape (8, 10, 1)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert output.shape == x.shape == torch.Size([8, 10, 1])


def test_batch_processing():
    """Test batch processing with batch size 16."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    x = torch.randn(16, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (16, 10, 1)


def test_multiple_batch_sizes():
    """Test that model accepts variable batch sizes (1, 4, 8, 32)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    for batch_size in [1, 4, 8, 32]:
        x = torch.randn(batch_size, 10, 1)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 10, 1)


def test_dynamic_sequence_length():
    """Test that model dynamically handles different sequence lengths (5, 10, 20, 30)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    for seq_len in [5, 10, 20, 30]:
        x = torch.randn(8, seq_len, 1)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (8, seq_len, 1)


def test_multi_layer_model():
    """Test multi-layer LSTMAutoencoder configuration (num_layers=2)."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=32, num_layers=2)
    assert model.encoder.num_layers == 2
    assert model.decoder.num_layers == 2
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert output.shape == x.shape == torch.Size([8, 10, 1])


def test_hidden_dimension_configuration():
    """Test different hidden_dim configurations (16, 32, 64)."""
    for hidden_dim in [16, 32, 64]:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=hidden_dim, num_layers=1)
        x = torch.randn(4, 10, 1)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 10, 1)


def test_input_dimension_configuration():
    """Test parameterization for input_dim > 1."""
    model = LSTMAutoencoder(input_dim=3, hidden_dim=32, num_layers=1)
    x = torch.randn(4, 10, 3)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (4, 10, 3)


def test_gradient_flow():
    """Test autograd compatibility and gradient computation."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=32, num_layers=1)
    x = torch.randn(4, 10, 1, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None


def test_model_parameters_exist():
    """Test that model has trainable parameters."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    params = list(model.parameters())
    assert len(params) > 0
    assert any(p.requires_grad for p in params)


def test_output_feature_dimension():
    """Test final Linear layer reconstructs exact feature dimension."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert output.shape[-1] == 1


def test_no_nan_or_infinity_output():
    """Test forward pass produces finite values without NaN or Inf."""
    model = LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=1)
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        output = model(x)
    assert torch.isfinite(output).all()


def test_constructor_invalid_parameters():
    """Test constructor raises ValueError on invalid architectural dimensions."""
    with pytest.raises(ValueError):
        LSTMAutoencoder(input_dim=0, hidden_dim=64, num_layers=1)
    with pytest.raises(ValueError):
        LSTMAutoencoder(input_dim=1, hidden_dim=0, num_layers=1)
    with pytest.raises(ValueError):
        LSTMAutoencoder(input_dim=1, hidden_dim=64, num_layers=0)
