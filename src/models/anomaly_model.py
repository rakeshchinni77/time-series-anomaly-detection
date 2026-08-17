"""
PyTorch LSTM Autoencoder Architecture for Time-Series Anomaly Detection.

The LSTM Autoencoder compresses an input time series sequence into a latent
hidden-state representation, repeats the latent vector across the temporal dimension,
and reconstructs the original input sequence using a decoder LSTM and linear projection.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder neural network for univariate or multivariate time series reconstruction.

    Args:
        input_dim (int): Number of input features per time step.
        hidden_dim (int): Dimension of the LSTM hidden state and latent space.
        num_layers (int, optional): Number of stacked LSTM layers. Defaults to 1.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(LSTMAutoencoder, self).__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Final output Linear projection layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time-series reconstruction.

        Args:
            x (torch.Tensor): 3D input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Reconstructed 3D tensor of shape (batch_size, seq_len, input_dim).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seq_len, features), got shape {tuple(x.shape)}")

        batch_size, seq_len, _ = x.size()

        # Encoder: compress sequence into hidden state
        _, (hidden, _) = self.encoder(x)

        # Extract final hidden state of the top layer
        latent = hidden[-1]  # shape: (batch_size, hidden_dim)

        # Repeat latent representation across the sequence dimension using .repeat()
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)  # shape: (batch_size, seq_len, hidden_dim)

        # Decoder: reconstruct sequence from repeated latent representation
        decoder_output, _ = self.decoder(decoder_input)  # shape: (batch_size, seq_len, hidden_dim)

        # Linear projection back to original feature dimension
        reconstruction = self.output_layer(decoder_output)  # shape: (batch_size, seq_len, input_dim)

        return reconstruction
