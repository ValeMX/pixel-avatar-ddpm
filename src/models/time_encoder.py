import math

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Time encoder that encodes time steps into a higher-dimensional space.

    The time encoder uses a multi-layer perceptron (MLP) to transform
    the input time steps into a higher-dimensional representation.
    The MLP consists of two linear layers with SiLU activation functions in between.

    Args:
        input_dimension: Dimension of the input time steps.
        output_dimension: Dimension of the output encoded time steps.
    """

    def __init__(self, input_dimension: int, output_dimension: int):
        """Initialize the time encoder.

        Args:
            input_dimension: Dimension of the input time steps.
            output_dimension: Dimension of the output encoded time steps.
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        super(TimeEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.SiLU(),
            nn.Linear(output_dimension, output_dimension),
            nn.SiLU(),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the time encoder.

        Args:
            timesteps: A tensor of shape (batch_size, input_dimension)
                representing the input time steps.

        Returns:
            A tensor of shape (batch_size, output_dimension) representing
            the encoded time steps.

        Raises:
            ValueError: If the input timesteps tensor is not 1D.
        """
        if len(timesteps.shape) != 1:
            raise ValueError("Expected timestep to be a 1D tensor.")

        encoded_timesteps = self.get_positional_encoding(timesteps)

        return self.mlp(encoded_timesteps)

    def get_positional_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal encoding for the given time steps.

        Args:
            timesteps: A tensor of shape (batch_size,) representing the time steps.

        Returns:
            A tensor of shape (batch_size, input_dimension) representing the sinusoidal encoding.
        """
        timesteps = timesteps.to(torch.float32)
        half = self.input_dimension // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None] * freqs[None]
        encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.input_dimension % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)

        return encoding
