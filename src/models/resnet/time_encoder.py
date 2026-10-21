import math

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Time Encoder class that encodes time steps into a higher-dimensional space.

    The Time Encoder uses sinusoidal positional encoding to represent time steps
    and then passes the encoded representation through a multi-layer perceptron (MLP)
    to produce the final output.
    """

    def __init__(self, input_dimension: int, output_dimension: int):
        """Initialize the Time Encoder module.

        Args:
            input_dimension (int): Dimension of the input time steps.
            output_dimension (int): Dimension of the output encoded time steps.
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
        """Forward pass of the Time Encoder.

        Args:
            timesteps (torch.Tensor): A tensor of shape (batch_size,) representing the input time steps.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, output_dimension) representing the encoded time steps.

        Raises:
            ValueError: If the input timesteps tensor is not 1D.
        """
        if len(timesteps.shape) != 1:
            raise ValueError("Expected timesteps to be a 1D tensor.")

        encoded_timesteps = self.get_positional_encoding(timesteps)

        return self.mlp(encoded_timesteps)

    def get_positional_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal encoding for the given time steps.

        Args:
            timesteps (torch.Tensor): A tensor of shape (batch_size,) representing the time steps.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, input_dimension) representing the sinusoidal encoding.
        """
        timesteps = timesteps.to(torch.float32)

        # Compute the frequencies for the sinusoidal encoding
        half = self.input_dimension // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )

        # Generate the sinusoidal encoding using the sine and cosine functions
        args = timesteps[:, None] * freqs[None]
        encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If the input dimension is odd, append a zero vector to the encoding
        if self.input_dimension % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)

        return encoding
