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
        super(TimeEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.SiLU(),
            nn.Linear(output_dimension, output_dimension),
            nn.SiLU(),
        )

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the time encoder.

        Args:
            time_steps: A tensor of shape (batch_size, input_dimension)
                representing the input time steps.

        Returns:
            A tensor of shape (batch_size, output_dimension) representing
            the encoded time steps.
        """
        return self.mlp(time_steps)
