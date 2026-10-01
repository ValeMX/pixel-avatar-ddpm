import torch
import torch.nn as nn
import torch.nn.functional as f


class ResNetBlock(nn.Module):
    """Residual block of the ResNet model.

    This block implements a residual connection by adding the input to the output
    of a sequence of operations including normalization, activation, and convolution.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        time_embedding_dimension: Dimension of the time embedding vector.
        num_groups: Number of groups for group normalization. Default is 32.
        kernel_size: Size of the convolutional kernel. Default is 3.
        padding: Padding added to all four sides of the input. Default is 1.
        bias: If True, adds a learnable bias to the convolutional layers. Default is True.

    Note:
        The forward pass of the ResNet block consists of the following steps:
        1. Normalize and activate the input tensor.
        2. Apply the first convolution to the activated tensor.
        3. Project the time embedding onto the tensor dimension.
        4. Add the time embedding to the tensor.
        5. Normalize and activate the tensor.
        6. Apply the second convolution to the activated tensor.
        7. Add the residual connection to the output tensor.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dimension: int,
        num_groups: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        """Initialize the normalization, activation, and convolution layers."""
        super(ResNetBlock, self).__init__()

        self.silu = nn.SiLU()

        self.normalization_input = nn.GroupNorm(num_groups, input_channels)
        self.normalization_output = nn.GroupNorm(num_groups, output_channels)

        self.time_encoder = nn.Linear(time_embedding_dimension, output_channels)

        self.convolution_input_output = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.convolution_output_output = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # If the input and output channels are different, we need to create a
        # convolutional layer to match the dimensions for the residual connection.
        if input_channels != output_channels:
            self.convolution_residual = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                padding=0,
                bias=bias,
            )
        else:
            self.convolution_residual = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet block.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
                representing the input feature map.
            time_embedding: Input tensor of shape (batch_size, time_embedding_dimension)
                representing the time embedding.

        Returns:
            Output tensor of shape (batch_size, output_channels, height, width)
                representing the output feature map.
        """

        # Preserve the input tensor for the residual connection.
        # Shape: (batch_size, input_channels, height, width)
        residue = x

        # First phase: normalize and activate the input tensor.
        # Shape before and after these operations:
        # (batch_size, input_channels, height, width)
        x = self.normalization_input(x)
        x = self.silu(x)
        # (batch_size, input_channels, height, width)

        # Second phase: apply the first convolution to the activated tensor.
        # Shape before and after this operation:
        # (batch_size, input_channels, height, width)
        x = self.convolution_input_output(x)
        # (batch_size, output_channels, height, width)

        # Third phase: project the time embedding on the feature map dimension.
        # Shape before and after this operation:
        # (batch_size, time_embedding_dimension)
        time_embedding = self.time_encoder(time_embedding)
        time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-1)
        # (batch_size, output_channels, height, width)

        # Fourth phase: add the time embedding to the feature map.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = x + time_embedding
        # (batch_size, output_channels, height, width)

        # Fifth phase: normalize and activate the feature map.
        # Shape before and after these operations:
        # (batch_size, output_channels, height, width)
        x = self.normalization_output(x)
        x = self.silu(x)
        # (batch_size, output_channels, height, width)

        # Sixth phase: apply the second convolution to the activated tensor.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = self.convolution_output_output(x)
        # (batch_size, output_channels, height, width)

        # Seventh phase: add the residual connection to the output tensor.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = x + self.convolution_residual(residue)
        # (batch_size, output_channels, height, width)

        return x
