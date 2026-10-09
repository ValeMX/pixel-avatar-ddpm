import torch
import torch.nn as nn
import torch.nn.functional as f


class DownSample(nn.Module):
    """Donwsample module that reduces the spatial dimensions of the input tensor.

    This module applies a convolutional layer with a specified stride to reduce the
    spatial dimensions (height and width) of the input tensor.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        kernel_size: Size of the convolutional kernel. Default is 3.
        stride: Stride of the convolution. Default is 2.
        padding: Padding added to all four sides of the input. Default is 1.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 2,
    ):
        """Initialize the DownSample module."""
        super(DownSample, self).__init__()
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=scale_factor,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DownSample module.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, output_channels, height/scale_factor, width/scale_factor).
        """
        return self.convolution(x)
