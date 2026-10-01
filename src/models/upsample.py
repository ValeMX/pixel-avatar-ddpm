import torch
import torch.nn as nn
import torch.nn.functional as f


class UpSample(nn.Module):
    """Upsample module that increases the spatial dimensions of the input tensor.

    This module applies an interpolation operation followed by a convolutional layer
    to increase the spatial dimensions (height and width) of the input tensor.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        scale_factor: Factor by which to increase the spatial dimensions. Default is 2.
        mode: Interpolation mode to use for upsampling. Default is 'bilinear'.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 2,
        mode: str = "bilinear",
    ):
        """Initialize the UpSample module."""
        super(UpSample, self).__init__()

        self.interpolation = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=True,
        )

        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpSample module.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, output_channels, height*scale_factor, width*scale_factor).
        """
        x = self.interpolation(x)
        x = self.convolution(x)
        return x
