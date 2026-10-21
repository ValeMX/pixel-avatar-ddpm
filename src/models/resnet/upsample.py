import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):
    """UpSample class that increases the spatial dimensions of the input tensor.

    The UpSample class applies an interpolation operation followed by a convolutional layer
    to increase the spatial dimensions (height and width) of the input tensor.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 2,
        mode: str = "nearest",
    ):
        """Initialize the UpSample module.

        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of channels in the output tensor.
            scale_factor (int): Factor by which to scale the spatial dimensions. Default is 2.
            mode (str): Interpolation mode to use for upsampling. Default is "nearest".
        """
        super(UpSample, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode

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
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, height * scale_factor, width * scale_factor).
        """
        # Apply interpolation to increase the spatial dimensions of the input tensor
        # Shape before and after this operation:
        # (batch_size, input_channels, height, width)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        # (batch_size, input_channels, height * scale_factor, width * scale_factor)

        # Apply convolution to the upsampled tensor to adjust the number of channels
        # and refine the features after upsampling.
        # Shape before and after this operation:
        # (batch_size, input_channels, height * scale_factor, width * scale_factor)
        x = self.convolution(x)
        # (batch_size, output_channels, height * scale_factor, width * scale_factor)

        return x
