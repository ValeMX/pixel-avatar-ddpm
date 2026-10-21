import torch
import torch.nn as nn
import torch.nn.functional as f


class DownSample(nn.Module):
    """DownSample class that reduces the spatial dimensions of the input tensor.

    The DownSample class applies a convolutional layer with a stride greater than 1
    to reduce the spatial dimensions (height and width) of the input tensor.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 2,
    ):
        """Initialize the DownSample module.

        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of channels in the output tensor.
            scale_factor (int): Factor by which to reduce the spatial dimensions. Default is 2.
        """
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
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, height / scale_factor, width / scale_factor).
        """
        return self.convolution(x)
