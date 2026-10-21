import torch
import torch.nn as nn
import torch.nn.functional as f


class ResidualBlock(nn.Module):
    """Residual Block class of the ResNet model.

    This block implements a residual connection by adding the input to the output
    of a sequence of operations including normalizations, activations, and convolutions.

    Note:
        The forward pass of the Residual block consists of the following steps:
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
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize the Residual Block.

        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of channels in the output tensor.
            time_embedding_dimension (int): Dimension of the time embedding.
            num_groups (int): Number of groups for Group Normalization. Default is 32.
            kernel_size (int): Size of the convolutional kernel. Default is 3.
            padding (int): Padding for the convolutional layers. Default is 1.
            dropout (float): Dropout rate. Default is 0.1.
            bias (bool): Whether to include a bias term in the convolutional layers. Default is True.
        """
        super(ResidualBlock, self).__init__()

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

        self.dropout = nn.Dropout(dropout)

        self.convolution_output_output = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # If the input and output channels are different, it is necessary
        # to apply a 1x1 convolution to the input tensor to match the output channels
        # before adding it to the output tensor in the residual connection.
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

    def forward(self, x: torch.Tensor, time_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width) representing the input feature map.
            time_embeddings (torch.Tensor): Input tensor of shape (batch_size, time_embedding_dimension) representing the time embedding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, height, width) representing the output feature map.
        """
        # Preserve the input tensor for the residual connection.
        # Shape: (batch_size, input_channels, height, width)
        residue = x

        # 1. Normalize and activate the input tensor.
        # Shape before and after these operations:
        # (batch_size, input_channels, height, width)
        x = self.normalization_input(x)
        x = self.silu(x)
        # (batch_size, input_channels, height, width)

        # 2. Apply the first convolution to the activated tensor.
        # Note: This operation changes the number of channels from input_channels to output_channels.
        # Shape before and after this operation:
        # (batch_size, input_channels, height, width)
        x = self.convolution_input_output(x)
        # (batch_size, output_channels, height, width)

        # 3. Project the time embedding on the feature map dimension.
        # Shape before and after this operation:
        # (batch_size, time_embedding_dimension)
        t_emb = self.time_encoder(time_embeddings)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        # (batch_size, output_channels, height, width)

        # 4. Add the time embedding to the feature map via broadcasting.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = x + t_emb
        # (batch_size, output_channels, height, width)

        # 5. Normalize and activate the feature map.
        # Shape before and after these operations:
        # (batch_size, output_channels, height, width)
        x = self.normalization_output(x)
        x = self.silu(x)
        # (batch_size, output_channels, height, width)

        # 6. Apply dropout to the activated tensor.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = self.dropout(x)
        # (batch_size, output_channels, height, width)

        # 7. Apply the second convolution to the activated tensor.
        # Note: This operation maintains the number of channels as output_channels and refines the feature map.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = self.convolution_output_output(x)
        # (batch_size, output_channels, height, width)

        # 8. Add the residual connection to the output tensor.
        # Note: If the input and output channels are different, the input tensor is projected to match the output channels.
        # Shape before and after this operation:
        # (batch_size, output_channels, height, width)
        x = x + self.convolution_residual(residue)
        # (batch_size, output_channels, height, width)

        return x
