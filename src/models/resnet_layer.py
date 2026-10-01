import torch
import torch.nn as nn
import torch.nn.functional as f

from resnet_block import ResNetBlock
from upsample import UpSample
from downsample import DownSample


class ResNetLayer(nn.Sequential):
    """ResNet layer that consists of a sequence of blocks.

    This layer can contain ResNet blocks, downsampling layers, and upsampling layers.
    It is designed to process input tensors through a series of transformations and
    pass time embeddings through the network.

    """

    def forward(self, input, *args):
        """Forward pass through the ResNet layer.

        Args:
            input: Input tensor of shape (batch_size, channels, height, width).
            *args: Additional arguments, including the time embedding.

        Returns:
            Output tensor after processing through the ResNet layer.
        """
        time_embedding = args[0]
        x = input
        for block in self:
            if isinstance(block, ResNetBlock):
                x = block(x, time_embedding)
            else:
                x = block(x)
        return x
