import torch.nn as nn
import torch.nn.functional as f

from .residual_block import ResidualBlock
from .attention_block import AttentionBlock


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
            *args: Time embedding and text embedding.

        Returns:
            Output tensor after processing through the ResNet layer.
        """
        time_embedding = args[0]
        text_embedding = args[1]
        x = input
        for block in self:
            if isinstance(block, ResidualBlock):
                x = block(x, time_embedding)
            elif isinstance(block, AttentionBlock):
                x = block(x, text_embedding)
            else:
                x = block(x)
        return x
