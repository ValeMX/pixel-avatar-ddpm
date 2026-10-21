import torch.nn as nn

from .residual_block import ResidualBlock
from .attention_block import AttentionBlock


class ResNetLayer(nn.Sequential):
    """ResNet Layer class that processes input tensors through a series of ResNet blocks, Attention blocks,
        Downsample layers, and Upsample layers.

    This layer is a nn.Sequential child class designed to process input tensors through a series of transformations and
    pass time embeddings to the ResNet blocks and text embeddings to the Attention blocks.
    """

    def forward(self, input, *args):
        """Forward pass through the ResNet layer.

        Args:
            input: Input tensor of shape (batch_size, channels, height, width).
            *args: Time embedding, text embedding and text mask (optional).

        Returns:
            Output tensor after processing through the ResNet layer.
        """
        time_embedding = args[0]
        text_embedding = args[1]
        text_mask = args[2] if len(args) > 2 else None
        x = input
        for block in self:
            if isinstance(block, ResidualBlock):
                x = block(x, time_embedding)
            elif isinstance(block, AttentionBlock):
                x = block(x, text_embedding, text_mask)
            else:
                x = block(x)
        return x
