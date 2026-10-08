import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention block of the ResNet that applies self-attention and cross-attention to the input.

    This block consists of a series of operations that include normalization, convolution, self-attention,
    cross-attention with a text embedding, and a feed-forward neural network. The block also includes
    residual connections to preserve the input information throughout the operations.

    Note:
        The forward pass of the Attention block consists of the following steps:
        1. Normalize and convolve the input tensor to generate the query tensor.
        2. Reshape the query tensor for attention computation.
        3. Compute self-attention on the query tensor.
        4. Compute cross-attention with the text embedding.
        5. Apply a feed-forward neural network to the cross-attention output.
        6. Reshape the output tensor back to the original shape of the input tensor.
    """

    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 32,
        text_embedding_dimension: int = 64,
        num_heads: int = 4,
        feedforward_dimension: int = 256,
    ):
        """Initialize the attention block.

        Args:
            channels: Number of channels in the input feature maps.
            num_groups: Number of groups for group normalization. Default is 32.
            text_embedding_dimension: Dimension of the text embedding. Default is 64.
            num_heads: Number of attention heads. Default is 4.
            feedforward_dimension: Dimension of the feed-forward neural network. Default is 256.
        """
        super(AttentionBlock, self).__init__()
        self.convolution_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.groupnorm = nn.GroupNorm(num_groups, channels)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.text_projection = nn.Linear(text_embedding_dimension, channels)

        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.layernorm3 = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, feedforward_dimension),
            nn.SiLU(),
            nn.Linear(feedforward_dimension, channels),
        )

        self.convolution_output = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:

        # Preserve the input tensor for the final residual connection.
        # Shape: (batch_size, channels, height, width)
        residue_final = x

        # 1. Apply the normalization and convolution to the input tensor to generate the query tensor.
        # Shape before and after this operation:
        # (batch_size, channels, height, width)
        x = self.groupnorm(x)
        x = self.convolution_input(x)

        # 2. Reshape the query tensor to prepare for attention computation.
        # Note: In order to apply the attention mechanism, we need to reshape the query tensor
        # to have the shape (batch_size, sequence_length, channels), where sequence_length = height * width.
        # Shape before and after this operation:
        # (batch_size, channels, height, width)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).permute(0, 2, 1)
        # (batch_size, sequence_length, channels)

        # 3. Save the query tensor for the residual connection after self-attention computation.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        residue_attention = x
        # (batch_size, sequence_length, channels)

        # 4. Normalize and compute the self-attention of the query tensor.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = self.layernorm1(x)
        x, _ = self.attention1(x, x, x)
        # (batch_size, sequence_length, channels)

        # 5. Add the residual connection from the query tensor to the self-attention output.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = x + residue_attention
        # (batch_size, sequence_length, channels)

        # 6. Save the self-attention output for the residual connection after cross-attention computation.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        residue_attention = x
        # (batch_size, sequence_length, channels)

        # 7. Project the text embedding to match the channel dimension of the query tensor.
        # Shape before and after this operation:
        # (batch_size, sequence_length, text_embedding_dimension)
        text_embedding = self.text_projection(text_embedding)
        # (batch_size, sequence_length, channels)

        # 8. Normalize and compute the cross-attention of the self-attention output with the text embedding.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = self.layernorm2(x)
        x, _ = self.attention2(x, text_embedding, text_embedding)
        # (batch_size, sequence_length, channels)

        # 9. Add the residual connection from the self-attention output to the cross-attention output.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = x + residue_attention
        # (batch_size, sequence_length, channels)

        # 10. Save the cross-attention output for the residual connection after the feed-forward neural network.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        residue_attention = x
        # (batch_size, sequence_length, channels)

        # 11. Normalize and apply the feed-forward neural network to the cross-attention output.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = self.layernorm3(x)
        x = self.feedforward(x)
        # (batch_size, sequence_length, channels)

        # 12. Add the residual connection from the cross-attention output to the feed-forward neural network output.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = x + residue_attention
        # (batch_size, sequence_length, channels)

        # 13. Reshape the output tensor back to the original shape of the input tensor.
        # Shape before and after this operation:
        # (batch_size, sequence_length, channels)
        x = x.permute(0, 2, 1).view(n, c, h, w)
        # (batch_size, channels, height, width)

        # 14. Apply the final convolution to the output tensor.
        # Shape before and after this operation:
        # (batch_size, channels, height, width)
        x = self.convolution_output(x)
        # (batch_size, channels, height, width)

        # 15. Add the final residual connection from the input tensor to the output tensor.
        # Shape before and after this operation:
        # (batch_size, channels, height, width)
        x = x + residue_final
        # (batch_size, channels, height, width)

        return x
