import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoderBlock(nn.Module):
    """Text encoder block that encodes input text via a multi-head self-attention and a feed-forward layer."""

    def __init__(
        self,
        hidden_dimension: int = 64,
        num_heads: int = 4,
        feedforward_dimension: int = 256,
    ):
        """Initialize the text encoder block.

        Args:
            hidden_dimension: Dimension of the input and output.
            num_heads: Number of attention heads.
            feedforward_dimension: Dimension of the feed-forward layer.
        """
        super(TextEncoderBlock, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.feedforward_dimension = feedforward_dimension

        # Multi-head self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dimension,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, feedforward_dimension),
            nn.SiLU(),
            nn.Linear(feedforward_dimension, hidden_dimension),
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dimension)
        self.layer_norm2 = nn.LayerNorm(hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the text encoder block.

        Args:
            x: A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the input text.

        Returns:
            A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the encoded text.
        """
        # 1. Compute multi-head self-attention of the input
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        attention_output, _ = self.self_attention(x, x, x)
        # (batch_size, sequence_length, hidden_dimension)

        # 2. Add the attention output to the input and apply layer normalization
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = self.layer_norm1(x + attention_output)
        # (batch_size, sequence_length, hidden_dimension)

        # 3. Pass the result through the feed-forward layer
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        feedforward_output = self.feed_forward(x)
        # (batch_size, sequence_length, hidden_dimension)

        # 4. Add the feed-forward output to the input and apply layer normalization
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = self.layer_norm2(x + feedforward_output)
        # (batch_size, sequence_length, hidden_dimension)

        return x
