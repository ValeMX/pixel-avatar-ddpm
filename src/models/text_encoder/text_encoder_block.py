import torch
import torch.nn as nn


class TextEncoderBlock(nn.Module):
    """Text Encoder Block class that encodes input text via a multi-head self-attention layer
        and a feed-forward layer with residual connections.
    """

    def __init__(
        self,
        hidden_dimension: int = 64,
        num_heads: int = 4,
        feedforward_dimension: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize the Text Encoder Block.

        Args:
            hidden_dimension (int): Dimension of the input and output. Default is 64.
            num_heads (int): Number of attention heads. Default is 4.
            feedforward_dimension (int): Dimension of the feed-forward layer. Default is 256.
            dropout (float): Dropout rate. Default is 0.1.
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
        )

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, feedforward_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, hidden_dimension),
        )

        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(hidden_dimension)
        self.layer_norm2 = nn.LayerNorm(hidden_dimension)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the Text Encoder Block.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the input text.
            padding_mask (torch.Tensor, optional): A tensor of shape (batch_size, sequence_length) representing the padding mask. Default is None.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the encoded text.
        """
        # 1. Compute multi-head self-attention of the input
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        attention_output, _ = self.self_attention(x, x, x, need_weights=False)
        # (batch_size, sequence_length, hidden_dimension)

        # 2. Add the attention output to its input and apply layer normalization
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = self.layer_norm1(x + self.dropout(attention_output))
        # (batch_size, sequence_length, hidden_dimension)

        # 3. Pass the result through the feed-forward layer
        # Note: The feed-forward layer projects the input to a higher-dimensional space feedforward_dimension
        #       and then back to the original dimension hidden_dimension.
        #       This allows the model to learn more complex representations.
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        feedforward_output = self.feed_forward(x)
        # (batch_size, sequence_length, hidden_dimension)

        # 4. Add the feed-forward output to its input and apply layer normalization
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = self.layer_norm2(x + self.dropout(feedforward_output))
        # (batch_size, sequence_length, hidden_dimension)

        return x
