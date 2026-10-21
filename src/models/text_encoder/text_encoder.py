import math
import torch
import torch.nn as nn

from .text_encoder_block import TextEncoderBlock


class TextEncoder(nn.Module):
    """Text Encoder class that encodes input text into a latent space."""

    def __init__(
        self,
        vocabulary_size: int,
        hidden_dimension: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize the Text Encoder.

        Args:
            vocabulary_size: Size of the vocabulary.
            hidden_dimension: Dimension of the hidden layers. Default is 64.
            heads: Number of attention heads. Default is 4.
            layers: Number of layers in the encoder. Default is 2.
            dropout: Dropout rate. Default is 0.1.
        """
        super(TextEncoder, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.layers = layers

        self.embedding = nn.Embedding(vocabulary_size, hidden_dimension)
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                TextEncoderBlock(
                    hidden_dimension=hidden_dimension,
                    num_heads=heads,
                    feedforward_dimension=hidden_dimension * 4,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the Text Encoder.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length) representing the input text tokens indices.
            padding_mask (torch.Tensor, optional): A tensor of shape (batch_size, sequence_length) representing the padding mask.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the encoded text.
        """
        # 1. Embed the input tokens into a latent space of dimension hidden_dimension
        # Shape before and after this operation:
        # (batch_size, sequence_length)
        x = self.embedding(x)
        # (batch_size, sequence_length, hidden_dimension)

        # 2. Add positional encoding and apply dropout to the embedded tokens
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = x + self.get_positional_encoding(torch.arange(x.size(1), device=x.device))
        x = self.dropout(x)
        # (batch_size, sequence_length, hidden_dimension)

        # 3. Pass the result through a series of encoder blocks
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        for block in self.encoder_blocks:
            x = block(x, padding_mask)
        # (batch_size, sequence_length, hidden_dimension)

        return x

    def get_positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal encoding for the given token positions.

        Args:
            positions (torch.Tensor): A tensor of shape (sequence_length,) representing the token positions.

        Returns:
            torch.Tensor: A tensor of shape (sequence_length, hidden_dimension) representing the sinusoidal encoding.
        """
        positions = positions.to(torch.float32)

        # Compute the frequencies for the sinusoidal encoding
        half = self.hidden_dimension // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=positions.device
            )
            / half
        )

        # Compute the sinusoidal encoding using sine and cosine functions
        args = positions[:, None] * freqs[None]
        encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If the hidden dimension is odd, append a zero vector to the encoding
        if self.hidden_dimension % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)

        return encoding
