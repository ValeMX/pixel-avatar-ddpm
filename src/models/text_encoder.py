import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_encoder_block import TextEncoderBlock


class TextEncoder(nn.Module):
    """Text encoder that encodes input text into a latent space."""

    def __init__(
        self,
        vocabulary_size: int,
        hidden_dimension: int,
        heads: int,
        layers: int,
    ):
        """Initialize the text encoder. TODO: default values for the parameters?

        Args:
            vocabulary_size: Size of the vocabulary.
            hidden_dimension: Dimension of the hidden layers.
            heads: Number of attention heads.
            layers: Number of layers in the encoder.
        """
        super(TextEncoder, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.layers = layers

        self.embedding = nn.Embedding(vocabulary_size, hidden_dimension)

        self.encoder_blocks = nn.ModuleList(
            [
                TextEncoderBlock(
                    hidden_dimension=hidden_dimension,
                    num_heads=heads,
                    feedforward_dimension=hidden_dimension * 4,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        """Forward pass of the text encoder.

        Args:
            x: A tensor of shape (batch_size, sequence_length) representing the input text.

        Returns:
            A tensor of shape (batch_size, sequence_length, hidden_dimension) representing the encoded text.
        """
        # 1. Embed the input tokens into a latent space of dimension hidden_dimension
        # Shape before and after this operation:
        # (batch_size, sequence_length)
        x = self.embedding(x)
        # (batch_size, sequence_length, hidden_dimension)

        # 2. Add positional encoding TODO Forse si può calcolare solo una volta nell'init
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        x = x + self.get_positional_encoding(torch.arange(x.size(1), device=x.device))
        # (batch_size, sequence_length, hidden_dimension)

        # 3. Pass the result through a series of encoder blocks
        # Shape before and after this operation:
        # (batch_size, sequence_length, hidden_dimension)
        for block in self.encoder_blocks:
            x = block(x)
        # (batch_size, sequence_length, hidden_dimension)

        return x

    def get_positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal encoding for the given token positions.

        Args:
            positions: A tensor of shape (sequence_length,) representing the token positions.

        Returns:
            A tensor of shape (sequence_length, hidden_dimension) representing the sinusoidal encoding.
        """
        positions = positions.to(torch.float32)
        half = self.hidden_dimension // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=positions.device
            )
            / half
        )
        args = positions[:, None] * freqs[None]
        encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.hidden_dimension % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)

        return encoding


if __name__ == "__main__":
    # Example usage of the TextEncoder
    vocabulary_size = 10000
    hidden_dimension = 64
    heads = 4
    layers = 2

    model = TextEncoder(
        vocabulary_size=vocabulary_size,
        hidden_dimension=hidden_dimension,
        heads=heads,
        layers=layers,
    )

    # Create a random input tensor with shape (batch_size, sequence_length)
    input_tensor = torch.randint(0, vocabulary_size, (2, 12))

    print("Input shape:", input_tensor.shape)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    print("Output shape:", output_tensor.shape)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
