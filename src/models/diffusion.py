import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .text_encoder import TextEncoder


class Diffusion(nn.Module):
    """Diffusion model that combines a ResNet for image processing and a TextEncoder for text processing."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int,
        spatial_resolutions: int,
        time_embedding_dimension: int,
        text_embedding_dimension: int,
        heads: int,
        vocabulary_size: int,
        text_encoder_layers: int,
        dropout: float = 0.1,
        padding_id: int = 0,
    ):
        """Initialize the Diffusion model.

        Args:
            input_channels: Number of channels in the input images.
            output_channels: Number of channels in the output images.
            base_channels: Number of channels in the base feature maps.
            spatial_resolutions: Number of spatial resolution levels.
            time_embedding_dimension: Dimension of the time embedding.
            text_embedding_dimension: Dimension of the text embedding.
            heads: Number of attention heads.
            vocabulary_size: Size of the vocabulary for text encoding.
            text_encoder_layers: Number of layers in the text encoder.
            dropout: Dropout rate. Default is 0.1.
            padding_id: Padding token ID for text encoding. Default is 0.
        """
        super(Diffusion, self).__init__()

        self.padding_id = padding_id

        self.resnet = ResNet(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            spatial_resolutions=spatial_resolutions,
            time_embedding_dimension=time_embedding_dimension,
            text_embedding_dimension=text_embedding_dimension,
            heads=heads,
            dropout=dropout,
        )

        self.text_encoder = TextEncoder(
            vocabulary_size=vocabulary_size,
            hidden_dimension=text_embedding_dimension,
            heads=heads,
            layers=text_encoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the diffusion model.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
            t: Time step tensor of shape (batch_size, 1).
            text: Text tensor of shape (batch_size, sequence_length).

        Returns:
            Output tensor of shape (batch_size, output_channels, height, width).
        """
        text_mask = text == self.padding_id
        print(f"Text mask shape: {text_mask.shape}, Text mask sum: {text_mask.sum().item()}")
        text_embedding = self.text_encoder(text, text_mask)
        print(f"Text embedding shape: {text_embedding.shape}")
        return self.resnet(x, t, text_embedding, text_mask)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage of the Diffusion model
    diffusion_model = Diffusion(
        input_channels=3,
        output_channels=3,
        base_channels=32,
        spatial_resolutions=2,
        time_embedding_dimension=16,
        text_embedding_dimension=8,
        heads=4,
        vocabulary_size=10000,
        text_encoder_layers=2,
    ).to(device)

    # Create dummy inputs
    batch_size = 32
    height, width = 32, 32
    sequence_length = 70

    x = torch.randn(batch_size, 3, height, width).to(device)  # Input images
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # Time steps
    text = torch.randint(0, 10000, (batch_size, sequence_length)).to(device)  # Text input

    # Forward pass
    output = diffusion_model(x, t, text)
    print("Output shape:", output.shape)
