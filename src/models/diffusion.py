import torch
import torch.nn as nn

from .resnet import ResNet
from .text_encoder import TextEncoder


class Diffusion(nn.Module):
    """Diffusion class that combines a ResNet for image denoising and a Text Encoder for text conditioning."""

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
            input_channels (int): Number of channels in the input images.
            output_channels (int): Number of channels in the output images.
            base_channels (int): Number of channels in the base feature maps.
            spatial_resolutions (int): Number of spatial resolution levels.
            time_embedding_dimension (int): Dimension of the time embedding.
            text_embedding_dimension (int): Dimension of the text embedding.
            heads (int): Number of attention heads.
            vocabulary_size (int): Size of the vocabulary for text encoding.
            text_encoder_layers (int): Number of layers in the text encoder.
            dropout (float): Dropout rate. Default is 0.1.
            padding_id (int): Padding token ID for text encoding. Default is 0.
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
        """Forward pass of the Diffusion model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width) representing the input noisy images.
            t (torch.Tensor): Time step tensor of shape (batch_size,) representing the time steps.
            text (torch.Tensor): Text tokens indices tensor of shape (batch_size, sequence_length) representing the input text.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, height, width) representing the predicted noise.
        """
        text_mask = text == self.padding_id
        text_embedding = self.text_encoder(text, text_mask)
        return self.resnet(x, t, text_embedding, text_mask)
