import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .text_encoder import TextEncoder


class Diffusion(nn.Module):
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
    ):
        super(Diffusion, self).__init__()

        self.resnet = ResNet(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            spatial_resolutions=spatial_resolutions,
            time_embedding_dimension=time_embedding_dimension,
            text_embedding_dimension=text_embedding_dimension,
            heads=heads,
        )

        self.text_encoder = TextEncoder(
            vocabulary_size=vocabulary_size,
            hidden_dimension=text_embedding_dimension,
            heads=heads,
            layers=text_encoder_layers,
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
        text_embedding = self.text_encoder(text)
        return self.resnet(x, t, text_embedding)
