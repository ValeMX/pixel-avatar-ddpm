import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_block import AttentionBlock
from .downsample import DownSample
from .residual_block import ResidualBlock
from .resnet_layer import ResNetLayer
from .time_encoder import TimeEncoder
from .upsample import UpSample


class ResNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        spatial_resolutions: int = 2,
        time_embedding_dimension: int = 128,
        text_embedding_dimension: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize the ResNet model.

        Args:
            input_channels: Number of channels in the input images. Default is 3.
            output_channels: Number of channels in the output images. Default is 3.
            base_channels: Number of channels in the base feature maps. Default is 64.
            spatial_resolutions: Number of spatial resolution levels. Default is 2.
            time_embedding_dimension: Dimension of the time embedding. Default is 128.
            text_embedding_dimension: Dimension of the text embedding. Default is 64.
            heads: Number of attention heads. Default is 4.
            dropout: Dropout rate. Default is 0.1.
        """
        super(ResNet, self).__init__()

        # Time embedding is generated using a linear layer that maps the time step to a higher-dimensional space
        self.time_encoder = TimeEncoder(base_channels, time_embedding_dimension)

        # Initial convolution layer to map the input image to the base feature maps
        self.initial_convolution = nn.Conv2d(
            input_channels, base_channels, kernel_size=3, padding=1
        )

        # Encoding is done by a series of ResNet blocks followed by downsampling layers
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(spatial_resolutions):
            in_ch, out_ch = base_channels * (2**i), base_channels * (2 ** (i + 1))
            # TODO provare a fare 2 res-att prima di downsample
            self.encoder_layers.append(
                ResNetLayer(
                    ResidualBlock(
                        input_channels=in_ch,
                        output_channels=out_ch,
                        time_embedding_dimension=time_embedding_dimension,
                        dropout=dropout,
                    ),
                    AttentionBlock(
                        channels=out_ch,
                        text_embedding_dimension=text_embedding_dimension,
                        num_heads=heads,
                        feedforward_dimension=out_ch * 4,
                    ),
                )
            )

            self.downsample_layers.append(
                DownSample(
                    input_channels=out_ch,
                    output_channels=out_ch,
                    scale_factor=2,
                )
            )

        # Middle block is a ResNet block that processes the features at the lowest spatial resolution
        mid_ch = base_channels * (2**spatial_resolutions)
        self.middle_block = ResNetLayer(
            ResidualBlock(
                input_channels=mid_ch,
                output_channels=mid_ch,
                time_embedding_dimension=time_embedding_dimension,
                dropout=dropout,
            ),
            AttentionBlock(
                channels=mid_ch,
                text_embedding_dimension=text_embedding_dimension,
                num_heads=heads,
                feedforward_dimension=mid_ch * 4,
            ),
            ResidualBlock(
                input_channels=mid_ch,
                output_channels=mid_ch,
                time_embedding_dimension=time_embedding_dimension,
                dropout=dropout,
            ),
        )

        # Decoding is done by a series of ResNet blocks followed by upsampling layers
        # Note: The input channels for the decoder layers are doubled because
        # of the skip connections from the encoder layers
        self.upsample_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(spatial_resolutions - 1, -1, -1):
            up_ch = base_channels * (2 ** (i + 1))
            in_ch, out_ch = base_channels * (2 ** (i + 2)), base_channels * (2**i)

            self.upsample_layers.append(
                UpSample(input_channels=up_ch, output_channels=up_ch, scale_factor=2)
            )

            self.decoder_layers.append(
                ResNetLayer(
                    ResidualBlock(
                        input_channels=in_ch,
                        output_channels=out_ch,
                        time_embedding_dimension=time_embedding_dimension,
                        dropout=dropout,
                    ),
                    AttentionBlock(
                        channels=out_ch,
                        text_embedding_dimension=text_embedding_dimension,
                        num_heads=heads,
                        feedforward_dimension=out_ch * 4,
                    ),
                )
            )

        # Final convolution layer to map the features to the desired output channels
        self.final_convolution = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Generate the time embedding from the time step
        time_embedding = self.time_encoder(timestep)

        # Initial convolution to map the input image to the base channels
        x = self.initial_convolution(x)

        # Encoder: pass through the ResNet blocks and downsampling layers, storing skip connections
        skips = []
        for layer, downsample in zip(self.encoder_layers, self.downsample_layers):
            x = layer(x, time_embedding, text_embedding)
            skips.append(x)
            x = downsample(x)
           
        # Middle block: process the features at the lowest spatial resolution
        x = self.middle_block(x, time_embedding, text_embedding)

        # Decoder: pass through the ResNet blocks and upsampling layers, using skip connections from the encoder
        for layer, upsample in zip(self.decoder_layers, self.upsample_layers):
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_embedding, text_embedding)

        # Final convolution to map the features to the desired output channels
        x = self.final_convolution(x)
        return x


if __name__ == "__main__":
    # Example usage of the ResNet model
    model = ResNet(
        input_channels=3,
        output_channels=3,
        base_channels=64,
        spatial_resolutions=2,
        time_embedding_dimension=256,
        text_embedding_dimension=64,
        heads=4,
    )

    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(2, 3, 32, 32)
    timesteps_tensor = torch.tensor([3, 7])
    text_embeddings = torch.randn(2, 12, 64)

    # Forward pass through the model
    output_tensor = model(input_tensor, timesteps_tensor, text_embeddings)

    # Print the output shape and model summary
    print("Output shape:", output_tensor.shape)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
