import math

import torch
import torch.nn as nn
import torch.nn.functional as f

from downsample import DownSample
from resnet_block import ResNetBlock
from resnet_layer import ResNetLayer
from time_encoder import TimeEncoder
from upsample import UpSample


class ResNet(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        spatial_resolutions: int = 2,
        time_embedding_dimension: int = 128,
    ):
        """Initialize the ResNet model.

        Args:
            image_size: Spatial size of the input images. Default is 32.
            input_channels: Number of channels in the input images. Default is 3.
            output_channels: Number of channels in the output images. Default is 3.
            base_channels: Number of channels in the base feature maps. Default is 64.
            spatial_resolutions: Number of spatial resolution levels. Default is 2.
            time_embedding_dimension: Dimension of the time embedding. Default is 128.
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
        for i in range(spatial_resolutions):
            in_ch = base_channels * (2**i)
            out_ch = base_channels * (2 ** (i + 1))

            self.encoder_layers.append(
                ResNetLayer(
                    ResNetBlock(
                        input_channels=in_ch,
                        output_channels=out_ch,
                        time_embedding_dimension=time_embedding_dimension,
                    ),
                    DownSample(
                        input_channels=out_ch,
                        output_channels=out_ch,
                        scale_factor=2,
                    ),
                )
            )

        # Middle block is a ResNet block that processes the features at the lowest spatial resolution
        mid_ch = base_channels * (2**spatial_resolutions)
        self.middle_block = ResNetBlock(
            input_channels=mid_ch,
            output_channels=mid_ch,
            time_embedding_dimension=time_embedding_dimension,
        )

        # Decoding is done by a series of ResNet blocks followed by upsampling layers
        # Note: The input channels for the decoder layers are doubled because
        # of the skip connections from the encoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(spatial_resolutions - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 2))
            out_ch = base_channels * (2**i)

            self.decoder_layers.append(
                ResNetLayer(
                    ResNetBlock(
                        input_channels=in_ch,
                        output_channels=out_ch,
                        time_embedding_dimension=time_embedding_dimension,
                    ),
                    UpSample(
                        input_channels=out_ch,
                        output_channels=out_ch,
                        scale_factor=2,
                    ),
                )
            )

        # Final convolution layer to map the features to the desired output channels
        self.final_convolution = nn.Conv2d(
            base_channels, output_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Generate the time embedding from the time step
        time_embedding = self.time_encoder(timestep)

        # Initial convolution to map the input image to the base channels
        x = self.initial_convolution(x)

        # Encoder: pass through the ResNet blocks and downsampling layers, storing skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x, time_embedding)
            skip_connections.append(x)

        # Middle block: process the features at the lowest spatial resolution
        x = self.middle_block(x, time_embedding)

        # Decoder: pass through the ResNet blocks and upsampling layers, using skip connections from the encoder
        for layer in self.decoder_layers:
            # Concatenate the skip connection with the current features
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = layer(x, time_embedding)

        # Final convolution to map the features to the desired output channels
        x = self.final_convolution(x)
        return x


if __name__ == "__main__":
    # Example usage of the ResNet model
    model = ResNet(
        image_size=32,
        input_channels=3,
        output_channels=3,
        base_channels=64,
        spatial_resolutions=2,
        time_embedding_dimension=256,
    )

    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(2, 3, 32, 32)
    timesteps_tensor = torch.tensor([3, 7])

    # Forward pass through the model
    output_tensor = model(input_tensor, timesteps_tensor)

    # Print the output shape and model summary
    print("Output shape:", output_tensor.shape)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
