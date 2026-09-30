import torch
import torch.nn as nn
import torch.nn.functional as f

from resnet_block import ResNetBlock
from time_encoder import TimeEncoder

class ResNet(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 64,
        spatial_resolutions: int = 2,
    ):
        """Initialize the ResNet model.

        Args:
            image_size: Spatial size of the input images.
            input_channels: Number of channels in the input images.
            output_channels: Number of channels in the output images.
            base_channels: Number of channels in the base feature maps.
            spatial_resolutions: Number of spatial resolution levels.
        """
        super(ResNet, self).__init__()

        # Hyperparameters initialization
        self.image_size = image_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.spatial_resolutions = spatial_resolutions

        # Model architecture initialization
        self.residual_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    input_channels=base_channels,
                    output_channels=base_channels,
                    time_embedding_dimension=128,
                    num_groups=8,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                for i in range(spatial_resolutions)
            ]
        )
