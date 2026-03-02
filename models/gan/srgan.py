"""SRGAN — Photo-Realistic Single Image Super-Resolution Using a GAN.

Ledig et al., CVPR 2017.

Generator: ResNet-based backbone with pixel-shuffle upsampling.
Discriminator: VGGDiscriminator (accessed separately during training).
Loss: pixel (MSE) + perceptual (VGG22) + adversarial (BCE).
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model
from models.gan.discriminators import VGGDiscriminator


# ---------------------------------------------------------------------------
# Generator building blocks
# ---------------------------------------------------------------------------

class SRGANResBlock(nn.Module):
    """SRGAN residual block: Conv-BN-PReLU-Conv-BN + skip."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SRGANUpBlock(nn.Module):
    """Conv → PixelShuffle(×2) → PReLU."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@register_model("srgan")
class SRGAN(BaseSRModel):
    """SRGAN generator.

    The discriminator is held separately as ``srgan.discriminator`` and
    accessed by the GANTrainer.

    Args:
        scale: Upsampling factor (2 or 4, since each UpBlock does ×2).
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature map width.
        num_resblocks: Number of residual blocks (default 16).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_resblocks: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 9, padding=4),
            nn.PReLU(),
        )

        self.body = nn.Sequential(
            *[SRGANResBlock(num_features) for _ in range(num_resblocks)],
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
        )

        assert scale in (2, 4), "SRGAN supports scale 2 or 4 (each UpBlock is ×2)"
        up_blocks = [SRGANUpBlock(num_features) for _ in range(scale // 2)]
        self.upsample = nn.Sequential(*up_blocks)

        self.tail = nn.Conv2d(num_features, out_channels, 9, padding=4)

        # Discriminator — instantiated here for easy access from trainer
        self.discriminator = VGGDiscriminator(in_channels=out_channels)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
