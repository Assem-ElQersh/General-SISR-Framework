"""Real-ESRGAN — Training Real-World Blind Super-Resolution with Pure Synthetic Data.

Wang et al., ICCV 2021.

Key differences from ESRGAN:
    1. No clean paired HR/LR training data — LR is synthesized from HR via
       the multi-order degradation pipeline in data/degradations.py
    2. Uses a U-Net discriminator (with spectral norm) for pixel-level realism
    3. Otherwise identical RRDB generator as ESRGAN

The degradation pipeline is applied externally (in the trainer / dataloader),
so this class is structurally identical to ESRGAN but registered under a
separate name and paired with the UNetDiscriminator.
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model
from models.gan.esrgan import RRDB
from models.gan.discriminators import UNetDiscriminator
from models.cnn.edsr import make_upsampler


@register_model("real_esrgan")
class RealESRGAN(BaseSRModel):
    """Real-ESRGAN generator.

    Identical RRDB architecture to ESRGAN but coupled with the
    UNetDiscriminator for spatially-aware discrimination.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature width (default 64).
        num_rrdb: Number of RRDB blocks (default 23).
        growth_channels: Dense layer growth rate (default 32).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        growth_channels: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[RRDB(num_features, growth_channels) for _ in range(num_rrdb)],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )
        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )

        # U-Net discriminator for pixel-level realism
        self.discriminator = UNetDiscriminator(in_channels=out_channels)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
