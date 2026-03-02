"""ESRGAN — Enhanced Super-Resolution Generative Adversarial Networks.

Wang et al., ECCVW 2018.

Key improvements over SRGAN:
    1. RRDB (Residual in Residual Dense Block) generator — no BN
    2. Relativistic average GAN (RaGAN) discriminator loss
    3. Perceptual loss from VGG features before activation (richer gradients)
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model
from models.gan.discriminators import VGGDiscriminator
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# Dense Block
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    def __init__(self, in_features: int, growth: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, growth, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.conv(x))


class DenseBlock(nn.Module):
    """Densely connected block: each layer receives all previous features."""

    def __init__(self, num_features: int = 64, growth: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = num_features
        for _ in range(num_layers):
            self.layers.append(DenseLayer(in_ch, growth))
            in_ch += growth
        # 1×1 conv to project back to num_features
        self.final = nn.Conv2d(in_ch, num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return self.final(torch.cat(features, dim=1))


# ---------------------------------------------------------------------------
# RRDB — Residual in Residual Dense Block
# ---------------------------------------------------------------------------

class RRDB(nn.Module):
    """Three nested Dense Blocks with residual scaling."""

    _RES_SCALE = 0.2

    def __init__(self, num_features: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.db1 = DenseBlock(num_features, growth)
        self.db2 = DenseBlock(num_features, growth)
        self.db3 = DenseBlock(num_features, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x) * self._RES_SCALE + x
        out = self.db2(out) * self._RES_SCALE + out
        out = self.db3(out) * self._RES_SCALE + out
        return out * self._RES_SCALE + x


# ---------------------------------------------------------------------------
# ESRGAN generator
# ---------------------------------------------------------------------------

@register_model("esrgan")
class ESRGAN(BaseSRModel):
    """ESRGAN generator (RRDB-based).

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

        # Discriminator (accessed by GANTrainer)
        self.discriminator = VGGDiscriminator(in_channels=out_channels)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
