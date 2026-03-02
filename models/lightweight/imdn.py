"""IMDN — Lightweight Image Super-Resolution with Information Multi-Distillation Network.

Hui et al., ACM MM 2019.

Key ideas:
    1. Information Multi-Distillation Block (IMDB):
       Split channels → partial distillation at each step → concatenate retained features
    2. Contrast-aware Channel Attention (CCA) on the fused features
    3. PixelShuffle upsampling at the end (LR-space feature extraction)

This makes IMDN very lightweight (~700K params for ×4) and real-time capable.
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Contrast-aware Channel Attention
# ---------------------------------------------------------------------------

class CCALayer(nn.Module):
    """Channel attention that uses mean + std (contrast-aware) pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        descriptor = mean + std
        return x * self.fc(descriptor)


# ---------------------------------------------------------------------------
# Information Multi-Distillation Block (IMDB)
# ---------------------------------------------------------------------------

class IMDB(nn.Module):
    """Information Multi-Distillation Block.

    Splits features at each conv step:
        - A fraction (distillation_rate) is retained directly
        - The rest flows through the next conv

    Final output: concat of all retained + final processed features → CCA.
    """

    def __init__(
        self,
        num_features: int = 64,
        distillation_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.distill_ch = int(num_features * distillation_rate)
        self.remaining_ch = num_features - self.distill_ch

        self.c1 = nn.Sequential(nn.Conv2d(num_features, num_features, 3, padding=1), nn.LeakyReLU(0.05))
        self.c2 = nn.Sequential(nn.Conv2d(self.remaining_ch, num_features, 3, padding=1), nn.LeakyReLU(0.05))
        self.c3 = nn.Sequential(nn.Conv2d(self.remaining_ch, num_features, 3, padding=1), nn.LeakyReLU(0.05))
        self.c4 = nn.Conv2d(self.remaining_ch, self.distill_ch, 3, padding=1)

        fusion_ch = self.distill_ch * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_ch, num_features, 1),
            CCALayer(num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, r1 = self.c1(x).split([self.distill_ch, self.remaining_ch], dim=1)
        d2, r2 = self.c2(r1).split([self.distill_ch, self.remaining_ch], dim=1)
        d3, r3 = self.c3(r2).split([self.distill_ch, self.remaining_ch], dim=1)
        d4 = self.c4(r3)
        fused = self.fuse(torch.cat([d1, d2, d3, d4], dim=1))
        return fused + x


# ---------------------------------------------------------------------------
# IMDN model
# ---------------------------------------------------------------------------

@register_model("imdn")
class IMDN(BaseSRModel):
    """Information Multi-Distillation Network.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature width (default 64).
        num_blocks: Number of IMDB blocks (default 6).
        distillation_rate: Fraction of channels distilled per step (default 0.25).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        distillation_rate: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[IMDB(num_features, distillation_rate) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        if scale in (2, 4):
            up_layers = []
            for _ in range(scale // 2):
                up_layers += [
                    nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                ]
            self.upsample = nn.Sequential(*up_layers)
        elif scale == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 9, 3, padding=1),
                nn.PixelShuffle(3),
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = self.conv_after_body(self.body(x)) + x
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
