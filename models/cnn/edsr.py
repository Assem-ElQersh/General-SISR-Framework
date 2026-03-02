"""EDSR — Enhanced Deep Super-Resolution Network.

Lim et al., CVPRW 2017 (winner of NTIRE 2017).

Key modifications over standard ResNet:
    - Remove Batch Normalization (prevents artifacts, saves memory)
    - Scale residual outputs by 0.1 (res_scale)
    - Larger feature width and more blocks than SRCNN/VDSR
    - Upsampling at the end (sub-pixel convolution)

Variants:
    "baseline": 16 ResBlocks, 64 features  (~1.5M params)
    "large":    32 ResBlocks, 256 features (~43M params)
"""

from typing import Optional

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block without Batch Normalization."""

    def __init__(self, num_features: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


def make_upsampler(num_features: int, scale: int) -> nn.Sequential:
    """Sub-pixel convolution upsampler for scale 2, 3, or 4."""
    layers = []
    if scale in (2, 4):
        for _ in range(scale // 2):
            layers += [
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
            ]
    elif scale == 3:
        layers += [
            nn.Conv2d(num_features, num_features * 9, 3, padding=1),
            nn.PixelShuffle(3),
        ]
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# EDSR model
# ---------------------------------------------------------------------------

@register_model("edsr")
class EDSR(BaseSRModel):
    """Enhanced Deep SR.

    Args:
        scale: Upsampling factor (2, 3, or 4).
        in_channels: Input channels (3 for RGB).
        out_channels: Output channels (3 for RGB).
        num_features: Feature map width.
        num_resblocks: Number of residual blocks.
        res_scale: Residual scaling factor.
        variant: "baseline" (uses provided args) or "large" (overrides to 256 ch, 32 blocks).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_resblocks: int = 16,
        res_scale: float = 0.1,
        variant: str = "baseline",
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        if variant == "large":
            num_features = 256
            num_resblocks = 32

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.body = nn.Sequential(
            *[ResBlock(num_features, res_scale) for _ in range(num_resblocks)],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
