"""DRRN — Image Super-Resolution via Deep Recursive Residual Network.

Tai et al., CVPR 2017.

Key idea — parameter efficiency through weight sharing:
    A single Recursive Block (RB) is applied num_iterations times.
    All iterations share the same weights → very deep effective network
    with a small parameter count.

Architecture:
    LR → Bicubic(×scale) → Conv(head)
       → [Recursive Block] × B, each block applied U times
       → Conv(tail)
       → + skip (bicubic input) → SR

Each Recursive Block:
    for u in range(num_units):
        residual = conv1(relu(conv2(relu(bn(x)))))   # shared weights
        x = x + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


class RecursiveUnit(nn.Module):
    """Shared-weight residual unit used inside the Recursive Block."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RecursiveBlock(nn.Module):
    """Applies a single RecursiveUnit `num_units` times (shared weights)."""

    def __init__(self, num_features: int, num_units: int) -> None:
        super().__init__()
        self.unit = RecursiveUnit(num_features)
        self.num_units = num_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_units):
            x = self.unit(x)
        return x


@register_model("drrn")
class DRRN(BaseSRModel):
    """Deep Recursive Residual Network.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature map width (default 128).
        num_blocks: Number of Recursive Blocks (default 1).
        num_units: Number of recursive applications per block (default 25).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 128,
        num_blocks: int = 1,
        num_units: int = 25,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.blocks = nn.ModuleList(
            [RecursiveBlock(num_features, num_units) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        skip = x
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        x = self.tail(x)
        return (x + skip).clamp(0.0, 1.0)
