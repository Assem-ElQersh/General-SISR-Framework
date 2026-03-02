"""RCAN — Image Super-Resolution Using Very Deep Residual Channel Attention Networks.

Zhang et al., ECCV 2018.

Hierarchy:
    RCAN
    └── Residual Group (RG) × num_groups
        └── Residual Channel Attention Block (RCAB) × num_rcab_per_group
            ├── Conv → ReLU → Conv
            └── Channel Attention: GAP → FC → ReLU → FC → Sigmoid → scale

The global long skip connection (LSC) skips the entire RG stack.
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# Channel Attention Block
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, num_features: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, num_features: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """Residual Group: num_rcab RCAB blocks + one conv + local skip."""

    def __init__(self, num_features: int, num_rcab: int, reduction: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(num_features, reduction) for _ in range(num_rcab)],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ---------------------------------------------------------------------------
# RCAN model
# ---------------------------------------------------------------------------

@register_model("rcan")
class RCAN(BaseSRModel):
    """Residual Channel Attention Network.

    Args:
        scale: Upsampling factor (2, 3, or 4).
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature map width (default 64).
        num_groups: Number of Residual Groups (default 10).
        num_rcab_per_group: RCAB blocks per group (default 20).
        reduction: Channel attention reduction ratio (default 16).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_groups: int = 10,
        num_rcab_per_group: int = 20,
        reduction: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.body = nn.Sequential(
            *[
                ResidualGroup(num_features, num_rcab_per_group, reduction)
                for _ in range(num_groups)
            ],
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
