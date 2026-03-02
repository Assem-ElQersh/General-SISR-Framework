"""EDSR3D — 3D port of EDSR for volumetric super-resolution.

All Conv2d layers replaced with Conv3d. The upsampler uses VoxelShuffle
instead of PixelShuffle.

Supports both:
    - Isotropic scaling: scale = 2  →  (×2, ×2, ×2)
    - Anisotropic scaling: scale = (1, 1, 4)  →  z-axis only (useful for MRI)
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register_model
from .base_3d import Base3DSRModel, ScaleFactor
from .voxel_shuffle import make_upsampler_3d


# ---------------------------------------------------------------------------
# 3D ResBlock (no BatchNorm, scaled residual)
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """3D residual block without BatchNorm."""

    def __init__(self, num_features: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_features, num_features, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


# ---------------------------------------------------------------------------
# EDSR3D model
# ---------------------------------------------------------------------------

@register_model("edsr3d")
class EDSR3D(Base3DSRModel):
    """Enhanced Deep 3D SR Network.

    Args:
        scale: Upsampling factor.
               - int: isotropic (e.g., 2 → all axes ×2)
               - tuple (sd, sh, sw): anisotropic (e.g., (1, 1, 4) → z-axis only)
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature map width (default 64).
        num_resblocks: Number of residual blocks (default 16).
        res_scale: Residual scaling factor (default 0.1).
    """

    def __init__(
        self,
        scale: ScaleFactor = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_resblocks: int = 16,
        res_scale: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.head = nn.Conv3d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock3D(num_features, res_scale) for _ in range(num_resblocks)],
            nn.Conv3d(num_features, num_features, 3, padding=1),
        )
        self.upsample = make_upsampler_3d(num_features, self.scale, out_channels)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = x + self.body(x)
        return self.upsample(x).clamp(0.0, 1.0)
