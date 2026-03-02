"""SRCNN3D — Direct 3D port of SRCNN for volumetric super-resolution.

Maps (B, C, D, H, W) → (B, C, D*s, H*s, W*s) via:
    1. Trilinear pre-upsample (replaces bicubic in 2D version)
    2. Three-layer 3D CNN (patch extraction → non-linear mapping → reconstruction)

Architecture mirrors SRCNN (Dong et al., 2014) exactly, with:
    Conv2d  → Conv3d
    bicubic → trilinear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register_model
from .base_3d import Base3DSRModel, ScaleFactor


@register_model("srcnn3d")
class SRCNN3D(Base3DSRModel):
    """3D SRCNN for volumetric SR.

    Args:
        scale: Upsampling factor (int for isotropic, tuple for anisotropic).
        in_channels: Input channels (e.g., 1 for grayscale MRI).
        out_channels: Output channels.
        f1: Patch extraction filter size (default 9).
        f2: Non-linear mapping filter size (default 5).
        f3: Reconstruction filter size (default 5).
        n1: Feature maps at extraction layer (default 64).
        n2: Feature maps at mapping layer (default 32).
    """

    def __init__(
        self,
        scale: ScaleFactor = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        f1: int = 9,
        f2: int = 5,
        f3: int = 5,
        n1: int = 64,
        n2: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.conv1 = nn.Conv3d(in_channels, n1, f1, padding=f1 // 2)
        self.conv2 = nn.Conv3d(n1, n2, f2, padding=f2 // 2)
        self.conv3 = nn.Conv3d(n2, out_channels, f3, padding=f3 // 2)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        sd, sh, sw = self.scale
        x = F.interpolate(lr, scale_factor=(sd, sh, sw), mode="trilinear", align_corners=False)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x).clamp(0.0, 1.0)
