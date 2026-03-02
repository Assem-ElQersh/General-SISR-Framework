"""MedSR — Anisotropic MRI Super-Resolution.

Specialized for the common clinical scenario where MRI has high in-plane
resolution (x, y) but low through-plane resolution (z-axis).

E.g., an MRI volume might have:
    - In-plane resolution:    0.5 mm × 0.5 mm  (high)
    - Through-plane (z) slice thickness: 4.0 mm  (low, ×8 interpolation needed)

Two approaches are provided:

1. Slice-wise 2D SR (``MedSRSliceWise``):
   - Process each z-slice with a 2D SR model
   - Apply SR only along z (recombine slices)
   - Fast baseline; no 3D context

2. Full 3D SR (``MedSR``):
   - Wraps EDSR3D with anisotropic scale (1, 1, r) for z-axis only
   - Captures 3D context while being efficient for z-SR tasks
   - Registered as ``"med_sr"``
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register_model
from models.cnn.edsr import EDSR
from .base_3d import Base3DSRModel
from .edsr3d import EDSR3D


# ---------------------------------------------------------------------------
# Slice-wise 2D SR baseline
# ---------------------------------------------------------------------------

class MedSRSliceWise(nn.Module):
    """Anisotropic SR: applies 2D SR to each axial slice independently.

    Input shape:  (B, C, D, H, W)
    Output shape: (B, C, D*scale_z, H, W)

    This treats each 2D slice as an independent image and does NOT use
    inter-slice 3D context.
    """

    def __init__(
        self,
        scale_z: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_resblocks: int = 8,
    ) -> None:
        super().__init__()
        self.scale_z = scale_z
        # 2D EDSR processes slices as (B*D, C, H, W)
        self.sr2d = EDSR(
            scale=scale_z,
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_resblocks=num_resblocks,
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = lr.shape
        # Flatten B and D into batch dimension for 2D processing
        slices = lr.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        sr_slices = self.sr2d(slices)   # (B*D, C, H, W)
        # Reshape back to volume
        sr = sr_slices.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        return sr  # (B, C, D, H, W) — note: no H,W change; only D is implicitly the same


# ---------------------------------------------------------------------------
# Full 3D MedSR (EDSR3D with anisotropic z-only scale)
# ---------------------------------------------------------------------------

@register_model("med_sr")
class MedSR(Base3DSRModel):
    """3D anisotropic MRI SR — upsamples z-axis only.

    Uses EDSR3D with scale=(1, 1, scale_z) so D grows by scale_z while
    H and W remain unchanged.

    Args:
        scale_z: Through-plane SR factor (default 4).
        in_channels: Input channels (1 for grayscale MRI).
        out_channels: Output channels.
        num_features: EDSR3D feature width (default 64).
        num_resblocks: EDSR3D depth (default 16).
        mode: "3d" for full 3D context, "slicewise" for 2D baseline.
    """

    def __init__(
        self,
        scale_z: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_resblocks: int = 16,
        mode: str = "3d",
        **kwargs,
    ) -> None:
        # scale tuple: (z, h, w) — only z-axis upsampling
        super().__init__(scale=(scale_z, 1, 1), in_channels=in_channels, out_channels=out_channels)
        self.mode = mode
        self.scale_z = scale_z

        if mode == "slicewise":
            self.model = MedSRSliceWise(scale_z, in_channels, out_channels, num_features, num_resblocks)
        else:
            self.model = EDSR3D(
                scale=(scale_z, 1, 1),
                in_channels=in_channels,
                out_channels=out_channels,
                num_features=num_features,
                num_resblocks=num_resblocks,
            )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr: (B, C, D, H, W) — low-resolution MRI volume

        Returns:
            (B, C, D*scale_z, H, W) — super-resolved along z-axis
        """
        if self.mode == "slicewise":
            # Slicewise model outputs (B, C, D, H, W) unchanged in D
            # Apply trilinear upsampling for z after 2D SR on each slice
            sr = self.model(lr)
            # Upsample z-axis with trilinear
            B, C, D, H, W = sr.shape
            sr = F.interpolate(
                sr, size=(D * self.scale_z, H, W),
                mode="trilinear", align_corners=False,
            )
            return sr.clamp(0.0, 1.0)
        else:
            return self.model(lr)
