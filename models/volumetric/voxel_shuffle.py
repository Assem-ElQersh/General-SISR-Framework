"""VoxelShuffle — 3D generalization of PixelShuffle.

Maps (B, C·sd·sh·sw, D, H, W) → (B, C, D·sd, H·sh, W·sw) by rearranging
feature map channels into spatial depth, height, and width dimensions.

This is the 3D analogue of F.pixel_shuffle for sub-voxel convolution networks.

Reference:
    Analogous to Shi et al., "Real-Time Single Image and Video SR Using an
    Efficient Sub-Pixel CNN", CVPR 2016 — extended to 3D volumes.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn


class VoxelShuffle(nn.Module):
    """3D Sub-Voxel Shuffle upsampler.

    Args:
        upscale_factor: int or (sd, sh, sw) upscale factors for D, H, W.
    """

    def __init__(self, upscale_factor: Union[int, Tuple[int, int, int]]) -> None:
        super().__init__()
        if isinstance(upscale_factor, int):
            self.sd, self.sh, self.sw = upscale_factor, upscale_factor, upscale_factor
        else:
            self.sd, self.sh, self.sw = upscale_factor

    @property
    def spatial_factor(self) -> int:
        return self.sd * self.sh * self.sw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C * sd * sh * sw, D, H, W)

        Returns:
            (B, C, D*sd, H*sh, W*sw)
        """
        B, C_in, D, H, W = x.shape
        sd, sh, sw = self.sd, self.sh, self.sw
        assert C_in % (sd * sh * sw) == 0, (
            f"Input channels {C_in} must be divisible by sd*sh*sw={sd*sh*sw}"
        )
        C = C_in // (sd * sh * sw)

        # (B, C, sd, sh, sw, D, H, W)
        x = x.view(B, C, sd, sh, sw, D, H, W)
        # (B, C, D, sd, H, sh, W, sw)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # (B, C, D*sd, H*sh, W*sw)
        return x.view(B, C, D * sd, H * sh, W * sw)

    def extra_repr(self) -> str:
        return f"upscale_factor=({self.sd}, {self.sh}, {self.sw})"


def make_upsampler_3d(
    num_features: int,
    scale: Union[int, Tuple[int, int, int]],
    out_channels: int = 1,
) -> nn.Sequential:
    """Build a 3D sub-voxel upsampler: Conv3d + VoxelShuffle."""
    if isinstance(scale, int):
        sd, sh, sw = scale, scale, scale
    else:
        sd, sh, sw = scale
    return nn.Sequential(
        nn.Conv3d(num_features, out_channels * sd * sh * sw, 3, padding=1),
        VoxelShuffle((sd, sh, sw)),
    )
