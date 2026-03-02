"""SRCNN — Learning a Deep Convolutional Network for Image Super-Resolution.

Dong et al., ECCV 2014 / TPAMI 2016.

Architecture:
    1. Bicubic upsampling to target size
    2. Patch extraction + representation:  Conv(f1×f1, n1)
    3. Non-linear mapping:                 Conv(f2×f2, n2)
    4. Reconstruction:                     Conv(f3×f3, C_out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.base_model import BaseSRModel
from models.registry import register_model


@register_model("srcnn")
class SRCNN(BaseSRModel):
    """SRCNN — shallow 3-layer convolutional SR network.

    Args:
        scale: Upsampling factor (2, 3, or 4).
        in_channels: Number of input image channels (1 for Y-channel, 3 for RGB).
        out_channels: Number of output channels (same as in_channels typically).
        num_channels: Feature widths for the two intermediate layers [n1, n2].
        kernel_sizes: Kernel sizes for the three conv layers [f1, f2, f3].
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if num_channels is None:
            num_channels = [64, 32]
        if kernel_sizes is None:
            kernel_sizes = [9, 1, 5]

        n1, n2 = num_channels
        f1, f2, f3 = kernel_sizes

        self.conv1 = nn.Conv2d(in_channels, n1, f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, out_channels, f3, padding=f3 // 2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return x.clamp(0.0, 1.0)
