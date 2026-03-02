"""FSRCNN — Accelerating the Super-Resolution Convolutional Neural Network.

Dong et al., ECCV 2016.

Key improvement over SRCNN:
    1. Feature extraction in LR space (no bicubic pre-upsample)
    2. A shrinking layer reduces channels before mapping
    3. An expanding layer restores channels after mapping
    4. Deconvolution (transposed conv) upsampler at the end

This is ~40× faster than SRCNN and supports arbitrary scale factors.

Architecture:
    LR → Conv(f1, d)          feature extraction
       → Conv(1×1, s)          shrinking
       → [Conv(f2, s)] × m    non-linear mapping
       → Conv(1×1, d)          expanding
       → DeConv(f3, C_out)    sub-pixel/deconv upsampler
"""

from typing import List

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model


@register_model("fsrcnn")
class FSRCNN(BaseSRModel):
    """Fast SRCNN.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        d: Feature map depth for extraction / expanding (default 56).
        s: Shrunk feature depth for mapping (default 12).
        m: Number of mapping layers (default 4).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, d, 5, padding=2),
            nn.PReLU(d),
        )

        # Shrinking
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s),
        )

        # Non-linear mapping
        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        self.mapping = nn.Sequential(*mapping)

        # Expanding
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.PReLU(d),
        )

        # Deconvolution upsampler
        self.upsample = nn.ConvTranspose2d(
            d, out_channels, kernel_size=9, stride=scale, padding=9 // 2,
            output_padding=scale - 1,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.feature_extract(lr)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        return self.upsample(x).clamp(0.0, 1.0)
