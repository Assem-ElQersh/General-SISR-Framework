"""ESPCN — Real-Time Single Image and Video Super-Resolution Using an
Efficient Sub-Pixel Convolutional Neural Network.

Shi et al., CVPR 2016.

Key idea:
    - All feature extraction happens in LR space (cheap)
    - The last layer outputs scale² channels
    - PixelShuffle rearranges them into an HR image

This avoids the expensive bicubic pre-upsampling of SRCNN / VDSR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


@register_model("espcn")
class ESPCN(BaseSRModel):
    """Efficient Sub-Pixel Convolutional Network.

    Args:
        scale: Upsampling factor (2, 3, or 4).
        in_channels: Input channels (1 for Y-channel).
        out_channels: Output channels.
        num_features: Intermediate feature width.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(num_features // 2, out_channels * scale * scale, 3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.body(lr)
        x = self.pixel_shuffle(x)
        return x.clamp(0.0, 1.0)
