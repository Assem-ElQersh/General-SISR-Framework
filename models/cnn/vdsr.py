"""VDSR — Accurate Image Super-Resolution Using Very Deep Convolutional Networks.

Kim et al., CVPR 2016.

Architecture:
    1. Bicubic upsampling to target size
    2. 20 conv layers (3×3, 64 features each), with ReLU
    3. Global residual: output = input + learned residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


@register_model("vdsr")
class VDSR(BaseSRModel):
    """Very Deep SR with global skip connection.

    Args:
        scale: Upsampling factor.
        in_channels: 1 (Y-channel) or 3 (RGB).
        out_channels: Same as in_channels.
        num_layers: Depth of the network (default 20).
        num_features: Feature map width (default 64).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 20,
        num_features: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        layers = [nn.Conv2d(in_channels, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(num_features, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(num_features, out_channels, 3, padding=1))

        self.body = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        residual = self.body(x)
        return (x + residual).clamp(0.0, 1.0)
