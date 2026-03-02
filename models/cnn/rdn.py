"""RDN — Residual Dense Network for Image Super-Resolution.

Zhang et al., CVPR 2018.

Key innovations:
    1. Residual Dense Block (RDB): each conv layer receives all previous
       layer outputs (dense connections within the block)
    2. Local Feature Fusion: 1×1 conv at end of each RDB combines all dense features
    3. Global Feature Fusion: fuses outputs from all RDBs
    4. Global Residual Learning: adds shallow features to the fused output

Architecture:
    LR → SFENet (shallow feature extraction, 2 conv layers)
       → [RDB × num_blocks] with dense local + global connections
       → Dense Feature Fusion (1×1 conv + 3×3 conv)
       → PixelShuffle upsampler
       → Output conv
"""

from typing import List

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# Residual Dense Block
# ---------------------------------------------------------------------------

class RDB(nn.Module):
    """Residual Dense Block.

    Each layer outputs are concatenated and fed to all subsequent layers.
    A 1×1 local feature fusion conv reduces back to num_features.
    """

    def __init__(
        self,
        num_features: int,
        growth_rate: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = num_features + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
        # Local Feature Fusion
        self.lff = nn.Conv2d(num_features + num_layers * growth_rate, num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return x + self.lff(torch.cat(features, dim=1))


# ---------------------------------------------------------------------------
# RDN model
# ---------------------------------------------------------------------------

@register_model("rdn")
class RDN(BaseSRModel):
    """Residual Dense Network for SISR.

    Args:
        scale: Upsampling factor (2, 3, or 4).
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Feature map width G0 (default 64).
        num_blocks: Number of RDB blocks D (default 16).
        growth_rate: Dense layer growth rate G (default 32).
        num_layers: Dense layers per RDB C (default 8).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        growth_rate: int = 32,
        num_layers: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        # Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Residual Dense Blocks
        self.rdbs = nn.ModuleList([
            RDB(num_features, growth_rate, num_layers)
            for _ in range(num_blocks)
        ])

        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_blocks * num_features, num_features, 1),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Upsampling + output
        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        f1 = self.sfe1(lr)
        x = self.sfe2(f1)

        rdb_outs = []
        for rdb in self.rdbs:
            x = rdb(x)
            rdb_outs.append(x)

        x = self.gff(torch.cat(rdb_outs, dim=1)) + f1
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)
