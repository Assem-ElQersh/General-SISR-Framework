"""DASR — Unsupervised Degradation Representation Learning for Blind SR.

Liang et al., CVPR 2021.

Key contributions:
    1. Contrastive degradation encoder: learns a compact degradation embedding d
       from the LR image without any explicit kernel annotation.
    2. Degradation-adaptive SR: the generator is conditioned on d via
       SPADE-style spatial-adaptive normalization layers.

Architecture:

    Degradation Encoder:
        LR → ResNet encoder → GAP → 512-dim degradation vector d

    SR Generator (conditioned on d):
        LR → Conv head
           → [SPADE-ResBlock(d)] × depth
           → PixelShuffle upsampler
           → Conv tail

    Training (besides the standard pixel loss):
        Contrastive loss on d:
            Positive pairs: crops from the same degraded LR
            Negative pairs: crops from differently degraded LRs
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# SPADE-style Degradation-Adaptive Normalization
# ---------------------------------------------------------------------------

class DegradationNorm(nn.Module):
    """Adaptive normalization conditioned on a degradation vector d."""

    def __init__(self, num_features: int, d_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(d_dim, num_features)
        self.beta = nn.Linear(d_dim, num_features)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gamma = self.gamma(d)[:, :, None, None]
        beta = self.beta(d)[:, :, None, None]
        return x_norm * (1 + gamma) + beta


class DAResBlock(nn.Module):
    """Residual block with degradation-adaptive normalization."""

    def __init__(self, num_features: int, d_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.norm1 = DegradationNorm(num_features, d_dim)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.norm2 = DegradationNorm(num_features, d_dim)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.norm1(self.conv1(x), d), inplace=True)
        residual = self.norm2(self.conv2(residual), d)
        return x + residual


# ---------------------------------------------------------------------------
# Degradation Encoder (contrastive)
# ---------------------------------------------------------------------------

class DegradationEncoder(nn.Module):
    """Lightweight CNN encoder that outputs a degradation embedding."""

    def __init__(self, in_channels: int = 3, d_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, d_dim)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        feat = self.net(lr).flatten(1)
        return F.normalize(self.proj(feat), dim=-1)


# ---------------------------------------------------------------------------
# DASR model
# ---------------------------------------------------------------------------

@register_model("dasr")
class DASR(BaseSRModel):
    """Degradation-Adaptive SR with contrastive representation learning.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Generator feature width (default 64).
        d_dim: Degradation embedding dimension (default 512).
        num_blocks: Number of DA-ResBlocks (default 16).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        d_dim: int = 512,
        num_blocks: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.degradation_encoder = DegradationEncoder(in_channels, d_dim)
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.ModuleList([DAResBlock(num_features, d_dim) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(
        self, lr: torch.Tensor, d: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if d is None:
            d = self.degradation_encoder(lr)
        feat = self.head(lr)
        x = feat
        for block in self.body:
            x = block(x, d)
        x = self.conv_after_body(x) + feat
        x = self.upsample(x)
        return self.tail(x).clamp(0.0, 1.0)

    def encode_degradation(self, lr: torch.Tensor) -> torch.Tensor:
        """Expose degradation encoder for contrastive training."""
        return self.degradation_encoder(lr)
