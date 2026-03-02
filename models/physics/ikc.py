"""IKC — Blind Super-Resolution with Iterative Kernel Correction.

Gu et al., CVPR 2019.

Problem:
    Unknown blur kernel makes SR ill-posed. If the kernel were known, a
    kernel-conditioned SR network (SFTMD) could use it explicitly.
    IKC alternates:
        1. SR step:   SFTMD(LR, k) → SR estimate
        2. Correct step: Corrector(SR, k) → refined kernel estimate k'

Architecture:

    SFTMD (SR with Feature-level Modulation with Degradation):
        LR + spatial kernel feature map (via SFT layers) → SR

    Predictor: estimates initial kernel from LR alone
        LR → Conv encoder → FCN → k_init

    Corrector: refines the kernel from the SR output
        concat[LR, SR, k] → Conv encoder → FCN → Δk → k + Δk

    Inference loop:
        k = Predictor(LR)
        for i in range(num_iter):
            SR = SFTMD(LR, k)
            k  = Corrector(LR, SR, k)

SFT (Spatial Feature Transform):
    Condition a feature map F on a spatial map M:
        SFT(F, M) = alpha(M) * F + beta(M)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# SFT layer
# ---------------------------------------------------------------------------

class SFTLayer(nn.Module):
    """Spatial Feature Transform conditioned on a kernel feature map."""

    def __init__(self, num_features: int, k_features: int) -> None:
        super().__init__()
        self.alpha_net = nn.Sequential(
            nn.Conv2d(k_features, num_features, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_features, num_features, 1),
        )
        self.beta_net = nn.Sequential(
            nn.Conv2d(k_features, num_features, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_features, num_features, 1),
        )

    def forward(self, feat: torch.Tensor, k_map: torch.Tensor) -> torch.Tensor:
        # k_map may need to be spatially broadcast
        k_map = F.interpolate(k_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.alpha_net(k_map) * feat + self.beta_net(k_map)


class SFTResBlock(nn.Module):
    """Residual block with SFT conditioning."""

    def __init__(self, num_features: int, k_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.sft1 = SFTLayer(num_features, k_features)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.sft2 = SFTLayer(num_features, k_features)

    def forward(self, feat: torch.Tensor, k_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.sft1(self.conv1(feat), k_map), inplace=True)
        x = self.sft2(self.conv2(x), k_map)
        return feat + x, k_map


# ---------------------------------------------------------------------------
# SFTMD — SR network with kernel modulation
# ---------------------------------------------------------------------------

class SFTMD(nn.Module):
    """Kernel-conditioned SR network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        k_features: int,
        num_blocks: int,
        scale: int,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.k_proj = nn.Sequential(
            nn.Linear(k_features, num_features),
            nn.ReLU(inplace=True),
        )
        self.body = nn.ModuleList([SFTResBlock(num_features, num_features) for _ in range(num_blocks)])
        self.conv_after = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample = make_upsampler(num_features, scale)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.k_features = k_features

    def forward(self, lr: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B = lr.shape[0]
        k_vec = k.view(B, -1)
        if k_vec.shape[1] != self.k_features:
            k_vec = F.adaptive_avg_pool1d(k_vec.unsqueeze(1), self.k_features).squeeze(1)
        k_feat = self.k_proj(k_vec)[:, :, None, None].expand(-1, -1, 1, 1)

        feat = self.head(lr)
        x = feat
        for block in self.body:
            x, _ = block(x, k_feat)
        x = self.conv_after(x) + feat
        x = self.upsample(x)
        return self.tail(x)


# ---------------------------------------------------------------------------
# Kernel Predictor and Corrector
# ---------------------------------------------------------------------------

class KernelPredictor(nn.Module):
    """Predict initial blur kernel from LR image."""

    def __init__(self, in_channels: int, k_features: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, k_features)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.fc(self.enc(lr).flatten(1))


class KernelCorrector(nn.Module):
    """Refine kernel estimate from LR, SR, and current kernel."""

    def __init__(self, in_channels: int, k_features: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + k_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k_features),
        )

    def forward(self, lr: torch.Tensor, sr: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        sr_down = F.interpolate(sr, size=lr.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.enc(torch.cat([lr, sr_down], dim=1)).flatten(1)
        return k + self.fc(torch.cat([feat, k.view(k.shape[0], -1)], dim=1))


# ---------------------------------------------------------------------------
# IKC model
# ---------------------------------------------------------------------------

@register_model("ikc")
class IKC(BaseSRModel):
    """Iterative Kernel Correction for Blind SR.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: SR network feature width (default 64).
        k_features: Kernel latent dimension (default 70 ≈ 7×7×√2).
        num_blocks: SR network depth (default 16).
        num_iter: Correction iterations at inference (default 4).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        k_features: int = 70,
        num_blocks: int = 16,
        num_iter: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.num_iter = num_iter
        self.predictor = KernelPredictor(in_channels, k_features)
        self.corrector = KernelCorrector(in_channels, k_features)
        self.sftmd = SFTMD(in_channels, out_channels, num_features, k_features, num_blocks, scale)

    def forward(
        self, lr: torch.Tensor, k_init: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lr: (B, C, H, W) LR image
            k_init: optional initial kernel vector (B, k_features).
                    If None, predicted from LR.

        Returns:
            (sr, k_final): final SR image and estimated kernel.
        """
        if k_init is None:
            k = self.predictor(lr)
        else:
            k = k_init

        sr = None
        for _ in range(self.num_iter):
            sr = self.sftmd(lr, k)
            k = self.corrector(lr, sr, k)

        return sr.clamp(0.0, 1.0), k
