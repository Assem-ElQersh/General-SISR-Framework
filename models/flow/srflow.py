"""SRFlow — Learning the Super-Resolution Space with Normalizing Flow.

Lugmayr et al., ECCV 2020.

Core idea:
    Instead of learning a single deterministic mapping LR → HR, learn the
    full conditional distribution p(HR | LR) via an invertible flow network.

    At inference:
        - Sample z ~ N(0, I)
        - hr = Flow⁻¹(z | lr_features)    (diverse, perceptually sharp results)
        - Setting z = 0 gives the "mean" HR (comparable to EDSR quality)

Architecture:
    Encoder: RRDB network (same as ESRGAN generator) extracts LR features.
    Flow network: K coupling layers conditioned on RRDB features.
        Each coupling layer = ActNorm + Invertible 1×1 Conv + Affine Coupling

Training objective:
    Negative log-likelihood: -log p(HR | LR)
        = -log p_z(z) - sum_k log |det J_k|
    where z = Flow(HR | LR) and J_k is the Jacobian of layer k.

Important:
    - forward(lr) for inference (sampling)
    - forward_train(lr, hr) → (z, log_det) for training NLL loss
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.gan.esrgan import RRDB


# ---------------------------------------------------------------------------
# Flow building blocks
# ---------------------------------------------------------------------------

class ActNorm(nn.Module):
    """Activation Normalization — data-dependent scale+bias, invertible."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._initialized = False

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6
            self.bias.data = -mean / std
            self.scale.data = 1.0 / std
        self._initialized = True

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized:
            self._initialize(x)
        H, W = x.shape[2], x.shape[3]
        if not reverse:
            y = self.scale * (x + self.bias)
            log_det = self.scale.abs().log().sum() * H * W
        else:
            y = x / self.scale - self.bias
            log_det = -self.scale.abs().log().sum() * H * W
        return y, log_det


class Inv1x1Conv(nn.Module):
    """Invertible 1×1 convolution via LU decomposition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        W = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.register_buffer("W", W)
        self.register_buffer("W_inv", W.inverse())

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W_size = x.shape[2], x.shape[3]
        if not reverse:
            w = self.W.view(self.W.shape[0], self.W.shape[1], 1, 1)
            y = F.conv2d(x, w)
            log_det = torch.log(torch.abs(torch.det(self.W))) * H * W_size
        else:
            w_inv = self.W_inv.view(self.W_inv.shape[0], self.W_inv.shape[1], 1, 1)
            y = F.conv2d(x, w_inv)
            log_det = -torch.log(torch.abs(torch.det(self.W))) * H * W_size
        return y, log_det


class AffineCoupling(nn.Module):
    """Affine coupling layer conditioned on LR features."""

    def __init__(self, channels: int, cond_channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half + cond_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (channels - half) * 2, 3, padding=1),
        )
        # Initialize to identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        cond_resized = F.interpolate(cond, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        st = self.net(torch.cat([x1, cond_resized], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2, 3])
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2, 3])
        return torch.cat([x1, y2], dim=1), log_det


class FlowStep(nn.Module):
    """One flow step: ActNorm → Inv1×1 Conv → Affine Coupling."""

    def __init__(self, channels: int, cond_channels: int) -> None:
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.inv1x1 = Inv1x1Conv(channels)
        self.coupling = AffineCoupling(channels, cond_channels)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device)
        if not reverse:
            x, ld = self.actnorm(x)
            log_det += ld
            x, ld = self.inv1x1(x)
            log_det += ld
            x, ld = self.coupling(x, cond)
            log_det += ld
        else:
            x, ld = self.coupling(x, cond, reverse=True)
            log_det += ld
            x, ld = self.inv1x1(x, reverse=True)
            log_det += ld
            x, ld = self.actnorm(x, reverse=True)
            log_det += ld
        return x, log_det


# ---------------------------------------------------------------------------
# SRFlow model
# ---------------------------------------------------------------------------

@register_model("srflow")
class SRFlow(BaseSRModel):
    """SRFlow: Super-Resolution via Normalizing Flow.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_rrdb: Number of RRDB encoder blocks (default 8).
        num_flow_steps: Number of flow steps K (default 16).
        rrdb_features: RRDB feature width (default 64).
        temperature: Sampling temperature at inference (default 0.8).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_rrdb: int = 8,
        num_flow_steps: int = 16,
        rrdb_features: int = 64,
        temperature: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.temperature = temperature

        # LR encoder — RRDB (no upsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, rrdb_features, 3, padding=1),
            *[RRDB(rrdb_features) for _ in range(num_rrdb)],
            nn.Conv2d(rrdb_features, rrdb_features, 3, padding=1),
        )

        # HR channels after squeezing (spatial → channel): C * scale²
        hr_channels = out_channels * scale * scale
        self.flow_steps = nn.ModuleList([
            FlowStep(hr_channels, rrdb_features)
            for _ in range(num_flow_steps)
        ])

    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H*s, W*s) → (B, C*s², H, W)  via pixel-unshuffle."""
        return F.pixel_unshuffle(x, self.scale)

    def _unsqueeze(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C*s², H, W) → (B, C, H*s, W*s)  via pixel-shuffle."""
        return F.pixel_shuffle(z, self.scale)

    def encode(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map HR to latent z and return log-determinant sum."""
        cond = self.encoder(lr)
        z = self._squeeze(hr)
        total_log_det = torch.zeros(hr.shape[0], device=hr.device)
        for step in self.flow_steps:
            z, ld = step(z, cond)
            total_log_det += ld
        return z, total_log_det

    def decode(self, lr: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample HR from latent z (or random noise if z is None)."""
        cond = self.encoder(lr)
        if z is None:
            hr_h, hr_w = lr.shape[2] * self.scale, lr.shape[3] * self.scale
            hr_channels = self.out_channels * self.scale * self.scale
            z = torch.randn(
                lr.shape[0], hr_channels, hr_h // self.scale, hr_w // self.scale,
                device=lr.device,
            ) * self.temperature
        for step in reversed(self.flow_steps):
            z, _ = step(z, cond, reverse=True)
        return self._unsqueeze(z).clamp(0.0, 1.0)

    def forward_train(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training pass: returns (z, log_det) for NLL loss."""
        return self.encode(lr, hr)

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Inference: sample an HR image."""
        return self.decode(lr)
