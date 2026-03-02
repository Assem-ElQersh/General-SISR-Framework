"""UnrolledSR — Learned Unrolling of the HQS Algorithm for SR.

Inspired by ISTA-Net (Zhang et al., 2018) and deep unrolling for SR.

Idea:
    Unlike DPSR where the HQS outer loop is fixed and uses a generic denoiser,
    UnrolledSR learns each iteration stage end-to-end.

    Each stage has:
        1. Data Fidelity Update: x = x - α * Aᵀ(Ax - y)
           (gradient step on ||Ax - y||²)
        2. Learned Proximal Step: z = ProxNet(x)
           (lightweight ResNet that acts as a learned regularizer)

    The degradation operator A and its transpose Aᵀ are fixed
    (known blur kernel + bicubic down/up sampling).

Architecture:
    Each stage:
        ProxNet: 5-layer ResNet (~128 features)
        Step-size α_t: learned scalar per stage
    Total: num_stages × ProxNet (non-shared by default, can be shared)

Training:
    Standard pixel (L1/L2) loss with optional perceptual loss.
    No special training procedure required.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Lightweight ProxNet (learned prior)
# ---------------------------------------------------------------------------

class ProxNet(nn.Module):
    """Lightweight residual prior network for one unrolled stage."""

    def __init__(self, channels: int = 3, num_features: int = 64, num_layers: int = 5) -> None:
        super().__init__()
        layers = [nn.Conv2d(channels, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(num_features, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(num_features, channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


# ---------------------------------------------------------------------------
# Degradation operators (known kernel + bicubic)
# ---------------------------------------------------------------------------

def apply_degradation(
    x: torch.Tensor,
    kernel: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    """A(x) = downsample(blur(x, kernel), scale)."""
    B, C, H, W = x.shape
    kH, kW = kernel.shape[-2], kernel.shape[-1]
    pad_h, pad_w = kH // 2, kW // 2
    k = kernel.view(1, 1, kH, kW).expand(C, 1, kH, kW)
    blurred = F.conv2d(x.view(1, B * C, H, W), k.view(B * C if kernel.dim() == 4 else C, 1, kH, kW),
                       padding=(pad_h, pad_w), groups=B * C if kernel.dim() == 4 else C)
    blurred = blurred.view(B, C, H, W)
    return F.interpolate(blurred, scale_factor=1.0 / scale, mode="bicubic", align_corners=False)


def apply_transpose_degradation(
    y: torch.Tensor,
    kernel: torch.Tensor,
    scale: int,
    target_size,
) -> torch.Tensor:
    """Aᵀ(y) = blur(upsample(y, scale), kernel_flipped)."""
    y_up = F.interpolate(y, size=target_size, mode="bicubic", align_corners=False)
    B, C, H, W = y_up.shape
    kH, kW = kernel.shape[-2], kernel.shape[-1]
    pad_h, pad_w = kH // 2, kW // 2
    k_flip = kernel.flip(-1).flip(-2)
    k = k_flip.view(1, 1, kH, kW).expand(C, 1, kH, kW)
    out = F.conv2d(y_up.view(1, B * C, H, W), k.view(C, 1, kH, kW),
                   padding=(pad_h, pad_w), groups=C)
    return out.view(B, C, H, W)


# ---------------------------------------------------------------------------
# UnrolledSR model
# ---------------------------------------------------------------------------

@register_model("unrolled_sr")
class UnrolledSR(BaseSRModel):
    """Learned Unrolled HQS for Blind-Kernel SR.

    Args:
        scale: Upsampling factor.
        in_channels: Image channels.
        out_channels: Image channels.
        num_stages: Number of unrolled HQS stages (default 8).
        prox_features: ProxNet feature width (default 64).
        prox_layers: ProxNet depth (default 5).
        shared_prox: Whether to share ProxNet weights across stages (default False).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_stages: int = 8,
        prox_features: int = 64,
        prox_layers: int = 5,
        shared_prox: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.num_stages = num_stages

        if shared_prox:
            shared = ProxNet(in_channels, prox_features, prox_layers)
            self.prox_nets = nn.ModuleList([shared] * num_stages)
        else:
            self.prox_nets = nn.ModuleList([
                ProxNet(in_channels, prox_features, prox_layers) for _ in range(num_stages)
            ])
        # Learned step-sizes per stage
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)
        ])

    def forward(
        self,
        lr: torch.Tensor,
        kernel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lr:     (B, C, H//s, W//s) LR image
            kernel: (1, kH, kW) or (kH, kW) blur kernel.
                    If None, uses isotropic Gaussian with σ=1.6.
        """
        B, C, Hlr, Wlr = lr.shape
        H, W = Hlr * self.scale, Wlr * self.scale

        if kernel is None:
            kernel = self._default_kernel(lr.device, lr.dtype)

        # Initialize with bicubic upsample
        x = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)

        for i in range(self.num_stages):
            # Data fidelity gradient step
            Ax = apply_degradation(x, kernel, self.scale)
            residual = Ax - lr
            grad = apply_transpose_degradation(residual, kernel, self.scale, (H, W))
            x = x - self.alphas[i] * grad
            # Learned proximal step
            x = self.prox_nets[i](x)

        return x.clamp(0.0, 1.0)

    def _default_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = 15
        sigma = 1.6
        ax = torch.arange(k, device=device, dtype=dtype) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return (kernel / kernel.sum()).unsqueeze(0)  # (1, k, k)
