"""PhysicsLoss — forward model consistency loss for physics-informed SR.

Penalizes inconsistency between the SR output and the LR measurement:

    L_physics = weight × ||A(SR) - LR||²

where A = blur_kernel ⊛ downsample(·, scale).

This loss enforces that the SR image, when degraded with the same operator
that produced the LR input, reproduces the LR image exactly.

Optional: If no kernel is provided, falls back to pure bicubic downsampling.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """Forward model consistency loss.

    Args:
        scale: SR upsampling factor (used to build the degradation operator).
        weight: Loss weight (default 0.1).
        kernel_size: Gaussian kernel size used for default blur (default 15).
        kernel_sigma: Gaussian sigma for default blur (default 1.6).
    """

    def __init__(
        self,
        scale: int = 4,
        weight: float = 0.1,
        kernel_size: int = 15,
        kernel_sigma: float = 1.6,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.weight = weight
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma

    def _make_gaussian_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = self.kernel_size
        sigma = self.kernel_sigma
        ax = torch.arange(k, device=device, dtype=dtype) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _apply_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply spatial blur with the given kernel."""
        B, C, H, W = x.shape
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad_h, pad_w = kH // 2, kW // 2

        if kernel.dim() == 2:
            k = kernel.view(1, 1, kH, kW).expand(C, 1, kH, kW)
        else:
            # (B, kH, kW) — per-sample kernels: use average
            k = kernel.mean(0).view(1, 1, kH, kW).expand(C, 1, kH, kW)

        x_flat = x.view(1, B * C, H, W)
        blurred = F.conv2d(x_flat, k.repeat(B, 1, 1, 1) if B > 1 else k,
                           padding=(pad_h, pad_w), groups=B * C if B > 1 else C)
        return blurred.view(B, C, H, W)

    def _degrade(
        self, sr: torch.Tensor, kernel: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Degrade SR → approximate LR (blur + downsample)."""
        if kernel is not None:
            sr = self._apply_blur(sr, kernel)
        return F.interpolate(sr, scale_factor=1.0 / self.scale, mode="bicubic", align_corners=False)

    def forward(
        self,
        sr: torch.Tensor,
        lr: torch.Tensor,
        kernel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sr:     (B, C, H×s, W×s) Super-resolved image.
            lr:     (B, C, H, W)     Low-resolution measurement.
            kernel: (B, kH, kW) or (kH, kW) blur kernel.
                    If None, uses an isotropic Gaussian kernel.

        Returns:
            Scalar loss tensor.
        """
        if kernel is None:
            kernel = self._make_gaussian_kernel(sr.device, sr.dtype)

        sr_degraded = self._degrade(sr, kernel)
        return self.weight * F.mse_loss(sr_degraded, lr)
