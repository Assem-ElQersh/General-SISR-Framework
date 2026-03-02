"""DPSR — Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels.

Zhang et al., CVPR 2019.

Algorithm — Half-Quadratic Splitting (HQS):
    Original MAP problem: argmin_x  ||y - Ax||² + λ R(x)
        where A = blur+downsample, R(x) = − log p(x) (image prior)

    HQS introduces auxiliary variable z:
        argmin_x,z  ||y - Ax||² + μ ||x - z||² + λ R(z)

    Splitting into two alternating sub-problems:
        x-step: x = (AᵀA + μI)⁻¹ (Aᵀy + μz)
                   = FFT-based closed-form solution (since A = known blur)
        z-step: z = prox_{λ/μ R}(x)
                   = denoiser(x, σ=√(λ/μ))  [Plug-and-Play prior]

Key design choices:
    - x-step: solved in Fourier domain (cheap FFT inversion)
    - z-step: any pretrained Gaussian denoiser (here: simple DnCNN-lite)
    - μ is annealed from small (emphasize prior) to large (trust data)

Note:
    DPSR is not trained end-to-end in the traditional sense; only the
    denoiser is trained. The HQS outer loop is fixed.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Simple built-in denoiser (DnCNN-lite)
# ---------------------------------------------------------------------------

class DnCNN(nn.Module):
    """Lightweight DnCNN for the Plug-and-Play prior."""

    def __init__(self, in_channels: int = 3, num_layers: int = 7, num_features: int = 64) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels + 1, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W), sigma: (B,) noise level map."""
        sigma_map = sigma.view(-1, 1, 1, 1).expand_as(x[:, :1])
        inp = torch.cat([x, sigma_map], dim=1)
        return x - self.net(inp)  # residual denoiser


# ---------------------------------------------------------------------------
# FFT-based x-step
# ---------------------------------------------------------------------------

def fft_x_step(
    y: torch.Tensor,
    kernel: torch.Tensor,
    z: torch.Tensor,
    mu: float,
    scale: int,
) -> torch.Tensor:
    """Solve x-step in the Fourier domain.

    x = ifft( (conj(K̂) · Ŷ_up + μ Ẑ) / (|K̂|² + μ) )

    Args:
        y:      (B, C, H//s, W//s) LR image
        kernel: (B, kH, kW) or (kH, kW) blur kernel
        z:      (B, C, H, W) auxiliary variable (from denoiser)
        mu:     HQS penalty parameter
        scale:  SR scale factor

    Returns:
        x: (B, C, H, W) updated HR estimate
    """
    B, C, Hlr, Wlr = y.shape
    H, W = Hlr * scale, Wlr * scale

    # Upsample y to HR
    y_up = F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)

    # Pad kernel to (H, W) and compute FFT
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).expand(B, -1, -1)
    kH, kW = kernel.shape[-2], kernel.shape[-1]
    pad_h, pad_w = H - kH, W - kW
    k_pad = F.pad(kernel, (0, pad_w, 0, pad_h))
    K = torch.fft.rfft2(k_pad)

    # FFT of inputs
    Y = torch.fft.rfft2(y_up)
    Z = torch.fft.rfft2(z)

    # Solve: X = (conj(K) * Y + mu * Z) / (|K|^2 + mu)
    K_conj = K.conj().unsqueeze(1)  # (B, 1, H, W/2+1) for broadcast over C
    X = (K_conj * Y + mu * Z) / (K.abs().pow(2).unsqueeze(1) + mu)

    return torch.fft.irfft2(X, s=(H, W)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# DPSR model
# ---------------------------------------------------------------------------

@register_model("dpsr")
class DPSR(BaseSRModel):
    """Deep Plug-and-Play SR with HQS.

    Args:
        scale: Upsampling factor.
        in_channels: Image channels.
        out_channels: Image channels.
        denoiser_layers: DnCNN depth (default 7).
        denoiser_features: DnCNN feature width (default 64).
        num_iter: HQS iterations (default 15).
        mu_start: Initial μ (default 0.01).
        mu_end: Final μ (default 0.5).
        sigma_start: Initial denoising σ (default 0.5).
        sigma_end: Final denoising σ (default 0.05).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        denoiser_layers: int = 7,
        denoiser_features: int = 64,
        num_iter: int = 15,
        mu_start: float = 0.01,
        mu_end: float = 0.5,
        sigma_start: float = 0.5,
        sigma_end: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        self.num_iter = num_iter
        self.mu_start = mu_start
        self.mu_end = mu_end
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.denoiser = DnCNN(in_channels, denoiser_layers, denoiser_features)

    def forward(
        self,
        lr: torch.Tensor,
        kernel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lr:     (B, C, H//s, W//s) LR image
            kernel: (B, kH, kW) or (kH, kW) blur kernel.
                    If None, uses an isotropic Gaussian with σ=1.6.

        Returns:
            SR: (B, C, H, W)
        """
        B, C, Hlr, Wlr = lr.shape
        H, W = Hlr * self.scale, Wlr * self.scale

        if kernel is None:
            kernel = self._default_kernel(lr.device, lr.dtype)

        # Initialize x with bicubic upscale
        x = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)

        mus = torch.linspace(self.mu_start, self.mu_end, self.num_iter)
        sigmas = torch.linspace(self.sigma_start, self.sigma_end, self.num_iter)

        for i in range(self.num_iter):
            mu = mus[i].item()
            sigma = sigmas[i].to(lr.device)
            # z-step (denoiser)
            z = self.denoiser(x, sigma.expand(B))
            # x-step (FFT)
            x = fft_x_step(lr, kernel, z, mu, self.scale)

        return x.clamp(0.0, 1.0)

    def _default_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = 15
        sigma = 1.6
        ax = torch.arange(k, device=device, dtype=dtype) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return (kernel / kernel.sum()).unsqueeze(0)  # (1, k, k)
