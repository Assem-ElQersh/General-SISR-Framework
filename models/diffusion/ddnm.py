"""DDNM — Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model.

Wang et al., ICLR 2023.

Core idea — zero-shot SR without task-specific training:
    Any linear degradation can be written as y = Ax, where A is the degradation operator.
    A has a null-space N(A) = {z : Az = 0}.
    The solution x* = x_range + x_null, where:
        x_range = A† y  (range-space content — from the LR measurement)
        x_null  ∈ N(A)  (null-space content — hallucinated by the diffusion model)

    At each reverse diffusion step t:
        x̂₀ = denoiser(x_t, t)               (diffusion prediction)
        x̂₀ = A†y + (I - A†A)x̂₀            (DDNM correction)
        x_{t-1} = q(x_{t-1} | x_t, x̂₀)     (standard DDPM reverse step)

For SR:
    A = bicubic_downsample
    A† = bicubic_upsample (Moore-Penrose pseudo-inverse)
    A†A = bicubic_upsample ∘ bicubic_downsample

No additional training is required — only a pretrained unconditional DDPM.

Usage:
    1. Train (or use) a pretrained DDPM on HR images (e.g., SR3 or any U-Net DDPM)
    2. Load it into DDNM
    3. Run DDNM.forward(lr) — zero-shot super-resolution
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.diffusion.sr3 import UNet, TimeEmbedding, _linear_schedule, _cosine_schedule


@register_model("ddnm")
class DDNM(BaseSRModel):
    """Zero-Shot Diffusion SR via Null-Space Correction.

    Args:
        scale: SR upsampling factor.
        in_channels: Image channels.
        out_channels: Image channels (same as in_channels for SR).
        model_channels: U-Net base channel width.
        channel_mult: U-Net channel multipliers.
        num_res_blocks: ResBlocks per U-Net level.
        attention_resolutions: Spatial resolutions with attention.
        dropout: Dropout in ResBlocks.
        timesteps: Total diffusion timesteps.
        noise_schedule: "linear" or "cosine".
        beta_start: Beta start for linear schedule.
        beta_end: Beta end for linear schedule.
        pretrained_ckpt: Optional path to pretrained unconditional DDPM checkpoint.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mult=None,
        num_res_blocks: int = 2,
        attention_resolutions=None,
        dropout: float = 0.0,
        timesteps: int = 1000,
        noise_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        pretrained_ckpt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8]
        if attention_resolutions is None:
            attention_resolutions = [16, 8]

        self.timesteps = timesteps

        # Unconditional U-Net denoiser (same architecture as SR3, input = noisy HR only)
        self.unet = UNet(
            in_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
        self.out_proj = nn.Conv2d(model_channels, out_channels, 1)

        # Diffusion schedule
        if noise_schedule == "cosine":
            betas = _cosine_schedule(timesteps)
        else:
            betas = _linear_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

    def _load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[DDNM] Missing keys: {missing[:5]} ...")
        if unexpected:
            print(f"[DDNM] Unexpected keys: {unexpected[:5]} ...")

    # ------------------------------------------------------------------
    # Degradation operators (SR: bicubic)
    # ------------------------------------------------------------------

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        """Forward degradation: A(x) = bicubic downsample."""
        return F.interpolate(x, scale_factor=1.0 / self.scale, mode="bicubic", align_corners=False)

    def _At(self, y: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse: A†(y) = bicubic upsample."""
        B, C, H, W = y.shape
        return F.interpolate(y, size=(H * self.scale, W * self.scale), mode="bicubic", align_corners=False)

    def _ddnm_correction(self, x_hat0: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """Apply DDNM null-space correction:
            x̂₀ = A†y + (I - A†A) x̂₀
        """
        AtAx = self._At(self._A(x_hat0))
        return self._At(lr) + x_hat0 - AtAx

    # ------------------------------------------------------------------
    # Reverse diffusion with DDNM
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_x0(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps = self.out_proj(self.unet(x_t, t_tensor))
        sqrt_a = self.sqrt_alphas_cumprod[t]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t]
        return (x_t - sqrt_1ma * eps) / sqrt_a

    @torch.no_grad()
    def _ddpm_step(self, x_t: torch.Tensor, x_hat0: torch.Tensor, t: int) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_t = self.betas[t]

        mean = (
            alpha_prev.sqrt() * beta_t / (1 - alpha_t) * x_hat0
            + (1 - alpha_prev) ** 0.5 * (1 - beta_t) ** 0.5 / (1 - alpha_t) ** 0.5 * x_t
        )
        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + self.posterior_variance[t].sqrt() * noise
        return mean

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Zero-shot SR via DDNM."""
        B, C, H, W = lr.shape
        hr_h, hr_w = H * self.scale, W * self.scale
        x = torch.randn(B, self.out_channels, hr_h, hr_w, device=lr.device)

        for t in reversed(range(self.timesteps)):
            x_hat0 = self._predict_x0(x, t)
            x_hat0 = self._ddnm_correction(x_hat0, lr)
            x_hat0 = x_hat0.clamp(-1, 1)
            x = self._ddpm_step(x, x_hat0, t)

        return x.clamp(0.0, 1.0)

    def forward_train(self, lr: torch.Tensor, hr: torch.Tensor):
        """Optional fine-tuning: standard DDPM noise prediction loss."""
        B = hr.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=hr.device)
        noise = torch.randn_like(hr)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_a * hr + sqrt_1ma * noise
        noise_pred = self.out_proj(self.unet(x_t, t))
        return noise, noise_pred
