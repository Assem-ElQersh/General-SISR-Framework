"""SR3 — Image Super-Resolution via Iterative Refinement.

Saharia et al., IEEE TPAMI 2022.

Core idea:
    Treat SR as conditional image generation.
    A U-Net denoising model is conditioned on the LR image.
    At inference, start from Gaussian noise and iteratively denoise
    toward a plausible HR image guided by the LR.

DDPM formulation:
    Forward process: q(x_t | x_{t-1}) = N(sqrt(1-β_t)·x_{t-1}, β_t·I)
    Reverse process: p_θ(x_{t-1} | x_t, y) = N(μ_θ(x_t, y, t), Σ_t)
    where y = bicubic(LR) (the conditioning signal)

The U-Net input at training time: concatenate [x_t, y] on the channel dim.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1)
    )
    emb = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, model_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        self.model_channels = model_channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = _sinusoidal_embedding(t, self.model_channels)
        return self.net(emb)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32 if in_ch >= 32 else in_ch, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32 if out_ch >= 32 else out_ch, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention at low spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32 if channels >= 32 else channels, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        scale = C ** -0.5
        attn = (q.transpose(-1, -2) @ k * scale).softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# U-Net denoising backbone
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        channel_mult: List[int],
        num_res_blocks: int,
        attention_resolutions: List[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        time_emb_dim = model_channels * 4
        self.time_emb = TimeEmbedding(model_channels)

        self.inp_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        down_ch = model_channels
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch_list = [down_ch]
        current_res = 256  # nominal starting resolution

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(down_ch, out_ch, time_emb_dim, dropout))
                if current_res in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(out_ch))
                down_ch = out_ch
                ch_list.append(down_ch)
            if i < len(channel_mult) - 1:
                self.down_samples.append(Downsample(down_ch))
                ch_list.append(down_ch)
                current_res //= 2

        self.mid = nn.ModuleList([
            ResBlock(down_ch, down_ch, time_emb_dim, dropout),
            AttentionBlock(down_ch),
            ResBlock(down_ch, down_ch, time_emb_dim, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = ch_list.pop()
                self.up_blocks.append(ResBlock(down_ch + skip_ch, out_ch, time_emb_dim, dropout))
                if current_res in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(out_ch))
                down_ch = out_ch
            if i > 0:
                self.up_samples.append(Upsample(down_ch))
                current_res *= 2

        self.out_norm = nn.GroupNorm(32 if down_ch >= 32 else down_ch, down_ch)
        self.out_conv = nn.Conv2d(down_ch, model_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        h = self.inp_conv(x)
        skips = [h]

        down_block_idx = 0
        sample_idx = 0
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            skips.append(h)
            down_block_idx += 1

        for module in self.down_samples:
            h = module(h)
            skips.append(h)

        for module in self.mid:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)

        up_block_idx = 0
        up_sample_idx = 0
        for module in self.up_blocks:
            if isinstance(module, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        if len(self.up_samples) > up_sample_idx:
            h = self.up_samples[up_sample_idx](h)
            up_sample_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

def _linear_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_schedule(timesteps: int) -> torch.Tensor:
    s = 0.008
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return betas.clamp(0.0001, 0.9999)


# ---------------------------------------------------------------------------
# SR3 model
# ---------------------------------------------------------------------------

@register_model("sr3")
class SR3(BaseSRModel):
    """SR3 — Image Super-Resolution via Iterative Refinement.

    Args:
        scale: SR upsampling factor.
        in_channels: LR image channels.
        out_channels: HR image channels.
        model_channels: Base U-Net channel width.
        channel_mult: Channel multipliers per resolution level.
        num_res_blocks: ResBlocks per level.
        attention_resolutions: Spatial resolutions where attention is applied.
        dropout: Dropout probability in ResBlocks.
        timesteps: Total diffusion steps.
        noise_schedule: "linear" or "cosine".
        beta_start: Starting β for linear schedule.
        beta_end: Ending β for linear schedule.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mult: List[int] = None,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        dropout: float = 0.2,
        timesteps: int = 2000,
        noise_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8]
        if attention_resolutions is None:
            attention_resolutions = [16, 8]

        self.timesteps = timesteps

        # U-Net input: concatenate noisy HR + bicubic(LR) → 2*out_channels
        self.unet = UNet(
            in_channels=out_channels * 2,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
        self.out_proj = nn.Conv2d(model_channels, out_channels, 1)

        # Register diffusion buffers
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
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample x_t from x_0 using the closed-form diffusion formula."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_a * x_start + sqrt_1ma * noise

    def forward_train(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random t, add noise to HR, predict noise.

        Returns:
            noise: Ground-truth noise ε
            noise_pred: Predicted noise ε_θ(x_t, y, t)
        """
        B = hr.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=hr.device)
        noise = torch.randn_like(hr)
        x_t = self.q_sample(hr, t, noise)
        y = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        inp = torch.cat([x_t, y], dim=1)
        noise_pred = self.out_proj(self.unet(inp, t))
        return noise, noise_pred

    # ------------------------------------------------------------------
    # Inference (DDPM reverse)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self, x_t: torch.Tensor, t: int, y: torch.Tensor
    ) -> torch.Tensor:
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        inp = torch.cat([x_t, y], dim=1)
        eps = self.out_proj(self.unet(inp, t_tensor))

        beta_t = self.betas[t]
        sqrt_recip = (1.0 / self.alphas_cumprod[t]).sqrt()
        sqrt_recip_m1 = (1.0 / self.alphas_cumprod[t] - 1).sqrt()
        mean = sqrt_recip * (x_t - beta_t / self.sqrt_one_minus_alphas_cumprod[t] * eps)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            return mean + variance.sqrt() * noise
        return mean

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Full DDPM reverse chain for inference."""
        B, C, H, W = lr.shape
        hr_h, hr_w = H * self.scale, W * self.scale
        y = F.interpolate(lr, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
        x = torch.randn(B, self.out_channels, hr_h, hr_w, device=lr.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, y)
        return x.clamp(0.0, 1.0)
