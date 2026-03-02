"""MambaIR — A Simple Baseline for Image Restoration with State Space Model.

Guo et al., ECCV 2024.

Core idea:
    Replace self-attention (O(N²)) with a Selective State Space Model (SSM) scan (O(N)).
    Process image feature sequences in 4 directions (→ ← ↑ ↓) to capture
    global context with linear complexity.

SSM formulation (continuous):
    h'(t) = A h(t) + B x(t)
    y(t)  = C h(t)

Discretized (S4 / Mamba):
    h_t = Ā h_{t-1} + B̄ x_t
    y_t = C h_t + D x_t

Mamba adds input-dependent (selective) B, C, and Δ — making it data-dependent.

Architecture:
    LR → shallow Conv
       → [VSS Block × depth] × num_stages
       → Conv after body
       → PixelShuffle upsampler
       → Output Conv

VSS Block (Visual State Space Block):
    LayerNorm → VSS Module → skip
    VSS Module:
        Linear → SiLU (gate branch)
        Linear → Conv(dw) → SiLU → SSM scan → LayerNorm (feature branch)
        gate * feature → Linear → output

Note on mamba-ssm package:
    If `mamba-ssm` is installed, uses the optimized CUDA kernels.
    Otherwise falls back to a pure-PyTorch selective scan implementation.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# Pure-PyTorch selective scan (Mamba fallback)
# ---------------------------------------------------------------------------

def selective_scan_pytorch(
    u: torch.Tensor,    # (B, L, D)
    delta: torch.Tensor,  # (B, L, D)
    A: torch.Tensor,    # (D, N) — log A
    B: torch.Tensor,    # (B, L, N)
    C: torch.Tensor,    # (B, L, N)
    D: torch.Tensor,    # (D,)
) -> torch.Tensor:
    """Pure-PyTorch selective scan (simplified, for fallback use).

    This is an O(L) sequential scan; the CUDA kernel in mamba-ssm is faster.
    """
    B_size, L, D_size = u.shape
    N = A.shape[1]

    # Discretize A and B using ZOH with delta
    A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

    # Sequential scan
    h = torch.zeros(B_size, D_size, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t] * u[:, t, :].unsqueeze(-1)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, D)
    return y + u * D.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Mamba SSM layer
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """Single Mamba layer (selective SSM scan in one direction).

    Uses mamba-ssm CUDA kernels if available; falls back to pure PyTorch.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(-A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self._use_mamba_ssm = self._check_mamba_ssm()

    @staticmethod
    def _check_mamba_ssm() -> bool:
        try:
            import mamba_ssm  # noqa
            return True
        except ImportError:
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Depthwise conv
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)

        # SSM
        x_dbl = self.x_proj(x_conv)
        dt = x_dbl[:, :, : self.d_inner]
        B_ssm = x_dbl[:, :, self.d_inner : self.d_inner + self.d_state]
        C_ssm = x_dbl[:, :, self.d_inner + self.d_state :]
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        if self._use_mamba_ssm:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                y = selective_scan_fn(
                    x_conv.contiguous(), dt.contiguous(), A.contiguous(),
                    B_ssm.contiguous(), C_ssm.contiguous(), self.D.float(), z=None,
                    delta_bias=None, delta_softplus=False, return_last_state=False,
                )
            except Exception:
                y = selective_scan_pytorch(x_conv, dt, A, B_ssm, C_ssm, self.D)
        else:
            y = selective_scan_pytorch(x_conv, dt, A, B_ssm, C_ssm, self.D)

        y = y * self.act(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# VSS Block — 4-direction scan
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    """Visual State Space Block: applies Mamba scan in 4 directions."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 4 directional SSM layers
        self.ssm_h = MambaLayer(dim, d_state)   # horizontal →
        self.ssm_h_r = MambaLayer(dim, d_state) # horizontal ←
        self.ssm_v = MambaLayer(dim, d_state)   # vertical ↓
        self.ssm_v_r = MambaLayer(dim, d_state) # vertical ↑
        self.proj = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, dim)"""
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x).view(B, H, W, C)

        # Flatten in 4 directions
        seq_h = x.view(B, H * W, C)
        seq_h_r = x.flip(2).view(B, H * W, C)
        seq_v = x.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        seq_v_r = x.permute(0, 2, 1, 3).flip(1).contiguous().view(B, H * W, C)

        out = torch.cat([
            self.ssm_h(seq_h),
            self.ssm_h_r(seq_h_r).view(B, H, W, C).flip(2).view(B, H * W, C),
            self.ssm_v(seq_v).view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, H * W, C),
            self.ssm_v_r(seq_v_r).view(B, W, H, C).flip(1).permute(0, 2, 1, 3).contiguous().view(B, H * W, C),
        ], dim=-1)

        return shortcut + self.proj(out)


# ---------------------------------------------------------------------------
# MambaIR Stage
# ---------------------------------------------------------------------------

class MambaIRStage(nn.Module):
    """A stage of MambaIR: depth × VSSBlock + Conv + residual."""

    def __init__(self, dim: int, depth: int, d_state: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock(dim, d_state) for _ in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        for block in self.blocks:
            x = block(x, H, W)
        feat = x.transpose(1, 2).view(-1, self.conv.in_channels, H, W)
        feat = self.conv(feat)
        return shortcut + feat.view(-1, self.conv.out_channels, H * W).transpose(1, 2)


# ---------------------------------------------------------------------------
# MambaIR model
# ---------------------------------------------------------------------------

@register_model("mambair")
class MambaIR(BaseSRModel):
    """MambaIR — Image Restoration with State Space Models.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        embed_dim: Feature dimension (default 64).
        depths: Depth per stage (default [6, 6, 6, 6]).
        d_state: SSM state dimension N (default 16).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        depths: List[int] = None,
        d_state: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if depths is None:
            depths = [6, 6, 6, 6]

        self.embed_dim = embed_dim
        self.head = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.stages = nn.ModuleList([
            MambaIRStage(embed_dim, d, d_state) for d in depths
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = make_upsampler(embed_dim, scale)
        self.tail = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        _, _, H, W = lr.shape
        feat = self.head(lr)
        x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for stage in self.stages:
            x = stage(x, H, W)

        x = self.norm(x)
        feat2 = x.transpose(1, 2).view(-1, self.embed_dim, H, W)
        feat2 = self.conv_after_body(feat2) + feat
        out = self.upsample(feat2)
        return self.tail(out).clamp(0.0, 1.0)
