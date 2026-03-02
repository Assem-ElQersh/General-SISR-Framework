"""SwinIR — Image Restoration Using Swin Transformer.

Liang et al., ICCV 2021.

Architecture:
    1. Shallow feature extraction:  Conv(3×3)
    2. Deep feature extraction:     N × RSTB (Residual Swin Transformer Block)
        Each RSTB = L × STL (Swin Transformer Layer) + Conv + residual
        Each STL  = W-MSA (or SW-MSA) + LayerNorm + MLP + LayerNorm
    3. HQ image reconstruction:    Conv + PixelShuffle
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) → (num_windows*B, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class WindowMSA(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))  # 2, ws, ws
        coords_flat = grid.flatten(1)  # 2, ws²
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, ws², ws²
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1
        ).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer Layer (STL)
# ---------------------------------------------------------------------------

class SwinTransformerLayer(nn.Module):
    """One Swin Transformer Layer with W-MSA or SW-MSA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMSA(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        mask = torch.zeros(1, H, W, 1, device=device)
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in slices:
            for w in slices:
                mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = _window_partition(mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x).view(-1, H, W, self.dim)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        mask = self._compute_attn_mask(H, W, x.device)
        windows = _window_partition(x, self.window_size).view(-1, self.window_size ** 2, self.dim)
        attn_out = self.attn(windows, mask).view(-1, self.window_size, self.window_size, self.dim)
        x = _window_reverse(attn_out, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(-1, H * W, self.dim)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Residual Swin Transformer Block (RSTB)
# ---------------------------------------------------------------------------

class RSTB(nn.Module):
    """Residual Swin Transformer Block: L×STL + Conv + residual."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.conv(x.transpose(1, 2).view(-1, self.conv.in_channels, H, W))
        return shortcut + x.view(-1, self.conv.out_channels, H * W).transpose(1, 2)


# ---------------------------------------------------------------------------
# SwinIR model
# ---------------------------------------------------------------------------

@register_model("swinir")
class SwinIR(BaseSRModel):
    """SwinIR: Image Restoration Using Swin Transformer.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        img_size: Expected input spatial size (for padding alignment).
        window_size: Attention window size.
        embed_dim: Embedding / feature dimension.
        depths: Number of STL per RSTB.
        num_heads: Attention heads per RSTB.
        mlp_ratio: MLP expansion ratio.
        upsampler: "pixelshuffle" or "nearest+conv".
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 64,
        window_size: int = 8,
        embed_dim: int = 60,
        depths: List[int] = None,
        num_heads: List[int] = None,
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffle",
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if depths is None:
            depths = [6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6]

        self.window_size = window_size
        self.embed_dim = embed_dim

        self.head = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.pos_drop = nn.Dropout(0.0)

        self.layers = nn.ModuleList([
            RSTB(embed_dim, d, h, window_size, mlp_ratio)
            for d, h in zip(depths, num_heads)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        self.upsample = make_upsampler(embed_dim, scale)
        self.tail = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x, H_orig, W_orig = self._pad(lr)
        _, _, H, W = x.shape

        feat = self.head(x)  # (B, C, H, W)
        x_seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_seq = self.pos_drop(x_seq)

        for layer in self.layers:
            x_seq = layer(x_seq, H, W)

        x_seq = self.norm(x_seq)
        feat2 = x_seq.transpose(1, 2).view(-1, self.embed_dim, H, W)
        feat2 = self.conv_after_body(feat2) + feat
        feat2 = feat2[:, :, :H_orig, :W_orig]

        out = self.upsample(feat2)
        return self.tail(out).clamp(0.0, 1.0)
