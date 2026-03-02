"""HAT — Activating More Pixels in Image Super-Resolution Transformer.

Chen et al., CVPR 2023.

HAT addresses the limitation of pure window attention in SwinIR by mixing:
    1. Window self-attention (W-MSA)         — local structure
    2. Channel attention (squeeze-excitation) — inter-channel context
    3. Overlapping Cross-Attention Block (OCAB) — cross-window interaction

HAB (Hybrid Attention Block) = W-MSA + Channel Attention
OCAB (Overlapping Cross-Attention Block) = cross-window Q/K/V attention
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler
from models.transformer.swinir import (
    WindowMSA,
    _window_partition,
    _window_reverse,
)


# ---------------------------------------------------------------------------
# Channel Attention (reintroduced from CNN world)
# ---------------------------------------------------------------------------

class CALayer(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, num_features: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


# ---------------------------------------------------------------------------
# HAB — Hybrid Attention Block
# ---------------------------------------------------------------------------

class HAB(nn.Module):
    """Hybrid Attention Block: Window-MSA + Channel Attention.

    Operates on spatial token sequences (B, H*W, C).
    Channel attention is applied after reshaping back to (B, C, H, W).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 16,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMSA(dim, window_size, num_heads)

        # Channel attention branch
        self.conv_compress = nn.Conv2d(dim, dim // compress_ratio, 3, padding=1)
        self.ca = CALayer(dim // compress_ratio, reduction=squeeze_factor // compress_ratio)
        self.conv_expand = nn.Conv2d(dim // compress_ratio, dim, 3, padding=1)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
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
        mw = _window_partition(mask, self.window_size).view(-1, self.window_size ** 2)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # --- Window attention branch ---
        shortcut = x
        x_norm = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x_norm = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        mask = self._attn_mask(H, W, x.device)
        windows = _window_partition(x_norm, self.window_size).view(-1, self.window_size ** 2, C)
        attn_out = self.attn(windows, mask).view(-1, self.window_size, self.window_size, C)
        x_attn = _window_reverse(attn_out, self.window_size, H, W)
        if self.shift_size > 0:
            x_attn = torch.roll(x_attn, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x_attn.view(B, N, C)

        # --- Channel attention branch (applied on conv feature map) ---
        feat = x.transpose(1, 2).view(B, C, H, W)
        ca = self.conv_expand(self.ca(self.conv_compress(feat)))
        x = x + ca.view(B, C, N).transpose(1, 2)

        # --- MLP ---
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# OCAB — Overlapping Cross-Attention Block
# ---------------------------------------------------------------------------

class OCAB(nn.Module):
    """Overlapping Cross-Attention Block for cross-window interaction.

    Queries come from non-overlapping windows; keys and values come from
    larger, overlapping windows to capture longer-range context.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 16,
        overlap_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * (1 + overlap_ratio))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.unfold = nn.Unfold(
            kernel_size=self.overlap_win_size,
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x)

        # Queries: non-overlapping windows
        feat = x.transpose(1, 2).view(B, C, H, W)
        q_windows = _window_partition(feat.permute(0, 2, 3, 1), self.window_size)
        q = self.q(q_windows.view(-1, self.window_size ** 2, C))

        # Keys/values: overlapping windows via unfold
        kv_feat = self.unfold(feat).view(B, C, self.overlap_win_size ** 2, -1)
        kv_feat = kv_feat.permute(0, 3, 2, 1).contiguous()  # B, nW, ow², C
        nW = kv_feat.shape[1]
        kv = self.kv(kv_feat.view(B * nW, self.overlap_win_size ** 2, C))
        k, v = kv.chunk(2, dim=-1)

        # Attention
        q = q.view(B * nW, self.window_size ** 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B * nW, self.overlap_win_size ** 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B * nW, self.overlap_win_size ** 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nW, self.window_size ** 2, C)
        out = self.proj(out)
        out = _window_reverse(out.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)
        return shortcut + out.view(B, N, C)


# ---------------------------------------------------------------------------
# HAT Block (stack of HAB + one OCAB)
# ---------------------------------------------------------------------------

class HATBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        compress_ratio: int,
        squeeze_factor: int,
        overlap_ratio: float,
    ) -> None:
        super().__init__()
        self.hab_layers = nn.ModuleList([
            HAB(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
            )
            for i in range(depth)
        ])
        self.ocab = OCAB(dim, num_heads, window_size, overlap_ratio)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        for hab in self.hab_layers:
            x = hab(x, H, W)
        x = self.ocab(x, H, W)
        feat = x.transpose(1, 2).view(-1, self.conv.in_channels, H, W)
        feat = self.conv(feat)
        return shortcut + feat.view(-1, self.conv.out_channels, H * W).transpose(1, 2)


# ---------------------------------------------------------------------------
# HAT model
# ---------------------------------------------------------------------------

@register_model("hat")
class HAT(BaseSRModel):
    """Hybrid Attention Transformer for SISR.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        img_size: Expected input size (for padding).
        window_size: Attention window size (default 16).
        embed_dim: Feature dimension.
        depths: Depth of each HAT block.
        num_heads: Attention heads per block.
        mlp_ratio: MLP expansion ratio.
        compress_ratio: Channel compression in channel attention.
        squeeze_factor: SE reduction factor.
        overlap_ratio: Overlap ratio for OCAB.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 64,
        window_size: int = 16,
        embed_dim: int = 144,
        depths: List[int] = None,
        num_heads: List[int] = None,
        mlp_ratio: float = 2.0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        overlap_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if depths is None:
            depths = [6, 6, 6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6, 6, 6]

        self.window_size = window_size
        self.embed_dim = embed_dim

        self.head = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        self.layers = nn.ModuleList([
            HATBlock(
                dim=embed_dim,
                depth=d,
                num_heads=h,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                overlap_ratio=overlap_ratio,
            )
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

        feat = self.head(x)
        x_seq = feat.flatten(2).transpose(1, 2)

        for layer in self.layers:
            x_seq = layer(x_seq, H, W)

        x_seq = self.norm(x_seq)
        feat2 = x_seq.transpose(1, 2).view(-1, self.embed_dim, H, W)
        feat2 = self.conv_after_body(feat2) + feat
        feat2 = feat2[:, :, :H_orig, :W_orig]

        out = self.upsample(feat2)
        return self.tail(out).clamp(0.0, 1.0)
