"""Restormer — Efficient Transformer for High-Resolution Image Restoration.

Zamir et al., CVPR 2022.

Problem with standard transformers for SR:
    Self-attention complexity is O(N²) where N = H × W.
    For high-resolution images (e.g., 512×512), this is prohibitive.

Restormer's solution — Transposed Multi-Head Attention (MDTA):
    Compute attention across the channel dimension instead of spatial.
    Complexity: O(C² × H × W) instead of O(H² × W²).
    Since C << H*W for typical feature maps, this is much cheaper.

Gated-Dconv Feed-Forward Network (GDFN):
    Replaces standard FFN with depthwise conv + gating:
    GDFN(x) = Gate(DWConv(Linear(x))) × Linear(x)
    This encodes local context while the MDTA captures global context.

Architecture:
    LR → Shallow Conv (3×3)
       → Encoder: [transformer_block × depth_i + downsample] × num_enc_levels
       → Bottleneck transformer blocks
       → Decoder: [upsample + transformer_block × depth_i] × num_enc_levels
       → Conv after body
       → PixelShuffle upsampler
       → Output Conv
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseSRModel
from models.registry import register_model
from models.cnn.edsr import make_upsampler


# ---------------------------------------------------------------------------
# MDTA — Multi-Dconv Head Transposed Attention
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    """Transposed Multi-Head Attention (channel-wise)."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, heads, head_dim, H*W)
        q = q.reshape(B, self.num_heads, -1, H * W)
        k = k.reshape(B, self.num_heads, -1, H * W)
        v = v.reshape(B, self.num_heads, -1, H * W)

        # Normalize across spatial dimension
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: (head_dim, head_dim) matrix
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, -1, H, W)
        return self.proj(out)


# ---------------------------------------------------------------------------
# GDFN — Gated-Dconv Feed-Forward Network
# ---------------------------------------------------------------------------

class GDFN(nn.Module):
    """Gated feed-forward with depthwise conv."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(self.proj_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Restormer transformer block: LayerNorm → MDTA → LayerNorm → GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # applied as channel-wise
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def _norm(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._norm(self.norm1, x))
        x = x + self.ffn(self._norm(self.norm2, x))
        return x


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Restormer model
# ---------------------------------------------------------------------------

@register_model("restormer")
class Restormer(BaseSRModel):
    """Restormer: Efficient Transformer for High-Resolution Image Restoration.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        dim: Feature dimension (default 48).
        num_blocks: Transformer blocks per level (default [4, 6, 6, 8]).
        num_refinement_blocks: Bottleneck blocks (default 4).
        heads: Attention heads per level (default [1, 2, 4, 8]).
        ffn_expansion_factor: FFN expansion ratio (default 2.66).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: List[int] = None,
        num_refinement_blocks: int = 4,
        heads: List[int] = None,
        ffn_expansion_factor: float = 2.66,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1, bias=False)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck
        self.latent = nn.Sequential(*[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor) for _ in range(num_blocks[3])])

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor) for _ in range(num_refinement_blocks)])

        self.upsample = make_upsampler(dim * 2, scale)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=False)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        inp = self.patch_embed(lr)

        out_enc1 = self.encoder_level1(inp)
        inp_enc2 = self.down1_2(out_enc1)

        out_enc2 = self.encoder_level2(inp_enc2)
        inp_enc3 = self.down2_3(out_enc2)

        out_enc3 = self.encoder_level3(inp_enc3)
        inp_enc4 = self.down3_4(out_enc3)

        latent = self.latent(inp_enc4)

        inp_dec3 = self.up4_3(latent)
        inp_dec3 = self.reduce_chan_level3(torch.cat([inp_dec3, out_enc3], dim=1))
        out_dec3 = self.decoder_level3(inp_dec3)

        inp_dec2 = self.up3_2(out_dec3)
        inp_dec2 = self.reduce_chan_level2(torch.cat([inp_dec2, out_enc2], dim=1))
        out_dec2 = self.decoder_level2(inp_dec2)

        inp_dec1 = self.up2_1(out_dec2)
        inp_dec1 = torch.cat([inp_dec1, out_enc1], dim=1)
        out_dec1 = self.decoder_level1(inp_dec1)

        out = self.refinement(out_dec1)
        out = self.upsample(out)
        return self.output(out).clamp(0.0, 1.0)
