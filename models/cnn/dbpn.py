"""DBPN — Deep Back-Projection Networks for Super-Resolution.

Haris et al., CVPR 2018.

Core idea — iterative error feedback:
    Alternating up- and down-projection stages each produce a feature map.
    The error between projections is fed back, forcing the network to correct
    itself iteratively (analogous to iterative back-projection in classical SR).

Architecture:
    LR → initial feature extraction
       → [Up-projection stage → Down-projection stage] × T
       → reconstruction conv
       → SR

Up-projection stage:
    H_t = Deconv(L_{t-1})                (upscale)
    L_t' = Conv(H_t)                     (downscale back)
    e_t = L_{t-1} - L_t'               (residual error)
    H_t = H_t + Deconv(e_t)            (error correction)

Down-projection stage (symmetric):
    L_t = Conv(H_t)                      (downscale)
    H_t' = Deconv(L_t)                  (upscale back)
    e_t = H_t - H_t'                   (residual error)
    L_t = L_t + Conv(e_t)              (error correction)
"""

import torch
import torch.nn as nn

from models.base_model import BaseSRModel
from models.registry import register_model


# ---------------------------------------------------------------------------
# Projection stages
# ---------------------------------------------------------------------------

def _deconv(in_ch: int, out_ch: int, scale: int) -> nn.ConvTranspose2d:
    kernel = 2 * scale - scale % 2
    pad = (kernel - scale) // 2
    return nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=scale, padding=pad)


def _conv_down(in_ch: int, out_ch: int, scale: int) -> nn.Conv2d:
    kernel = 2 * scale - scale % 2
    pad = (kernel - scale) // 2
    return nn.Conv2d(in_ch, out_ch, kernel, stride=scale, padding=pad)


class UpProjection(nn.Module):
    """Up-projection stage with error feedback."""

    def __init__(self, in_ch: int, feat_ch: int, scale: int) -> None:
        super().__init__()
        self.up1 = nn.Sequential(_deconv(in_ch, feat_ch, scale), nn.PReLU(feat_ch))
        self.down = nn.Sequential(_conv_down(feat_ch, in_ch, scale), nn.PReLU(in_ch))
        self.up2 = nn.Sequential(_deconv(in_ch, feat_ch, scale), nn.PReLU(feat_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up1(x)
        l = self.down(h)
        e = x - l
        return h + self.up2(e)


class DownProjection(nn.Module):
    """Down-projection stage with error feedback."""

    def __init__(self, in_ch: int, feat_ch: int, scale: int) -> None:
        super().__init__()
        self.down1 = nn.Sequential(_conv_down(in_ch, feat_ch, scale), nn.PReLU(feat_ch))
        self.up = nn.Sequential(_deconv(feat_ch, in_ch, scale), nn.PReLU(in_ch))
        self.down2 = nn.Sequential(_conv_down(in_ch, feat_ch, scale), nn.PReLU(feat_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.down1(x)
        h = self.up(l)
        e = x - h
        return l + self.down2(e)


# ---------------------------------------------------------------------------
# DBPN model
# ---------------------------------------------------------------------------

@register_model("dbpn")
class DBPN(BaseSRModel):
    """Deep Back-Projection Network.

    Args:
        scale: Upsampling factor.
        in_channels: Input channels.
        out_channels: Output channels.
        num_features: Initial / reconstruction feature width (default 256).
        bp_features: Feature width inside projection stages (default 64).
        num_stages: Number of alternating up/down projection stages T (default 7).
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 256,
        bp_features: int = 64,
        num_stages: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)

        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.PReLU(128),
            nn.Conv2d(128, num_features, 1),
            nn.PReLU(num_features),
        )

        # Back-projection stages
        self.up_stages = nn.ModuleList()
        self.down_stages = nn.ModuleList()
        # First up stage takes num_features; subsequent ones take bp_features
        for t in range(num_stages):
            in_up = num_features if t == 0 else bp_features
            self.up_stages.append(UpProjection(in_up, bp_features, scale))
            if t < num_stages - 1:
                # Concatenation of all HR maps so far → 1×1 conv to bp_features
                self.down_stages.append(
                    nn.Sequential(
                        nn.Conv2d((t + 1) * bp_features, bp_features, 1),
                        nn.PReLU(bp_features),
                        DownProjection(bp_features, bp_features, scale),
                    )
                )

        # Reconstruction from all HR maps
        self.reconstruct = nn.Conv2d(num_stages * bp_features, out_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(lr)

        hr_maps = []
        for t in range(len(self.up_stages)):
            hr = self.up_stages[t](x)
            hr_maps.append(hr)
            if t < len(self.down_stages):
                x = self.down_stages[t](torch.cat(hr_maps, dim=1))

        return self.reconstruct(torch.cat(hr_maps, dim=1)).clamp(0.0, 1.0)
