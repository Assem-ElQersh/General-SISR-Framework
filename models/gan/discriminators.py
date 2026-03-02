"""Discriminator architectures for GAN-based SR.

Implementations:
    VGGDiscriminator  — Used in SRGAN / ESRGAN (patch-level realism)
    UNetDiscriminator — Used in Real-ESRGAN (pixel-level realism with spectral norm)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    use_bn: bool = True,
    spectral_norm: bool = False,
) -> nn.Sequential:
    conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=not use_bn)
    if spectral_norm:
        conv = nn.utils.spectral_norm(conv)
    layers: List[nn.Module] = [conv]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VGG-style discriminator (SRGAN / ESRGAN)
# ---------------------------------------------------------------------------

class VGGDiscriminator(nn.Module):
    """VGG-style discriminator.

    Processes HR patches and returns a scalar (or feature map) realism score.
    """

    def __init__(self, in_channels: int = 3, base_features: int = 64) -> None:
        super().__init__()
        f = base_features
        self.features = nn.Sequential(
            # No BN for the first layer
            nn.Conv2d(in_channels, f, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _conv_block(f, f, stride=2),
            _conv_block(f, f * 2),
            _conv_block(f * 2, f * 2, stride=2),
            _conv_block(f * 2, f * 4),
            _conv_block(f * 4, f * 4, stride=2),
            _conv_block(f * 4, f * 8),
            _conv_block(f * 8, f * 8, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(f * 8 * 6 * 6, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# U-Net discriminator with spectral normalization (Real-ESRGAN)
# ---------------------------------------------------------------------------

class UNetDiscriminator(nn.Module):
    """U-Net discriminator with spectral norm for pixel-level discrimination.

    Returns a spatial map of realism scores — better for high-frequency details.
    """

    def __init__(self, in_channels: int = 3, base_features: int = 64) -> None:
        super().__init__()
        f = base_features
        sn = nn.utils.spectral_norm

        # Encoder
        self.enc1 = nn.Sequential(sn(nn.Conv2d(in_channels, f, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(sn(nn.Conv2d(f, f * 2, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(sn(nn.Conv2d(f * 2, f * 4, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(sn(nn.Conv2d(f * 4, f * 8, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True))

        # Decoder
        self.dec3 = nn.Sequential(sn(nn.Conv2d(f * 8 + f * 4, f * 4, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.dec2 = nn.Sequential(sn(nn.Conv2d(f * 4 + f * 2, f * 2, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.dec1 = nn.Sequential(sn(nn.Conv2d(f * 2 + f, f, 3, padding=1)), nn.LeakyReLU(0.2, True))

        self.out = sn(nn.Conv2d(f, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))

        return self.out(d1)
