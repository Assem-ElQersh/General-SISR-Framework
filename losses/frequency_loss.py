"""Frequency domain loss (FFT-based spectral loss).

Encourages the SR network to match the frequency spectrum of the HR image,
which is complementary to pixel-domain losses.  High-frequency components
(edges, textures) are explicitly penalized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLoss(nn.Module):
    """L1 loss in the Fourier frequency domain.

    Args:
        weight: Loss weight in composite objective.
        loss_type: "L1" or "L2" for the frequency domain comparison.
    """

    def __init__(self, weight: float = 0.1, loss_type: str = "L1") -> None:
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type.upper()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_fft = torch.fft.rfft2(sr, norm="ortho")
        hr_fft = torch.fft.rfft2(hr, norm="ortho")

        sr_mag = torch.abs(sr_fft)
        hr_mag = torch.abs(hr_fft)

        if self.loss_type == "L1":
            loss = F.l1_loss(sr_mag, hr_mag)
        else:
            loss = F.mse_loss(sr_mag, hr_mag)

        return self.weight * loss
