"""Pixel-domain reconstruction losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelLoss(nn.Module):
    """L1 or L2 pixel loss.

    Args:
        loss_type: "L1" or "L2".
        weight: Loss weight in the composite objective.
        reduction: "mean" or "sum".
    """

    def __init__(
        self,
        loss_type: str = "L1",
        weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if loss_type.upper() not in ("L1", "L2"):
            raise ValueError(f"loss_type must be 'L1' or 'L2', got '{loss_type}'")
        self.loss_type = loss_type.upper()
        self.weight = weight
        self.reduction = reduction

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "L1":
            loss = F.l1_loss(sr, hr, reduction=self.reduction)
        else:
            loss = F.mse_loss(sr, hr, reduction=self.reduction)
        return self.weight * loss


class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1 / pseudo-Huber) loss.

    L = sqrt( (sr - hr)^2 + ε^2 ) − ε

    More robust to outliers than plain L1/L2.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-3) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        diff = sr - hr
        loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()
        return self.weight * loss
